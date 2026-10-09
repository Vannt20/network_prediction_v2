"""
Soft Actor-Critic cho RL-Gate (spec Mục 6.3). Mỗi luồng là một tác tử, policy và critic dùng chung.

Bộ đệm phát lại chỉ lưu (t, a_{t-1}, u_t, r_t): trạng thái là hàm tất định của (t, a_{t-1}) và cache,
nên đặc trưng được dựng lại khi lấy mẫu (không lưu tensor trạng thái đầy đủ).
"""
import copy
import time
import torch
import torch.nn.functional as F

from Graph_models.rl_gate.policy import GateEncoder, Actor, Critic, mix

DEFAULTS = {
    'gamma': 0.9, 'tau': 0.005, 'lr': 3e-4, 'batch_t': 64, 'n_envs': 16, 'ep_len': 256,
    'replay_t': 20000, 'd': 64, 'd_t': 32, 'H': 12, 'use_gnn': True, 'use_temporal': True,
}


def build_actor(K, F_static, graphs, cfg):
    enc = GateEncoder(K, F_static, cfg['H'], graphs if cfg['use_gnn'] else (), cfg['d_t'], cfg['d'],
                      cfg['use_gnn'], cfg['use_temporal'])
    return Actor(enc, cfg['d'])


class SAC:
    def __init__(self, K, F_static, graphs, cfg, device):
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        self.device = device
        self.actor = build_actor(K, F_static, graphs, c).to(device)
        enc_c = GateEncoder(K, F_static, c['H'], graphs if c['use_gnn'] else (), c['d_t'], c['d'],
                            c['use_gnn'], c['use_temporal'])
        self.critic = Critic(enc_c, c['d']).to(device)
        self.critic_t = copy.deepcopy(self.critic)
        for p in self.critic_t.parameters():
            p.requires_grad_(False)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=c['lr'])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=c['lr'])
        self.log_alpha = torch.tensor(-2.0, device=device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=c['lr'])
        self.target_ent = -K / 2.0
        self.K = K

    def set_norm(self, stats):
        for m in (self.actor.enc, self.critic.enc, self.critic_t.enc):
            m.set_norm(stats)

    def load_actor_critic(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_t.load_state_dict(state['critic'])
        self.log_alpha.data.fill_(float(state.get('log_alpha', -2.0)))

    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                'log_alpha': float(self.log_alpha.detach()), 'cfg': self.cfg}

    def train(self, task, n_updates, max_minutes=None, seed=0, log_every=250, verbose=True):
        """Huấn luyện trên một GateTask. Trả về lịch sử (list dict)."""
        c, dev, d = self.cfg, self.device, task.data
        g = torch.Generator(device='cpu').manual_seed(seed)
        T, N, K = d.T, d.N, self.K
        ep_len = max(2, min(c['ep_len'], T - 1))
        cap = min(c['replay_t'], n_updates * c['n_envs'] + c['n_envs'])
        buf_t = torch.zeros(cap, dtype=torch.long, device=dev)
        buf_ap = torch.zeros(cap, N, K, dtype=torch.float16, device=dev)
        buf_u = torch.zeros(cap, N, K, dtype=torch.float16, device=dev)
        buf_r = torch.zeros(cap, N, device=dev)
        ptr, size = 0, 0

        def new_start(n):
            return torch.randint(0, max(1, T - ep_len), (n,), generator=g).to(dev)
        ts = new_start(c['n_envs'])
        t_end = ts + ep_len
        a_prev = task.prior[ts].clone()
        hist, t0, alpha = [], time.time(), self.log_alpha.exp().item()
        acc = {'q': 0.0, 'pi': 0.0, 'r': 0.0, 'n': 0}
        for it in range(1, n_updates + 1):
            # 1) Tương tác: một bước cho mỗi episode song song
            with torch.no_grad():
                u, _ = self.actor(*task.state(ts, a_prev), with_logp=False)
                a = mix(task.logp[ts], u)
                r = task.reward(ts, a, a_prev)
            n = len(ts)
            idx = (torch.arange(n, device=dev) + ptr) % cap
            buf_t[idx], buf_ap[idx], buf_u[idx], buf_r[idx] = ts, a_prev.half(), u.half(), r
            ptr, size = (ptr + n) % cap, min(cap, size + n)
            acc['r'] += float(r.mean())
            ts, a_prev = ts + 1, a
            done = (ts >= t_end) | (ts >= T - 1)
            if done.any():
                k = int(done.sum())
                ts[done] = new_start(k)
                t_end[done] = ts[done] + ep_len
                a_prev[done] = task.prior[ts[done]]

            # 2) Cập nhật
            if size >= c['batch_t']:
                bi = torch.randint(0, size, (c['batch_t'],), generator=g).to(dev)
                bt, bap, bu, br = buf_t[bi], buf_ap[bi].float(), buf_u[bi].float(), buf_r[bi]
                ba = mix(task.logp[bt], bu)                                     # a_t = trạng thái tiếp theo
                bt1 = (bt + 1).clamp(max=T - 1)
                notdone = (bt + 1 < T).float()[:, None]
                with torch.no_grad():
                    s1 = task.state(bt1, ba)
                    u1, lp1 = self.actor(*s1)
                    q1t, q2t = self.critic_t(*s1, u1)
                    target = br + c['gamma'] * notdone * (torch.min(q1t, q2t) - alpha * lp1)
                s0 = task.state(bt, bap)
                q1, q2 = self.critic(*s0, bu)
                loss_q = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
                self.opt_c.zero_grad(set_to_none=True)
                loss_q.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.opt_c.step()

                un, lpn = self.actor(*s0)
                q1n, q2n = self.critic(*s0, un)
                loss_pi = (alpha * lpn - torch.min(q1n, q2n)).mean()
                self.opt_a.zero_grad(set_to_none=True)
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.opt_a.step()

                loss_al = -(self.log_alpha * (lpn.detach() + self.target_ent)).mean()
                self.opt_alpha.zero_grad(set_to_none=True)
                loss_al.backward()
                self.opt_alpha.step()
                alpha = self.log_alpha.exp().item()

                with torch.no_grad():
                    for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                        pt.mul_(1 - c['tau']).add_(c['tau'] * p)
                acc['q'] += float(loss_q.detach())
                acc['pi'] += float(loss_pi.detach())
            acc['n'] += 1
            if it % log_every == 0 or it == n_updates:
                rec = {'update': it, 'loss_q': acc['q'] / acc['n'], 'loss_pi': acc['pi'] / acc['n'],
                       'reward': acc['r'] / acc['n'], 'alpha': alpha, 'minutes': (time.time() - t0) / 60}
                hist.append(rec)
                if verbose:
                    print(f"      SAC {it:5d}/{n_updates} | Q {rec['loss_q']:.4f} | π {rec['loss_pi']:.4f} | "
                          f"r {rec['reward']:+.4f} | α {alpha:.4f} | {rec['minutes']:.1f} phút", flush=True)
                acc = {'q': 0.0, 'pi': 0.0, 'r': 0.0, 'n': 0}
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                if verbose:
                    print(f"      [CẮT THỜI GIAN] SAC dừng ở cập nhật {it}", flush=True)
                hist.append({'update': it, 'time_capped': True})
                break
        return hist
