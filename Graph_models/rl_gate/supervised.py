"""
Gate giám sát (gamma = 0) - đối chứng bắt buộc của RL-Gate (spec Mục 6.3).

Cùng kiến trúc actor, hành động tất định u = c * tanh(mu), a_{t-1} cố định bằng w_static (không có thành phần
tuần tự), tối thiểu hóa trực tiếp  SE/c + λ_KL KL + λ_floor sàn  bằng lan truyền ngược.
Dừng sớm theo 20% cuối của đoạn huấn luyện.
"""
import copy
import time
import torch

from Graph_models.rl_gate.policy import mix


def train_supervised(actor, task, n_updates=2000, lr=1e-3, batch_t=64, eval_every=100, patience=5,
                     max_minutes=None, seed=0, verbose=True):
    d = task.data
    dev = d.device
    g = torch.Generator(device='cpu').manual_seed(seed)
    cut = max(1, int(d.T * 0.8))
    es_t = torch.arange(cut, d.T, device=dev)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    best, best_state, wait, hist, t0 = float('inf'), copy.deepcopy(actor.state_dict()), 0, [], time.time()

    def loss_on(ts):
        a_prev = task.w[None].expand(len(ts), -1, -1)
        u, _ = actor(*task.state(ts, a_prev), deterministic=True, with_logp=False)
        a = mix(task.logw, u)
        se = ((a * d.P[ts]).sum(-1) - d.y[ts]) ** 2
        return (se / task.c + task.penalties(a, a_prev)).mean(), se.mean()

    for it in range(1, n_updates + 1):
        actor.train()
        ts = torch.randint(0, cut, (batch_t,), generator=g).to(dev)
        loss, _ = loss_on(ts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        opt.step()
        if (it % eval_every == 0 or it == n_updates) and len(es_t):
            actor.eval()
            with torch.no_grad():
                mses = [loss_on(es_t[i:i + 256])[1] * len(es_t[i:i + 256]) for i in range(0, len(es_t), 256)]
                v = float(sum(mses) / len(es_t))
            hist.append({'update': it, 'loss': float(loss.detach()), 'es_mse': v, 'minutes': (time.time() - t0) / 60})
            if v < best:
                best, best_state, wait = v, copy.deepcopy(actor.state_dict()), 0
            else:
                wait += 1
            if verbose:
                print(f"      SUP {it:5d}/{n_updates} | loss {float(loss.detach()):.4f} | ES MSE {v*1e3:.4f}e-3", flush=True)
            if wait >= patience:
                break
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            hist.append({'update': it, 'time_capped': True})
            break
    actor.load_state_dict(best_state)
    actor.eval()
    return hist
