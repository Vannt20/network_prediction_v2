"""
Huấn luyện RL-Gate (spec Mục 6.4 và 15) cho một dataset.

Prior (vòng 2): Hedge + sàn trên tập nhánh, tham số (eta, beta, alpha) chọn theo Huber trên đúng đoạn dùng để
học; alpha = 1 tương đương static. Policy học phần lệch u quanh prior: a_t = softmax(log p_t + u_t), nên khi u = 0
RL-Gate trùng prior. Biến thể static_prior dùng lại prior tĩnh của vòng 1 để đo phần đóng góp của prior Hedge.

  1. Tiền huấn luyện trên OOF (cache/v4*/{ds}_oof.pt), stacking và tham số prior fit trên chính OOF. Một lần / dataset.
  2. Lưới 4 cấu hình (λ_sw ∈ {0, 0.1} × f ∈ {0, 0.2}, λ_KL = 0.01) bằng CV 2 khối trên Val của run_0:
     tinh chỉnh trên khối A (stacking và prior fit trên A), chấm trên khối B, rồi đổi vai. Chọn theo Huber.
     Điểm CV từng khối (MSE và Huber) dùng trong quy tắc chọn tầng kết hợp. Chỉ chạy cho algo=sac, variant=main;
     gate giám sát (algo=sup) tính điểm CV với cấu hình đã chọn (cv.json) để tham khảo.
  3. Mỗi run: stacking và prior fit trên Val của run, tinh chỉnh trên toàn Val, chạy tuần tự qua Test
     (prior trên Test dùng tham số chọn trên Val, Hedge khởi động lại ở đầu Test).

algo: sac (RL-Gate) | sup (gate giám sát gamma = 0)
variant (ablation): main | no_gnn | no_floor | no_oof | no_temporal | no_kl | no_sigma | static_prior
Log: logs/gnn_rl*/rl_{algo}[_{variant}]_data_{ds}/{pretrain/, grid.json, run_r/}
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.gnn_rl_common import (setup_utf8, parse_list, ALL_DATASETS, EXTRAP_BRANCHES, load_cache, split_arrays,
                                    torch_load, oof_file, run_dir, save_json, load_json, save_combiner_result,
                                    set_seed, get_device)
from training.run_hedge import rl_branches, val_blocks, _fit_weights_selected, block_scores
from Graph_models.online_hedge import select_prior, prior_weights
from training.p1_decision import decision_file
from features.feature_store import load_raw_dataset
from features.graph_builder import build_flow_graphs, resolve_topology
from Graph_models.robust_stacking import RobustPerFlowStacking
from Graph_models.rl_gate.features import split_from_cache
from Graph_models.rl_gate.env import GateTask
from Graph_models.rl_gate.sac import SAC, build_actor, DEFAULTS
from Graph_models.rl_gate.supervised import train_supervised
from baselines_ml.run_ml_baselines import parse_run_ids

VARIANTS = ['main', 'no_gnn', 'no_floor', 'no_oof', 'no_temporal', 'no_kl', 'no_sigma', 'static_prior']
GRID = [{'lam_sw': s, 'floor': f} for s in (0.0, 0.1) for f in (0.0, 0.2)]
REWARD_DEFAULT = {'lam_sw': 0.1, 'lam_kl': 0.01, 'floor': 0.2, 'lam_floor': 1.0}
BUDGET = {'pretrain': 3000, 'finetune': 1000, 'grid_finetune': 500, 'sup': 2000,
          'max_min_pretrain': 10, 'max_min_finetune': 4, 'max_min_grid': 3, 'max_min_sup': 3}


def tag_of(algo, variant):
    return f"rl_{algo}" + ('' if variant == 'main' else f"_{variant}")


def variant_cfg(variant, reward):
    net = {'use_gnn': variant != 'no_gnn', 'use_temporal': variant != 'no_temporal'}
    rw = dict(reward)
    if variant == 'no_floor':
        rw['floor'] = 0.0
    if variant == 'no_kl':
        rw['lam_kl'] = 0.0
    return net, rw, variant != 'no_sigma'


class Runner:
    def __init__(self, ds, algo, variant, device, quick_check=False, budget=None):
        self.ds, self.algo, self.variant, self.device, self.qc = ds, algo, variant, device, quick_check
        self.tag = tag_of(algo, variant)
        self.prior_mode = 'static' if variant == 'static_prior' else 'hedge'
        self.base = run_dir(self.tag, ds, None, seq=False)
        self.budget = {**BUDGET, **(budget or {})}
        if quick_check:
            self.budget.update({'pretrain': 20, 'finetune': 10, 'grid_finetune': 5, 'sup': 20})
        dec = load_json(decision_file(ds))
        self.branches = rl_branches(ds)
        self.use_oof = dec['use_oof'] and variant != 'no_oof' and os.path.exists(oof_file(ds))
        if self.use_oof:
            missing = [b for b in self.branches if b not in torch_load(oof_file(ds))['branches']]
            if missing:
                print(f"[CẢNH BÁO] OOF thiếu nhánh {missing} -> RL-Gate chỉ học trên Val", flush=True)
                self.use_oof = False
        self.ext_idx = [k for k, b in enumerate(self.branches) if b in EXTRAP_BRANCHES]
        topo = resolve_topology(ds, 'auto')
        g = build_flow_graphs(ds, list(load_raw_dataset(ds).columns), topo)
        self.graphs = [g['A_route'], g['A_od']]
        main_grid = os.path.join(run_dir(tag_of('sac', 'main'), ds, None, seq=False), 'grid.json')
        reward = dict(REWARD_DEFAULT)
        if os.path.exists(main_grid):
            reward.update(load_json(main_grid)['best'])
        self.net, self.reward, self.use_sigma = variant_cfg(variant, reward)
        self.cfg = {**DEFAULTS, **self.net}
        if quick_check:
            self.cfg.update({'ep_len': 16, 'n_envs': 4, 'batch_t': 4})

    # ---------- dữ liệu ----------
    def _split(self, d, names):
        idx = [names.index(b) for b in self.branches]
        sd = split_from_cache(d, idx, H=self.cfg['H'], use_sigma=self.use_sigma, device=self.device)
        return sd.subset(0, min(sd.T, 80)) if self.qc else sd

    def _static(self, P, y):
        st = RobustPerFlowStacking(self.branches).fit(P, y)
        return st.per_flow_weights(y.shape[1]).T.copy(), st.selected       # [N, K]

    def _prior(self, P, y, W, params=None):
        """
        Prior cho một đoạn: P [K,T,N], y [T,N], W static [N,K]. Trả về (prior, tham số).
        Hedge + sàn khởi động lại ở đầu đoạn; params=None thì chọn trên chính đoạn này (Huber).
        """
        if self.prior_mode == 'static':
            return W, {'alpha': 1.0}
        params = params or select_prior(P, y, W.T)
        return prior_weights(P, y, W.T, params), params

    def _task(self, data, prior, reward=None):
        return GateTask(data, prior, self.ext_idx, **(reward or self.reward))

    # ---------- huấn luyện ----------
    def _new_agent(self, data, state=None):
        F_static = data.F
        if self.algo == 'sac':
            ag = SAC(len(self.branches), F_static, self.graphs, self.cfg, self.device)
            if state is not None:
                ag.load_actor_critic(state)
            else:
                ag.set_norm(data.norm_stats())
            return ag
        actor = build_actor(len(self.branches), F_static, self.graphs, self.cfg).to(self.device)
        if state is not None:
            actor.load_state_dict(state['actor'])
        else:
            actor.enc.set_norm(data.norm_stats())
        return actor

    def _fit(self, agent, task, n, max_min, seed):
        if self.algo == 'sac':
            return agent.train(task, n, max_minutes=max_min, seed=seed, verbose=not self.qc)
        return train_supervised(agent, task, n_updates=n, max_minutes=max_min, seed=seed, verbose=False)

    def _state(self, agent):
        return agent.state_dict() if self.algo == 'sac' else {'actor': agent.state_dict()}

    def _actor(self, agent):
        return agent.actor if self.algo == 'sac' else agent

    def _rollout(self, agent, task):
        actor = self._actor(agent).eval()
        return task.rollout(actor) if self.algo == 'sac' else task.rollout_static_prev(actor)

    def pretrain(self, skip_existing=True):
        f = os.path.join(self.base, 'pretrain', 'policy.pt')
        if not self.use_oof:
            return None
        if skip_existing and os.path.exists(f):
            return torch_load(f)
        set_seed(0)
        o = torch_load(oof_file(self.ds))
        data = self._split(o, o['branches'])
        P = o['P'].numpy()[[o['branches'].index(b) for b in self.branches]]
        y = o['y'].numpy()
        if self.qc:
            P, y = P[:, :data.T], y[:data.T]
        W, sel = self._static(P, y)
        prior, params = self._prior(P, y, W)
        task = self._task(data, prior)
        print(f"  [{self.tag} {self.ds.upper()}] tiền huấn luyện trên OOF ({data.T} bước), stacking {sel}, "
              f"prior {self.prior_mode} {params}", flush=True)
        agent = self._new_agent(data)
        t0 = time.time()
        n = self.budget['pretrain'] if self.algo == 'sac' else self.budget['sup']
        hist = self._fit(agent, task, n, self.budget['max_min_pretrain'], seed=0)
        state = {**self._state(agent), 'reward': self.reward, 'net': self.net, 'branches': self.branches,
                 'prior': self.prior_mode, 'prior_params': params, 'minutes': (time.time() - t0) / 60}
        os.makedirs(os.path.dirname(f), exist_ok=True)
        torch.save(state, f)
        pd.DataFrame(hist).to_csv(os.path.join(os.path.dirname(f), 'train_curve.csv'), index=False)
        return state

    def _train_on(self, data, prior, reward, pre, n_ft, max_min, seed):
        """Tinh chỉnh từ policy tiền huấn luyện (hoặc học từ đầu nếu không có OOF)."""
        agent = self._new_agent(data, pre)
        task = self._task(data, prior, reward)
        n = n_ft if pre is not None else (self.budget['pretrain'] if self.algo == 'sac' else self.budget['sup'])
        hist = self._fit(agent, task, n, max_min if pre is not None else self.budget['max_min_pretrain'], seed)
        return agent, hist

    def cv(self, pre, reward, run_id=0):
        """CV 2 khối trên Val của run_0 (giống run_hedge.cv_scores). Trả về {'mse': [2 khối], 'huber': [...]}."""
        c = load_cache(self.ds, run_id)
        val = self._split(c['val'], c['branches'])
        Pv, yv, _ = split_arrays(c, 'val', self.branches)
        Pv, yv = Pv[:, :val.T], yv[:val.T]
        sel = RobustPerFlowStacking(self.branches).fit(Pv, yv).selected
        out = {'mse': [], 'huber': []}
        n_ft = self.budget['grid_finetune'] if self.algo == 'sac' else self.budget['sup']
        for j, (a, b) in enumerate(val_blocks(val.T)):
            Wa = _fit_weights_selected(Pv[:, a], yv[a], sel).T.copy()
            prior_a, params = self._prior(Pv[:, a], yv[a], Wa)
            prior_b, _ = self._prior(Pv[:, b], yv[b], Wa, params)
            set_seed(100 + j)
            agent, _ = self._train_on(val.subset(int(a[0]), int(a[-1]) + 1), prior_a, reward, pre, n_ft,
                                      self.budget['max_min_grid'], seed=100 + j)
            pred, _ = self._rollout(agent, self._task(val.subset(int(b[0]), int(b[-1]) + 1), prior_b, reward))
            s_mse, s_hub = block_scores(pred, yv[b])
            out['mse'].append(s_mse)
            out['huber'].append(s_hub)
        return out

    def grid(self, pre, skip_existing=True):
        f = os.path.join(self.base, 'grid.json')
        if skip_existing and os.path.exists(f):
            return load_json(f)
        rows = []
        for gcfg in GRID:
            rw = {**self.reward, **gcfg}
            s = self.cv(pre, rw)
            rows.append({**gcfg, 'cv_mse': float(np.mean(s['mse'])), 'cv_huber': float(np.mean(s['huber'])),
                         'blocks': s})
            print(f"  [{self.tag} {self.ds.upper()}] lưới λ_sw={gcfg['lam_sw']} f={gcfg['floor']}: "
                  f"CV Huber {rows[-1]['cv_huber']*1e3:.4f}e-3 | MSE {rows[-1]['cv_mse']*1e3:.4f}e-3", flush=True)
        best = min(rows, key=lambda r: r['cv_huber'])
        out = {'grid': rows, 'best': {k: best[k] for k in GRID[0]}, 'cv_mse': best['cv_mse'],
               'cv_huber': best['cv_huber'], 'blocks': best['blocks'], 'lam_kl': self.reward['lam_kl'],
               'prior': self.prior_mode, 'criterion': 'huber'}
        save_json(out, f)
        self.reward.update(out['best'])
        return out

    def run(self, run_id, pre, skip_existing=True):
        d = run_dir(self.tag, self.ds, run_id, seq=False)
        if skip_existing and os.path.exists(os.path.join(d, 'test_metrics.csv')):
            print(f"  [SKIP] {d}", flush=True)
            return None
        set_seed(42 + run_id)
        c = load_cache(self.ds, run_id)
        val, test = self._split(c['val'], c['branches']), self._split(c['test'], c['branches'])
        Pv, yv, _ = split_arrays(c, 'val', self.branches)
        Pt, yt, lt = split_arrays(c, 'test', self.branches)
        Pv, yv, Pt, yt, lt = Pv[:, :val.T], yv[:val.T], Pt[:, :test.T], yt[:test.T], lt[:test.T]
        W, sel = self._static(Pv, yv)
        prior_v, params = self._prior(Pv, yv, W)
        prior_t, _ = self._prior(Pt, yt, W, params)
        t0 = time.time()
        agent, hist = self._train_on(val, prior_v, self.reward, pre, self.budget['finetune'] if self.algo == 'sac'
                                     else self.budget['sup'], self.budget['max_min_finetune'], seed=42 + run_id)
        task_t = self._task(test, prior_t)
        pred, Wt = self._rollout(agent, task_t)
        mse_prior = float(task_t.se_prior.mean())
        cfg = {'algo': self.algo, 'variant': self.variant, 'branches': self.branches, 'reward': self.reward,
               'net': self.net, 'use_oof': self.use_oof, 'use_sigma': self.use_sigma, 'static_selected': list(sel),
               'prior': self.prior_mode, 'prior_params': params, 'test_mse_prior': mse_prior,
               'minutes': (time.time() - t0) / 60, 'budget': self.budget}
        m = save_combiner_result(self.tag, self.ds, run_id, pred, yt, lt, config=cfg,
                                 weights=Wt if run_id == 0 else None)
        torch.save(self._state(agent), os.path.join(d, 'policy.pt'))
        pd.DataFrame(hist).to_csv(os.path.join(d, 'train_curve.csv'), index=False)
        dev_w = float((task_t.prior - torch.as_tensor(Wt, device=task_t.prior.device)).abs().sum(-1).mean())
        print(f"  [{self.tag} {self.ds.upper()} run_{run_id}] Test MSE={m['mse']*1e3:.4f}e-3 "
              f"(prior {self.prior_mode} {mse_prior*1e3:.4f}e-3) | lệch TB khỏi prior ‖a-p‖₁={dev_w:.3f} | "
              f"{cfg['minutes']:.1f} phút", flush=True)
        return m


def main(args):
    device = get_device()
    for ds in parse_list(args.datasets, ALL_DATASETS):
        R = Runner(ds, args.algo, args.variant, device, args.quick_check)
        print(f"\n[{R.tag} {ds.upper()}] nhánh {R.branches} | OOF={R.use_oof} | reward {R.reward} | net {R.net} | "
              f"device {device}", flush=True)
        pre = R.pretrain(skip_existing=args.skip_existing)
        if args.phase in ('all', 'grid') and args.algo == 'sac' and args.variant == 'main':
            R.grid(pre, skip_existing=args.skip_existing)
        if args.phase in ('all', 'cv') and args.algo == 'sup' and args.variant == 'main':
            f = os.path.join(R.base, 'cv.json')
            if not (args.skip_existing and os.path.exists(f)):
                s = R.cv(pre, R.reward)
                save_json({'cv_mse': float(np.mean(s['mse'])), 'cv_huber': float(np.mean(s['huber'])), 'blocks': s,
                           'reward': R.reward}, f)
        if args.phase in ('all', 'runs'):
            for r in parse_run_ids(args.run_ids):
                R.run(r, pre, skip_existing=args.skip_existing)


if __name__ == '__main__':
    setup_utf8()
    ap = argparse.ArgumentParser(description="Huấn luyện RL-Gate / gate giám sát (V4)")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_ids', default='0-2')
    ap.add_argument('--algo', default='sac', choices=['sac', 'sup'])
    ap.add_argument('--variant', default='main', choices=VARIANTS)
    ap.add_argument('--phase', default='all', choices=['all', 'pretrain', 'grid', 'cv', 'runs'])
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--quick_check', action='store_true')
    main(ap.parse_args())
