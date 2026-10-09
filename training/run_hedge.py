"""
Các tầng kết hợp không dùng học tăng cường, chạy trên cache (spec Mục 2.3 và 6.5):
  static       : RobustPerFlowStacking fit trên Val (như v3) trên đúng tập nhánh
  hedge        : Hedge per-flow, (eta, beta) chọn trên Val
  hedge_floor  : Hedge + sàn trọng số tĩnh, (eta, beta, alpha) chọn trên Val
  context_gate : cổng MLP ngữ cảnh v2 (evaluation/ablation_study.train_context_gate)

Ngoài kết quả Test, mỗi run ghi điểm CV 2 khối trên Val (cv_scores.json) cho static và Hedge,
dùng ở quy tắc chọn tầng kết hợp (spec Mục 6.4.5).

P0 (trên cache v3, 3 nhánh v3):  --cache_version v3 --tag_suffix _v3
P3 (trên cache v4, tập nhánh RL): --cache_version v4  (nhánh lấy từ results/gnn_rl/p1_decision_{ds}.json)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.gnn_rl_common import (setup_utf8, parse_list, ALL_DATASETS, RESULTS_V4, load_cache, split_arrays,
                                    save_combiner_result, save_json, load_json, run_dir)
from Graph_models.robust_stacking import RobustPerFlowStacking, fit_convex_weights, blend, DEFAULT_CONFIG
from Graph_models.online_hedge import hedge_predict, select_hedge
from baselines_ml.run_ml_baselines import parse_run_ids

ALL_METHODS = ['static', 'hedge', 'hedge_floor', 'context_gate']


def rl_branches(ds_key):
    """Tập nhánh K = 3 của RL-Gate cho dataset (spec Mục 4.7)."""
    from training.p1_decision import decision_file
    f = decision_file(ds_key)
    if not os.path.exists(f):
        raise FileNotFoundError(f"Chưa có {f}. Chạy: python training/p1_decision.py --datasets {ds_key}")
    return load_json(f)['rl_branches']


def fit_static(branches, Pv, yv):
    st = RobustPerFlowStacking(branches).fit(Pv, yv)
    return st, st.per_flow_weights(yv.shape[1])


def _fit_weights_selected(P, y, selected):
    loss, scope = selected
    res = fit_convex_weights(P, y, loss=loss, scope=scope, shrink_lambda=DEFAULT_CONFIG['shrink_lambda'],
                             huber_delta=DEFAULT_CONFIG['huber_delta'], iters=DEFAULT_CONFIG['iters'],
                             lr=DEFAULT_CONFIG['lr'])
    w = res['weights']
    return np.repeat(w[:, None], y.shape[1], 1) if w.ndim == 1 else w


def val_blocks(T):
    h = T // 2
    return [(np.arange(0, h), np.arange(h, T)), (np.arange(h, T), np.arange(0, h))]


def cv_scores(Pv, yv, selected):
    """CV 2 khối trên Val: fit trên khối A, chấm MSE trên khối B rồi đổi vai (static, Hedge, Hedge + sàn)."""
    out = {'static': [], 'hedge': [], 'hedge_floor': []}
    for a, b in val_blocks(len(yv)):
        w0 = _fit_weights_selected(Pv[:, a], yv[a], selected)
        out['static'].append(float(np.mean((blend(w0, Pv[:, b]) - yv[b]) ** 2)))
        h, hf, _ = select_hedge(Pv[:, a], yv[a], w0)
        ph, _ = hedge_predict(Pv[:, b], yv[b], w0, h['eta'], h['beta'])
        out['hedge'].append(float(np.mean((ph - yv[b]) ** 2)))
        pf, _ = hedge_predict(Pv[:, b], yv[b], w0, hf['eta'], hf['beta'], hf['alpha'])
        out['hedge_floor'].append(float(np.mean((pf - yv[b]) ** 2)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def run_combiners(datasets, run_ids, cache_version='v4', branches=None, methods=None, tag_suffix='',
                  static_tag=None, quick_check=False):
    methods = methods or ALL_METHODS
    rows = []
    for ds in datasets:
        for r in run_ids:
            c = load_cache(ds, r, cache_version)
            br = branches or (c['branches'] if cache_version == 'v3' else rl_branches(ds))
            Pv, yv, _ = split_arrays(c, 'val', br)
            Pt, yt, lt = split_arrays(c, 'test', br)
            if quick_check:
                Pv, yv, Pt, yt, lt = Pv[:, :60], yv[:60], Pt[:, :60], yt[:60], lt[:60]
            st, W = fit_static(br, Pv, yv)
            res = {}
            if 'static' in methods:
                tag = (static_tag or 'static') + tag_suffix
                res['static'] = save_combiner_result(tag, ds, r, blend(W, Pt), yt, lt,
                                                     config={'branches': br, 'selected': list(st.selected)})
                np.save(os.path.join(run_dir(tag, ds, r, seq=False), 'weights_static.npy'), W.astype(np.float32))
            if 'hedge' in methods or 'hedge_floor' in methods:
                h, hf, grid = select_hedge(Pv, yv, W)
                if 'hedge' in methods:
                    p, _ = hedge_predict(Pt, yt, W, h['eta'], h['beta'])
                    res['hedge'] = save_combiner_result('hedge' + tag_suffix, ds, r, p, yt, lt,
                                                        config={'branches': br, **h})
                if 'hedge_floor' in methods:
                    p, _ = hedge_predict(Pt, yt, W, hf['eta'], hf['beta'], hf['alpha'])
                    res['hedge_floor'] = save_combiner_result('hedge_floor' + tag_suffix, ds, r, p, yt, lt,
                                                              config={'branches': br, **hf})
                pd.DataFrame(grid).to_csv(os.path.join(run_dir('hedge' + tag_suffix, ds, r, seq=False),
                                                       'val_grid.csv'), index=False)
            if 'context_gate' in methods:
                from evaluation.ablation_study import train_context_gate
                ctx_v = c['val']['context'].numpy()
                ctx_t = c['test']['context'].numpy()
                if quick_check:
                    ctx_v, ctx_t = ctx_v[:60], ctx_t[:60]
                Wg = train_context_gate(ctx_v, Pv, yv, ctx_t, W.mean(axis=1), epochs=5 if quick_check else 200)
                p = np.sum(Wg * np.moveaxis(Pt, 0, -1), axis=-1)
                res['context_gate'] = save_combiner_result('context_gate' + tag_suffix, ds, r, p, yt, lt,
                                                           config={'branches': br})
            cv = {}
            if 'hedge' in methods:
                cv = cv_scores(Pv, yv, st.selected)
                save_json({'branches': br, 'selected': list(st.selected), 'cv_mse': cv},
                          os.path.join(run_dir('cv' + tag_suffix, ds, r, seq=False), 'cv_scores.json'))
            msg = " | ".join(f"{k}={v['mse']*1e3:.3f}" for k, v in res.items())
            print(f"  [{ds.upper()} run_{r}] {br} Test MSE e-3: {msg} | CV e-3: "
                  + " ".join(f"{k}={v*1e3:.3f}" for k, v in cv.items()), flush=True)
            for k, v in res.items():
                rows.append({'dataset': ds, 'run': r, 'method': k, 'mse': v['mse']})
    df = pd.DataFrame(rows)
    if len(df):
        print(df.groupby(['dataset', 'method'])['mse'].agg(runs='count', mse_e3=lambda x: x.mean() * 1e3).to_string())
    return df


if __name__ == '__main__':
    setup_utf8()
    ap = argparse.ArgumentParser(description="Tầng kết hợp static / Hedge / Hedge + sàn / cổng MLP trên cache")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_ids', default='0-9')
    ap.add_argument('--cache_version', default='v4', choices=['v3', 'v4'])
    ap.add_argument('--branches', default=None, help="Mặc định: v3 -> mọi nhánh trong cache; v4 -> p1_decision.json")
    ap.add_argument('--methods', default=','.join(ALL_METHODS))
    ap.add_argument('--tag_suffix', default='', help="Ví dụ '_v3' cho P0")
    ap.add_argument('--static_tag', default=None, help="Tên thư mục cho static (mặc định 'static'), ví dụ 'static_k4'")
    ap.add_argument('--quick_check', action='store_true')
    a = ap.parse_args()
    run_combiners(parse_list(a.datasets, ALL_DATASETS), parse_run_ids(a.run_ids), a.cache_version,
                  parse_list(a.branches, []) or None, parse_list(a.methods, ALL_METHODS), a.tag_suffix,
                  a.static_tag, a.quick_check)
