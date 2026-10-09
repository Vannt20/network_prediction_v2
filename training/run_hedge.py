"""
Các tầng kết hợp không dùng học tăng cường, chạy trên cache (spec Mục 2.3 và 6.5):
  static       : RobustPerFlowStacking fit trên Val (như v3) trên đúng tập nhánh
  hedge        : Hedge per-flow, (eta, beta) chọn trên Val
  hedge_floor  : Hedge + sàn trọng số tĩnh, (eta, beta, alpha) chọn trên Val
  hedge_prior  : Hedge + sàn, alpha ∈ {0..1} (1 = static), chọn theo Huber trên Val - đúng prior của RL-Gate
                 vòng 2, báo cáo riêng để tách phần RL đóng góp
  context_gate : cổng MLP ngữ cảnh v2 (evaluation/ablation_study.train_context_gate)

Ngoài kết quả Test, mỗi run ghi điểm CV 2 khối trên Val (cv_scores.json): MSE và Huber của từng khối cho
static, Hedge, Hedge + sàn, prior; dùng ở quy tắc chọn tầng kết hợp (spec Mục 6.4.5 và 15).

P0 (trên cache v3, 3 nhánh v3):  --cache_version v3 --tag_suffix _v3
P3 (cache V4 của vòng hiện tại, tập nhánh RL lấy từ results/gnn_rl*/p1_decision_{ds}.json)
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

from training.gnn_rl_common import (setup_utf8, parse_list, ALL_DATASETS, load_cache, split_arrays,
                                    save_combiner_result, save_json, load_json, run_dir, huber, CACHE_VERSION_V4)
from Graph_models.robust_stacking import RobustPerFlowStacking, fit_convex_weights, blend, DEFAULT_CONFIG
from Graph_models.online_hedge import hedge_predict, select_hedge, select_prior
from baselines_ml.run_ml_baselines import parse_run_ids

ALL_METHODS = ['static', 'hedge', 'hedge_floor', 'hedge_prior', 'context_gate']


def rl_branches(ds_key):
    """Tập nhánh của RL-Gate cho dataset (vòng 2: cả 4 nhánh, spec Mục 15)."""
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


def block_scores(pred, y):
    e = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.mean(e ** 2)), huber(e)


def cv_scores(Pv, yv, selected):
    """
    CV 2 khối trên Val: fit trên khối A, chấm trên khối B rồi đổi vai.
    Trả về {phương pháp: {'mse': [khối 1, khối 2], 'huber': [...]}} cho static, Hedge, Hedge + sàn, prior.
    """
    out = {m: {'mse': [], 'huber': []} for m in ('static', 'hedge', 'hedge_floor', 'hedge_prior')}

    def add(m, pred, y):
        s_mse, s_hub = block_scores(pred, y)
        out[m]['mse'].append(s_mse)
        out[m]['huber'].append(s_hub)
    for a, b in val_blocks(len(yv)):
        w0 = _fit_weights_selected(Pv[:, a], yv[a], selected)
        add('static', blend(w0, Pv[:, b]), yv[b])
        h, hf, _ = select_hedge(Pv[:, a], yv[a], w0)
        add('hedge', hedge_predict(Pv[:, b], yv[b], w0, h['eta'], h['beta'])[0], yv[b])
        add('hedge_floor', hedge_predict(Pv[:, b], yv[b], w0, hf['eta'], hf['beta'], hf['alpha'])[0], yv[b])
        pp = select_prior(Pv[:, a], yv[a], w0)
        add('hedge_prior', hedge_predict(Pv[:, b], yv[b], w0, pp['eta'], pp['beta'], pp['alpha'])[0], yv[b])
    return out


def run_combiners(datasets, run_ids, cache_version=CACHE_VERSION_V4, branches=None, methods=None, tag_suffix='',
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
            if 'hedge_prior' in methods:
                pp = select_prior(Pv, yv, W)
                p, _ = hedge_predict(Pt, yt, W, pp['eta'], pp['beta'], pp['alpha'])
                res['hedge_prior'] = save_combiner_result('hedge_prior' + tag_suffix, ds, r, p, yt, lt,
                                                          config={'branches': br, **pp, 'criterion': 'huber'})
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
                save_json({'branches': br, 'selected': list(st.selected),
                           'cv_mse': {m: float(np.mean(v['mse'])) for m, v in cv.items()},
                           'cv_huber': {m: float(np.mean(v['huber'])) for m, v in cv.items()},
                           'blocks': cv},
                          os.path.join(run_dir('cv' + tag_suffix, ds, r, seq=False), 'cv_scores.json'))
            msg = " | ".join(f"{k}={v['mse']*1e3:.3f}" for k, v in res.items())
            print(f"  [{ds.upper()} run_{r}] {br} Test MSE e-3: {msg} | CV Huber e-3: "
                  + " ".join(f"{k}={np.mean(v['huber'])*1e3:.4f}" for k, v in cv.items()), flush=True)
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
    ap.add_argument('--cache_version', default=CACHE_VERSION_V4, help="v3 (P0) hoặc cache V4 của vòng hiện tại")
    ap.add_argument('--branches', default=None, help="Mặc định: v3 -> mọi nhánh trong cache; v4 -> p1_decision.json")
    ap.add_argument('--methods', default=','.join(ALL_METHODS))
    ap.add_argument('--tag_suffix', default='', help="Ví dụ '_v3' cho P0")
    ap.add_argument('--static_tag', default=None, help="Tên thư mục cho static (mặc định 'static'), ví dụ 'static_k4'")
    ap.add_argument('--quick_check', action='store_true')
    a = ap.parse_args()
    run_combiners(parse_list(a.datasets, ALL_DATASETS), parse_run_ids(a.run_ids), a.cache_version,
                  parse_list(a.branches, []) or None, parse_list(a.methods, ALL_METHODS), a.tag_suffix,
                  a.static_tag, a.quick_check)
