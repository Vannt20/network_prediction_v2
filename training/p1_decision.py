"""
Quyết định sau P1 (spec Mục 4.7), ghi results/gnn_rl/p1_decision_{ds}.json (một file mỗi dataset để
hai tài khoản Kaggle không ghi đè nhau).

Với từng dataset: nếu Val MSE trung bình của OD-GraphFormer thấp hơn ST-WaveFormer
  -> tập nhánh RL-Gate = {odgraphformer, xgboost, lightgbm_res}, có OOF
  ngược lại -> 3 nhánh v3, RL-Gate chỉ học trên Val (--no_oof), không huấn luyện lại ST-WaveFormer.
"""
import os
import sys
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.gnn_rl_common import (setup_utf8, parse_list, ALL_DATASETS, RESULTS_V4, load_cache, split_arrays,
                                    save_json)
from baselines_ml.run_ml_baselines import parse_run_ids


def decision_file(ds_key):
    return os.path.join(RESULTS_V4, f'p1_decision_{ds_key}.json')


def decide(datasets, run_ids, force=None):
    out = {}
    for ds in datasets:
        mse = {}
        for r in run_ids:
            c = load_cache(ds, r, 'v4')
            Pv, yv, _ = split_arrays(c, 'val')
            for k, b in enumerate(c['branches']):
                mse.setdefault(b, []).append(float(np.mean((Pv[k] - yv) ** 2)))
            gml = [b for b in c['branches'] if b not in ('stwaveformer', 'odgraphformer')]
        m = {b: float(np.mean(v)) for b, v in mse.items()}
        use_odgf = m['odgraphformer'] < m['stwaveformer'] if force is None else force == 'odgraphformer'
        dl = 'odgraphformer' if use_odgf else 'stwaveformer'
        out[ds] = {'dataset': ds, 'rl_branches': [dl] + gml, 'use_oof': bool(use_odgf), 'val_mse_mean': m,
                   'n_runs': len(run_ids), 'forced': force}
        save_json(out[ds], decision_file(ds))
        print(f"[{ds.upper()}] Val MSE e-3: " + " | ".join(f"{b}={v*1e3:.3f}" for b, v in m.items())
              + f" -> nhánh học sâu cho RL-Gate: {dl}", flush=True)
    return out


if __name__ == '__main__':
    setup_utf8()
    ap = argparse.ArgumentParser(description="P1: chọn nhánh học sâu cho RL-Gate theo Val MSE")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_ids', default='0-9')
    ap.add_argument('--force', default=None, choices=['odgraphformer', 'stwaveformer'], help="Chỉ dùng cho quick_check")
    a = ap.parse_args()
    decide(parse_list(a.datasets, ALL_DATASETS), parse_run_ids(a.run_ids), a.force)
