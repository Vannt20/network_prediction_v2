"""
Cận trên (oracle) của tầng kết hợp - CÓ nhìn nhãn Test, chỉ dùng để phân tích dư địa (spec Mục 2.1).

  per_flow_fixed : mỗi luồng chọn 1 nhánh cố định tốt nhất trên Test
  per_step_global: mỗi bước chọn 1 nhánh cho toàn mạng
  per_step_flow  : mỗi (bước, luồng) chọn 1 nhánh
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

from training.gnn_rl_common import setup_utf8, parse_list, RESULTS_V4, ALL_DATASETS, upsert_csv
from baselines_ml.run_ml_baselines import parse_run_ids


def oracle_mse(P, y):
    P = np.asarray(P, dtype=np.float64)
    se = (P - np.asarray(y, dtype=np.float64)[None]) ** 2          # [K,T,N]
    return {
        'oracle_per_flow_fixed': float(se.mean(axis=1).min(axis=0).mean()),
        'oracle_per_step_global': float(se.mean(axis=2).min(axis=0).mean()),
        'oracle_per_step_flow': float(se.min(axis=0).mean()),
    }


def run_headroom(datasets, run_ids, cache_version='v3'):
    from training.precompute_cache import cache_path
    import torch
    rows = []
    for ds in datasets:
        for r in run_ids:
            f = cache_path(ds, r) if cache_version == 'v3' else os.path.join(parent_dir, 'cache', cache_version, f"{ds}_run_{r}.pt")
            with open(f, 'rb') as fh:
                c = torch.load(fh, map_location='cpu', weights_only=False)
            row = {'dataset': ds, 'run': r, **oracle_mse(c['test']['P'].numpy(), c['test']['y'].numpy())}
            rows.append(row)
        sub = pd.DataFrame([x for x in rows if x['dataset'] == ds])
        print(f"[{ds.upper()}] " + " | ".join(f"{k}={sub[k].mean()*1e3:.3f}e-3" for k in sub.columns if k.startswith('oracle')))
    return upsert_csv(pd.DataFrame(rows), os.path.join(RESULTS_V4, f'p0_headroom_{cache_version}.csv'))


if __name__ == '__main__':
    setup_utf8()
    ap = argparse.ArgumentParser(description="Oracle (cận trên) của tầng kết hợp trên cache")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_ids', default='0-9')
    ap.add_argument('--cache_version', default='v3')
    a = ap.parse_args()
    run_headroom(parse_list(a.datasets, ALL_DATASETS), parse_run_ids(a.run_ids), a.cache_version)
