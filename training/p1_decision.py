"""
Quyết định sau P1, ghi results/gnn_rl*/p1_decision_{ds}.json (một file mỗi dataset để hai tài khoản Kaggle
không ghi đè nhau).

Vòng 1 thay ST-WaveFormer bằng OD-GraphFormer khi Val MSE của nhánh đơn thấp hơn. Ở SDN, ST-WaveFormer đứng một
mình rất kém nhưng bổ sung tốt cho tổ hợp: bỏ nó làm static tăng từ 4,00 lên 4,68 (x10^-3). Vòng 2 (spec Mục 15)
giữ cả 4 nhánh cho RL-Gate và các tầng kết hợp; Static K = 4 tốt hơn v3 ở 3/3 run trên cả 3 dataset ở vòng 1.
File vẫn ghi Val MSE từng nhánh để tham khảo.
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


def decide(datasets, run_ids):
    out = {}
    for ds in datasets:
        mse, branches = {}, None
        for r in run_ids:
            c = load_cache(ds, r)
            branches = c['branches']
            Pv, yv, _ = split_arrays(c, 'val')
            for k, b in enumerate(branches):
                mse.setdefault(b, []).append(float(np.mean((Pv[k] - yv) ** 2)))
        m = {b: float(np.mean(v)) for b, v in mse.items()}
        out[ds] = {'dataset': ds, 'rl_branches': list(branches), 'use_oof': True, 'val_mse_mean': m,
                   'n_runs': len(run_ids), 'rule': 'vòng 2: giữ cả 4 nhánh'}
        save_json(out[ds], decision_file(ds))
        print(f"[{ds.upper()}] Val MSE e-3: " + " | ".join(f"{b}={v*1e3:.3f}" for b, v in m.items())
              + f" -> RL-Gate dùng {len(branches)} nhánh: {branches}", flush=True)
    return out


if __name__ == '__main__':
    setup_utf8()
    ap = argparse.ArgumentParser(description="P1: tập nhánh cho RL-Gate và các tầng kết hợp")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_ids', default='0-2')
    a = ap.parse_args()
    decide(parse_list(a.datasets, ALL_DATASETS), parse_run_ids(a.run_ids))
