"""
Kiểm tra đồng bộ trước khi chạy ST-Adaptive-Ensemble v3 (nhẹ, không huấn luyện, không suy diễn).

    python preflight_check.py --run_ids 5-9
    python preflight_check.py --run_ids 0-9

Kiểm tra cho từng dataset x run:
  - ST-WaveFormer: log hợp lệ (có seed, số bước test khớp dữ liệu hiện tại), seed = 42 + run_id,
    lr = 0.001, số epoch <= 200, và early stopping đúng patience 30 (suy ra từ train_metrics.csv).
  - Nhánh học máy (champion + lightgbm_res): có model.bin, test_metrics.csv, y_real_data.npy đúng số mẫu.
  - Cache v3 (nếu có): đúng danh sách nhánh và khớp dấu vân tay MD5 của mô hình hiện tại.
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from features.feature_store import DATASET_CONFIGS
from run_experiments import check_existing_run, expected_test_windows, SEEDS
from baselines_ml.run_ml_baselines import parse_run_ids
from training.precompute_cache import cache_path, default_branches, source_fingerprint
from Graph_models.robust_stacking import DEFAULT_CONFIG

EXPECTED_EPOCHS = 200
EXPECTED_PATIENCE = 30
EXPECTED_LR = 1e-3


def check_dl(ds, r, n_win):
    sl = DATASET_CONFIGS[ds]['seq_len']
    d = os.path.join(ROOT, 'logs', f"stwaveformer_data_{ds}_seq_{sl}", f"run_{r}")
    ok, reason = check_existing_run(d, n_win)
    if not ok:
        return False, reason
    m = pd.read_csv(os.path.join(d, 'test_metrics.csv')).iloc[0]
    issues = []
    if int(m['seed']) != SEEDS[r % len(SEEDS)]:
        issues.append(f"seed={int(m['seed'])} (kỳ vọng {SEEDS[r % len(SEEDS)]})")
    if 'lr' in m and abs(float(m['lr']) - EXPECTED_LR) > 1e-12:
        issues.append(f"lr={m['lr']}")
    tf = os.path.join(d, 'train_metrics.csv')
    if os.path.exists(tf):
        h = pd.read_csv(tf)
        n_ep = int(h['epoch'].max())
        best_ep = int(h.loc[h['val_loss'].idxmin(), 'epoch'])
        if n_ep > EXPECTED_EPOCHS:
            issues.append(f"{n_ep} epoch > {EXPECTED_EPOCHS}")
        elif n_ep < EXPECTED_EPOCHS and n_ep - best_ep != EXPECTED_PATIENCE:
            issues.append(f"dừng ở epoch {n_ep}, best {best_ep} -> patience {n_ep - best_ep} (kỳ vọng {EXPECTED_PATIENCE})")
        info = f"epoch {n_ep}/{EXPECTED_EPOCHS}, best {best_ep}"
    else:
        issues.append('thiếu train_metrics.csv')
        info = ''
    return (not issues), ('; '.join(issues) if issues else f"ok ({info})")


def check_ml(ds, name, r, n_rows):
    d = os.path.join(ROOT, 'logs', f"{name}_data_{ds}_shared", f"run_{r}")
    missing = [f for f in ['model.bin', 'test_metrics.csv', 'y_real_data.npy', 'y_pred_data.npy']
               if not os.path.exists(os.path.join(d, f))]
    if missing:
        return False, f"thiếu {missing}"
    n = np.load(os.path.join(d, 'y_real_data.npy'), mmap_mode='r').size
    if n != n_rows:
        return False, f"y_real có {n} mẫu, kỳ vọng {n_rows}"
    return True, 'ok'


def check_cache(ds, r, branches):
    f = cache_path(ds, r)
    if not os.path.exists(f):
        return None, 'chưa có (sẽ được tạo)'
    import torch
    c = torch.load(f, map_location='cpu')
    if c.get('branches') != branches:
        return False, f"nhánh {c.get('branches')} khác {branches} (sẽ tính lại)"
    if c.get('sources') != source_fingerprint(branches, ds, r):
        return False, 'không khớp dấu vân tay mô hình (sẽ tính lại)'
    return True, 'khớp'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_ids', default='5-9')
    ap.add_argument('--datasets', default='sdn,geant,abilene')
    args = ap.parse_args()
    ids = parse_run_ids(args.run_ids)
    branches = default_branches()
    ml_branches = [b for b in branches if b != 'stwaveformer']

    import xgboost, lightgbm, torch
    print(f"Thư viện: torch {torch.__version__} (CUDA={torch.cuda.is_available()}) | "
          f"xgboost {xgboost.__version__} | lightgbm {lightgbm.__version__}")
    print(f"Nhánh: {branches} | Run IDs: {ids}")
    print(f"Tầng kết hợp (cố định): {DEFAULT_CONFIG}\n")

    blockers = 0
    for ds in [d.strip() for d in args.datasets.split(',')]:
        cfg = DATASET_CONFIGS[ds]
        n_win = expected_test_windows(ds, cfg['seq_len'])
        print(f"=== {ds.upper()} (seq_len={cfg['seq_len']}, {cfg['flows']} luồng, {n_win} bước test)")
        for r in ids:
            ok_dl, msg_dl = check_dl(ds, r, n_win)
            line = [f"  run_{r} (seed {SEEDS[r]}): ST-WaveFormer {'OK' if ok_dl else 'LỖI'} – {msg_dl}"]
            for b in ml_branches:
                ok_ml, msg_ml = check_ml(ds, b, r, n_win * cfg['flows'])
                if not ok_ml and b == 'lightgbm_res':
                    msg_ml += ' (sẽ được huấn luyện ở Bước 2, CPU)'
                elif not ok_ml:
                    blockers += 1
                line.append(f"      {b}: {'OK' if ok_ml else 'CHƯA'} – {msg_ml}")
            ok_c, msg_c = check_cache(ds, r, branches)
            line.append(f"      cache v3: {msg_c}")
            if not ok_dl:
                blockers += 1
            print('\n'.join(line))
    print("\n" + ("SẴN SÀNG: không có lỗi chặn." if blockers == 0 else
                  f"CÓ {blockers} LỖI CHẶN: ST-WaveFormer cần huấn luyện lại trên GPU (Kaggle) hoặc thiếu mô hình champion."))
    return 0 if blockers == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
