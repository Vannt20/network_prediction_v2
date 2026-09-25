import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import json
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

from features.feature_store import DATASET_CONFIGS
from Graph_models.robust_stacking import RobustPerFlowStacking
from baselines_ml.metrics import calc_metrics_numpy
from baselines_ml.run_ml_baselines import parse_run_ids
from evaluation.stats_tests import diebold_mariano
from training.precompute_cache import cache_path

torch.set_num_threads(min(4, os.cpu_count() or 1))

MODEL_TAG = 'st_adaptive_ensemble'


def load_cache(ds_key, run_id):
    f = cache_path(ds_key, run_id)
    if not os.path.exists(f):
        raise FileNotFoundError(f"Chưa có cache {f}. Chạy: python training/precompute_cache.py --datasets {ds_key} --run_ids {run_id}")
    with open(f, 'rb') as fh:
        return torch.load(fh, map_location='cpu')


def ensemble_metrics(y_pred, y_real, P_test, branches):
    """Chỉ số chính + MSE trong/ngoài biên Train + so sánh Diebold-Mariano với từng nhánh."""
    m = calc_metrics_numpy(y_pred, y_real)
    in_range = y_real <= 1.0
    m['mse_in_range'] = float(np.mean((y_pred - y_real)[in_range] ** 2))
    m['n_out_of_range'] = int((~in_range).sum())
    m['frac_err_out_of_range'] = float(((y_pred - y_real)[~in_range] ** 2).sum() / (((y_pred - y_real) ** 2).sum() + 1e-12))
    for k, b in enumerate(branches):
        m[f'branch_mse_{b}'] = float(np.mean((P_test[k] - y_real) ** 2))
        m[f'branch_mse_in_range_{b}'] = float(np.mean((P_test[k] - y_real)[in_range] ** 2))
        dm, p = diebold_mariano(y_real, y_pred, P_test[k])
        m[f'dm_stat_vs_{b}'] = dm
        m[f'dm_p_vs_{b}'] = p
    return m


def train_stacking_for_run(dataset_name, run_id, config=None, logdir=None, verbose=True):
    ds_key = dataset_name.lower()
    seq_len = DATASET_CONFIGS[ds_key]['seq_len']
    cache = load_cache(ds_key, run_id)
    branches = cache['branches']
    P_val, y_val = cache['val']['P'].numpy(), cache['val']['y'].numpy()
    P_test, y_test = cache['test']['P'].numpy(), cache['test']['y'].numpy()
    N = y_val.shape[1]

    t0 = time.perf_counter()
    stack = RobustPerFlowStacking(branches, config=config).fit(P_val, y_val)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = stack.predict(P_test)
    blend_ms = (time.perf_counter() - t1) * 1000.0 / max(1, int(np.ceil(len(y_test) / 64)))

    metrics = ensemble_metrics(y_pred, y_test, P_test, branches)
    branch_times = [cache['inference_time_ms'].get(b, np.nan) for b in branches]
    metrics['blend_time_ms'] = float(blend_ms)
    # Thời gian suy diễn toàn mô hình (các nhánh chạy tuần tự + tầng kết hợp), ms / lô 64
    metrics['inference_time_ms'] = float(np.nansum(branch_times) + blend_ms)
    metrics['stacking_fit_time_s'] = float(fit_time)
    metrics['selected_loss'], metrics['selected_scope'] = stack.selected
    W = stack.per_flow_weights(N)
    for k, b in enumerate(branches):
        metrics[f'mean_w_{b}'] = float(W[k].mean())

    if logdir is None:
        logdir = os.path.join(parent_dir, 'logs', f"{MODEL_TAG}_data_{ds_key}_seq_{seq_len}", f"run_{run_id}")
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'stacking.json'), 'w', encoding='utf-8') as f:
        json.dump(stack.state_dict(), f, indent=2, ensure_ascii=False)
    np.save(os.path.join(logdir, 'weights.npy'), W)
    np.save(os.path.join(logdir, 'y_pred_data.npy'), y_pred.astype(np.float32))
    np.save(os.path.join(logdir, 'y_real_data.npy'), y_test.astype(np.float32))
    pd.DataFrame([metrics]).to_csv(os.path.join(logdir, 'test_metrics.csv'), index=False)

    if verbose:
        cv = ", ".join(f"{k[0]}-{k[1]}={v*1e3:.4f}" for k, v in (stack.cv_scores or {}).items())
        print(f"    run_{run_id}: chọn {stack.selected} | CV(Huber e-3): {cv}", flush=True)
        br = " | ".join(f"{b}={metrics[f'branch_mse_{b}']*1e3:.3f}" for b in branches)
        ws = " ".join(f"{b}={metrics[f'mean_w_{b}']:.3f}" for b in branches)
        print(f"      Test MSE e-3: ENSEMBLE={metrics['mse']*1e3:.3f} | {br}", flush=True)
        print(f"      Trọng số TB: {ws}", flush=True)
    return metrics


def train_all_stacking(datasets=None, runs=5, run_ids=None, config=None):
    if datasets is None or 'all' in datasets:
        datasets = ['sdn', 'geant', 'abilene']
    run_id_list = parse_run_ids(run_ids, runs)
    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(" HUẤN LUYỆN TẦNG KẾT HỢP: ROBUST PER-FLOW CONVEX STACKING (ST-ADAPTIVE-ENSEMBLE v3)")
    print(f" Datasets: {datasets} | Run IDs: {run_id_list}")
    print("=" * 80)

    for ds in datasets:
        print(f"\n---> {ds.upper()}")
        rows = []
        for r in run_id_list:
            m = train_stacking_for_run(ds, r, config=config)
            m['run'] = r
            rows.append(m)
        df = pd.DataFrame(rows)
        out_csv = os.path.join(results_dir, f"results_STAdaptiveEnsemble_data_{ds}.csv")
        if os.path.exists(out_csv):
            old = pd.read_csv(out_csv)
            if 'run' in old.columns:
                df = pd.concat([old[~old['run'].isin(df['run'])], df], ignore_index=True)
        df = df.sort_values('run').reset_index(drop=True)
        df.to_csv(out_csv, index=False)
        cur = df[df['run'].isin(run_id_list)]
        print(f"[*] {ds.upper()} ({len(cur)} run): ENSEMBLE MSE = {cur['mse'].mean()*1e3:.3f} ± {cur['mse'].std()*1e3:.3f} e-3")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện tầng stacking v3 trên Validation cache")
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--run_ids', type=str, default=None, help="Ví dụ '5-9' (ghi đè --runs)")
    args = parser.parse_args()
    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ['sdn', 'geant', 'abilene']
    train_all_stacking(datasets=d_list, runs=args.runs, run_ids=args.run_ids)
