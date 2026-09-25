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
from Graph_models.robust_stacking import RobustPerFlowStacking, blend
from Graph_models.contextual_gate import PerFlowContextualMetaGating
from baselines_ml.metrics import calc_metrics_numpy
from baselines_ml.run_ml_baselines import parse_run_ids
from training.train_stacking import load_cache, MODEL_TAG

torch.set_num_threads(min(4, os.cpu_count() or 1))

# Mô tả cấu hình (phục vụ bảng luận văn)
CONFIG_DESC = {
    'full': 'Đầy đủ: 3 nhánh + stacking bền vững theo luồng (cấu hình chọn bằng CV)',
    'no_stwaveformer': 'Bỏ nhánh học sâu (chỉ 2 nhánh học máy)',
    'no_lightgbm_res': 'Bỏ nhánh LightGBM-Residual',
    'global_only': 'Chỉ trọng số chung toàn mạng',
    'perflow_only': 'Luôn dùng trọng số theo luồng',
    'static_average': 'Trung bình tĩnh 1/K',
    'best_single_val': 'Chọn một nhánh tốt nhất theo MSE Validation',
    'context_gate': 'Cổng MLP ngữ cảnh (v2) thay cho stacking',
}


def train_context_gate(ctx_val, P_val, y_val, ctx_test, prior_w, epochs=200, lr=1e-2, patience=20):
    """Cổng MLP ngữ cảnh v2 (K nhánh), loss MSE, dừng sớm trên 20% cuối của Val."""
    torch.manual_seed(0)
    Xv = torch.as_tensor(ctx_val, dtype=torch.float32)
    Pv = torch.as_tensor(np.moveaxis(P_val, 0, -1), dtype=torch.float32)  # [T, N, K]
    yv = torch.as_tensor(y_val, dtype=torch.float32)
    T = len(Xv)
    cut = int(T * 0.8)
    gate = PerFlowContextualMetaGating(context_dim=Xv.shape[-1], hidden_dim=32, num_branches=P_val.shape[0],
                                       init_prior=np.log(np.asarray(prior_w) + 1e-4))
    opt = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, wait = float('inf'), None, 0
    for _ in range(epochs):
        gate.train()
        loss = ((torch.sum(gate(Xv[:cut]) * Pv[:cut], -1) - yv[:cut]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        gate.eval()
        with torch.no_grad():
            hl = ((torch.sum(gate(Xv[cut:]) * Pv[cut:], -1) - yv[cut:]) ** 2).mean().item()
        if hl < best:
            best, best_state, wait = hl, {k: v.clone() for k, v in gate.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    gate.load_state_dict(best_state)
    gate.eval()
    with torch.no_grad():
        return gate(torch.as_tensor(ctx_test, dtype=torch.float32)).numpy()  # [T, N, K]


def evaluate_ablation_run(dataset_name, run_id=0):
    ds_key = dataset_name.lower()
    seq_len = DATASET_CONFIGS[ds_key]['seq_len']
    cache = load_cache(ds_key, run_id)
    branches = cache['branches']
    P_val, y_val = cache['val']['P'].numpy(), cache['val']['y'].numpy()
    P_test, y_test = cache['test']['P'].numpy(), cache['test']['y'].numpy()
    K = len(branches)

    preds = {}

    # 1. full: dùng lại đúng trọng số đã học ở train_stacking (không học lại)
    st_file = os.path.join(parent_dir, 'logs', f"{MODEL_TAG}_data_{ds_key}_seq_{seq_len}", f"run_{run_id}", 'stacking.json')
    if os.path.exists(st_file):
        with open(st_file, 'r', encoding='utf-8') as f:
            w_full = np.asarray(json.load(f)['weights'], dtype=np.float32)
    else:
        w_full = RobustPerFlowStacking(branches).fit(P_val, y_val).weights
    preds['full'] = blend(w_full, P_test)

    # 2. Bỏ từng nhánh: cùng thủ tục chọn cấu hình trên K-1 nhánh
    for k, b in enumerate(branches):
        keep = [j for j in range(K) if j != k]
        st = RobustPerFlowStacking([branches[j] for j in keep]).fit(P_val[keep], y_val)
        preds[f'no_{b}'] = st.predict(P_test[keep])

    # 3. Phạm vi trọng số
    preds['global_only'] = RobustPerFlowStacking(branches, force_scope='global').fit(P_val, y_val).predict(P_test)
    preds['perflow_only'] = RobustPerFlowStacking(branches, force_scope='perflow').fit(P_val, y_val).predict(P_test)

    # 4. Trung bình tĩnh và chọn một nhánh
    preds['static_average'] = P_test.mean(axis=0)
    k_best = int(np.argmin([np.mean((P_val[k] - y_val) ** 2) for k in range(K)]))
    preds['best_single_val'] = P_test[k_best]

    # 5. Cổng MLP ngữ cảnh (v2) với K nhánh
    w_glob = RobustPerFlowStacking(branches, force_loss='mse', force_scope='global').fit(P_val, y_val).weights
    W_gate = train_context_gate(cache['val']['context'].numpy(), P_val, y_val, cache['test']['context'].numpy(), w_glob)
    preds['context_gate'] = np.sum(W_gate * np.moveaxis(P_test, 0, -1), axis=-1)

    records = []
    for cfg, yp in preds.items():
        m = calc_metrics_numpy(yp, y_test)
        m['mse_in_range'] = float(np.mean((yp - y_test)[y_test <= 1.0] ** 2))
        m.update({'config': cfg, 'dataset': ds_key.upper(), 'run': run_id})
        if cfg == 'best_single_val':
            m['note'] = branches[k_best]
        records.append(m)
    return records


def run_ablation_experiments(datasets=None, runs=5, run_ids=None):
    if datasets is None or 'all' in datasets:
        datasets = ['sdn', 'geant', 'abilene']
    run_id_list = parse_run_ids(run_ids, runs)
    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(" ABLATION STUDY - ST-ADAPTIVE-ENSEMBLE v3")
    print(f" Datasets: {datasets} | Run IDs: {run_id_list}")
    print("=" * 80)

    new_records = []
    for ds in datasets:
        for r in run_id_list:
            print(f"  [*] {ds.upper()} run_{r}", flush=True)
            new_records.extend(evaluate_ablation_run(ds, run_id=r))
    df_new = pd.DataFrame(new_records)

    out_csv = os.path.join(results_dir, 'ablation_results.csv')
    df_all = df_new
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        key = ['dataset', 'run', 'config']
        if set(key).issubset(old.columns):
            merged = old.merge(df_new[key], on=key, how='left', indicator=True)
            df_all = pd.concat([old[(merged['_merge'] == 'left_only').values], df_new], ignore_index=True)
    df_all.to_csv(out_csv, index=False)

    cur = df_all[df_all['run'].isin(run_id_list) & df_all['dataset'].isin([d.upper() for d in datasets])]
    summary = cur.groupby(['dataset', 'config']).agg(
        runs=('run', 'nunique'),
        mean_mse=('mse', lambda x: np.mean(x) * 1000.0),
        std_mse=('mse', lambda x: np.std(x, ddof=1) * 1000.0 if len(x) > 1 else 0.0),
        mean_mae=('mae', lambda x: np.mean(x) * 1000.0),
        mean_mse_in_range=('mse_in_range', lambda x: np.mean(x) * 1000.0),
    ).reset_index()
    summary['description'] = summary['config'].map(
        lambda c: CONFIG_DESC.get(c, f"Bỏ nhánh {c[3:]} (champion)" if c.startswith('no_') else c))
    summary_csv = os.path.join(results_dir, 'ablation_summary.csv')
    summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')

    print("\nBẢNG TỔNG HỢP ABLATION (MSE, MAE x 10^-3):")
    print(summary.drop(columns=['description']).to_string(index=False))
    print(f"\n-> {out_csv}\n-> {summary_csv}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ablation study v3")
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--run_ids', type=str, default=None, help="Ví dụ '5-9' (ghi đè --runs)")
    args = parser.parse_args()
    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ['sdn', 'geant', 'abilene']
    run_ablation_experiments(datasets=d_list, runs=args.runs, run_ids=args.run_ids)
