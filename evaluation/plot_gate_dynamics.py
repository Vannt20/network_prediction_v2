"""
Hình vẽ phân tích trọng số tầng kết hợp và hành vi theo luồng (ST-Adaptive-Ensemble v3).
Mọi hình đều được vẽ từ kết quả thực nghiệm (results/, logs/, cache/v3/); không có dữ liệu giả lập.
Nếu thiếu dữ liệu, hình tương ứng được bỏ qua và in cảnh báo.
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.feature_store import DATASET_CONFIGS
from training.precompute_cache import cache_path

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

LABEL = {'stwaveformer': 'ST-WaveFormer', 'xgboost': 'XGBoost', 'lightgbm_res': 'LightGBM-Residual',
         'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}
COLOR = {'stwaveformer': '#1f77b4', 'xgboost': '#2ca02c', 'lightgbm_res': '#d62728',
         'lightgbm': '#9467bd', 'catboost': '#8c564b'}
DATASETS = ['sdn', 'geant', 'abilene']


def _ens_logdir(ds):
    return os.path.join(parent_dir, 'logs', f"st_adaptive_ensemble_data_{ds}_seq_{DATASET_CONFIGS[ds]['seq_len']}")


def _available_runs(ds):
    runs = []
    for d in glob.glob(os.path.join(_ens_logdir(ds), 'run_*')):
        if os.path.exists(os.path.join(d, 'weights.npy')):
            runs.append(int(os.path.basename(d).split('_')[1]))
    return sorted(runs)


def _branches(ds, run_id):
    import torch
    return torch.load(cache_path(ds, run_id), map_location='cpu')['branches']


def plot_weights_across_datasets(results_dir):
    """Hình 1: trọng số trung bình của từng nhánh trên mỗi bộ dữ liệu (trung bình qua luồng và run)."""
    rows = []
    for ds in DATASETS:
        runs = _available_runs(ds)
        if not runs:
            continue
        br = _branches(ds, runs[0])
        W = np.mean([np.load(os.path.join(_ens_logdir(ds), f'run_{r}', 'weights.npy')) for r in runs], axis=0)
        for k, b in enumerate(br):
            rows.append({'dataset': ds.upper(), 'branch': b, 'w': W[k].mean() * 100})
    if not rows:
        print("[SKIP] Hình 1: chưa có weights.npy")
        return
    df = pd.DataFrame(rows)
    branches = list(dict.fromkeys(df['branch']))
    dss = list(dict.fromkeys(df['dataset']))
    x = np.arange(len(dss))
    width = 0.8 / len(branches)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    for i, b in enumerate(branches):
        vals = [df[(df.dataset == d) & (df.branch == b)]['w'].sum() for d in dss]
        bars = ax.bar(x + (i - (len(branches) - 1) / 2) * width, vals, width, label=LABEL.get(b, b),
                      color=COLOR.get(b), edgecolor='black', linewidth=0.5)
        for rect, v in zip(bars, vals):
            ax.annotate(f'{v:.1f}%', (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 2),
                        textcoords='offset points', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(dss)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Trọng số trung bình (%)')
    ax.set_title('Trọng số của tầng kết hợp theo bộ dữ liệu')
    ax.legend(frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(results_dir, 'hinh_1_trong_so_nhanh_theo_dataset.png')
    plt.savefig(out)
    plt.close()
    print(f"[*] {out}")


def plot_flow_weight_heatmap(results_dir, ds='sdn'):
    """Hình 2: trọng số theo từng luồng OD (ma trận nguồn x đích) cho từng nhánh."""
    runs = _available_runs(ds)
    if not runs:
        print(f"[SKIP] Hình 2 ({ds}): chưa có weights.npy")
        return
    br = _branches(ds, runs[0])
    W = np.mean([np.load(os.path.join(_ens_logdir(ds), f'run_{r}', 'weights.npy')) for r in runs], axis=0)
    n_nodes = int(round(np.sqrt(W.shape[1])))
    if n_nodes * n_nodes != W.shape[1]:
        print(f"[SKIP] Hình 2 ({ds}): số luồng không phải bình phương số nút")
        return
    fig, axes = plt.subplots(1, len(br), figsize=(5.2 * len(br), 4.6), dpi=300)
    for k, (ax, b) in enumerate(zip(np.atleast_1d(axes), br)):
        sns.heatmap(W[k].reshape(n_nodes, n_nodes), ax=ax, vmin=0, vmax=1, cmap='YlGnBu',
                    cbar=k == len(br) - 1, square=True, xticklabels=range(1, n_nodes + 1),
                    yticklabels=range(1, n_nodes + 1))
        ax.set_title(LABEL.get(b, b))
        ax.set_xlabel('Nút đích')
        ax.set_ylabel('Nút nguồn' if k == 0 else '')
    fig.suptitle(f'Trọng số theo luồng OD – {ds.upper()} (trung bình {len(runs)} run)')
    plt.tight_layout()
    out = os.path.join(results_dir, f'hinh_2_heatmap_trong_so_luong_{ds}.png')
    plt.savefig(out)
    plt.close()
    print(f"[*] {out}")


def plot_spike_case_study(results_dir, ds='sdn', half_window=30):
    """Hình 3: đoạn Test có bước tăng đột biến lớn nhất – thực tế vs ensemble vs từng nhánh (dữ liệu thật)."""
    import torch
    runs = _available_runs(ds)
    if not runs:
        print(f"[SKIP] Hình 3 ({ds}): chưa có kết quả ensemble")
        return
    r = runs[0]
    c = torch.load(cache_path(ds, r), map_location='cpu')
    y = c['test']['y'].numpy()
    P = c['test']['P'].numpy()
    y_ens = np.load(os.path.join(_ens_logdir(ds), f'run_{r}', 'y_pred_data.npy'))
    jump = np.abs(np.diff(y, axis=0))
    t_star, flow = np.unravel_index(np.argmax(jump), jump.shape)
    t_star += 1
    lo, hi = max(0, t_star - half_window), min(len(y), t_star + half_window)
    t = np.arange(lo, hi)
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.plot(t, y[lo:hi, flow], 'k-', lw=2, label='Thực tế')
    ax.plot(t, y_ens[lo:hi, flow], 'r--', lw=1.8, label='ST-Adaptive-Ensemble v3')
    for k, b in enumerate(c['branches']):
        ax.plot(t, P[k, lo:hi, flow], lw=1, alpha=0.8, color=COLOR.get(b), label=LABEL.get(b, b))
    ax.axvline(t_star, color='orange', alpha=0.4, lw=6)
    ax.set_xlabel('Bước thời gian trên tập Test')
    ax.set_ylabel('Lưu lượng (Min–Max)')
    ax.set_title(f'{ds.upper()} – luồng #{flow}, bước tăng đột biến lớn nhất (run_{r})')
    ax.legend(fontsize=8, ncol=3, frameon=True, framealpha=0.9, loc='upper left')
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(results_dir, f'hinh_3_case_study_{ds}.png')
    plt.savefig(out)
    plt.close()
    print(f"[*] {out}")


def plot_out_of_range_error(results_dir):
    """Hình 4: tỷ trọng sai số bình phương đến từ các điểm vượt biên Train (y > 1) trên tập Test."""
    import torch
    rows = []
    for ds in DATASETS:
        runs = _available_runs(ds)
        if not runs:
            continue
        r = runs[0]
        c = torch.load(cache_path(ds, r), map_location='cpu')
        y = c['test']['y'].numpy()
        oor = y > 1.0
        preds = {b: c['test']['P'][k].numpy() for k, b in enumerate(c['branches'])}
        preds['ensemble'] = np.load(os.path.join(_ens_logdir(ds), f'run_{r}', 'y_pred_data.npy'))
        for b, p in preds.items():
            e = (p - y) ** 2
            rows.append({'dataset': f"{ds.upper()}\n({oor.mean()*100:.2f}% điểm)", 'model': b,
                         'share': e[oor].sum() / e.sum() * 100})
    if not rows:
        print("[SKIP] Hình 4: chưa có cache/ensemble")
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=300)
    models = list(dict.fromkeys(df['model']))
    dss = list(dict.fromkeys(df['dataset']))
    x = np.arange(len(dss))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [df[(df.dataset == d) & (df.model == m)]['share'].sum() for d in dss]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
               label='Ensemble v3' if m == 'ensemble' else LABEL.get(m, m),
               color='#ff7f0e' if m == 'ensemble' else COLOR.get(m), edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(dss)
    ax.set_ylabel('Tỷ trọng tổng sai số bình phương (%)')
    ax.set_title('Đóng góp của các điểm vượt biên Train (y > 1) vào MSE trên tập Test')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(results_dir, 'hinh_4_sai_so_diem_vuot_bien.png')
    plt.savefig(out)
    plt.close()
    print(f"[*] {out}")


def plot_all_thesis_figures(results_dir=None):
    results_dir = results_dir or os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_weights_across_datasets(results_dir)
    plot_flow_weight_heatmap(results_dir, 'sdn')
    plot_spike_case_study(results_dir, 'sdn')
    plot_out_of_range_error(results_dir)


if __name__ == '__main__':
    plot_all_thesis_figures()
