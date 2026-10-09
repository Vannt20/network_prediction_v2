"""
P5 - Phân tích hành vi của RL-Gate trên Test, run_0 (spec Mục 8).

- Quỹ đạo trọng số a_t^k của vài luồng (luồng có đỉnh lớn nhất, luồng ổn định nhất), vẽ chồng y và các dự báo.
- Đặc trưng phản ứng của từng luồng: tương quan giữa tổng trọng số nhánh ngoại suy và cờ spike, E||Δa||_1,
  KL trung bình so với w_static, tỷ lệ bước lệch khỏi static > 0,1.
- Phân cụm luồng (k-means, k chọn bằng silhouette trong 2..6).
Đầu ra: results/gnn_rl/hinh_5_quy_dao_trong_so_{ds}.png, hinh_6_phan_cum_luong_{ds}.png, p5_flow_behavior_{ds}.csv
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

from training.gnn_rl_common import (setup_utf8, parse_list, ALL_DATASETS, RESULTS_V4, EXTRAP_BRANCHES, load_cache,
                                    split_arrays, run_dir, load_json)

FEATURES = ['corr_ext_spike', 'mean_switch', 'mean_kl', 'frac_dev_01', 'mean_ext_weight']


def flow_behavior(A, W, ctx, ext_idx):
    """A [T,N,K] trọng số động, W [N,K] static, ctx [T,N,4] (cờ spike ở kênh 1)."""
    T, N, K = A.shape
    ext = A[..., ext_idx].sum(-1) if ext_idx else np.zeros((T, N))
    spike = ctx[..., 1]
    rows = []
    for i in range(N):
        s = spike[:, i]
        corr = float(np.corrcoef(ext[:, i], s)[0, 1]) if s.std() > 0 and ext[:, i].std() > 0 else 0.0
        kl = (A[:, i] * (np.log(np.clip(A[:, i], 1e-8, None)) - np.log(np.clip(W[i], 1e-8, None)))).sum(-1)
        dev = np.abs(A[:, i] - W[i]).sum(-1)
        rows.append({'flow': i, 'corr_ext_spike': corr,
                     'mean_switch': float(np.abs(np.diff(A[:, i], axis=0)).sum(-1).mean()),
                     'mean_kl': float(kl.mean()), 'frac_dev_01': float((dev > 0.1).mean()),
                     'mean_ext_weight': float(ext[:, i].mean())})
    return pd.DataFrame(rows)


def cluster(df, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(df[FEATURES].fillna(0).values)
    best = (None, -1.0, None)
    for k in range(2, min(6, len(df) - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(X, lab)
        if s > best[1]:
            best = (k, s, lab)
    df = df.copy()
    df['cluster'] = best[2] if best[2] is not None else 0
    return df, best[0], best[1]


def analyse(ds, run_id=0, tag='rl_sac'):
    d = run_dir(tag, ds, run_id, seq=False)
    f = os.path.join(d, 'weights_test.npz')
    if not os.path.exists(f):
        print(f"[{ds.upper()}] chưa có {f}, bỏ qua P5")
        return None
    cfg = load_json(os.path.join(d, 'config.json'))
    branches = cfg['branches']
    A = np.load(f)['w'].astype(np.float32)                                 # [T, N, K]
    c = load_cache(ds, run_id)
    P, y, last = split_arrays(c, 'test', branches)
    ctx = c['test']['context'].numpy()
    from Graph_models.robust_stacking import RobustPerFlowStacking
    Pv, yv, _ = split_arrays(c, 'val', branches)
    W = RobustPerFlowStacking(branches).fit(Pv, yv).per_flow_weights(y.shape[1]).T   # [N, K]
    T = A.shape[0]
    P, y, last, ctx = P[:, :T], y[:T], last[:T], ctx[:T]
    ext_idx = [k for k, b in enumerate(branches) if b in EXTRAP_BRANCHES]
    df = flow_behavior(A, W, ctx, ext_idx)
    df, k, sil = cluster(df)
    from features.spatial_features import parse_od_columns
    from features.feature_store import load_raw_dataset
    pairs = parse_od_columns(list(load_raw_dataset(ds).columns))
    df['src'] = [s for s, _ in pairs]
    df['dst'] = [t for _, t in pairs]
    df['y_std'] = y.std(axis=0)
    df['y_max'] = y.max(axis=0)
    os.makedirs(RESULTS_V4, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_V4, f'p5_flow_behavior_{ds}.csv'), index=False)
    summ = df.groupby('cluster')[FEATURES + ['y_std', 'y_max']].mean()
    summ['n_flows'] = df.groupby('cluster').size()
    summ.to_csv(os.path.join(RESULTS_V4, f'p5_clusters_{ds}.csv'))
    print(f"[{ds.upper()}] k = {k} cụm (silhouette {sil:.3f})")
    print(summ.round(4).to_string())

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[CẢNH BÁO] không nạp được matplotlib ({e}); bỏ qua hình, đã ghi CSV")
        return df

    # Hình 5: quỹ đạo trọng số
    act = y.std(axis=0) > 0
    i_peak = int(np.argmax(y.max(axis=0)))
    stab = np.where(act, y.std(axis=0), np.inf)
    i_stable = int(np.argmin(stab))
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    for col, (i, name) in enumerate([(i_peak, 'đỉnh lớn nhất'), (i_stable, 'ổn định nhất')]):
        ax = axes[0, col]
        ax.plot(y[:, i], color='black', lw=1.0, label='thực tế')
        for kk, b in enumerate(branches):
            ax.plot(P[kk, :, i], lw=0.7, alpha=0.7, label=b)
        ax.plot((A[:, i] * np.moveaxis(P[:, :, i], 0, -1)).sum(-1), color='red', lw=0.9, label='RL-Gate')
        ax.set_title(f"Luồng OD_{pairs[i][0]}-{pairs[i][1]} ({name})")
        ax.legend(fontsize=7, loc='upper right')
        ax2 = axes[1, col]
        ax2.stackplot(np.arange(T), A[:, i].T, labels=branches, alpha=0.8)
        for kk in range(len(branches)):
            ax2.axhline(np.cumsum(W[i])[kk], color='k', lw=0.4, ls='--')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('trọng số')
        ax2.legend(fontsize=7, loc='lower right')
    fig.suptitle(f"{ds.upper()} - quỹ đạo trọng số RL-Gate trên Test (run_{run_id}); nét đứt: trọng số static")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_V4, f'hinh_5_quy_dao_trong_so_{ds}.png'), dpi=150)
    plt.close(fig)

    # Hình 6: phân cụm
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df['mean_kl'], df['corr_ext_spike'], c=df['cluster'], s=12 + 60 * df['frac_dev_01'], cmap='tab10')
    ax.set_xlabel('KL trung bình so với static')
    ax.set_ylabel('tương quan (trọng số nhánh ngoại suy, cờ spike)')
    ax.set_title(f"{ds.upper()} - phân cụm luồng theo phản ứng của RL-Gate (k={k})")
    fig.colorbar(sc, ax=ax, label='cụm')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_V4, f'hinh_6_phan_cum_luong_{ds}.png'), dpi=150)
    plt.close(fig)
    return df


if __name__ == '__main__':
    setup_utf8()
    ap = argparse.ArgumentParser(description="P5: phân tích hành vi RL-Gate")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_id', type=int, default=0)
    a = ap.parse_args()
    for ds in parse_list(a.datasets, ALL_DATASETS):
        analyse(ds, a.run_id)
