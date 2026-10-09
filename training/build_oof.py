"""
P2 - Dự báo out-of-fold (OOF) một fold, dùng chung cho mọi run (spec Mục 5.2).

Mỗi nhánh học trên 60% đầu của Train (10% cuối phần này để dừng sớm), dự báo 40% cuối Train.
Seed 42 (cấu hình của run_0). Scaler là MinMaxScaler fit trên toàn Train như v3.
Vòng 2 có thêm ST-WaveFormer (RL-Gate dùng cả 4 nhánh): cùng mô hình và vòng huấn luyện của v3
(run_experiments.train_and_eval_model), giới hạn số epoch để giữ ngân sách 1 ngày.
Kết quả: cache/v4*/{ds}_oof.pt
  {'branches', 'train_frac', 'P': [K,T,N], 'y', 'last', 'context', 'sigma', 'tod', 'dow'}
"""
import os
import sys
import time
import argparse
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.gnn_rl_common import (setup_utf8, parse_list, ALL_DATASETS, LOGS_V4, RESULTS_V4, ODGF_MAX_STEPS,
                                    oof_file, cache_file, load_splits, comb_array, get_device, save_json, oof_ranges)
from features.feature_store import DATASET_CONFIGS, prepare_feature_store
from features.temporal_features import extract_context_features_torch

DEFAULT_BRANCHES = ['stwaveformer', 'xgboost', 'lightgbm_res', 'odgraphformer']   # cùng thứ tự với cache v4
ST_EPOCHS, ST_PATIENCE = 60, 10


def oof_dir(branch, ds_key):
    return os.path.join(LOGS_V4, 'oof', f'{branch}_data_{ds_key}')


def odgf_oof(ds_key, train_frac, quick_check, skip_existing, max_steps=None):
    from training.train_od_graphformer import build_parser, default_config, prepare, train_one
    out = oof_dir('odgraphformer', ds_key)
    if not (skip_existing and os.path.exists(os.path.join(out, 'pred_oof.npy'))):
        args = build_parser().parse_args([])
        args.graphs = ['route', 'od', 'adp']
        args.train_frac = train_frac
        args.max_steps = max_steps or int(ODGF_MAX_STEPS[ds_key] * train_frac)
        device = get_device()
        cfg = default_config(ds_key, args)
        cfg['tag'] = 'odgraphformer_oof'
        combs, graphs = prepare(ds_key, cfg, device)
        train_one(ds_key, 0, cfg, combs, graphs, out, device, quick_check=quick_check)
    return (np.load(os.path.join(out, 'pred_oof.npy')), np.load(os.path.join(out, 'sigma_oof.npy')),
            np.load(os.path.join(out, 'y_oof.npy')))


def st_oof(ds_key, train_frac, quick_check, skip_existing, epochs=ST_EPOCHS, patience=ST_PATIENCE):
    """ST-WaveFormer học trên 60% đầu Train, dự báo 40% cuối (không cắt về >= 0, giống dự báo trong cache v3)."""
    out = oof_dir('stwaveformer', ds_key)
    f = os.path.join(out, 'pred_oof.npy')
    if skip_existing and os.path.exists(f):
        return np.load(f)
    from torch.utils.data import DataLoader
    from run_experiments import TrafficDataset, train_and_eval_model, set_seed, device
    from Graph_models.st_waveformer import STWaveFormer
    cfg = DATASET_CONFIGS[ds_key]
    L, N = cfg['seq_len'], cfg['flows']
    sp = load_splits(ds_key)
    comb = comb_array(sp['train'])
    fit_t, es_t, oof_t = oof_ranges(len(comb), L, train_frac)
    if quick_check:
        fit_t, es_t, epochs, patience = fit_t[:128], es_t[:32], 2, 2
    view = np.lib.stride_tricks.sliding_window_view(comb, L, axis=0)       # [T-L+1, N, 3, L]; cửa sổ j <-> đích j + L

    def loader(t, shuffle):
        x = np.ascontiguousarray(np.transpose(view[t - L], (0, 3, 1, 2)))   # [M, L, N, 3]
        return DataLoader(TrafficDataset('stwaveformer', x, comb[t, :, 0], device), batch_size=64, shuffle=shuffle)
    set_seed(42)
    model = STWaveFormer(input_dim=N, num_nodes=cfg['nodes'], seq_len=L, d_model=64, num_layers=2)
    t0 = time.time()
    train_and_eval_model(model, loader(fit_t, True), loader(es_t, False), loader(oof_t, False), scaler=sp['scaler'],
                         columns=sp['columns'], epochs=epochs, patience=patience, logdir=out,
                         model_name='STWaveFormer', dataset_name=ds_key, run_id=0, seed=42)
    pred = np.load(os.path.join(out, 'y_pred_data.npy')).astype(np.float32)
    assert np.allclose(np.load(os.path.join(out, 'y_real_data.npy')), comb[oof_t, :, 0], atol=1e-6), \
        "Lệch nhãn OOF của ST-WaveFormer"
    for raw in ('y_pred_data_raw.npy', 'y_real_data_raw.npy'):        # bản thang gốc không dùng, bỏ cho nhẹ repo
        if os.path.exists(os.path.join(out, raw)):
            os.remove(os.path.join(out, raw))
    np.save(f, pred)
    save_json({'branch': 'stwaveformer', 'dataset': ds_key, 'train_frac': train_frac, 'seed': 42,
               'cut': int(oof_t[0]), 'epochs_max': epochs, 'patience': patience, 'n_fit_steps': len(fit_t),
               'n_es_steps': len(es_t), 'n_oof_steps': len(oof_t), 'fit_time_s': time.time() - t0},
              os.path.join(out, 'config.json'))
    print(f"    [stwaveformer] OOF {pred.shape[0]} bước, {(time.time() - t0) / 60:.1f} phút", flush=True)
    return pred


def gbdt_oof(name, ds_key, train_frac, quick_check, skip_existing, fs=None):
    from baselines_ml.run_ml_baselines import get_model_instance
    out = oof_dir(name, ds_key)
    f = os.path.join(out, 'pred_oof.npy')
    if skip_existing and os.path.exists(f):
        return np.load(f), fs
    if fs is None:
        fs = prepare_feature_store(ds_key, align_to_seq_len=True, data_dir=os.path.join(parent_dir, 'data'))
    (X_tr, y_tr), _, _, meta = fs
    L, N = DATASET_CONFIGS[ds_key]['seq_len'], meta['num_flows']
    T_train = len(meta['train_norm'])
    fit_t, es_t, oof_t = oof_ranges(T_train, L, train_frac)
    assert len(y_tr) // N == T_train - L, "Lệch số cửa sổ Train giữa dữ liệu bảng và chuỗi"
    rows = lambda t: slice((t[0] - L) * N, (t[-1] - L + 1) * N)          # cửa sổ j <-> bước đích j + L
    Xf, yf = X_tr[rows(fit_t)], y_tr[rows(fit_t)]
    Xe, ye = X_tr[rows(es_t)], y_tr[rows(es_t)]
    Xo = X_tr[rows(oof_t)]
    if quick_check:
        Xf, yf, Xe, ye = Xf[:2000], yf[:2000], Xe[:500], ye[:500]
    t0 = time.time()
    inst = get_model_instance(name, seed=42, quick_check=quick_check, feature_names=meta['feature_names'])
    inst.fit(Xf, yf, X_val=Xe, y_val=ye)
    pred = np.clip(np.asarray(inst.predict(Xo), dtype=np.float64), 0.0, None).reshape(-1, N).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    np.save(f, pred)
    save_json({'branch': name, 'dataset': ds_key, 'train_frac': train_frac, 'seed': 42, 'cut': int(oof_t[0]),
               'n_fit_steps': len(fit_t), 'n_es_steps': len(es_t), 'n_oof_steps': pred.shape[0],
               'fit_time_s': time.time() - t0}, os.path.join(out, 'config.json'))
    print(f"    [{name}] OOF {pred.shape[0]} bước, {time.time() - t0:.0f} s", flush=True)
    return pred, fs


def build_oof(ds_key, branches=None, train_frac=0.6, quick_check=False, skip_existing=False, max_steps=None,
              st_epochs=ST_EPOCHS, st_patience=ST_PATIENCE, assemble_only=False):
    branches = branches or DEFAULT_BRANCHES
    if assemble_only:
        # Chỉ ghép dự báo đã có (không bao giờ huấn luyện ở bước này, tránh tự huấn luyện ST trên CPU)
        miss = [b for b in branches if not os.path.exists(os.path.join(oof_dir(b, ds_key), 'pred_oof.npy'))]
        if miss:
            raise FileNotFoundError(f"[OOF {ds_key.upper()}] thiếu dự báo OOF của {miss}, không ghép được")
        skip_existing = True
    L = DATASET_CONFIGS[ds_key]['seq_len']
    sp = load_splits(ds_key)
    T_train = len(sp['train']['x'])
    cut = int(T_train * train_frac)
    print(f"[OOF {ds_key.upper()}] học trên {cut} bước đầu Train, dự báo {T_train - cut} bước cuối | nhánh {branches}",
          flush=True)
    P, sigma, fs, y_ref = [], None, None, None
    for b in branches:
        if b == 'odgraphformer':
            p, sigma, y_ref = odgf_oof(ds_key, train_frac, quick_check, skip_existing, max_steps)
        elif b == 'stwaveformer':
            p = st_oof(ds_key, train_frac, quick_check, skip_existing, st_epochs, st_patience)
        else:
            p, fs = gbdt_oof(b, ds_key, train_frac, quick_check, skip_existing, fs)
        P.append(p)
    if not assemble_only and set(branches) != set(DEFAULT_BRANCHES):
        # Job của riêng một vài nhánh: chỉ lưu dự báo của nhánh đó, không ghi file OOF chung thiếu nhánh
        print(f"[OOF {ds_key.upper()}] đã có dự báo OOF của {branches}; file chung do bước --assemble_only ghép",
              flush=True)
        return None
    T_oof = min(len(p) for p in P)
    P = np.stack([p[:T_oof] for p in P]).astype(np.float32)
    x = sp['train']['x']
    tau = np.arange(cut, cut + T_oof)
    y = x[tau]
    if y_ref is not None:
        assert np.allclose(y_ref[:T_oof], y, atol=1e-6), "Lệch nhãn OOF giữa OD-GraphFormer và dữ liệu"
    comb = torch.from_numpy(comb_array(sp['train']))
    win = comb[torch.as_tensor(tau)[:, None] + torch.arange(-5, 0)[None, :]]            # [T, 5, N, 3]
    context = extract_context_features_torch(win)
    data = {
        'dataset': ds_key, 'branches': branches, 'train_frac': train_frac, 'cut': cut, 'seq_len': L,
        'P': torch.from_numpy(P), 'y': torch.from_numpy(y.astype(np.float32)),
        'last': torch.from_numpy(x[tau - 1].astype(np.float32)), 'context': context,
        'sigma': torch.from_numpy((sigma[:T_oof] if sigma is not None else
                                   np.full_like(y, np.nan)).astype(np.float32)),
        'tod': torch.from_numpy(sp['train']['tod'][tau]), 'dow': torch.from_numpy(sp['train']['dow'][tau]),
    }
    os.makedirs(os.path.dirname(oof_file(ds_key)), exist_ok=True)
    with open(oof_file(ds_key), 'wb') as f:
        torch.save(data, f)
    mse = {b: float(np.mean((P[k] - y) ** 2)) for k, b in enumerate(branches)}
    print(f"[OOF {ds_key.upper()}] MSE e-3: " + " | ".join(f"{b}={v*1e3:.3f}" for b, v in mse.items())
          + f" -> {oof_file(ds_key)}", flush=True)
    return data


def compare_oof_val(ds_key, run_ids=(0,)):
    """Nghiệm thu P2: tỷ số MSE(OOF)/MSE(Val) và thứ hạng nhánh (spec Mục 5.4)."""
    import pandas as pd
    from scipy import stats
    from training.gnn_rl_common import torch_load, load_cache, split_arrays
    o = torch_load(oof_file(ds_key))
    rows = []
    for r in run_ids:
        c = load_cache(ds_key, r)
        Pv, yv, _ = split_arrays(c, 'val', o['branches'])
        for k, b in enumerate(o['branches']):
            eo = np.abs(o['P'][k].numpy() - o['y'].numpy()).reshape(-1)
            ev = np.abs(Pv[k] - yv).reshape(-1)
            rng = np.random.default_rng(0)
            ks = stats.ks_2samp(rng.choice(eo, min(20000, eo.size), replace=False),
                                rng.choice(ev, min(20000, ev.size), replace=False))
            rows.append({'dataset': ds_key, 'run': r, 'branch': b, 'mse_oof': float((eo ** 2).mean()),
                         'mse_val': float((ev ** 2).mean()), 'ratio': float((eo ** 2).mean() / (ev ** 2).mean()),
                         'ks_stat': float(ks.statistic), 'ks_p': float(ks.pvalue)})
    df = pd.DataFrame(rows)
    rank_o = df.groupby('branch')['mse_oof'].mean().rank()
    rank_v = df.groupby('branch')['mse_val'].mean().rank()
    df['same_rank'] = bool((rank_o == rank_v).all())
    os.makedirs(RESULTS_V4, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_V4, f'p2_oof_vs_val_{ds_key}.csv'), index=False)
    print(df.to_string(index=False))
    return df


if __name__ == '__main__':
    setup_utf8()
    ap = argparse.ArgumentParser(description="P2: dự báo OOF một fold trên Train (dùng chung cho mọi run)")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--branches', default=','.join(DEFAULT_BRANCHES))
    ap.add_argument('--train_frac', type=float, default=0.6)
    ap.add_argument('--max_steps', type=int, default=None, help="Số bước OD-GraphFormer (mặc định 0,6 x ngân sách)")
    ap.add_argument('--st_epochs', type=int, default=ST_EPOCHS, help="Số epoch tối đa của ST-WaveFormer khi làm OOF")
    ap.add_argument('--st_patience', type=int, default=ST_PATIENCE)
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--quick_check', action='store_true')
    ap.add_argument('--compare_only', action='store_true', help="Chỉ so OOF với Val (cần cache v4)")
    ap.add_argument('--assemble_only', action='store_true', help="Chỉ ghép dự báo OOF đã có, không huấn luyện")
    a = ap.parse_args()
    for ds in parse_list(a.datasets, ALL_DATASETS):
        if not a.compare_only:
            build_oof(ds, parse_list(a.branches, DEFAULT_BRANCHES), a.train_frac, a.quick_check, a.skip_existing,
                      a.max_steps, a.st_epochs, a.st_patience, a.assemble_only)
        if os.path.exists(oof_file(ds)) and (a.compare_only or os.path.exists(cache_file(ds, 0))):
            compare_oof_val(ds)
