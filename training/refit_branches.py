"""
Huấn luyện lại (refit) 3 nhánh của ST-Adaptive-Ensemble v3 trên Train+Val, không có tập Val để early stopping.

- Chia dữ liệu và MinMaxScaler giữ nguyên v3 (scaler fit CHỈ trên Train, lấy từ prepare_feature_store).
- Dữ liệu huấn luyện = chuỗi Train+Val liên tục; cửa sổ trượt / đặc trưng tạo trên chuỗi này.
  Test vẫn tạo riêng từ đoạn Test như v3 (không có cửa sổ vắt qua ranh giới Val/Test).
- Số vòng huấn luyện lấy từ lần huấn luyện gốc của chính run đó:
    stwaveformer : E* = argmin(val_loss) + 1 trong logs/stwaveformer_data_{ds}_seq_{L}/run_{r}/train_metrics.csv,
                   CosineAnnealingLR với T_max = 200 (số epoch tối đa gốc), lưu trọng số ở epoch cuối.
    xgboost      : best_iteration + 1 trong logs/xgboost_data_{ds}_shared/run_{r}/model.bin
    lightgbm_res : best_iteration_ của logs/lightgbm_res_data_{ds}_shared/run_{r}/model.bin
- Dự báo Test lưu float32 [T_test, N] cùng thứ tự với cache/v3/{ds}_run_{r}.pt['test'];
  y_test của pipeline mới phải khớp cache (sai lệch < 1e-6), nếu không thì dừng.

Kết quả: logs_refit/{model}_data_{ds}/run_{r}/{test_pred.npy, meta.json[, model.pth, train_metrics.csv]}.
meta.json được ghi sau cùng, nên có meta.json (quick_check = false) nghĩa là đơn vị đã xong.

Ví dụ:  python training/refit_branches.py --model xgboost --dataset geant --run 3
        python training/refit_branches.py --model stwaveformer --dataset sdn --run 0 --quick_check
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import argparse
import json
import time

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.feature_store import prepare_feature_store, DATASET_CONFIGS
from features.spatial_features import compute_spatial_neighbors
from features.temporal_features import extract_temporal_features_matrix

MODELS = ['stwaveformer', 'xgboost', 'lightgbm_res']
OUT_ROOT = 'logs_refit'
DL_MAX_EPOCHS = 200          # số epoch tối đa của lần huấn luyện gốc (T_max của CosineAnnealingLR)
QUICK_EPOCHS, QUICK_TREES = 2, 20
Y_TOL = 1e-6


def unit_dir(model, ds, run_id, root=None):
    return os.path.join(root or os.path.join(parent_dir, OUT_ROOT), f"{model}_data_{ds}", f"run_{run_id}")


def orig_dir(model, ds, run_id):
    if model == 'stwaveformer':
        return os.path.join(parent_dir, 'logs', f"stwaveformer_data_{ds}_seq_{DATASET_CONFIGS[ds]['seq_len']}", f"run_{run_id}")
    return os.path.join(parent_dir, 'logs', f"{model}_data_{ds}_shared", f"run_{run_id}")


def unit_done(model, ds, run_id, root=None):
    """Đơn vị đã xong khi có meta.json của lần chạy thật (không phải quick_check) và test_pred.npy hợp lệ."""
    d = unit_dir(model, ds, run_id, root)
    try:
        with open(os.path.join(d, 'meta.json'), encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('quick_check'):
            return False
        pred = np.load(os.path.join(d, 'test_pred.npy'), mmap_mode='r')
        return pred.shape == tuple(meta['shape']) and pred.dtype == np.float32
    except (OSError, ValueError, KeyError):
        return False


# ---------------------------------------------------------------- số vòng huấn luyện của lần chạy gốc
def n_rounds(model, ds, run_id):
    """(số epoch / số cây, đường dẫn nguồn) đọc từ lần huấn luyện gốc của run."""
    d = orig_dir(model, ds, run_id)
    if model == 'stwaveformer':
        f = os.path.join(d, 'train_metrics.csv')
        return int(pd.read_csv(f)['val_loss'].values.argmin()) + 1, f
    f = os.path.join(d, 'model.bin')
    if model == 'xgboost':
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(f)
        return int(booster.attributes()['best_iteration']) + 1, f
    import joblib
    m = joblib.load(f)['model']
    return int(m.best_iteration_ or m.n_estimators_), f          # best_iteration_ = 0 khi không có early stopping


# ---------------------------------------------------------------- dữ liệu
def load_data(ds):
    """Gọi prepare_feature_store của v3 (giữ nguyên split, scaler fit trên Train)."""
    return prepare_feature_store(ds, align_to_seq_len=True, data_dir=os.path.join(parent_dir, 'data'))


def tabular_split(traffic, tod, dow, ds, meta):
    """
    Tái tạo process_split của prepare_feature_store cho một chuỗi bất kỳ (ở đây là Train+Val liên tục).
    Hàm cũ là hàm lồng nên không gọi lại được; check_tabular_replica bảo đảm hai bản cho kết quả trùng khớp.
    """
    seq_len = DATASET_CONFIGS[ds]['seq_len']
    feats, k_valid, names = extract_temporal_features_matrix(traffic, tod, dow, k_lags=meta['k_lags'], windows=(3, 5, 10))
    if seq_len > k_valid:
        off = seq_len - k_valid
        feats = {n: feats[n][off:] for n in names}
        k_valid = seq_len
    T_len = len(traffic)
    nbr_in, nbr_out = compute_spatial_neighbors(traffic[k_valid - 1: T_len - 1], meta['M_in'], meta['M_out'])
    feats['neighbor_sum_in'], feats['neighbor_sum_out'] = nbr_in, nbr_out
    all_names = list(names) + ['neighbor_sum_in', 'neighbor_sum_out']
    assert ['flow_id'] + all_names == list(meta['feature_names']), "Thứ tự đặc trưng khác v3"
    y = traffic[k_valid:T_len]
    T_valid, N = y.shape
    X = np.column_stack([np.tile(np.arange(N, dtype=np.int32), T_valid)] +
                        [feats[n].reshape(-1) for n in all_names]).astype(np.float32)
    return X, y.reshape(-1).astype(np.float32)


def check_tabular_replica(ds, meta, X_train, y_train):
    X, y = tabular_split(meta['train_norm'], meta['tod_train'], meta['dow_train'], ds, meta)
    if X.shape != X_train.shape or not np.array_equal(X, X_train) or not np.array_equal(y, y_train):
        raise RuntimeError(f"[{ds}] tabular_split không tái tạo đúng đặc trưng Train của v3")


def trainval_series(meta):
    cat = np.concatenate
    return (cat([meta['train_norm'], meta['val_norm']]), cat([meta['tod_train'], meta['tod_val']]),
            cat([meta['dow_train'], meta['dow_val']]))


def cache_y_test(ds, run_id):
    import torch
    c = torch.load(os.path.join(parent_dir, 'cache', 'v3', f"{ds}_run_{run_id}.pt"), map_location='cpu', weights_only=False)
    k = c['branches']
    return c['test']['y'].numpy(), c['test']['P'].numpy(), k


def check_y_test(y_new, y_cache, tag):
    if y_new.shape != y_cache.shape:
        raise RuntimeError(f"{tag}: y_test có kích thước {y_new.shape}, cache/v3 là {y_cache.shape}")
    diff = float(np.max(np.abs(y_new.astype(np.float64) - y_cache.astype(np.float64))))
    if diff >= Y_TOL:
        raise RuntimeError(f"{tag}: y_test lệch cache/v3 {diff:.2e} (>= {Y_TOL:g}) -> dừng")
    return diff


# ---------------------------------------------------------------- GBDT
def refit_gbdt(model, ds, run_id, quick_check=False, log=print):
    from baselines_ml.run_ml_baselines import get_model_instance
    seed = 42 + run_id
    n, src = n_rounds(model, ds, run_id)
    if quick_check:
        n = QUICK_TREES
    (X_train, y_train), _, (X_te, y_te), meta = load_data(ds)
    check_tabular_replica(ds, meta, X_train, y_train)
    del X_train, y_train
    X_tv, y_tv = tabular_split(*trainval_series(meta), ds, meta)
    T_test, N = meta['T_test'], meta['num_flows']
    y_cache, P_cache, branches = cache_y_test(ds, run_id)
    diff = check_y_test(y_te.reshape(T_test, N), y_cache, f"[{model} {ds} run_{run_id}]")
    log(f"{model} {ds} run_{run_id}: Train+Val {X_tv.shape}, {n} cây (gốc: {src}), seed {seed}")

    inst = get_model_instance(model, seed=seed, feature_names=meta['feature_names'])
    inst.params['n_estimators'] = n             # siêu tham số khác giữ nguyên v3
    t0 = time.time()
    inst.fit(X_tv, y_tv)                        # không truyền Val -> không early stopping, đúng n cây
    fit_s = time.time() - t0
    pred = np.asarray(inst.predict(X_te), dtype=np.float32).reshape(T_test, N)
    extra = {'n_trees': n, 'fit_time_s': round(fit_s, 1)}
    return pred, y_cache, P_cache[branches.index(model)], extra, src, diff, len(y_tv)


# ---------------------------------------------------------------- ST-WaveFormer
def refit_stwaveformer(ds, run_id, quick_check=False, log=print, out_dir=None):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from run_experiments import set_seed, SEEDS
    from training.precompute_cache import create_sliding_windows
    from Graph_models.st_waveformer import STWaveFormer

    cfg = DATASET_CONFIGS[ds]
    seed = SEEDS[run_id % len(SEEDS)]
    e_star, src = n_rounds('stwaveformer', ds, run_id)
    epochs = QUICK_EPOCHS if quick_check else e_star
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(seed)                              # cùng thứ tự như run_experiments: seed -> dữ liệu -> mô hình -> loader
    _, _, _, meta = load_data(ds)
    x_tr, y_tr = create_sliding_windows(*trainval_series(meta), cfg['seq_len'])
    x_te, y_te = create_sliding_windows(meta['test_norm'], meta['tod_test'], meta['dow_test'], cfg['seq_len'])
    y_cache, P_cache, branches = cache_y_test(ds, run_id)
    diff = check_y_test(y_te.numpy(), y_cache, f"[stwaveformer {ds} run_{run_id}]")
    del meta

    model = STWaveFormer(input_dim=cfg['flows'], num_nodes=cfg['nodes'], seq_len=cfg['seq_len'], d_model=64, num_layers=2)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)
    lossfn = nn.SmoothL1Loss(beta=0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DL_MAX_EPOCHS, eta_min=1e-6)
    model.to(device)
    log(f"stwaveformer {ds} run_{run_id}: Train+Val {tuple(x_tr.shape)}, E*={e_star} (gốc: {src}), "
        f"huấn luyện {epochs} epoch, seed {seed}, {device}")

    def squeeze(out):
        if out.dim() == 4:
            out = out[:, :, :, -1]
        if out.dim() == 3 and out.size(1) == 1:
            out = out.squeeze(1)
        return out

    history, t_run = [], time.time()
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = lossfn(squeeze(model(x)), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        spe = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == 'cuda' else 0.0
        history.append({'epoch': epoch + 1, 'seed': seed, 'lr': lr, 'train_loss': float(np.mean(losses)), 's_epoch': spe})
        log(f"Epoch {epoch + 1}/{epochs} | LR {lr:.6f} | train_loss {np.mean(losses):.6f} | {spe:.1f} s/epoch | VRAM {vram:.1f} GB")

    model.eval()
    outs = []
    with torch.no_grad():
        for (bx,) in DataLoader(TensorDataset(x_te), batch_size=64, shuffle=False):
            outs.append(squeeze(model(bx.to(device))).cpu())
    pred = torch.cat(outs).numpy().astype(np.float32)          # thang Min-Max, không cắt (giống cache v3)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'model.pth'), 'wb') as f:
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, 'train_metrics.csv'), index=False)
    extra = {'n_epochs': epochs, 'e_star': e_star, 'T_max': DL_MAX_EPOCHS, 'fit_time_s': round(time.time() - t_run, 1),
             's_epoch_mean': round(float(np.mean([h['s_epoch'] for h in history])), 2), 'device': str(device)}
    return pred, y_cache, P_cache[branches.index('stwaveformer')], extra, src, diff, int(x_tr.shape[0])


# ---------------------------------------------------------------- một đơn vị (mô hình, dataset, run)
def _versions():
    out = {}
    for m in ['numpy', 'torch', 'xgboost', 'lightgbm', 'sklearn']:
        try:
            out[m] = __import__(m).__version__
        except Exception:
            pass
    return out


def run_unit(model, ds, run_id, quick_check=False, root=None, log=print):
    out = unit_dir(model, ds, run_id, root)
    os.makedirs(out, exist_ok=True)
    meta_f = os.path.join(out, 'meta.json')
    if os.path.exists(meta_f):
        os.remove(meta_f)                       # meta.json là dấu hoàn tất -> xóa trước khi chạy lại
    t0 = time.time()
    if model == 'stwaveformer':
        pred, y, p_orig, extra, src, diff, n_tr = refit_stwaveformer(ds, run_id, quick_check, log, out)
    else:
        pred, y, p_orig, extra, src, diff, n_tr = refit_gbdt(model, ds, run_id, quick_check, log)
    if pred.shape != y.shape:
        raise RuntimeError(f"Dự báo {pred.shape} khác y_test {y.shape}")
    if not np.isfinite(pred).all():
        raise RuntimeError(f"Dự báo có {int((~np.isfinite(pred)).sum())} giá trị NaN/inf")
    mse = float(np.mean((pred.astype(np.float64) - y) ** 2))
    mse_orig = float(np.mean((p_orig.astype(np.float64) - y) ** 2))
    tmp = os.path.join(out, 'test_pred.tmp.npy')
    np.save(tmp, pred)
    os.replace(tmp, os.path.join(out, 'test_pred.npy'))
    meta = {'model': model, 'dataset': ds, 'run': run_id, 'seed': 42 + run_id, 'quick_check': bool(quick_check),
            'shape': list(pred.shape), 'n_train_samples': n_tr, 'rounds_source': os.path.relpath(src, parent_dir).replace('\\', '/'),
            'test_mse': mse, 'test_mse_orig_branch': mse_orig, 'y_test_max_abs_diff': diff,
            'time_s': round(time.time() - t0, 1), 'finished_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
            'versions': _versions(), **extra}
    with open(meta_f, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    rounds = f"E*={extra['n_epochs']}" if model == 'stwaveformer' else f"{extra['n_trees']} cây"
    log(f"DONE {model} {ds} run_{run_id}: {rounds}, test MSE={mse * 1e3:.4f}e-3 (gốc {mse_orig * 1e3:.4f}e-3), {meta['time_s']:.0f} s")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Refit một nhánh của v3 trên Train+Val")
    ap.add_argument('--model', required=True, choices=MODELS)
    ap.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS))
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--quick_check', action='store_true', help=f"{QUICK_EPOCHS} epoch / {QUICK_TREES} cây")
    ap.add_argument('--root', default=None, help=f"thư mục kết quả (mặc định {OUT_ROOT}/)")
    args = ap.parse_args()
    run_unit(args.model, args.dataset, args.run, args.quick_check, args.root,
             log=lambda s: print(s, flush=True))


if __name__ == '__main__':
    main()
