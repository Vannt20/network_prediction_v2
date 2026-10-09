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
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.feature_store import prepare_feature_store, DATASET_CONFIGS
from features.temporal_features import extract_context_features_torch
from Graph_models.st_waveformer import STWaveFormer
from baselines_ml.run_ml_baselines import get_model_instance, parse_run_ids, MODEL_CLASSES
from run_experiments import check_existing_run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DL_BRANCHES = ['stwaveformer']
# Nhánh của hướng V4 (logs/gnn_rl/...), chỉ dùng với cache_version khác 'v3'
V4_DL_BRANCHES = ['odgraphformer']
CACHE_VERSION = 'v3'


def cache_path(ds_key, run_id, version=CACHE_VERSION):
    return os.path.join(parent_dir, 'cache', version, f"{ds_key}_run_{run_id}.pt")


def _v4_logdir(name, ds_key, run_id, logs_dir):
    # Log của V4 theo vòng thực nghiệm (logs/gnn_rl, logs/gnn_rl_r2, ...); logs_dir giữ lại cho tương thích chữ ký
    from training.gnn_rl_common import LOGS_V4
    seq_len = DATASET_CONFIGS[ds_key]['seq_len']
    return os.path.join(LOGS_V4, f"{name}_data_{ds_key}_seq_{seq_len}", f"run_{run_id}")


def create_sliding_windows(traffic_norm, tod_arr, dow_arr, seq_len):
    """
    Tạo tensor cửa sổ trượt [M, seq_len, N, 3] và nhãn y [M, N],
    khớp với TrafficDataset trong run_experiments.py.
    """
    N = traffic_norm.shape[1]
    tod_2d = np.tile(tod_arr[:, None], (1, N)) if tod_arr.ndim == 1 else tod_arr
    dow_2d = np.tile(dow_arr[:, None], (1, N)) if dow_arr.ndim == 1 else dow_arr
    comb = np.stack([traffic_norm, tod_2d, dow_2d], axis=-1).astype(np.float32)
    xs = np.lib.stride_tricks.sliding_window_view(comb[:-1], window_shape=seq_len, axis=0)  # [M, N, 3, seq]
    xs = np.ascontiguousarray(np.transpose(xs, (0, 3, 1, 2)))
    ys = comb[seq_len:, :, 0]
    return torch.from_numpy(xs), torch.from_numpy(np.ascontiguousarray(ys))


def get_champion_name(results_dir):
    json_path = os.path.join(results_dir, 'champion_ml_model.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f).get('champion_model', 'xgboost')
        except Exception:
            pass
    return 'xgboost'


def default_branches(results_dir=None):
    results_dir = results_dir or os.path.join(parent_dir, 'results')
    return ['stwaveformer', get_champion_name(results_dir), 'lightgbm_res']


def source_fingerprint(branches, ds_key, run_id, logs_dir=None):
    """MD5 của file mô hình từng nhánh -> cache chỉ được tái sử dụng khi đúng các mô hình đã tạo ra nó."""
    logs_dir = logs_dir or os.path.join(parent_dir, 'logs')
    seq_len = DATASET_CONFIGS[ds_key]['seq_len']
    fp = {}
    for b in branches:
        if b in DL_BRANCHES:
            f = os.path.join(logs_dir, f"{b}_data_{ds_key}_seq_{seq_len}", f"run_{run_id}", 'best_model.pth')
        elif b in V4_DL_BRANCHES:
            f = os.path.join(_v4_logdir(b, ds_key, run_id, logs_dir), 'best_model.pth')
        else:
            f = os.path.join(logs_dir, f"{b}_data_{ds_key}_shared", f"run_{run_id}", 'model.bin')
        if os.path.exists(f):
            h = hashlib.md5()
            with open(f, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    h.update(chunk)
            fp[b] = h.hexdigest()
        else:
            fp[b] = None
    return fp


def _infer(model, x_win, batch_size=64):
    loader = DataLoader(TensorDataset(x_win), batch_size=batch_size, shuffle=False)
    outs = []
    with torch.no_grad():
        for (bx,) in loader:
            outs.append(model(bx.to(device)).cpu())
    return torch.cat(outs, dim=0).numpy()


def _dl_branch_preds(name, ds_key, run_id, cfg, x_val_win, y_test_win, x_test_win, logs_dir):
    seq_len, num_nodes, num_flows = cfg['seq_len'], cfg['nodes'], cfg['flows']
    logdir = os.path.join(logs_dir, f"{name}_data_{ds_key}_seq_{seq_len}", f"run_{run_id}")
    valid, reason = check_existing_run(logdir, len(y_test_win))
    if not valid:
        raise FileNotFoundError(
            f"[{name} {ds_key.upper()} run_{run_id}] không dùng được: {reason}. "
            f"Huấn luyện trên GPU: python run_experiments.py --model STWaveFormer --dataset {ds_key} "
            f"--run_ids {run_id} --epochs 200 --patience 30 --skip_existing")

    model = STWaveFormer(input_dim=num_flows, num_nodes=num_nodes, seq_len=seq_len, d_model=64, num_layers=2)
    with open(os.path.join(logdir, 'best_model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.to(device).eval()

    p_val = _infer(model, x_val_win)

    # Test: tái sử dụng dự báo đã lưu CHỈ KHI nhãn của log khớp tuyệt đối với nhãn hiện tại
    y_real_log = np.load(os.path.join(logdir, 'y_real_data.npy'))
    y_pred_log = np.load(os.path.join(logdir, 'y_pred_data.npy'))
    y_now = y_test_win.numpy()
    if y_real_log.shape == y_now.shape and np.allclose(y_real_log, y_now, atol=1e-6):
        p_test = y_pred_log.astype(np.float32)
        src = 'log'
    else:
        p_test = _infer(model, x_test_win)
        src = 'suy diễn lại'

    logged = pd.read_csv(os.path.join(logdir, 'test_metrics.csv')).iloc[0]
    mse_now = float(np.mean((p_test - y_now) ** 2))
    if abs(mse_now - float(logged['mse'])) > 0.01 * float(logged['mse']):
        print(f"      [CẢNH BÁO] {name}: MSE test tính lại {mse_now*1e3:.4f}e-3 khác log {float(logged['mse'])*1e3:.4f}e-3", flush=True)
    print(f"      [{name}] dự báo test lấy từ: {src}", flush=True)
    return p_val.astype(np.float32), p_test, float(logged.get('inference_time_ms', np.nan))


def _v4_dl_branch_preds(name, ds_key, run_id, y_val_win, y_test_win, logs_dir):
    """Nhánh V4 (OD-GraphFormer): dự báo Val/Test và σ̂ đã lưu lúc huấn luyện, kiểm tra khớp nhãn."""
    logdir = _v4_logdir(name, ds_key, run_id, logs_dir)
    need = ['pred_val.npy', 'sigma_val.npy', 'y_pred_data.npy', 'sigma_test.npy', 'y_real_data.npy']
    miss = [f for f in need if not os.path.exists(os.path.join(logdir, f))]
    if miss:
        raise FileNotFoundError(
            f"[{name} {ds_key.upper()} run_{run_id}] thiếu {miss} trong {logdir}. Huấn luyện: "
            f"python training/train_od_graphformer.py --datasets {ds_key} --run_ids {run_id}")

    def ld(f):
        return np.load(os.path.join(logdir, f)).astype(np.float32)
    y_log, y_now = ld('y_real_data.npy'), y_test_win.numpy()
    if y_log.shape != y_now.shape or not np.allclose(y_log, y_now, atol=1e-6):
        raise RuntimeError(f"[{name} {ds_key.upper()} run_{run_id}] nhãn Test trong log không khớp dữ liệu hiện tại")
    p_val, p_test = ld('pred_val.npy'), ld('y_pred_data.npy')
    if p_val.shape != tuple(y_val_win.shape):
        raise RuntimeError(f"[{name}] pred_val.npy có kích thước {p_val.shape}, cần {tuple(y_val_win.shape)}")
    inf = float(pd.read_csv(os.path.join(logdir, 'test_metrics.csv')).iloc[0].get('inference_time_ms', np.nan))
    return p_val, p_test, inf, ld('sigma_val.npy'), ld('sigma_test.npy')


def _raw_predict(name, inst, X):
    """Dự báo chưa cắt về >= 0 (để hiệu chỉnh hệ số chặn chính xác)."""
    if name == 'lightgbm_res':
        return inst.predict(X)
    return np.asarray(inst.model.predict(X), dtype=np.float64).reshape(-1)


def _ml_branch_preds(name, ds_key, run_id, X_va, X_te, y_te_flat, feature_names, T_val, T_test, N, logs_dir):
    logdir = os.path.join(logs_dir, f"{name}_data_{ds_key}_shared", f"run_{run_id}")
    model_file = os.path.join(logdir, 'model.bin')
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"[{name} {ds_key.upper()} run_{run_id}] chưa có {model_file}. Chạy (CPU): "
            f"python baselines_ml/run_ml_baselines.py --models {name} --datasets {ds_key} --run_ids {run_id} --skip_existing")
    inst = get_model_instance(name, seed=42 + run_id, feature_names=feature_names).load(model_file)
    raw_val, raw_test = _raw_predict(name, inst, X_va), _raw_predict(name, inst, X_te)

    # Đối chiếu với dự báo test đã lưu lúc huấn luyện. Mô hình huấn luyện ở môi trường khác (ví dụ Kaggle,
    # phiên bản XGBoost khác) có thể bị đọc sai hệ số chặn base_score khi nạp lại -> lệch một hằng số.
    # Sai khác cho phép: vài mẫu rơi đúng ngưỡng tách của cây có thể đổi nhánh giữa hai phiên bản thư viện.
    offset = 0.0
    p_test_logged = None
    pred_log, real_log = os.path.join(logdir, 'y_pred_data.npy'), os.path.join(logdir, 'y_real_data.npy')
    if os.path.exists(pred_log) and os.path.exists(real_log):
        logged = np.load(pred_log).reshape(-1).astype(np.float64)
        y_log = np.load(real_log).reshape(-1)
        if logged.shape == raw_test.shape and np.allclose(y_log, y_te_flat, atol=1e-6):
            mask = (logged > 0) & (raw_test > 0)
            if mask.any():
                offset = float(np.median(logged[mask] - raw_test[mask]))
            rec = np.clip(raw_test + offset, 0.0, None)
            n_bad = int(np.sum(np.abs(rec - logged) > 1e-5))
            mse_log = float(np.mean((logged - y_te_flat) ** 2))
            mse_rec = float(np.mean((rec - y_te_flat) ** 2))
            rel = abs(mse_rec - mse_log) / max(mse_log, 1e-12)
            if n_bad > 1e-4 * len(logged) or rel > 1e-4:
                raise RuntimeError(
                    f"[{name} {ds_key.upper()} run_{run_id}] mô hình nạp lại không tái lập được dự báo lúc huấn luyện "
                    f"({n_bad} điểm lệch, ΔMSE tương đối {rel:.1e}). Cần huấn luyện lại mô hình này.")
            if abs(offset) > 1e-6 or n_bad:
                print(f"      [{name}] nạp model.bin: hiệu chỉnh hệ số chặn offset={offset:+.6f}; "
                      f"{n_bad}/{len(logged)} điểm lệch ngưỡng tách, ΔMSE tương đối {rel:.1e} -> Test dùng dự báo lúc huấn luyện",
                      flush=True)
            p_test_logged = logged
    p_val = np.clip(raw_val + offset, 0.0, None).reshape(T_val, N).astype(np.float32)
    p_test_src = p_test_logged if p_test_logged is not None else np.clip(raw_test + offset, 0.0, None)
    p_test = p_test_src.reshape(T_test, N).astype(np.float32)
    inf = np.nan
    tm = os.path.join(logdir, 'test_metrics.csv')
    if os.path.exists(tm):
        inf = float(pd.read_csv(tm).iloc[0].get('inference_time_ms', np.nan))
    return p_val, p_test, inf


def precompute_dataset_cache(dataset_name, runs=5, run_ids=None, branches=None, quick_check=False, skip_existing=False,
                             cache_version=CACHE_VERSION):
    ds_key = dataset_name.lower()
    cfg = DATASET_CONFIGS[ds_key]
    seq_len, num_flows = cfg['seq_len'], cfg['flows']
    logs_dir = os.path.join(parent_dir, 'logs')
    branches = branches or default_branches()
    run_id_list = parse_run_ids(run_ids, runs)
    is_v3 = cache_version == 'v3'
    for b in branches:
        if b not in DL_BRANCHES and b not in MODEL_CLASSES and (is_v3 or b not in V4_DL_BRANCHES):
            raise ValueError(f"Nhánh không hỗ trợ: {b}")

    todo = []
    for r in run_id_list:
        f = cache_path(ds_key, r, cache_version)
        if skip_existing and os.path.exists(f):
            try:
                old = torch.load(f, map_location='cpu')
                if old.get('branches') == branches and old.get('sources') == source_fingerprint(branches, ds_key, r, logs_dir):
                    print(f"  [SKIP] Cache khớp mô hình hiện tại: {f}", flush=True)
                    continue
                print(f"  [RECOMPUTE] Cache {f} không khớp nhánh/mô hình hiện tại -> tính lại", flush=True)
            except Exception:
                pass
        todo.append(r)
    if not todo:
        return

    print(f"\n=======================================================", flush=True)
    print(f"[*] PRECOMPUTE CACHE {cache_version}: {ds_key.upper()} | Nhánh: {branches} | Run IDs: {todo}", flush=True)
    print(f"=======================================================", flush=True)

    (_, _), (X_va, y_va), (X_te, y_te), meta = prepare_feature_store(
        ds_key, align_to_seq_len=True, data_dir=os.path.join(parent_dir, 'data'))
    x_val_win, y_val_win = create_sliding_windows(meta['val_norm'], meta['tod_val'], meta['dow_val'], seq_len)
    x_test_win, y_test_win = create_sliding_windows(meta['test_norm'], meta['tod_test'], meta['dow_test'], seq_len)
    T_val, T_test = len(y_val_win), len(y_test_win)

    # Kiểm tra đồng bộ giữa dữ liệu dạng bảng (nhánh ML) và dạng cửa sổ (nhánh DL)
    assert len(X_va) == T_val * num_flows and len(X_te) == T_test * num_flows, "Lệch số mẫu tab/window"
    assert np.allclose(y_va.reshape(T_val, num_flows), y_val_win.numpy(), atol=1e-6), "Lệch nhãn Val tab/window"
    assert np.allclose(y_te.reshape(T_test, num_flows), y_test_win.numpy(), atol=1e-6), "Lệch nhãn Test tab/window"

    if quick_check:
        lim = 10
        x_val_win, y_val_win, x_test_win, y_test_win = x_val_win[:lim], y_val_win[:lim], x_test_win[:lim], y_test_win[:lim]
        X_va, X_te = X_va[:lim * num_flows], X_te[:lim * num_flows]
        T_val = T_test = lim

    split_extra = {}
    for split, xw in [('val', x_val_win), ('test', x_test_win)]:
        split_extra[split] = {
            'context': extract_context_features_torch(xw),      # [T, N, 4] cho ablation cổng MLP
            'last': xw[:, -1, :, 0].clone(),                    # lag_1 - phân tích điểm vượt biên Train
        }
        if not is_v3:
            # Thời điểm của bước cần dự báo (cửa sổ j dự báo bước j + seq_len của split)
            n_win = len(xw)
            split_extra[split]['tod'] = torch.from_numpy(np.asarray(meta[f'tod_{split}'][seq_len:], dtype=np.float32)[:n_win])
            split_extra[split]['dow'] = torch.from_numpy(np.asarray(meta[f'dow_{split}'][seq_len:], dtype=np.float32)[:n_win])

    for run_id in todo:
        print(f"  --> run_{run_id}", flush=True)
        P_val, P_test, inf_times = [], [], {}
        sigma = {}
        # V4: lấy lại nguyên dự báo của các nhánh v3 từ cache v3 khi đúng mô hình đã tạo ra nó (khớp tuyệt đối với v3)
        v3c, fp = None, {}
        if not is_v3 and os.path.exists(cache_path(ds_key, run_id, 'v3')):
            v3c = torch.load(cache_path(ds_key, run_id, 'v3'), map_location='cpu', weights_only=False)
            fp = source_fingerprint(branches, ds_key, run_id, logs_dir)
        for b in branches:
            if v3c is not None and b in v3c['branches'] and fp.get(b) and v3c['sources'].get(b) == fp[b]:
                k = v3c['branches'].index(b)
                pv, pt = v3c['val']['P'][k].numpy()[:T_val], v3c['test']['P'][k].numpy()[:T_test]
                it = v3c['inference_time_ms'].get(b, np.nan)
                print(f"      [{b}] lấy từ cache v3 (cùng mô hình)", flush=True)
            elif b in V4_DL_BRANCHES and not is_v3:
                pv, pt, it, sv, stt = _v4_dl_branch_preds(b, ds_key, run_id, y_val_win, y_test_win, logs_dir)
                if quick_check:
                    pv, pt, sv, stt = pv[:T_val], pt[:T_test], sv[:T_val], stt[:T_test]
                sigma = {'val': sv, 'test': stt}
            elif b in DL_BRANCHES:
                pv, pt, it = _dl_branch_preds(b, ds_key, run_id, cfg, x_val_win, y_test_win, x_test_win, logs_dir)
                if quick_check:
                    pt = pt[:T_test]
            else:
                pv, pt, it = _ml_branch_preds(b, ds_key, run_id, X_va, X_te, y_te[:len(X_te)],
                                              meta['feature_names'], T_val, T_test, num_flows, logs_dir)
            P_val.append(pv)
            P_test.append(pt)
            inf_times[b] = it
            print(f"      [{b}] val MSE={np.mean((pv - y_val_win.numpy()) ** 2)*1e3:.3f}e-3 | "
                  f"test MSE={np.mean((pt - y_test_win.numpy()) ** 2)*1e3:.3f}e-3", flush=True)

        if not is_v3:
            for split, T in (('val', T_val), ('test', T_test)):
                s = sigma.get(split, np.full((T, num_flows), np.nan, dtype=np.float32))
                split_extra[split]['sigma'] = torch.from_numpy(np.asarray(s, dtype=np.float32))
        data = {
            'version': cache_version,
            'dataset': ds_key,
            'run_id': run_id,
            'branches': branches,
            'inference_time_ms': inf_times,
            'sources': source_fingerprint(branches, ds_key, run_id, logs_dir),
            'val': {'P': torch.from_numpy(np.stack(P_val)), 'y': y_val_win.clone(), **split_extra['val']},
            'test': {'P': torch.from_numpy(np.stack(P_test)), 'y': y_test_win.clone(), **split_extra['test']},
        }
        out = cache_path(ds_key, run_id, cache_version)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            torch.save(data, f)
        print(f"      [CACHE] Đã lưu: {out}", flush=True)


def run_precompute(datasets=None, runs=5, run_ids=None, branches=None, quick_check=False, skip_existing=False,
                   cache_version=CACHE_VERSION):
    if datasets is None or 'all' in datasets:
        datasets = ['sdn', 'geant', 'abilene']
    for ds in datasets:
        precompute_dataset_cache(ds, runs=runs, run_ids=run_ids, branches=branches,
                                 quick_check=quick_check, skip_existing=skip_existing, cache_version=cache_version)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Precompute bộ đệm dự báo Val/Test của các nhánh (v3)")
    parser.add_argument('--datasets', type=str, default='all', help="sdn,geant,abilene hoặc 'all'")
    parser.add_argument('--runs', type=int, default=5, help="Số run: run_0..run_{runs-1}")
    parser.add_argument('--run_ids', type=str, default=None, help="Ví dụ '5-9' (ghi đè --runs)")
    parser.add_argument('--branches', type=str, default=None,
                        help="Danh sách nhánh, mặc định: stwaveformer,<champion>,lightgbm_res")
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--quick_check', action='store_true')
    parser.add_argument('--cache_version', type=str, default=CACHE_VERSION,
                        help="v3 (mặc định, hành vi cũ) hoặc v4 (thêm nhánh odgraphformer, sigma, tod, dow)")
    args = parser.parse_args()
    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ['sdn', 'geant', 'abilene']
    b_list = [b.strip() for b in args.branches.split(',')] if args.branches else None
    run_precompute(datasets=d_list, runs=args.runs, run_ids=args.run_ids, branches=b_list,
                   quick_check=args.quick_check, skip_existing=args.skip_existing, cache_version=args.cache_version)
