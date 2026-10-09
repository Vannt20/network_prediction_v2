"""
Tiện ích dùng chung cho hướng V4 (ST-Adaptive-Ensemble-RL: OD-GraphFormer + RL-Gate).

Mọi sản phẩm của V4 ghi ra thư mục riêng để không đụng tới v3:
  logs/gnn_rl/...   results/gnn_rl/...   cache/v4/...
"""
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.feature_store import DATASET_CONFIGS, load_raw_dataset

ROOT = parent_dir
LOGS_V4 = os.path.join(ROOT, 'logs', 'gnn_rl')
RESULTS_V4 = os.path.join(ROOT, 'results', 'gnn_rl')
CACHE_V4 = os.path.join(ROOT, 'cache', 'v4')
ALL_DATASETS = ['sdn', 'geant', 'abilene']

# Nhánh dự báo được vượt biên Train (dự báo phần dư so với lag_1) - dùng cho ràng buộc sàn của RL-Gate
EXTRAP_BRANCHES = ('lightgbm_res', 'odgraphformer')

# Mục tiêu thiết kế ngân sách 1 ngày (spec Mục 4.3)
ODGF_MAX_STEPS = {'sdn': 1500, 'geant': 3000, 'abilene': 3000}
ODGF_MAX_MINUTES = {'sdn': 10, 'geant': 20, 'abilene': 15}


def set_seed(seed):
    """Giống run_experiments.set_seed (không import run_experiments để tránh nạp các mô hình cũ)."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_utf8():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, 'reconfigure'):
            try:
                s.reconfigure(encoding='utf-8')
            except Exception:
                pass


def parse_list(s, default):
    if s is None or s == '' or s == 'all':
        return list(default)
    return [x.strip() for x in str(s).split(',') if x.strip()]


def run_dir(tag, ds_key, run_id=None, seq=True):
    """logs/gnn_rl/{tag}_data_{ds}[_seq_L]/run_{r}"""
    name = f"{tag}_data_{ds_key}" + (f"_seq_{DATASET_CONFIGS[ds_key]['seq_len']}" if seq else '')
    d = os.path.join(LOGS_V4, name)
    return d if run_id is None else os.path.join(d, f"run_{run_id}")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_splits(ds_key, data_dir=None):
    """
    Chia 70/10/20 theo thời gian + MinMaxScaler fit trên Train, giống hệt
    features.feature_store.prepare_feature_store và run_experiments.prepare_dataset,
    nhưng không dựng ma trận đặc trưng bảng (tiết kiệm RAM cho Abilene).
    Trả về dict: {'train'|'val'|'test': {'x': [T,N], 'tod': [T], 'dow': [T]}, 'columns', 'scaler', 'steps_per_day'}
    """
    df = load_raw_dataset(ds_key, data_dir=data_dir or os.path.join(ROOT, 'data'))
    total = len(df)
    n_tr, n_va = int(total * 0.7), int(total * 0.1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    tr = scaler.fit_transform(df.iloc[:n_tr]).astype(np.float32)
    va = scaler.transform(df.iloc[n_tr:n_tr + n_va]).astype(np.float32)
    te = scaler.transform(df.iloc[n_tr + n_va:]).astype(np.float32)
    idx = df.index
    tod = ((idx.hour * 60.0 + idx.minute) / 1440.0).values.astype(np.float32)
    dow = (idx.dayofweek / 7.0).values.astype(np.float32)
    step_min = float(np.median(np.diff(idx.values).astype('timedelta64[s]').astype(np.float64))) / 60.0
    spd = int(round(1440.0 / step_min)) if step_min > 0 else 288
    sl = {'train': slice(0, n_tr), 'val': slice(n_tr, n_tr + n_va), 'test': slice(n_tr + n_va, total)}
    out = {'columns': list(df.columns), 'scaler': scaler, 'steps_per_day': max(1, spd)}
    for name, x in (('train', tr), ('val', va), ('test', te)):
        out[name] = {'x': x, 'tod': tod[sl[name]], 'dow': dow[sl[name]]}
    return out


def comb_array(split):
    """[T, N, 3]: traffic, tod, dow (giống kênh đầu vào của TrafficDataset)."""
    x = split['x']
    N = x.shape[1]
    return np.stack([x, np.repeat(split['tod'][:, None], N, 1), np.repeat(split['dow'][:, None], N, 1)],
                    axis=-1).astype(np.float32)


def gather_windows(comb_t, target_idx, seq_len):
    """
    comb_t: tensor [T, N, C]; target_idx: LongTensor [B] (chỉ số bước cần dự báo trong split).
    Trả về x [B, L, N, C] = comb[t-L .. t-1] và y [B, N] = comb[t, :, 0].
    Cửa sổ thứ j của v3 ứng với target_idx = j + L.
    """
    import torch
    offs = torch.arange(-seq_len, 0, device=comb_t.device)
    win = target_idx[:, None] + offs[None, :]
    return comb_t[win], comb_t[target_idx, :, 0]


def cache_file(ds_key, run_id, version='v4'):
    return os.path.join(ROOT, 'cache', version, f"{ds_key}_run_{run_id}.pt")


def oof_file(ds_key):
    return os.path.join(CACHE_V4, f"{ds_key}_oof.pt")


def torch_load(path):
    import torch
    with open(path, 'rb') as f:
        return torch.load(f, map_location='cpu', weights_only=False)


def load_cache(ds_key, run_id, version='v4'):
    f = cache_file(ds_key, run_id, version)
    if not os.path.exists(f):
        raise FileNotFoundError(f"Chưa có cache {f}. Chạy: python training/precompute_cache.py --cache_version {version} "
                                f"--datasets {ds_key} --run_ids {run_id}")
    return torch_load(f)


def split_arrays(cache, split, branches=None):
    """(P [K,T,N], y [T,N], last [T,N]) lấy theo danh sách nhánh (mặc định: mọi nhánh trong cache)."""
    d = cache[split]
    P = d['P'].numpy() if hasattr(d['P'], 'numpy') else np.asarray(d['P'])
    if branches is not None:
        P = P[[cache['branches'].index(b) for b in branches]]
    y = d['y'].numpy() if hasattr(d['y'], 'numpy') else np.asarray(d['y'])
    last = d['last'].numpy() if hasattr(d['last'], 'numpy') else np.asarray(d['last'])
    return P.astype(np.float32), y.astype(np.float32), last.astype(np.float32)


def save_combiner_result(tag, ds_key, run_id, pred, y, last, config=None, weights=None, save_pred=False):
    """
    Ghi kết quả của một tầng kết hợp / mô hình trên Test:
      test_metrics.csv (chỉ số chính + phụ), se_t.npy (MSE theo bước, đủ cho kiểm định Diebold-Mariano),
      config.json; tùy chọn y_pred_data.npy và weights_test.npz (float16).
    """
    import pandas as pd
    from evaluation.extra_metrics import extra_metrics
    d = run_dir(tag, ds_key, run_id, seq=False)
    os.makedirs(d, exist_ok=True)
    m = extra_metrics(pred, y, last)
    m['run'] = run_id
    pd.DataFrame([m]).to_csv(os.path.join(d, 'test_metrics.csv'), index=False)
    se_t = ((np.asarray(pred, np.float64) - np.asarray(y, np.float64)) ** 2).mean(axis=1)
    np.save(os.path.join(d, 'se_t.npy'), se_t)
    if config is not None:
        save_json(config, os.path.join(d, 'config.json'))
    if save_pred:
        np.save(os.path.join(d, 'y_pred_data.npy'), np.asarray(pred, np.float32))
    if weights is not None:
        np.savez_compressed(os.path.join(d, 'weights_test.npz'), w=np.asarray(weights, np.float16))
    return m


def oof_ranges(T_train, seq_len, train_frac=0.6):
    """
    Chỉ số bước đích (trong đoạn Train) cho OOF một fold (spec Mục 5.2):
      fit = [L, inner), es = [inner, cut) (10% cuối phần học, để dừng sớm), oof = [cut, T_train).
    Cửa sổ thứ j của dữ liệu bảng ứng với bước đích j + L.
    """
    cut = int(T_train * train_frac)
    inner = seq_len + int((cut - seq_len) * 0.9)
    return np.arange(seq_len, inner), np.arange(inner, cut), np.arange(cut, T_train)


def upsert_csv(df, path, key=('dataset', 'run')):
    """Ghi df vào path, chỉ thay các dòng trùng khóa (nhiều dataset / tài khoản ghi chung một file không đè nhau)."""
    import pandas as pd
    if os.path.exists(path):
        old = pd.read_csv(path)
        if all(k in old.columns for k in key):
            idx = pd.MultiIndex.from_frame(df[list(key)].astype(str))
            keep = ~pd.MultiIndex.from_frame(old[list(key)].astype(str)).isin(idx)
            df = pd.concat([old[keep], df], ignore_index=True)
    df = df.sort_values(list(key)).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def jump_mask(y, last, q=0.001):
    """Mặt nạ cố định theo dữ liệu: True ở các điểm KHÔNG thuộc q lớn nhất của |y - lag_1|."""
    j = np.abs(np.asarray(y, dtype=np.float64) - np.asarray(last, dtype=np.float64))
    k = int(np.floor(j.size * q))
    if k <= 0:
        return np.ones_like(j, dtype=bool)
    thr = np.partition(j.reshape(-1), j.size - k)[j.size - k]
    keep = j < thr
    # Trường hợp nhiều giá trị bằng ngưỡng: bỏ đúng k điểm lớn nhất theo thứ tự ổn định
    if keep.sum() != j.size - k:
        order = np.argsort(-j.reshape(-1), kind='stable')
        keep = np.ones(j.size, dtype=bool)
        keep[order[:k]] = False
        keep = keep.reshape(j.shape)
    return keep
