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
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from features.spatial_features import build_od_topology_matrices, compute_spatial_neighbors
from features.temporal_features import extract_temporal_features_matrix


DATASET_CONFIGS = {
    'sdn': {'nodes': 14, 'flows': 196, 'seq_len': 60, 'k_lags': 15},
    'geant': {'nodes': 23, 'flows': 529, 'seq_len': 24, 'k_lags': 12},
    'abilene': {'nodes': 12, 'flows': 144, 'seq_len': 24, 'k_lags': 12}
}


def load_raw_dataset(dataset_name, data_dir=None):
    """
    Nạp dữ liệu từ thư mục data/ và thực hiện Data Hygiene:
    - Đảm bảo tính đơn điệu của chuỗi thời gian (df.index.is_monotonic_increasing).
    - Khử trùng lặp (ví dụ 288 mốc trùng ở Abilene), đưa về chuỗi thời gian hoàn hảo.
    """
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')

    fpath = os.path.join(data_dir, f'{dataset_name}.csv')
    if not os.path.exists(fpath):
        fpath = os.path.join(data_dir, f'{dataset_name.upper()}.csv')
    if not os.path.exists(fpath):
        fpath = os.path.join(data_dir, f'{dataset_name.lower()}.csv')

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu cho dataset: {dataset_name} tại {fpath}")

    df = pd.read_csv(fpath, parse_dates=['time'])
    df = df.set_index(['time'])

    if not df.index.is_monotonic_increasing:
        n_before = len(df)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        n_after = len(df)
        print(f"[{dataset_name.upper()}] Data Hygiene: Đã khử {n_before - n_after} dòng trùng lặp, "
              f"đưa chuỗi về {n_after} bước thời gian đơn điệu hoàn hảo.")

    return df


def prepare_feature_store(dataset_name, split_ratio=(0.7, 0.1, 0.2), k_lags=None, align_to_seq_len=True, data_dir=None):
    """
    Chuẩn bị toàn diện dữ liệu Tabular theo chiến lược multi_output: shared_model:
    1. Chia Train (70%), Val (10%), Test (20%) theo thứ tự thời gian.
    2. Fit MinMaxScaler [0, 1] trên tập Train, transform Val và Test.
    3. Tính ma trận topo OD và neighbor_in, neighbor_out.
    4. Trích xuất lags, rolling statistics, diff, TOD, DOW.
    5. Đóng gói thành ma trận 2D X [M, D] và y [M] (M = T_valid * N) với 'flow_id' là biến categorical.
    """
    ds_key = dataset_name.lower()
    cfg = DATASET_CONFIGS.get(ds_key, {'k_lags': 12, 'seq_len': 24})
    if k_lags is None:
        k_lags = cfg['k_lags']
    seq_len = cfg.get('seq_len', 24)

    df = load_raw_dataset(ds_key, data_dir=data_dir)
    columns = list(df.columns)
    N = len(columns)
    total_steps = len(df)

    r_train, r_val, _ = split_ratio
    train_size = int(total_steps * r_train)
    val_size = int(total_steps * r_val)

    train_df = df.iloc[0:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_norm = scaler.fit_transform(train_df).astype(np.float32)
    val_norm = scaler.transform(val_df).astype(np.float32)
    test_norm = scaler.transform(test_df).astype(np.float32)

    # Contextual Time features: Time-of-day [0, 1) & Day-of-week [0, 1)
    time_idx = df.index
    tod = (time_idx.hour * 60.0 + time_idx.minute) / 1440.0
    dow = time_idx.dayofweek / 7.0

    tod_train = tod.values[0:train_size].astype(np.float32)
    tod_val = tod.values[train_size:train_size + val_size].astype(np.float32)
    tod_test = tod.values[train_size + val_size:].astype(np.float32)

    dow_train = dow.values[0:train_size].astype(np.float32)
    dow_val = dow.values[train_size:train_size + val_size].astype(np.float32)
    dow_test = dow.values[train_size + val_size:].astype(np.float32)

    # Ma trận topo không gian OD
    M_in, M_out = build_od_topology_matrices(columns)

    def process_split(traffic_split, tod_split, dow_split):
        # 1. Trích xuất đặc trưng chuỗi thời gian
        t_feats, k_valid, feat_names = extract_temporal_features_matrix(
            traffic_split, tod_split, dow_split, k_lags=k_lags, windows=(3, 5, 10)
        )
        if align_to_seq_len and seq_len > k_valid:
            # Cắt bớt phần đầu để khớp chính xác với điểm bắt đầu của sliding window
            offset = seq_len - k_valid
            for name in feat_names:
                t_feats[name] = t_feats[name][offset:]
            k_valid = seq_len

        T_len = len(traffic_split)
        T_valid = T_len - k_valid

        # 2. Đặc trưng không gian tại t-1 (từ k_valid-1 đến T_len-2)
        x_lag1 = traffic_split[k_valid - 1: T_len - 1] # [T_valid, N]
        nbr_in, nbr_out = compute_spatial_neighbors(x_lag1, M_in, M_out)
        t_feats['neighbor_sum_in'] = nbr_in
        t_feats['neighbor_sum_out'] = nbr_out

        all_names = list(feat_names) + ['neighbor_sum_in', 'neighbor_sum_out']

        # 3. Target y: lưu lượng tại thời điểm t (từ k_valid đến T_len)
        y_split = traffic_split[k_valid: T_len] # [T_valid, N]

        # 4. Flatten sang dạng shared_model: [M, D] với M = T_valid * N
        # Gom các đặc trưng theo thứ tự
        feature_cols = []
        for name in all_names:
            # t_feats[name]: [T_valid, N] -> làm phẳng thành [T_valid * N]
            feature_cols.append(t_feats[name].reshape(-1))

        # flow_id: [0, 1, ..., N-1] lặp lại T_valid lần
        flow_ids = np.tile(np.arange(N, dtype=np.int32), T_valid)

        # Đặt flow_id ở cột đầu tiên
        X_mat = np.column_stack([flow_ids] + feature_cols).astype(np.float32)
        y_vec = y_split.reshape(-1).astype(np.float32)

        column_names = ['flow_id'] + all_names
        return X_mat, y_vec, column_names, T_valid

    X_train, y_train, col_names, T_train = process_split(train_norm, tod_train, dow_train)
    X_val, y_val, _, T_val = process_split(val_norm, tod_val, dow_val)
    X_test, y_test, _, T_test = process_split(test_norm, tod_test, dow_test)

    metadata = {
        'dataset_name': ds_key,
        'columns': columns,
        'num_flows': N,
        'k_lags': k_lags,
        'feature_names': col_names,
        'categorical_features': ['flow_id'],
        'scaler': scaler,
        'M_in': M_in,
        'M_out': M_out,
        'T_train': T_train,
        'T_val': T_val,
        'T_test': T_test,
        'train_norm': train_norm,
        'val_norm': val_norm,
        'test_norm': test_norm,
        'tod_train': tod_train,
        'tod_val': tod_val,
        'tod_test': tod_test,
        'dow_train': dow_train,
        'dow_val': dow_val,
        'dow_test': dow_test
    }

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata


RELATIVE_PREFIXES = ('lag_', 'roll_mean_', 'roll_max_', 'roll_min_')


def to_relative_features(X, feature_names):
    """
    Biến đổi ma trận đặc trưng về dạng tương đối so với giá trị gần nhất lag_1
    (phục vụ nhánh LightGBM-Residual):
    - lag_i, roll_mean_w, roll_max_w, roll_min_w  ->  giá trị - lag_1
    - Giữ nguyên lag_1 (thông tin mức), roll_std, diff, tod, dow, lân cận, flow_id.
    Nhờ đó cây học "mức thay đổi" thay vì "mức tuyệt đối", nên dự báo không bị
    chặn bởi miền giá trị của tập Train.

    Trả về:
        X_rel: [M, D] cùng thứ tự cột với X
        lag1:  [M] giá trị lag_1 để khôi phục dự báo  y_hat = lag1 + delta_hat
    """
    idx = {n: i for i, n in enumerate(feature_names)}
    lag1 = X[:, idx['lag_1']].astype(np.float32)
    X_rel = X.astype(np.float32, copy=True)
    for name, i in idx.items():
        if name != 'lag_1' and name.startswith(RELATIVE_PREFIXES):
            X_rel[:, i] = X[:, i] - lag1
    return X_rel, lag1


class WindowTabularFeatureBuilder:
    """
    Bộ chuyển đổi trực tiếp từ raw window tensor [B, T, N, 3] sang ma trận đặc trưng
    cho mô hình cây [B * N, D] trong giai đoạn suy diễn (inference).
    """
    def __init__(self, metadata):
        self.metadata = metadata
        self.k_lags = metadata['k_lags']
        self.M_in = metadata['M_in']
        self.M_out = metadata['M_out']
        self.N = metadata['num_flows']
        self.col_names = metadata['feature_names']

    def __call__(self, x_window):
        """
        x_window: tensor [B, T, N, 3] hoặc numpy array [B, T, N, 3]
        - Kênh 0: Traffic [0, 1]
        - Kênh 1: TOD [0, 1)
        - Kênh 2: DOW [0, 1)
        """
        import torch
        if isinstance(x_window, torch.Tensor):
            x_np = x_window.detach().cpu().numpy()
        else:
            x_np = np.asarray(x_window)

        if x_np.ndim == 3:
            # [B, T, N]
            traffic = x_np
            tod = np.zeros((x_np.shape[0], self.N), dtype=np.float32)
            dow = np.zeros((x_np.shape[0], self.N), dtype=np.float32)
        else:
            traffic = x_np[:, :, :, 0]
            tod = x_np[:, -1, :, 1]
            dow = x_np[:, -1, :, 2]

        B, T, N = traffic.shape
        eps = 1e-6
        windows = (3, 5, 10)

        # 1. Lags từ cuối lùi về: lag_1 = x[:, -1, :], lag_2 = x[:, -2, :], ...
        feature_dict = {}
        for i in range(1, self.k_lags + 1):
            feature_dict[f"lag_{i}"] = traffic[:, -i, :] # [B, N]

        # 2. Rate of change
        lag1 = feature_dict["lag_1"]
        lag2 = feature_dict["lag_2"]
        feature_dict["diff_1"] = lag1 - lag2
        feature_dict["diff_pct"] = (lag1 - lag2) / (lag2 + eps)

        # 3. Rolling stats trên các mốc thời gian cuối của cửa sổ
        for w in windows:
            w_slice = traffic[:, -w:, :] # [B, w, N]
            feature_dict[f"roll_mean_{w}"] = np.mean(w_slice, axis=1) # [B, N]
            feature_dict[f"roll_std_{w}"] = np.std(w_slice, axis=1)
            feature_dict[f"roll_max_{w}"] = np.max(w_slice, axis=1)
            feature_dict[f"roll_min_{w}"] = np.min(w_slice, axis=1)

        # 4. TOD, DOW
        feature_dict["tod"] = tod
        feature_dict["dow"] = dow

        # 5. Spatial neighbors tại t-1 (lag 1)
        nbr_in, nbr_out = compute_spatial_neighbors(lag1, self.M_in, self.M_out)
        feature_dict["neighbor_sum_in"] = nbr_in
        feature_dict["neighbor_sum_out"] = nbr_out

        # Flatten sang [B * N, D]
        # Bỏ 'flow_id' trong dict vì flow_id sẽ được ghép đầu
        feat_matrices = []
        for name in self.col_names:
            if name == 'flow_id':
                continue
            feat_matrices.append(feature_dict[name].reshape(-1))

        # flow_id: [0, 1, ..., N-1] lặp lại B lần
        flow_ids = np.tile(np.arange(N, dtype=np.int32), B)
        X_out = np.column_stack([flow_ids] + feat_matrices).astype(np.float32)
        return X_out
