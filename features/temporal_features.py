import numpy as np


def extract_temporal_features_matrix(traffic_arr, tod_arr, dow_arr, k_lags=12, windows=(3, 5, 10), eps=1e-6):
    """
    Trích xuất ma trận đặc trưng chuỗi thời gian thuần NumPy (C-speed) cho toàn bộ chuỗi:
    - Lags: lag_1 ... lag_k
    - Rolling stats: roll_mean, roll_std, roll_max, roll_min trên các cửa sổ w in {3, 5, 10}
    - Rate of change: diff_1 = x[t-1] - x[t-2], diff_pct = diff_1 / (x[t-2] + eps)
    - Contextual time: tod[t], dow[t]
    
    Tham số:
        traffic_arr: [T, N] - mảng lưu lượng đã chuẩn hóa [0, 1]
        tod_arr: [T, N] hoặc [T] - time of day [0, 1]
        dow_arr: [T, N] hoặc [T] - day of week [0, 1]
        k_lags: số bước trễ (SDN: 15, GEANT/ABILENE: 12)
        windows: các độ dài cửa sổ trượt (3, 5, 10)
    
    Trả về:
        feats: dictionary gồm các mảng 2D kích thước [T_valid, N]
        t_start: index bắt đầu hợp lệ (k_lags)
        feature_names: danh sách tên các đặc trưng
    """
    T, N = traffic_arr.shape
    max_w = max(windows)
    k = max(k_lags, max_w + 1)
    T_valid = T - k
    
    if tod_arr.ndim == 1:
        tod_2d = np.tile(tod_arr[:, None], (1, N))
    else:
        tod_2d = tod_arr
        
    if dow_arr.ndim == 1:
        dow_2d = np.tile(dow_arr[:, None], (1, N))
    else:
        dow_2d = dow_arr
        
    feats = {}
    feature_names = []
    
    # 1. Lags: lag_1 ... lag_k_lags (x[t-1], x[t-2], ...)
    for i in range(1, k_lags + 1):
        name = f"lag_{i}"
        # t chạy từ k đến T-1 -> t-i chạy từ k-i đến T-i
        feats[name] = traffic_arr[k - i : T - i].copy()
        feature_names.append(name)
        
    # 2. Rate of change (sai phân & tỷ lệ sai phân)
    lag_1 = feats["lag_1"]
    lag_2 = feats["lag_2"]
    diff_1 = lag_1 - lag_2
    diff_pct = diff_1 / (lag_2 + eps)
    
    feats["diff_1"] = diff_1
    feature_names.append("diff_1")
    feats["diff_pct"] = diff_pct
    feature_names.append("diff_pct")
    
    # 3. Rolling statistics (Mean, Std, Max, Min)
    # Dùng numpy sliding window view cực nhanh
    swv = np.lib.stride_tricks.sliding_window_view(traffic_arr, window_shape=windows[-1] + 5, axis=0)
    
    for w in windows:
        # Cần cửa sổ kết thúc tại t-1: [t-w : t]
        # Cho từng t in [k, T):
        # Ta cắt trích xuất window shape [T_valid, N, w]
        w_slices = []
        for offset in range(w, 0, -1):
            w_slices.append(traffic_arr[k - offset : T - offset])
        w_stack = np.stack(w_slices, axis=-1) # [T_valid, N, w]
        
        m_mean = np.mean(w_stack, axis=-1)
        m_std = np.std(w_stack, axis=-1)
        m_max = np.max(w_stack, axis=-1)
        m_min = np.min(w_stack, axis=-1)
        
        feats[f"roll_mean_{w}"] = m_mean
        feats[f"roll_std_{w}"] = m_std
        feats[f"roll_max_{w}"] = m_max
        feats[f"roll_min_{w}"] = m_min
        
        feature_names.extend([f"roll_mean_{w}", f"roll_std_{w}", f"roll_max_{w}", f"roll_min_{w}"])
        
    # 4. Contextual time
    feats["tod"] = tod_2d[k : T].copy()
    feature_names.append("tod")
    feats["dow"] = dow_2d[k : T].copy()
    feature_names.append("dow")
    
    return feats, k, feature_names


def extract_context_features_torch(x, eps=1e-6):
    """
    Trích xuất vector ngữ cảnh 4 chiều cho mạng Gate:
    x: tensor [B, T, N, C] (hoặc [B, T, N])
       C=0: traffic, C=1: tod, C=2: dow
    Trả về:
       context_features: tensor [B, N, 4] gồm:
       [local_volatility, spike_flag, tod, dow]
    """
    import torch
    
    if x.dim() == 4:
        traffic = x[:, :, :, 0] # [B, T, N]
        if x.shape[-1] >= 3:
            tod = x[:, -1, :, 1] # [B, N]
            dow = x[:, -1, :, 2] # [B, N]
        else:
            tod = torch.zeros_like(traffic[:, -1, :])
            dow = torch.zeros_like(traffic[:, -1, :])
    else:
        traffic = x
        tod = torch.zeros_like(traffic[:, -1, :])
        dow = torch.zeros_like(traffic[:, -1, :])
        
    B, T, N = traffic.shape
    
    # 1. Local volatility (độ lệch chuẩn trong cửa sổ ngắn gần nhất)
    w_len = min(5, T)
    short_window = traffic[:, -w_len:, :] # [B, w_len, N]
    local_volatility = torch.std(short_window, dim=1) # [B, N]
    
    # 2. Spike flag: phát hiện gai xung đột biến
    # So sánh giá trị mốc cuối với trung bình mốc trước
    last_val = traffic[:, -1, :] # [B, N]
    prev_val = traffic[:, -2, :] if T > 1 else last_val
    mean_val = torch.mean(short_window, dim=1)
    
    # Spike khi chênh lệch > 2 * std hoặc tốc độ tăng vọt > 50%
    diff_abs = torch.abs(last_val - prev_val)
    spike_cond = (diff_abs > (2.0 * local_volatility + 1e-4)) | (diff_abs / (prev_val + 1e-4) > 0.5)
    spike_flag = spike_cond.float() # [B, N]
    
    # Gom 4 kênh: [B, N, 4]
    context = torch.stack([local_volatility, spike_flag, tod, dow], dim=-1)
    return context
