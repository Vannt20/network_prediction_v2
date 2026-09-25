import time
import numpy as np
import torch

EPS = 1e-8


def calc_metrics_numpy(preds, labels):
    """
    Tính các chỉ số đánh giá bằng NumPy:
    - MSE, MAE, RMSE, RSE, MAPE (masked), WAPE, sMAPE
    """
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    
    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(mse)
    
    denom_rse = np.sum((labels - np.mean(labels)) ** 2) + EPS
    rse = np.sum((preds - labels) ** 2) / denom_rse
    
    # WAPE (Weighted Absolute Percentage Error) - Chuẩn mực chống nổ mẫu số khi traffic rỗi
    wape = np.sum(np.abs(preds - labels)) / (np.sum(np.abs(labels)) + EPS)
    
    # sMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2.0 * np.abs(preds - labels) / (np.abs(preds) + np.abs(labels) + EPS)) * 100.0

    # Masked MAPE cho các giá trị nhãn đủ lớn để tránh chia cho số cận 0
    mask = np.abs(labels) > 1e-4
    if np.any(mask):
        mape = np.mean(np.abs((preds[mask] - labels[mask]) / labels[mask]))
    else:
        mape = 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'rse': float(rse),
        'mape': float(mape),
        'wape': float(wape),
        'smape': float(smape)
    }


def measure_inference_time(predict_fn, X, batch_size=64, n_warmup=2, n_runs=5):
    """
    Đo thời gian suy diễn trung bình (ms / batch).
    """
    N_samples = len(X)
    if N_samples == 0:
        return 0.0
        
    num_batches = int(np.ceil(N_samples / batch_size))
    # Chạy warmup
    for _ in range(n_warmup):
        sample_batch = X[:batch_size]
        predict_fn(sample_batch)
        
    # Đo thời gian
    times = []
    for _ in range(n_runs):
        idx = np.random.randint(0, max(1, N_samples - batch_size))
        batch = X[idx : idx + batch_size]
        t0 = time.perf_counter()
        predict_fn(batch)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0) # in ms
        
    return float(np.mean(times))
