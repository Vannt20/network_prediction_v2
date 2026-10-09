"""
Bộ điều phối online Hedge theo từng luồng (baseline bắt buộc của RL-Gate, spec Mục 2.3).

    L_{t,i}^k = beta * L_{t-1,i}^k + (p_{t-1,i}^k - y_{t-1,i})^2      # chỉ dùng nhãn đến t-1
    w_{t,i}^k ∝ w0_i^k * exp(-eta * L_{t,i}^k)
    Hedge + sàn: a_{t,i} = alpha * w0_i + (1 - alpha) * w_{t,i}

L = 0 ở đầu mỗi đoạn (Val và Test không liền nhau vì cách nhau seq_len bước).
Vì dự báo tuyến tính theo trọng số, dự báo của Hedge + sàn = alpha * static + (1 - alpha) * Hedge.
"""
import numpy as np
from scipy.signal import lfilter

ETA_GRID = (1, 3, 10, 30, 100, 300, 1000)
BETA_GRID = (0.9, 0.95, 0.98, 0.99, 0.995)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75)


def _as_kn(w0, K, N):
    w0 = np.asarray(w0, dtype=np.float64)
    if w0.ndim == 1:
        w0 = np.repeat(w0[:, None], N, axis=1)
    assert w0.shape == (K, N)
    return w0


def hedge_weights(P, y, w0, eta, beta):
    """P: [K,T,N], y: [T,N], w0: [K] hoặc [K,N] -> trọng số [K,T,N] (nhân quả)."""
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    K, T, N = P.shape
    w0 = _as_kn(w0, K, N)
    se = (P - y[None]) ** 2                                   # e_t^2 [K,T,N]
    se_prev = np.concatenate([np.zeros((K, 1, N)), se[:, :-1]], axis=1)   # e_{t-1}^2, = 0 ở t = 0
    L = lfilter([1.0], [1.0, -beta], se_prev, axis=1)        # L_t = beta L_{t-1} + e_{t-1}^2
    logits = np.log(np.clip(w0, 1e-12, None))[:, None, :] - eta * L
    logits -= logits.max(axis=0, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=0, keepdims=True)


def hedge_predict(P, y, w0, eta, beta, alpha=0.0):
    P = np.asarray(P, dtype=np.float64)
    K, T, N = P.shape
    w = hedge_weights(P, y, w0, eta, beta)
    if alpha:
        w = alpha * _as_kn(w0, K, N)[:, None, :] + (1.0 - alpha) * w
    return np.einsum('ktn,ktn->tn', w, P), w


def select_hedge(P_val, y_val, w0, etas=ETA_GRID, betas=BETA_GRID, alphas=ALPHA_GRID):
    """Chọn (eta, beta) cho Hedge và (eta, beta, alpha) cho Hedge + sàn theo MSE khi chạy tuần tự qua Val."""
    P_val = np.asarray(P_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    K, T, N = P_val.shape
    static = np.einsum('kn,ktn->tn', _as_kn(w0, K, N), P_val)
    grid = []
    for eta in etas:
        for beta in betas:
            ph, _ = hedge_predict(P_val, y_val, w0, eta, beta)
            for alpha in (0.0,) + tuple(a for a in alphas if a > 0):
                p = alpha * static + (1 - alpha) * ph
                grid.append({'eta': eta, 'beta': beta, 'alpha': alpha, 'mse': float(np.mean((p - y_val) ** 2))})
    hedge = min((g for g in grid if g['alpha'] == 0.0), key=lambda g: g['mse'])
    floor = min(grid, key=lambda g: g['mse'])
    return hedge, floor, grid
