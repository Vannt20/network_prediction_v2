import numpy as np
from scipy import stats


def diebold_mariano(y_true, pred_a, pred_b, power=2):
    """
    Kiểm định Diebold-Mariano so sánh độ chính xác dự báo của A và B trên cùng tập Test.
    - Chuỗi chênh lệch tổn thất theo thời gian: d_t = mean_i |e_A|^p - mean_i |e_B|^p (trung bình qua các luồng).
    - Phương sai dài hạn ước lượng theo Newey-West với độ trễ h = floor(T^(1/3)).
    Trả về (dm_stat, p_value hai phía). dm_stat < 0 nghĩa là A có sai số thấp hơn B.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    ea = np.abs(np.asarray(pred_a, dtype=np.float64) - y_true) ** power
    eb = np.abs(np.asarray(pred_b, dtype=np.float64) - y_true) ** power
    d = (ea - eb).reshape(ea.shape[0], -1).mean(axis=1)
    T = len(d)
    d_mean = d.mean()
    dc = d - d_mean
    h = int(np.floor(T ** (1.0 / 3.0)))
    lrv = np.dot(dc, dc) / T
    for lag in range(1, h + 1):
        gamma = np.dot(dc[lag:], dc[:-lag]) / T
        lrv += 2.0 * (1.0 - lag / (h + 1.0)) * gamma
    if lrv <= 0:
        return 0.0, 1.0
    dm = d_mean / np.sqrt(lrv / T)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def paired_ttest(a, b):
    """t-test ghép cặp theo run (a - b). Trả về (t, p); t < 0 nghĩa là a thấp hơn."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)
