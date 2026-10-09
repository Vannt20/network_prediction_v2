"""
Chỉ số chính + chỉ số phụ cho hướng V4 (spec Mục 2.2). Mọi giá trị trên thang MinMax.

- mse_in_range : MSE trên các điểm y <= 1 (trong biên Train)
- mse_trim_jump: MSE sau khi bỏ 0,1% điểm có |y - lag_1| lớn nhất. Mặt nạ cố định theo dữ liệu,
                 dùng chung cho mọi mô hình (không cắt theo sai số riêng của từng mô hình).
- top1_se_share: tỷ lệ tổng SE do 1% điểm sai số lớn nhất của chính mô hình
- rise_se_share: trong top 1% điểm theo SE, phần SE ở pha tăng (y > lag_1)
"""
import numpy as np

from baselines_ml.metrics import calc_metrics_numpy


def _jump_mask(y, last, q):
    from training.gnn_rl_common import jump_mask
    return jump_mask(y, last, q)


def extra_metrics(pred, y, last, keep_mask=None, q_trim=0.001):
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    last = np.asarray(last, dtype=np.float64)
    m = calc_metrics_numpy(pred, y)
    se = (pred - y) ** 2
    in_range = y <= 1.0
    m['mse_in_range'] = float(se[in_range].mean()) if in_range.any() else float('nan')
    if keep_mask is None:
        keep_mask = _jump_mask(y, last, q_trim)
    m['mse_trim_jump'] = float(se[keep_mask].mean())
    flat = se.reshape(-1)
    k = max(1, int(np.ceil(flat.size * 0.01)))
    top = np.argpartition(flat, flat.size - k)[flat.size - k:]
    tot = flat.sum() + 1e-30
    m['top1_se_share'] = float(flat[top].sum() / tot)
    rise = (y > last).reshape(-1)[top]
    m['rise_se_share'] = float(flat[top][rise].sum() / (flat[top].sum() + 1e-30))
    m['n_out_of_range'] = int((~in_range).sum())
    return m
