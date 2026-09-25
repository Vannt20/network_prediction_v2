import numpy as np
import torch
import torch.nn as nn


class STAdaptiveEnsemble(nn.Module):
    """
    KIẾN TRÚC MÔ HÌNH HỌC KẾT HỢP ST-ADAPTIVE-ENSEMBLE v3
    Kết hợp K nhánh dự báo (mặc định 3) bằng tầng stacking lồi theo từng luồng OD:

    1. Nhánh học sâu (duy nhất): ST-WaveFormer (Dynamic GCN + Temporal Attention + RevIN)
    2. Nhánh học máy 1: GBDT Champion (XGBoost) trên kho đặc trưng dạng bảng
    3. Nhánh học máy 2: LightGBM-Residual (học phần dư so với lag_1, ngoại suy được ngoài biên Train)
    4. Tầng kết hợp: y_hat[:, i] = sum_k w[k, i] * y_k[:, i] với w[:, i] thuộc đơn hình (Graph_models/robust_stacking.py)

    Tất cả nhánh dự báo trên cùng thang Min-Max [0, 1] (nhánh DL đã khôi phục thang đo qua RevIN).
    Lớp này dùng cho suy diễn trực tuyến từ cửa sổ thô; huấn luyện tầng kết hợp được thực hiện offline
    trên bộ đệm dự báo (training/train_stacking.py).
    """
    def __init__(self, branches, weights):
        """
        branches: list các module/callable nhận x [B, T, N, 3] và trả về [B, N]
        weights:  mảng [K] hoặc [K, N] trọng số lồi đã học trên Validation
        """
        super(STAdaptiveEnsemble, self).__init__()
        self.branches = nn.ModuleList([b for b in branches if isinstance(b, nn.Module)])
        self._callables = list(branches)
        w = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
        self.register_buffer('weights', w)

    def predict_components(self, x):
        preds = []
        for b in self._callables:
            with torch.no_grad():
                preds.append(b(x).detach())
        P = torch.stack(preds, dim=0)                       # [K, B, N]
        if self.weights.dim() == 1:
            y_hat = torch.einsum('k,kbn->bn', self.weights, P)
        else:
            y_hat = torch.einsum('kn,kbn->bn', self.weights, P)
        return {'y_hat': y_hat, 'components': P, 'weights': self.weights}

    def forward(self, x):
        return self.predict_components(x)['y_hat']
