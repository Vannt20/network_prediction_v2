import torch
import torch.nn as nn
import numpy as np


class MLBranch(nn.Module):
    """
    Wrapper đóng băng mô hình GBDT quán quân từ Module A (Shared Model across all flows).
    Nhận đầu vào tensor chuỗi thời gian [B, T, N, 3], trích xuất ma trận đặc trưng tabular,
    dự đoán bằng cây và trả về tensor [B, N] trong không gian MinMax [0, 1].
    """
    def __init__(self, model_instance, feature_builder):
        super(MLBranch, self).__init__()
        self.model = model_instance          # Model GBDT (LightGBM/CatBoost/XGBoost) đã fit
        self.feature_builder = feature_builder

    def forward(self, raw_window):
        """
        raw_window: tensor [B, T, N, 3] hoặc [B, T, N]
        Trả về:
            y_pred: tensor [B, N] trên cùng device với raw_window
        """
        device = raw_window.device if isinstance(raw_window, torch.Tensor) else torch.device('cpu')
        B = raw_window.shape[0]
        N = raw_window.shape[2] if raw_window.dim() >= 3 else raw_window.shape[1]

        # Trích xuất ma trận đặc trưng [B * N, D]
        X = self.feature_builder(raw_window)

        # Dự đoán bằng mô hình cây đã fit (không tính gradient)
        with torch.no_grad():
            preds_np = self.model.predict(X) # [B * N]
            preds_np = np.clip(preds_np, 0.0, None)
            preds_tensor = torch.from_numpy(preds_np).float().to(device)
            out = preds_tensor.view(B, N) # [B, N]

        return out
