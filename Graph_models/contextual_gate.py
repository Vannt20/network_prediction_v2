import torch
import torch.nn as nn
import torch.nn.functional as F


class PerFlowContextualMetaGating(nn.Module):
    """
    Cổng MLP ngữ cảnh theo luồng (phiên bản v2) - chỉ giữ lại làm cấu hình đối chứng trong ablation
    của ST-Adaptive-Ensemble v3.
    - Đầu vào: vector ngữ cảnh [B, N, context_dim] (local_volatility, spike_flag, tod, dow, ...).
    - Đầu ra: trọng số Softmax [B, N, K] cho K nhánh (sum = 1 theo từng luồng).
    - init_prior: logits khởi tạo cho lớp cuối (ví dụ log của trọng số stacking toàn mạng).
    """
    def __init__(self, context_dim=4, hidden_dim=32, num_branches=3, dropout=0.05, init_prior=None):
        super(PerFlowContextualMetaGating, self).__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_branches = num_branches

        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_branches)
        )
        self.init_weights(init_prior)

    def init_weights(self, init_prior=None):
        nn.init.xavier_uniform_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)
        nn.init.xavier_uniform_(self.gate_net[3].weight)
        if init_prior is not None:
            with torch.no_grad():
                self.gate_net[3].bias.copy_(torch.as_tensor(init_prior, dtype=torch.float32))
        else:
            nn.init.zeros_(self.gate_net[3].bias)

    def forward(self, context_features):
        logits = self.gate_net(context_features)  # [B, N, K]
        return F.softmax(logits, dim=-1)
