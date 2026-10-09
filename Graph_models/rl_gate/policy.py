"""
Policy của RL-Gate (spec Mục 6.2).

Encoder: Transformer nhỏ trên H token lịch sử sai số (+ token CLS) -> ghép đặc trưng tĩnh, a_{t-1}, log w_static
         -> MLP -> 1 lớp GNN trên {A_route, A_od} để luồng i nhận tín hiệu sớm từ luồng lân cận.
Actor  : u = c * tanh(z), z ~ N(mu, sigma); a = softmax(log w_static + u). Lớp cuối của mu khởi tạo 0
         nên hành động tất định ban đầu trùng trọng số stacking v3.
Critic : 2 head Q(s, u) dùng chung một encoder riêng của critic.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

U_SCALE = 3.0
LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0


class GateEncoder(nn.Module):
    def __init__(self, K, F_static, H, graphs=(), d_t=32, d=64, use_gnn=True, use_temporal=True):
        super().__init__()
        self.K, self.H = K, H
        self.use_temporal, self.use_gnn = use_temporal, use_gnn and len(graphs) > 0
        self.register_buffer('static_mean', torch.zeros(F_static))
        self.register_buffer('static_std', torch.ones(F_static))
        self.register_buffer('hist_scale', torch.ones(K + 2))
        in_dim = F_static + 2 * K
        if use_temporal:
            self.tok_proj = nn.Linear(K + 2, d_t)
            self.pos = nn.Parameter(torch.zeros(1, H + 1, d_t))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_t))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d_t, 2, 2 * d_t, 0.0, batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            in_dim += d_t
        self.mlp = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d))
        if self.use_gnn:
            for j, A in enumerate(graphs):
                self.register_buffer(f'A_{j}', torch.as_tensor(A, dtype=torch.float32))
            self.n_graphs = len(graphs)
            self.gnn = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in graphs])
            self.gnn_norm = nn.LayerNorm(d)

    def set_norm(self, stats):
        for k, v in stats.items():
            getattr(self, k).copy_(torch.as_tensor(v, dtype=torch.float32))

    def forward(self, static, hist, a_prev, logw):
        """static [B,N,F], hist [B,N,H,K+2], a_prev [B,N,K], logw [N,K] -> h [B,N,d]"""
        B, N, _ = static.shape
        xs = [(static - self.static_mean) / self.static_std, a_prev, logw.expand(B, N, self.K)]
        if self.use_temporal:
            tok = hist / self.hist_scale
            pad = tok[..., -1] < 0.5                                         # [B,N,H] True = chưa có lịch sử
            tok = self.tok_proj(tok.reshape(B * N, self.H, -1))
            tok = torch.cat([self.cls.expand(B * N, 1, -1), tok], dim=1) + self.pos
            mask = torch.cat([torch.zeros(B * N, 1, dtype=torch.bool, device=tok.device), pad.reshape(B * N, self.H)], 1)
            xs.append(self.temporal(tok, src_key_padding_mask=mask)[:, 0].reshape(B, N, -1))
        h = self.mlp(torch.cat(xs, dim=-1))
        if self.use_gnn:
            agg = sum(self.gnn[j](torch.einsum('ij,bjd->bid', getattr(self, f'A_{j}'), h)) for j in range(self.n_graphs))
            h = self.gnn_norm(h + F.relu(agg))
        return h


class Actor(nn.Module):
    def __init__(self, encoder, d=64, init_log_std=math.log(0.3)):
        super().__init__()
        self.enc = encoder
        K = encoder.K
        self.mu = nn.Linear(d, K)
        self.log_std = nn.Linear(d, K)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, init_log_std)

    def forward(self, static, hist, a_prev, logw, deterministic=False, with_logp=True):
        h = self.enc(static, hist, a_prev, logw)
        mu = self.mu(h)
        if deterministic:
            return U_SCALE * torch.tanh(mu), None
        std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        z = mu + std * torch.randn_like(mu)
        u = U_SCALE * torch.tanh(z)
        if not with_logp:
            return u, None
        logp = (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - torch.log(U_SCALE * (1 - torch.tanh(z) ** 2) + 1e-6).sum(-1)
        return u, logp


class Critic(nn.Module):
    def __init__(self, encoder, d=64):
        super().__init__()
        self.enc = encoder
        K = encoder.K
        self.q1 = nn.Sequential(nn.Linear(d + K, d), nn.ReLU(), nn.Linear(d, 1))
        self.q2 = nn.Sequential(nn.Linear(d + K, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, static, hist, a_prev, logw, u):
        h = torch.cat([self.enc(static, hist, a_prev, logw), u / U_SCALE], dim=-1)
        return self.q1(h).squeeze(-1), self.q2(h).squeeze(-1)


def mix(logw, u):
    """a = softmax(log w_static + u) theo trục nhánh."""
    return F.softmax(logw + u, dim=-1)
