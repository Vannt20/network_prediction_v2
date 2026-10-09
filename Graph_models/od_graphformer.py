"""
OD-GraphFormer: nhánh học sâu của ST-Adaptive-Ensemble-RL (spec Mục 4.2).

  1. Chuẩn hóa phần dư theo cửa sổ: z = asinh((x - x_last) / s), s = max(1.4826 * MAD_window, eps_s)
  2. Patch theo thời gian từng luồng (PatchTST), Transformer dùng chung trọng số giữa các luồng
  3. Nhúng danh tính: E_src + E_dst + E_flow + E_tod + E_dow (+ mức x_last, log s)
  4. Khối không gian: diffusion conv 2 bước trên {A_route, A_od, A_adp} + self-attention giữa các luồng
     có structural bias kiểu Graphormer (b_m^head * A_m[i, j])
  5. Hai head: Δz -> ŷ = x_last + Δz * s ; log σ̂² (trên thang MinMax)

Lớp cuối của head Δ khởi tạo bằng 0 nên mô hình ban đầu trùng Persistence.
Cờ ablation: residual=False (mức tuyệt đối + RevIN-like theo trung bình), graphs (bỏ từng đồ thị),
spatial_attn=False, use_identity=False, patch_len=1.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBlock(nn.Module):
    def __init__(self, d, num_flows, fixed_graphs, use_adp, adp_dim, nhead, dropout, spatial_attn, diffusion_steps=2):
        super().__init__()
        self.fixed_names = list(fixed_graphs.keys())
        for name, A in fixed_graphs.items():
            self.register_buffer(f'A_{name}', torch.as_tensor(A, dtype=torch.float32))
        self.use_adp = use_adp
        self.steps = diffusion_steps
        n_graphs = len(self.fixed_names) + int(use_adp)
        self.diff_lin = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_graphs * diffusion_steps)])
        self.norm1 = nn.LayerNorm(d)
        self.spatial_attn = spatial_attn
        if spatial_attn:
            self.nhead = nhead
            self.qkv = nn.Linear(d, 3 * d)
            self.attn_out = nn.Linear(d, d)
            self.graph_bias = nn.Parameter(torch.zeros(len(self.fixed_names), nhead))
            self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.norm3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def graphs(self, A_adp):
        gs = [getattr(self, f'A_{n}') for n in self.fixed_names]
        if self.use_adp:
            gs.append(A_adp)
        return gs

    def forward(self, h, A_adp):
        # h: [B, N, d]
        x = self.norm1(h)
        out, j = 0.0, 0
        for A in self.graphs(A_adp):
            xk = x
            for _ in range(self.steps):
                xk = torch.einsum('ij,bjd->bid', A.to(xk.dtype), xk)
                out = out + self.diff_lin[j](xk)
                j += 1
        h = h + self.drop(F.gelu(out)) if j else h
        if self.spatial_attn:
            x = self.norm2(h)
            B, N, d = x.shape
            q, k, v = self.qkv(x).view(B, N, 3, self.nhead, d // self.nhead).permute(2, 0, 3, 1, 4)
            bias = None
            if self.fixed_names:
                A = torch.stack([getattr(self, f'A_{n}') for n in self.fixed_names])          # [M, N, N]
                bias = torch.einsum('mh,mij->hij', self.graph_bias, A).unsqueeze(0).to(q.dtype)  # [1, H, N, N]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                               dropout_p=self.drop.p if self.training else 0.0)
            h = h + self.drop(self.attn_out(a.transpose(1, 2).reshape(B, N, d)))
        return h + self.drop(self.ffn(self.norm3(h)))


class ODGraphFormer(nn.Module):
    def __init__(self, num_flows, seq_len, src_idx, dst_idx, num_nodes, graphs, steps_per_day=288,
                 d_model=64, n_temporal=2, n_spatial=2, nhead=4, dropout=0.1, patch_len=6, patch_stride=None,
                 eps_s=1e-3, residual=True, use_identity=True, use_adp=True, spatial_attn=True, adp_dim=10):
        """
        graphs: dict tên -> ma trận [N, N] đã chuẩn hóa (ví dụ {'route': A_route, 'od': A_od}); có thể rỗng.
        src_idx, dst_idx: chỉ số nút nguồn/đích của từng luồng (list độ dài N).
        """
        super().__init__()
        self.N, self.L, self.d = num_flows, seq_len, d_model
        self.eps_s, self.residual, self.use_identity = eps_s, residual, use_identity
        self.spd = int(steps_per_day)
        self.patch_len = int(patch_len)
        self.stride = int(patch_stride or max(1, self.patch_len // 2))
        self.n_patch = (seq_len - self.patch_len) // self.stride + 1
        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout, batch_first=True, norm_first=True,
                                         activation='gelu')
        self.temporal = nn.TransformerEncoder(enc, n_temporal, enable_nested_tensor=False)
        self.level_proj = nn.Linear(2, d_model)
        self.register_buffer('src_idx', torch.as_tensor(src_idx, dtype=torch.long))
        self.register_buffer('dst_idx', torch.as_tensor(dst_idx, dtype=torch.long))
        if use_identity:
            self.E_src = nn.Embedding(num_nodes, d_model)
            self.E_dst = nn.Embedding(num_nodes, d_model)
            self.E_flow = nn.Embedding(num_flows, d_model)
            self.E_tod = nn.Embedding(self.spd, d_model)
            self.E_dow = nn.Embedding(7, d_model)
            for e in (self.E_src, self.E_dst, self.E_flow, self.E_tod, self.E_dow):
                nn.init.normal_(e.weight, std=0.02)
        self.use_adp = use_adp
        if use_adp:
            self.adp1 = nn.Parameter(torch.randn(num_flows, adp_dim) * 0.1)
            self.adp2 = nn.Parameter(torch.randn(adp_dim, num_flows) * 0.1)
        self.blocks = nn.ModuleList([
            SpatialBlock(d_model, num_flows, graphs, use_adp, adp_dim, nhead, dropout, spatial_attn)
            for _ in range(n_spatial)])
        self.norm = nn.LayerNorm(d_model)
        self.head_delta = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.head_logvar = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.head_delta[-1].weight)
        nn.init.zeros_(self.head_delta[-1].bias)

    def _scale(self, xv):
        """xv [B, L, N] -> (ref [B, N], s [B, N])"""
        if self.residual:
            ref = xv[:, -1]
            med = xv.median(dim=1).values
            s = 1.4826 * (xv - med[:, None]).abs().median(dim=1).values
        else:
            ref = xv.mean(dim=1)
            s = xv.std(dim=1)
        return ref, s.clamp_min(self.eps_s)

    def forward(self, x):
        """x: [B, L, N, 3] (traffic MinMax, tod, dow) -> (ŷ [B, N], log σ̂² [B, N])"""
        xv = x[..., 0].float()
        B = xv.shape[0]
        ref, s = self._scale(xv)
        z = torch.asinh((xv - ref[:, None]) / s[:, None])                     # [B, L, N]
        p = z.permute(0, 2, 1).unfold(-1, self.patch_len, self.stride)        # [B, N, P, patch]
        tok = self.patch_proj(p).reshape(B * self.N, self.n_patch, self.d) + self.pos
        h = self.temporal(tok)[:, -1].reshape(B, self.N, self.d)
        h = h + self.level_proj(torch.stack([ref, torch.log(s)], dim=-1))
        if self.use_identity:
            tod_last = x[:, -1, 0, 1].float()
            slot = (torch.floor(tod_last * self.spd).long() + 1) % self.spd    # bước cần dự báo = t-1 + 1
            dow = torch.floor(x[:, -1, 0, 2].float() * 7 + 1e-4).long().clamp(0, 6)
            ident = self.E_src(self.src_idx) + self.E_dst(self.dst_idx) + self.E_flow.weight
            h = h + ident[None] + (self.E_tod(slot) + self.E_dow(dow))[:, None, :]
        A_adp = F.softmax(F.relu(self.adp1 @ self.adp2), dim=1) if self.use_adp else None
        for blk in self.blocks:
            h = blk(h, A_adp)
        h = self.norm(h)
        dz = self.head_delta(h).squeeze(-1)
        y_hat = ref + dz * s
        log_var = self.head_logvar(h).squeeze(-1) + 2.0 * torch.log(s)
        return y_hat, log_var


def build_odgf_from_config(cfg, graphs):
    """Dựng mô hình từ config.json đã lưu (dùng lại khi suy diễn / precompute cache)."""
    keys = ['num_flows', 'seq_len', 'src_idx', 'dst_idx', 'num_nodes', 'steps_per_day', 'd_model', 'n_temporal',
            'n_spatial', 'nhead', 'dropout', 'patch_len', 'patch_stride', 'eps_s', 'residual', 'use_identity',
            'use_adp', 'spatial_attn']
    kw = {k: cfg[k] for k in keys if k in cfg}
    used = {g: graphs[g] for g in cfg.get('fixed_graphs', []) if g in graphs}
    return ODGraphFormer(graphs=used, **kw)
