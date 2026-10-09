"""
Trạng thái của RL-Gate cho từng (bước t, luồng i), chỉ dùng nhãn đến t-1 (spec Mục 6.2).

Đặc trưng tĩnh theo t (không phụ thuộc hành động), F = 3K + 12 chiều:
  p_t^k - x_{t-1} (K) | độ lệch chuẩn của p_t^k theo k (1) | sqrt EWMA e^2, decay 0.9 và 0.99 (2K)
  | log σ̂_t, cờ có σ̂ (2) | context v3 (4) | sin/cos tod, sin/cos dow của bước t (4) | cờ x_{t-1} > 1 (1)
Token lịch sử tau = 1..H: [e_{t-tau}^1..K, y_{t-tau} - y_{t-tau-1}, mask] (K + 2)
log w_static (K) và a_{t-1} (K) được ghép khi gọi state(), vì phụ thuộc prior và hành động.
"""
import math
import numpy as np
import torch

EW_DECAYS = (0.9, 0.99)


def _t(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class SplitData:
    """Dữ liệu một đoạn (OOF / Val / Test) đã sắp xếp theo [T, N, K] cho RL-Gate."""

    def __init__(self, P, y, last, context, sigma=None, tod=None, dow=None, H=12, use_sigma=True, device='cpu'):
        P = np.asarray(P, dtype=np.float32)                       # [K, T, N]
        K, T, N = P.shape
        self.K, self.T, self.N, self.H, self.device = K, T, N, H, device
        self.P = _t(np.moveaxis(P, 0, -1), device)                # [T, N, K]
        self.y = _t(y, device)
        self.last = _t(last, device)
        e = self.P - self.y[..., None]                            # e_t^k [T, N, K]
        dy = (self.y - self.last)[..., None]                       # y_t - y_{t-1} (last_t = y_{t-1} trong split)
        tok = torch.cat([e, dy, torch.ones_like(dy)], dim=-1)      # [T, N, K + 2]
        self.tok_pad = torch.cat([torch.zeros(H, N, K + 2, device=device), tok], dim=0)   # tok_pad[s + H] = tok[s]

        # EWMA nhân quả của e^2: ew[t] = d * ew[t-1] + (1 - d) * e[t-1]^2, ew[0] = 0
        e2 = (e ** 2).cpu().numpy()
        ews = []
        for d in EW_DECAYS:
            ew = np.zeros_like(e2)
            for t in range(1, T):
                ew[t] = d * ew[t - 1] + (1 - d) * e2[t - 1]
            ews.append(np.sqrt(ew))
        ew = _t(np.concatenate(ews, axis=-1), device)              # [T, N, 2K]

        d_lag = self.P - self.last[..., None]
        spread = self.P.std(dim=-1, keepdim=True)
        if sigma is not None and use_sigma and np.isfinite(np.asarray(sigma)).any():
            s = _t(np.nan_to_num(np.asarray(sigma, dtype=np.float32), nan=1.0), device)
            sig = torch.stack([torch.log(s.clamp_min(1e-6)), torch.ones_like(s)], dim=-1)
        else:
            sig = torch.zeros(T, N, 2, device=device)
        ctx = _t(context, device)                                  # [T, N, 4]
        tod = _t(tod if tod is not None else np.zeros(T), device)
        dow = _t(dow if dow is not None else np.zeros(T), device)
        cal = torch.stack([torch.sin(2 * math.pi * tod), torch.cos(2 * math.pi * tod),
                           torch.sin(2 * math.pi * dow), torch.cos(2 * math.pi * dow)], dim=-1)
        cal = cal[:, None, :].expand(T, N, 4)
        over = (self.last > 1.0).float()[..., None]
        self.static = torch.cat([d_lag, spread, ew, sig, ctx, cal, over], dim=-1)   # [T, N, 3K + 12]
        self.F = self.static.shape[-1]

    def hist(self, ts):
        """ts: LongTensor [B] -> token lịch sử [B, N, H, K + 2] (tau = 1 gần nhất)."""
        idx = ts[:, None] + self.H - torch.arange(1, self.H + 1, device=ts.device)[None, :]
        return self.tok_pad[idx].permute(0, 2, 1, 3)

    def norm_stats(self):
        """Thống kê chuẩn hóa (fit trên đoạn dùng để huấn luyện)."""
        s = self.static.reshape(-1, self.F)
        tok = self.tok_pad[self.H:].reshape(-1, self.K + 2)
        h_std = tok.std(dim=0).clamp_min(1e-6)
        h_std[-1] = 1.0
        return {'static_mean': s.mean(0), 'static_std': s.std(0).clamp_min(1e-6),
                'hist_scale': h_std}

    def subset(self, t0, t1):
        """Đoạn con [t0, t1) cho CV 2 khối. Giữ lịch sử thật trước t0 (đó là quá khứ, vẫn nhân quả)."""
        out = object.__new__(SplitData)
        out.__dict__.update(self.__dict__)
        out.P, out.y, out.last, out.static = self.P[t0:t1], self.y[t0:t1], self.last[t0:t1], self.static[t0:t1]
        out.tok_pad = self.tok_pad[t0:self.H + t1]
        out.T = t1 - t0
        return out


def split_from_cache(d, branch_idx, H=12, use_sigma=True, device='cpu'):
    """d: cache['val'] / cache['test'] / OOF dict; branch_idx: chỉ số nhánh theo thứ tự cần dùng."""
    P = d['P'].numpy()[branch_idx]
    g = lambda k: (d[k].numpy() if hasattr(d[k], 'numpy') else d[k]) if k in d else None
    return SplitData(P, g('y'), g('last'), g('context'), g('sigma'), g('tod'), g('dow'), H=H,
                     use_sigma=use_sigma, device=device)
