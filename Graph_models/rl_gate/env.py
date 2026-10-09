"""
Môi trường phát lại của RL-Gate (spec Mục 6.2). Hành động không làm thay đổi lưu lượng nên
môi trường là chuỗi lịch sử được phát lại; mọi luồng chạy song song, policy dùng chung.

Phần thưởng theo từng luồng:
  r_{t,i} = -[(ŷ - y)^2 - (ŷ_static - y)^2] / c  - λ_sw ||a_t - a_{t-1}||_1
            - λ_KL KL(a_t || w_static) - λ_floor max(0, f - Σ_{k∈Ext} a_t^k)
"""
import torch

from Graph_models.rl_gate.policy import mix


class GateTask:
    def __init__(self, data, w_static, ext_idx, lam_sw=0.1, lam_kl=0.01, floor=0.2, lam_floor=1.0, c_scale=None):
        """data: SplitData; w_static: tensor [N, K] (tổng theo K bằng 1); ext_idx: chỉ số nhánh ngoại suy được."""
        self.data = data
        self.w = torch.as_tensor(w_static, dtype=torch.float32, device=data.device).clamp_min(1e-6)
        self.w = self.w / self.w.sum(-1, keepdim=True)
        self.logw = torch.log(self.w)
        self.ext_idx = list(ext_idx)
        self.lam_sw, self.lam_kl, self.floor, self.lam_floor = lam_sw, lam_kl, floor, lam_floor
        ys = (self.w[None] * data.P).sum(-1)
        self.se_static = (ys - data.y) ** 2                                    # [T, N]
        self.c = float(c_scale if c_scale is not None else self.se_static.mean().clamp_min(1e-12))

    def state(self, ts, a_prev):
        d = self.data
        return d.static[ts], d.hist(ts), a_prev, self.logw

    def penalties(self, a, a_prev):
        kl = (a * (torch.log(a.clamp_min(1e-8)) - self.logw)).sum(-1)
        pen = self.lam_kl * kl
        if self.lam_sw:
            pen = pen + self.lam_sw * (a - a_prev).abs().sum(-1)
        if self.lam_floor and self.floor > 0 and self.ext_idx:
            pen = pen + self.lam_floor * torch.relu(self.floor - a[..., self.ext_idx].sum(-1))
        return pen

    def reward(self, ts, a, a_prev):
        d = self.data
        y_hat = (a * d.P[ts]).sum(-1)
        se = (y_hat - d.y[ts]) ** 2
        return -(se - self.se_static[ts]) / self.c - self.penalties(a, a_prev)

    @torch.no_grad()
    def rollout(self, actor, deterministic=True, chunk=None):
        """Chạy tuần tự qua toàn đoạn (a_{-1} = w_static). Trả về (ŷ [T,N], a [T,N,K])."""
        d = self.data
        a_prev = self.w[None].clone()
        preds, ws = [], []
        for t in range(d.T):
            ts = torch.tensor([t], device=d.device)
            u, _ = actor(*self.state(ts, a_prev), deterministic=deterministic, with_logp=False)
            a = mix(self.logw, u)
            preds.append((a[0] * d.P[t]).sum(-1))
            ws.append(a[0])
            a_prev = a
        return torch.stack(preds).cpu().numpy(), torch.stack(ws).cpu().numpy()

    @torch.no_grad()
    def rollout_static_prev(self, actor, bs=256):
        """Cho gate không phụ thuộc a_{t-1} (gate giám sát): a_{t-1} cố định bằng w_static, chạy theo lô."""
        d = self.data
        preds, ws = [], []
        for t0 in range(0, d.T, bs):
            ts = torch.arange(t0, min(d.T, t0 + bs), device=d.device)
            a_prev = self.w[None].expand(len(ts), -1, -1)
            u, _ = actor(*self.state(ts, a_prev), deterministic=True, with_logp=False)
            a = mix(self.logw, u)
            preds.append((a * d.P[ts]).sum(-1))
            ws.append(a)
        return torch.cat(preds).cpu().numpy(), torch.cat(ws).cpu().numpy()
