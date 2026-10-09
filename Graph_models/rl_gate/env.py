"""
Môi trường phát lại của RL-Gate (spec Mục 6.2). Hành động không làm thay đổi lưu lượng nên
môi trường là chuỗi lịch sử được phát lại; mọi luồng chạy song song, policy dùng chung.

Prior p_t (trọng số mà policy co về khi không có tín hiệu):
  vòng 1: trọng số stacking tĩnh w_static [N, K]
  vòng 2: Hedge + sàn [T, N, K] (thay đổi theo bước, nhân quả) - spec Mục 15
Hành động a_t = softmax(log p_t + u_t); u = 0 thì a_t = p_t.

Phần thưởng theo từng luồng:
  r_{t,i} = -[(ŷ - y)^2 - (ŷ_prior - y)^2] / c  - λ_sw ||a_t - a_{t-1}||_1
            - λ_KL KL(a_t || p_t) - λ_floor max(0, f - Σ_{k∈Ext} a_t^k)
"""
import torch

from Graph_models.rl_gate.policy import mix


class GateTask:
    def __init__(self, data, prior, ext_idx, lam_sw=0.1, lam_kl=0.01, floor=0.2, lam_floor=1.0, c_scale=None):
        """
        data : SplitData
        prior: [N, K] (tĩnh) hoặc [T, N, K] (thay đổi theo bước); tổng theo K bằng 1
        ext_idx: chỉ số các nhánh ngoại suy được (ràng buộc sàn)
        """
        self.data = data
        p = torch.as_tensor(prior, dtype=torch.float32, device=data.device).clamp_min(1e-6)
        p = p / p.sum(-1, keepdim=True)
        if p.dim() == 2:
            p = p[None].expand(data.T, -1, -1)
        assert p.shape[0] == data.T, f"prior có {p.shape[0]} bước, dữ liệu có {data.T}"
        self.prior = p                                                          # [T, N, K]
        self.logp = torch.log(p)
        self.ext_idx = list(ext_idx)
        self.lam_sw, self.lam_kl, self.floor, self.lam_floor = lam_sw, lam_kl, floor, lam_floor
        y_prior = (p * data.P).sum(-1)
        self.se_prior = (y_prior - data.y) ** 2                                 # [T, N]
        self.c = float(c_scale if c_scale is not None else self.se_prior.mean().clamp_min(1e-12))

    def state(self, ts, a_prev):
        d = self.data
        return d.static[ts], d.hist(ts), a_prev, self.logp[ts]

    def penalties(self, a, a_prev, logp):
        kl = (a * (torch.log(a.clamp_min(1e-8)) - logp)).sum(-1)
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
        return -(se - self.se_prior[ts]) / self.c - self.penalties(a, a_prev, self.logp[ts])

    @torch.no_grad()
    def rollout(self, actor, deterministic=True):
        """Chạy tuần tự qua toàn đoạn (a_{-1} = p_0). Trả về (ŷ [T,N], a [T,N,K])."""
        d = self.data
        a_prev = self.prior[:1].clone()
        preds, ws = [], []
        for t in range(d.T):
            ts = torch.tensor([t], device=d.device)
            u, _ = actor(*self.state(ts, a_prev), deterministic=deterministic, with_logp=False)
            a = mix(self.logp[ts], u)
            preds.append((a[0] * d.P[t]).sum(-1))
            ws.append(a[0])
            a_prev = a
        return torch.stack(preds).cpu().numpy(), torch.stack(ws).cpu().numpy()

    @torch.no_grad()
    def rollout_static_prev(self, actor, bs=256):
        """Cho gate không phụ thuộc a_{t-1} (gate giám sát): a_{t-1} cố định bằng prior p_t, chạy theo lô."""
        d = self.data
        preds, ws = [], []
        for t0 in range(0, d.T, bs):
            ts = torch.arange(t0, min(d.T, t0 + bs), device=d.device)
            u, _ = actor(*self.state(ts, self.prior[ts]), deterministic=True, with_logp=False)
            a = mix(self.logp[ts], u)
            preds.append((a * d.P[ts]).sum(-1))
            ws.append(a)
        return torch.cat(preds).cpu().numpy(), torch.cat(ws).cpu().numpy()
