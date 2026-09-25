"""
Tầng kết hợp của ST-Adaptive-Ensemble v3: Stacking lồi bền vững theo luồng (Robust Per-Flow Convex Stacking).

Dự báo cuối cho luồng i tại bước t là tổ hợp lồi của K nhánh:
    y_hat[t, i] = sum_k w[i, k] * y_k[t, i],     w[i, :] = softmax(z + u_i)
- z   [K]    : trọng số chung toàn mạng
- u_i [K]    : độ lệch riêng của luồng i, bị co rút về 0 bởi lambda * ||u_i||^2
Hàm mục tiêu: loss(y_hat, y) + lambda * mean_i ||u_i||^2, loss in {MSE, Huber(delta)}.

Cấu hình (loss, scope) được chọn bằng kiểm định chéo theo khối thời gian trên tập Validation,
với quy tắc ưu tiên cấu hình đơn giản (cố định TRƯỚC khi đánh giá trên Test):
  1. Tiêu chí CV là Huber(delta) - không để vài điểm cực trị trong Val chi phối lựa chọn.
  2. Trong nhóm 'global': mặc định Huber; chỉ chọn MSE khi CV giảm >= min_gain.
  3. Chỉ chuyển sang 'perflow' khi CV giảm >= min_gain so với cấu hình global đã chọn;
     trong nhóm perflow áp dụng lại quy tắc 2.
"""
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_CONFIG = {
    'huber_delta': 0.05,     # delta Huber cho loss và tiêu chí CV (thang Min-Max)
    'shrink_lambda': 1e-3,   # hệ số co rút u_i về trọng số chung
    'n_folds': 2,            # số khối thời gian trên Val
    'min_gain': 0.01,        # ngưỡng cải thiện tương đối (1%) để chọn cấu hình phức tạp hơn
    'iters': 500,
    'lr': 0.05,
}


def _loss_fn(name, delta):
    if name == 'mse':
        return lambda a, b: ((a - b) ** 2).mean()
    if name == 'huber':
        return lambda a, b: F.huber_loss(a, b, delta=delta)
    raise ValueError(f"Loss không hỗ trợ: {name}")


def fit_convex_weights(P, y, loss='mse', scope='global', shrink_lambda=1e-3, huber_delta=0.05,
                       iters=500, lr=0.05, z_init=None):
    """
    P: [K, T, N] dự báo của K nhánh; y: [T, N].
    Trả về dict {'z': [K], 'u': [K, N] hoặc None, 'weights': [K] hoặc [K, N]}.
    """
    P_t = torch.as_tensor(P, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    K, T, N = P_t.shape
    lf = _loss_fn(loss, huber_delta)

    z0 = torch.zeros(K) if z_init is None else torch.as_tensor(z_init, dtype=torch.float32).clone()
    z = z0.clone().requires_grad_(True)
    params = [z]
    u = None
    if scope == 'perflow':
        # Bước 1: học trọng số chung làm tâm co rút
        g = fit_convex_weights(P, y, loss=loss, scope='global', huber_delta=huber_delta, iters=iters, lr=lr)
        z = torch.as_tensor(g['z'], dtype=torch.float32).clone().requires_grad_(False)
        u = torch.zeros(K, N, requires_grad=True)
        params = [u]

    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(iters):
        if u is None:
            w = F.softmax(z, dim=0)
            y_hat = torch.einsum('k,ktn->tn', w, P_t)
            obj = lf(y_hat, y_t)
        else:
            w = F.softmax(z[:, None] + u, dim=0)
            y_hat = torch.einsum('kn,ktn->tn', w, P_t)
            obj = lf(y_hat, y_t) + shrink_lambda * (u ** 2).mean()
        opt.zero_grad()
        obj.backward()
        opt.step()

    with torch.no_grad():
        if u is None:
            w = F.softmax(z, dim=0).numpy()
            return {'z': z.detach().numpy(), 'u': None, 'weights': w}
        w = F.softmax(z[:, None] + u, dim=0).numpy()
        return {'z': z.numpy(), 'u': u.detach().numpy(), 'weights': w}


def blend(weights, P):
    """weights: [K] hoặc [K, N]; P: [K, T, N] -> [T, N]."""
    P = np.asarray(P)
    if weights.ndim == 1:
        return np.einsum('k,ktn->tn', weights, P)
    return np.einsum('kn,ktn->tn', weights, P)


def huber_score(pred, y, delta):
    return float(F.huber_loss(torch.as_tensor(pred, dtype=torch.float32),
                              torch.as_tensor(y, dtype=torch.float32), delta=delta))


CANDIDATES = [('huber', 'global'), ('mse', 'global'), ('huber', 'perflow'), ('mse', 'perflow')]


def blocked_cv_scores(P, y, cfg=None):
    """CV theo khối thời gian liên tiếp trên Val: fit trên các khối còn lại, chấm điểm trên khối giữ lại."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    T = y.shape[0]
    edges = np.linspace(0, T, cfg['n_folds'] + 1).astype(int)
    scores = {}
    for loss, scope in CANDIDATES:
        fold_scores = []
        for f in range(cfg['n_folds']):
            te = np.arange(edges[f], edges[f + 1])
            tr = np.setdiff1d(np.arange(T), te)
            res = fit_convex_weights(P[:, tr], y[tr], loss=loss, scope=scope,
                                     shrink_lambda=cfg['shrink_lambda'], huber_delta=cfg['huber_delta'],
                                     iters=cfg['iters'], lr=cfg['lr'])
            fold_scores.append(huber_score(blend(res['weights'], P[:, te]), y[te], cfg['huber_delta']))
        scores[(loss, scope)] = float(np.mean(fold_scores))
    return scores


def select_config(scores, min_gain=0.01):
    """Quy tắc ưu tiên cấu hình đơn giản (xem docstring đầu module)."""
    def pick(scope):
        h, m = scores[('huber', scope)], scores[('mse', scope)]
        return ('mse', scope) if m < h * (1.0 - min_gain) else ('huber', scope)

    best_global = pick('global')
    best_perflow = pick('perflow')
    if scores[best_perflow] < scores[best_global] * (1.0 - min_gain):
        return best_perflow
    return best_global


class RobustPerFlowStacking:
    """
    Đối tượng tầng kết hợp: fit trên Validation, áp dụng cho Test.
    Có thể ép cấu hình (force_loss / force_scope) để phục vụ ablation.
    """
    def __init__(self, branch_names, config=None, force_loss=None, force_scope=None):
        self.branch_names = list(branch_names)
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.force_loss = force_loss
        self.force_scope = force_scope
        self.cv_scores = None
        self.selected = None
        self.weights = None

    def fit(self, P_val, y_val):
        P_val = np.asarray(P_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        if self.force_loss and self.force_scope:
            self.selected = (self.force_loss, self.force_scope)
        else:
            self.cv_scores = blocked_cv_scores(P_val, y_val, self.cfg)
            scores = self.cv_scores
            if self.force_scope:
                scores = {k: v for k, v in scores.items() if k[1] == self.force_scope}
                h, m = scores[('huber', self.force_scope)], scores[('mse', self.force_scope)]
                self.selected = ('mse', self.force_scope) if m < h * (1 - self.cfg['min_gain']) else ('huber', self.force_scope)
            else:
                self.selected = select_config(scores, self.cfg['min_gain'])
        loss, scope = self.selected
        res = fit_convex_weights(P_val, y_val, loss=loss, scope=scope,
                                 shrink_lambda=self.cfg['shrink_lambda'], huber_delta=self.cfg['huber_delta'],
                                 iters=self.cfg['iters'], lr=self.cfg['lr'])
        self.weights = res['weights']
        return self

    def per_flow_weights(self, num_flows):
        """Luôn trả về [K, N] (kể cả khi scope = global) để phân tích hành vi theo luồng."""
        if self.weights.ndim == 1:
            return np.repeat(self.weights[:, None], num_flows, axis=1)
        return self.weights

    def predict(self, P):
        return blend(self.weights, P)

    def state_dict(self):
        return {
            'branch_names': self.branch_names,
            'config': self.cfg,
            'selected': list(self.selected) if self.selected else None,
            'cv_scores': {f"{k[0]}-{k[1]}": v for k, v in (self.cv_scores or {}).items()},
            'weights': np.asarray(self.weights).tolist(),
        }
