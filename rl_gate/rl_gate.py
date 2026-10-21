"""
RL-Gate: tác tử chọn trọng số kết hợp cho từng (bước t, luồng i), dùng chung một policy cho mọi luồng.

    z[t,i,:] = log w_v3[i,:] + sum_m beta_m(s) * (-eta_m * L_m[t,i,:]) + delta(s)
    a[t,i,:] = softmax(z[t,i,:]),     y_hat[t,i] = sum_k a[t,i,k] * p_k[t,i]

- w_v3   : trọng số stacking lồi của v3 (prior, "bộ nhớ dài hạn"). Khi beta = 0 và delta = 0, mô hình trùng v3.
- L_m    : sai số lũy kế có suy giảm của từng nhánh, chỉ dùng nhãn đến t-1 (các kênh Hedge nhanh/chậm, SE/AE).
- beta_m : mức tin vào kênh bộ nhớ m, delta: độ lệch logit; cả hai là hàm tuyến tính của trạng thái s[t,i].
- Reward r[t,i] = -(y_hat - y)^2 đã biết dạng giải tích nên policy được học bằng deterministic policy gradient
  với critic chính xác (không cần ước lượng Q); REINFORCE được giữ làm phương án đối chứng.

Công tắc an toàn (safety switch) theo thời gian: tại t chỉ dùng RL-Gate nếu tổng SE của nó trên Test tính đến t-1
thấp hơn v3 ít nhất `margin` (mặc định 1%, giống quy tắc chọn cấu hình của tầng stacking v3); ngược lại dùng v3.
"""
import numpy as np

CHANNELS = [('se', 100, 0.9), ('se', 100, 0.98), ('ae', 10, 0.98)]   # (loại sai số, eta, hệ số suy giảm)


def softmax_k(z):
    """Softmax theo trục nhánh (trục -2) của mảng [..., K, N]."""
    z = z - z.max(-2, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-2, keepdims=True)


def blend(W, P):
    return np.einsum('kn,ktn->tn', W, P)


def state_features(P, y, last, context):
    """Trạng thái s[t,i] [T,N,10], chỉ dùng thông tin có trước khi biết y_t."""
    K, T, N = P.shape
    err_prev = np.zeros((K, T, N))
    err_prev[:, 1:] = np.abs(P[:, :-1] - y[None, :-1])
    feats = [np.arcsinh(10 * (P - last[None])),      # 3: hướng/độ lớn thay đổi mà từng nhánh dự báo so với lag_1
             np.arcsinh(10 * P.std(0))[None],        # 1: mức bất đồng giữa các nhánh
             np.arcsinh(100 * err_prev),             # 3: sai số tuyệt đối bước trước của từng nhánh
             context.transpose(2, 0, 1)[:2],         # 2: local_volatility, spike_flag (cache v3)
             np.arcsinh(10 * last)[None]]            # 1: mức lưu lượng hiện tại
    return np.concatenate(feats, 0).transpose(1, 2, 0)


def hedge_memories(P, y, channels=CHANNELS, warmup=None):
    """H [T,M,K,N] = -eta_m * L_m(t). warmup=(P,y) của đoạn liền trước (Val) để khởi động bộ nhớ cho Test."""
    T = P.shape[1]
    if warmup is not None:
        P = np.concatenate([warmup[0], P], 1)
        y = np.concatenate([warmup[1], y], 0)
    K, total, N = P.shape
    etas = np.array([eta for _, eta, _ in channels])[:, None, None]
    H = np.empty((total, len(channels), K, N))
    L = np.zeros((len(channels), K, N))
    for t in range(total):
        H[t] = -etas * L
        e = P[:, t] - y[t]
        for m, (kind, _, decay) in enumerate(channels):
            L[m] = L[m] * decay + (e * e if kind == 'se' else np.abs(e))
    return H[total - T:]


def hedge_predict(P, y, W, eta, decay):
    """Baseline Hedge theo luồng (một kênh SE), khởi tạo từ trọng số v3."""
    H = hedge_memories(P, y, [('se', eta, decay)])[:, 0]
    a = softmax_k(np.log(W + 1e-6)[None] + H)
    return np.einsum('tkn,ktn->tn', a, P)


class LinearPolicy:
    """Policy tuyến tính: beta = 2*sigmoid(U s + b) in (0, 2)^M, delta = V s + d (M*F + M + K*F + K = 66 tham số)."""

    def __init__(self, n_features, n_branches, n_channels, beta_init=0.5):
        self.U = np.zeros((n_channels, n_features))
        self.b = np.full(n_channels, np.log(beta_init / (2 - beta_init)))
        self.V = np.zeros((n_branches, n_features))
        self.d = np.zeros(n_branches)

    def params(self):
        return [self.U, self.b, self.V, self.d]

    def set_params(self, params):
        self.U, self.b, self.V, self.d = [np.array(p, dtype=float) for p in params]

    def logits(self, S, H, prior_logit):
        sig = 1 / (1 + np.exp(-(np.einsum('tnf,mf->tmn', S, self.U) + self.b[None, :, None])))
        delta = np.einsum('tnf,kf->tkn', S, self.V) + self.d[None, :, None]
        return prior_logit[None] + np.einsum('tmn,tmkn->tkn', 2 * sig, H) + delta, sig

    def act(self, S, H, prior_logit):
        z, sig = self.logits(S, H, prior_logit)
        return softmax_k(z), 2 * sig

    def predict(self, S, H, prior_logit, P):
        return (self.act(S, H, prior_logit)[0] * P.transpose(1, 0, 2)).sum(1)


def _backprop(policy, S, H, sig, gz, l2):
    gV = np.einsum('tkn,tnf->kf', gz, S) + 2 * l2 * policy.V
    gd = gz.sum((0, 2))
    gs = np.einsum('tkn,tmkn->tmn', gz, H) * 2 * sig * (1 - sig)
    gU = np.einsum('tmn,tnf->mf', gs, S) + 2 * l2 * policy.U
    return [gU, gs.sum((0, 2)), gV, gd]


def dpg_gradient(policy, S, H, prior_logit, P, y, l2=0.0):
    """Gradient của -reward trung bình theo tham số policy (reward đã biết nên đạo hàm qua hành động là chính xác)."""
    z, sig = policy.logits(S, H, prior_logit)
    a = softmax_k(z)
    Pt = P.transpose(1, 0, 2)
    y_hat = (a * Pt).sum(1)
    r = y_hat - y
    gz = (2 * r / r.size)[:, None] * a * (Pt - y_hat[:, None])
    return float(np.mean(r ** 2)), _backprop(policy, S, H, sig, gz, l2)


def reinforce_gradient(policy, S, H, prior_logit, P, y, rng, sigma=0.3, l2=0.0):
    """REINFORCE với baseline self-critical: nhiễu Gauss trên logit, chỉ dùng giá trị reward của hành động lấy mẫu."""
    z, sig = policy.logits(S, H, prior_logit)
    Pt = P.transpose(1, 0, 2)
    eps = rng.standard_normal(z.shape)
    reward_sample = -(((softmax_k(z + sigma * eps) * Pt).sum(1) - y) ** 2)
    reward_base = -(((softmax_k(z) * Pt).sum(1) - y) ** 2)
    adv = reward_sample - reward_base
    adv = adv / (np.abs(adv).mean() + 1e-12)
    gz = -(adv[:, None] * eps / sigma) / adv.size
    return _backprop(policy, S, H, sig, gz, l2)


class Adam:
    def __init__(self, params, lr):
        self.m = [np.zeros_like(np.asarray(p, float)) for p in params]
        self.v = [np.zeros_like(m) for m in self.m]
        self.t = 0
        self.lr = lr

    def step(self, params, grads):
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            m_hat = self.m[i] / (1 - 0.9 ** self.t)
            v_hat = self.v[i] / (1 - 0.999 ** self.t)
            out.append(p - self.lr * m_hat / (np.sqrt(v_hat) + 1e-12))
        return out


def train_policy(S, H, prior_logit, P, y, holdout, mode='dpg', mask=(1, 1, 1, 1), l2=1e-4, lr=0.02,
                 epochs=300, eval_every=10, seed=0, beta_init=0.5):
    """Huấn luyện full-batch trên phần đầu Val; holdout=(S,H,P,y) phần cuối Val để early stopping."""
    policy = LinearPolicy(S.shape[-1], P.shape[0], H.shape[1], beta_init)
    opt = Adam(policy.params(), lr)
    rng = np.random.default_rng(seed)
    best_loss, best_params, best_epoch = np.inf, [p.copy() for p in policy.params()], 0
    for epoch in range(epochs + 1):
        if epoch % eval_every == 0:
            hold_loss = dpg_gradient(policy, holdout[0], holdout[1], prior_logit, holdout[2], holdout[3])[0]
            if hold_loss < best_loss:
                best_loss, best_params, best_epoch = hold_loss, [p.copy() for p in policy.params()], epoch
        if epoch == epochs:
            break
        if mode == 'dpg':
            grads = dpg_gradient(policy, S, H, prior_logit, P, y, l2)[1]
        else:
            grads = reinforce_gradient(policy, S, H, prior_logit, P, y, rng, l2=l2)
        policy.set_params(opt.step(policy.params(), [g * m for g, m in zip(grads, mask)]))
    policy.set_params(best_params)
    return policy, best_epoch


def safety_switch(pred_static, pred_rl, y, margin=0.01):
    """Tại t dùng RL-Gate khi tổng SE (toàn mạng, bước 0..t-1) của nó < (1-margin) * tổng SE của v3."""
    loss_static = loss_rl = 0.0
    out = np.empty_like(y)
    use_rl = np.zeros(y.shape[0], dtype=bool)
    for t in range(y.shape[0]):
        use_rl[t] = loss_rl < (1 - margin) * loss_static
        out[t] = pred_rl[t] if use_rl[t] else pred_static[t]
        loss_static += ((pred_static[t] - y[t]) ** 2).sum()
        loss_rl += ((pred_rl[t] - y[t]) ** 2).sum()
    return out, use_rl
