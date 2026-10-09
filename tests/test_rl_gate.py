import numpy as np
import torch

from Graph_models.rl_gate.features import SplitData
from Graph_models.rl_gate.env import GateTask
from Graph_models.rl_gate.sac import SAC, build_actor, DEFAULTS
from Graph_models.rl_gate.supervised import train_supervised

K, T, N = 3, 60, 5


def _arrays(seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random((T, N)).astype(np.float32)
    last = np.concatenate([rng.random((1, N)), y[:-1]]).astype(np.float32)
    P = (y[None] + rng.normal(0, 0.1, (K, T, N))).astype(np.float32)
    ctx = rng.random((T, N, 4)).astype(np.float32)
    sigma = rng.random((T, N)).astype(np.float32) + 0.1
    w = rng.dirichlet(np.ones(K), size=N).astype(np.float32)    # [N, K]
    return P, y, last, ctx, sigma, w


def _actor(data, perturb=True):
    cfg = {**DEFAULTS, 'd': 16, 'd_t': 8, 'H': 4}
    graphs = [np.eye(N, dtype=np.float32)]
    actor = build_actor(K, data.F, graphs, cfg)
    actor.enc.set_norm(data.norm_stats())
    if perturb:
        torch.manual_seed(1)
        with torch.no_grad():
            actor.mu.weight.normal_(0, 0.5)
    return actor.eval()


def _data(P, y, last, ctx, sigma):
    return SplitData(P, y, last, ctx, sigma, np.linspace(0, 1, T), np.zeros(T), H=4)


def test_init_policy_equals_static():
    P, y, last, ctx, sigma, w = _arrays()
    data = _data(P, y, last, ctx, sigma)
    task = GateTask(data, w, ext_idx=[2])
    pred, A = task.rollout(_actor(data, perturb=False))
    np.testing.assert_allclose(A, np.broadcast_to(w, A.shape), atol=1e-6)
    np.testing.assert_allclose(pred, (w[None] * np.moveaxis(P, 0, -1)).sum(-1), atol=1e-5)


def test_rollout_causal():
    """Thay nhãn từ bước t0 trở đi thì hành động tại các bước <= t0 không đổi."""
    P, y, last, ctx, sigma, w = _arrays()
    t0 = 30
    y2, last2 = y.copy(), last.copy()
    y2[t0:] += 5.0
    last2[t0 + 1:] = y2[t0:-1]                     # last_t = y_{t-1}
    d1, d2 = _data(P, y, last, ctx, sigma), _data(P, y2, last2, ctx, sigma)
    actor = _actor(d1)
    actor2 = actor                                 # cùng tham số và cùng thống kê chuẩn hóa
    _, A1 = GateTask(d1, w, [2]).rollout(actor)
    _, A2 = GateTask(d2, w, [2]).rollout(actor2)
    np.testing.assert_allclose(A1[:t0 + 1], A2[:t0 + 1], atol=1e-6)
    assert not np.allclose(A1[t0 + 1:], A2[t0 + 1:], atol=1e-4)
    _, B1 = GateTask(d1, w, [2]).rollout_static_prev(actor)
    _, B2 = GateTask(d2, w, [2]).rollout_static_prev(actor2)
    np.testing.assert_allclose(B1[:t0 + 1], B2[:t0 + 1], atol=1e-6)


def test_reward_zero_for_static_without_penalty():
    P, y, last, ctx, sigma, w = _arrays()
    task = GateTask(_data(P, y, last, ctx, sigma), w, [2], lam_sw=0.1, lam_kl=0.01, floor=0.0)
    ts = torch.arange(5)
    a = task.w[None].expand(5, N, K)
    r = task.reward(ts, a, a)
    torch.testing.assert_close(r, torch.zeros_like(r), atol=1e-6, rtol=0)


def test_sac_and_supervised_run():
    P, y, last, ctx, sigma, w = _arrays()
    data = _data(P, y, last, ctx, sigma)
    task = GateTask(data, w, [2])
    ag = SAC(K, data.F, [np.eye(N, dtype=np.float32)],
             {'d': 16, 'd_t': 8, 'H': 4, 'n_envs': 2, 'batch_t': 4, 'ep_len': 10}, 'cpu')
    ag.set_norm(data.norm_stats())
    hist = ag.train(task, 6, verbose=False, log_every=3)
    assert len(hist) >= 1
    actor = _actor(data, perturb=False)
    train_supervised(actor, task, n_updates=5, eval_every=5, verbose=False)
    pred, A = task.rollout_static_prev(actor)
    assert pred.shape == (T, N) and np.allclose(A.sum(-1), 1.0, atol=1e-5)
