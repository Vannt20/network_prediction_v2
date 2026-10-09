import numpy as np
import torch

from Graph_models.online_hedge import select_prior, prior_weights, select_hedge
from evaluation.report_gnn_rl import choose_combiner
from training.train_od_graphformer import choose_anchor


def test_choose_combiner_requires_both_blocks():
    st = [1.0, 1.0]
    # Thấp hơn trung bình nhưng thua ở một khối -> không được chọn
    assert choose_combiner({'static': st, 'hedge': [0.5, 1.2]}) == 'static'
    # Thắng cả hai khối nhưng không đủ 1% -> không được chọn
    assert choose_combiner({'static': st, 'hedge': [0.995, 0.996]}) == 'static'
    # Hai tầng đủ điều kiện -> chọn tầng có trung bình thấp hơn
    assert choose_combiner({'static': st, 'hedge': [0.9, 0.9], 'rl_sac': [0.8, 0.95]}) == 'rl_sac'


def test_select_prior_can_fall_back_to_static():
    rng = np.random.default_rng(0)
    T, N = 80, 4
    y = rng.random((T, N))
    # Nhánh 0 tốt ổn định, nhánh 1 tốt ở nửa đầu rồi tệ hẳn -> static (đã đúng) là lựa chọn an toàn
    P = np.stack([y + rng.normal(0, 0.01, (T, N)), y + np.where(np.arange(T)[:, None] < 40, 0.0, 0.5)])
    w0 = np.tile(np.array([[1.0], [1e-6]]), (1, N))
    p = select_prior(P, y, w0)
    assert set(p) == {'eta', 'beta', 'alpha'}
    w = prior_weights(P, y, w0, p)
    assert w.shape == (T, N, 2) and np.allclose(w.sum(-1), 1.0, atol=1e-6)


def test_select_hedge_mse_unchanged_by_criterion_field():
    rng = np.random.default_rng(1)
    y = rng.random((50, 3))
    P = y[None] + rng.normal(0, [[[0.05]], [[0.2]]], (2, 50, 3))
    w0 = np.full((2, 3), 0.5)
    h, f, grid = select_hedge(P, y, w0)
    assert h['score'] == h['mse'] and min(g['mse'] for g in grid if g['alpha'] == 0) == h['mse']


def test_choose_anchor():
    T, N, L = 200, 3, 10
    rng = np.random.default_rng(0)
    level = rng.random(N)
    noisy = level + rng.normal(0, 0.2, (T, N))            # nhiễu quanh một mức -> neo trung bình tốt hơn
    smooth = np.cumsum(rng.normal(0, 0.001, (T, N)), 0) + level   # bước ngẫu nhiên mịn -> neo x_{t-1} tốt hơn
    for x, want in ((noisy, 'mean'), (smooth, 'last')):
        comb = torch.from_numpy(np.stack([x, np.zeros_like(x), np.zeros_like(x)], -1))
        assert choose_anchor(comb, L)['auto'] == want
