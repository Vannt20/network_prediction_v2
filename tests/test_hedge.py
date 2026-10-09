import numpy as np

from Graph_models.online_hedge import hedge_predict, hedge_weights


def _data(K=3, T=50, N=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random((T, N))
    P = y[None] + rng.normal(0, [[[0.05]], [[0.1]], [[0.2]]][:K], (K, T, N))
    w0 = rng.dirichlet(np.ones(K), size=N).T          # [K, N]
    return P, y, w0


def test_eta0_equals_static():
    P, y, w0 = _data()
    pred, _ = hedge_predict(P, y, w0, eta=0.0, beta=0.9)
    np.testing.assert_allclose(pred, np.einsum('kn,ktn->tn', w0, P), atol=1e-12)


def test_alpha1_equals_static():
    P, y, w0 = _data()
    pred, _ = hedge_predict(P, y, w0, eta=100.0, beta=0.9, alpha=1.0)
    np.testing.assert_allclose(pred, np.einsum('kn,ktn->tn', w0, P), atol=1e-12)


def test_causal():
    P, y, w0 = _data()
    w = hedge_weights(P, y, w0, eta=50.0, beta=0.95)
    y2 = y.copy()
    y2[30:] = 99.0                                      # thay đổi nhãn từ bước 30
    w2 = hedge_weights(P, y2, w0, eta=50.0, beta=0.95)
    np.testing.assert_allclose(w[:, :31], w2[:, :31])   # trọng số tại t <= 30 chỉ dùng nhãn đến t-1
    assert not np.allclose(w[:, 31:], w2[:, 31:])


def test_weights_sum_to_one():
    P, y, w0 = _data()
    w = hedge_weights(P, y, w0, eta=1000.0, beta=0.99)
    np.testing.assert_allclose(w.sum(0), 1.0, atol=1e-9)
