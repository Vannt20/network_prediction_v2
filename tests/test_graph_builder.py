import numpy as np

from features.graph_builder import routing_matrix, sym_norm, _connected


def test_routing_line_graph():
    # 0 - 1 - 2, luồng (0,2) đi qua cả 2 liên kết, luồng (1,1) tự thân không đi qua liên kết nào
    E = [(0, 1), (1, 2)]
    pairs = [(0, 2), (1, 1), (0, 1)]
    R, dist, n_ecmp = routing_matrix(3, E, pairs)
    assert R.shape == (2, 3)
    np.testing.assert_allclose(R[:, 0], [1, 1])
    np.testing.assert_allclose(R[:, 1], [0, 0])
    np.testing.assert_allclose(R[:, 2], [1, 0])
    assert n_ecmp == 0 and dist[0, 2] == 2


def test_routing_ecmp_square():
    # Hình vuông 0-1-3, 0-2-3: luồng (0,3) có 2 đường ngắn nhất -> mỗi liên kết mang 0,5
    E = [(0, 1), (0, 2), (1, 3), (2, 3)]
    R, _, n_ecmp = routing_matrix(4, E, [(0, 3)])
    np.testing.assert_allclose(R[:, 0], [0.5, 0.5, 0.5, 0.5])
    assert n_ecmp == 1


def test_sym_norm_symmetric():
    A = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=float)
    S = sym_norm(A)
    np.testing.assert_allclose(S, S.T, atol=1e-6)
    assert (np.diag(S) > 0).all()


def test_connected():
    assert _connected(3, {(0, 1), (1, 2)})
    assert not _connected(3, {(0, 1)})
