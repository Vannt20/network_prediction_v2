import numpy as np
import torch

from Graph_models.od_graphformer import ODGraphFormer


def _model(N=6, L=12, residual=True, **kw):
    nodes = 3
    src = [i // nodes for i in range(N)]
    dst = [i % nodes for i in range(N)]
    A = np.eye(N, dtype=np.float32)
    return ODGraphFormer(num_flows=N, seq_len=L, src_idx=src, dst_idx=dst, num_nodes=nodes,
                         graphs={'route': A, 'od': A}, steps_per_day=288, d_model=16, n_temporal=1, n_spatial=1,
                         nhead=2, patch_len=4, residual=residual, **kw)


def _x(B=3, L=12, N=6):
    torch.manual_seed(0)
    x = torch.rand(B, L, N, 3)
    x[..., 1] = 0.5
    x[..., 2] = 3 / 7
    return x


def test_shapes_and_persistence_at_init():
    m = _model().eval()
    x = _x()
    y, lv = m(x)
    assert y.shape == (3, 6) and lv.shape == (3, 6)
    # head Δ khởi tạo 0 -> ŷ = x_{t-1} (Persistence)
    torch.testing.assert_close(y, x[:, -1, :, 0])


def test_backward_and_ablation_flags():
    for kw in ({}, {'spatial_attn': False}, {'use_identity': False}, {'use_adp': False}):
        m = _model(**kw)
        y, lv = m(_x())
        (y.sum() + lv.sum()).backward()
    m = _model(residual=False)
    y, _ = m(_x())
    assert torch.isfinite(y).all()


def test_no_graphs():
    m = ODGraphFormer(num_flows=4, seq_len=12, src_idx=[0, 0, 1, 1], dst_idx=[0, 1, 0, 1], num_nodes=2, graphs={},
                      d_model=16, n_temporal=1, n_spatial=1, nhead=2, patch_len=4, use_adp=False)
    y, _ = m(_x(N=4))
    assert y.shape == (3, 4)
