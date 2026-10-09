import os
import numpy as np
import pytest

from training.gnn_rl_common import jump_mask, oof_ranges, ROOT
from evaluation.extra_metrics import extra_metrics


def test_jump_mask_shared_and_size():
    rng = np.random.default_rng(0)
    y = rng.random((100, 20))
    last = rng.random((100, 20))
    keep = jump_mask(y, last, q=0.01)
    assert (~keep).sum() == 20                            # bỏ đúng 1% điểm
    j = np.abs(y - last)
    assert j[~keep].min() >= j[keep].max()
    m1 = extra_metrics(y + 0.1, y, last, keep_mask=keep)
    m2 = extra_metrics(y - 0.1, y, last)
    assert m1['mse_trim_jump'] == pytest.approx(0.01) and m2['mse_trim_jump'] == pytest.approx(0.01)


def test_extra_metrics_shares():
    y = np.zeros((10, 10))
    last = np.zeros((10, 10))
    pred = np.zeros((10, 10))
    pred[0, 0] = 1.0                                       # một điểm sai, nằm ở pha giảm (y <= last)
    m = extra_metrics(pred, y, last)
    assert m['top1_se_share'] == pytest.approx(1.0)
    assert m['rise_se_share'] == pytest.approx(0.0)


def test_oof_ranges():
    fit, es, oof = oof_ranges(1000, 24, 0.6)
    assert fit[0] == 24 and oof[0] == 600 and oof[-1] == 999
    assert fit[-1] + 1 == es[0] and es[-1] + 1 == oof[0]
    assert len(es) == 600 - 24 - int((600 - 24) * 0.9)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'cache', 'v4', 'sdn_run_0.pt')), reason="chưa có cache v4")
def test_cache_v4_matches_v3():
    from training.gnn_rl_common import torch_load
    a = torch_load(os.path.join(ROOT, 'cache', 'v3', 'sdn_run_0.pt'))
    b = torch_load(os.path.join(ROOT, 'cache', 'v4', 'sdn_run_0.pt'))
    for s in ('val', 'test'):
        assert b['branches'][:3] == a['branches']
        assert (a[s]['P'] == b[s]['P'][:3]).all()
        assert (a[s]['y'] == b[s]['y']).all() and (a[s]['context'] == b[s]['context']).all()
        for k in ('sigma', 'tod', 'dow'):
            assert k in b[s] and len(b[s][k]) == len(b[s]['y'])
