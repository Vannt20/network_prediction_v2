"""
P0 - Tái lập số của ST-Adaptive-Ensemble v3 từ cache/v3 (spec Mục 2.4).

Với mỗi (dataset, run): fit lại RobustPerFlowStacking trên Val như train_stacking.py, dự báo Test,
so MSE với results/results_STAdaptiveEnsemble_data_{ds}.csv (lệch cho phép 0,1% tương đối).
Đồng thời ghi chỉ số phụ của v3, ba nhánh và Persistence vào logs/gnn_rl/ để làm mốc.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.gnn_rl_common import (setup_utf8, parse_list, ALL_DATASETS, RESULTS_V4, load_cache, split_arrays,
                                    save_combiner_result, save_json, load_json, run_dir, upsert_csv)
from Graph_models.robust_stacking import RobustPerFlowStacking
from baselines_ml.run_ml_baselines import parse_run_ids

TOL = 1e-3


def reproduce(datasets, run_ids):
    rows, bad = [], []
    for ds in datasets:
        ref_csv = os.path.join(parent_dir, 'results', f'results_STAdaptiveEnsemble_data_{ds}.csv')
        ref = pd.read_csv(ref_csv).set_index('run') if os.path.exists(ref_csv) else None
        for r in run_ids:
            c = load_cache(ds, r, 'v3')
            branches = c['branches']
            Pv, yv, _ = split_arrays(c, 'val')
            Pt, yt, lt = split_arrays(c, 'test')
            st = RobustPerFlowStacking(branches).fit(Pv, yv)
            pred = st.predict(Pt)
            m = save_combiner_result('v3_static', ds, r, pred, yt, lt,
                                     config={'branches': branches, 'selected': list(st.selected)})
            np.save(os.path.join(run_dir('v3_static', ds, r, seq=False), 'weights_static.npy'),
                    st.per_flow_weights(yt.shape[1]).astype(np.float32))
            for k, b in enumerate(branches):
                save_combiner_result(f'branch_{b}', ds, r, Pt[k], yt, lt)
            save_combiner_result('persistence', ds, r, lt, yt, lt)
            ref_mse = float(ref.loc[r, 'mse']) if ref is not None and r in ref.index else float('nan')
            rel = abs(m['mse'] - ref_mse) / ref_mse if ref_mse == ref_mse else float('nan')
            ok = rel <= TOL if rel == rel else None
            if ok is False:
                bad.append((ds, r, rel))
            rows.append({'dataset': ds, 'run': r, 'mse': m['mse'], 'mse_ref': ref_mse, 'rel_diff': rel, 'ok': ok,
                         'selected': '-'.join(st.selected)})
            print(f"  [{ds.upper()} run_{r}] v3 MSE tái lập={m['mse']*1e3:.4f}e-3 | CSV={ref_mse*1e3:.4f}e-3 | "
                  f"lệch {rel:.1e} | {st.selected}", flush=True)
        sub = pd.DataFrame([x for x in rows if x['dataset'] == ds])
        print(f"[{ds.upper()}] TB {len(sub)} run: {sub['mse'].mean()*1e3:.3f}e-3 (CSV {sub['mse_ref'].mean()*1e3:.3f}e-3)")
    df = upsert_csv(pd.DataFrame(rows), os.path.join(RESULTS_V4, 'p0_reproduce_v3.csv'))
    bad_all = [[r['dataset'], int(r['run']), float(r['rel_diff'])] for _, r in df.iterrows() if r['ok'] is False
               or str(r['ok']) == 'False']
    save_json({'tolerance': TOL, 'n_bad': len(bad_all), 'bad': bad_all}, os.path.join(RESULTS_V4, 'p0_reproduce_v3.json'))
    if bad:
        print(f"[CẢNH BÁO] {len(bad)} run lệch hơn {TOL:.0e}: {bad}")
    return df


if __name__ == '__main__':
    setup_utf8()
    ap = argparse.ArgumentParser(description="P0: tái lập ST-Adaptive-Ensemble v3 từ cache/v3")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_ids', default='0-9')
    a = ap.parse_args()
    reproduce(parse_list(a.datasets, ALL_DATASETS), parse_run_ids(a.run_ids))
