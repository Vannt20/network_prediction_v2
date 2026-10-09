"""
P4 - Báo cáo cuối của hướng V4 (spec Mục 6.4.5 và 7).

1. Quy tắc chọn tầng kết hợp theo từng dataset (cố định trên Val, run_0):
   RL-Gate nếu S_rl < min(S_static, S_hedge)·0,99; nếu không Hedge nếu S_hedge < S_static·0,99; nếu không static.
   -> results/gnn_rl/p3_selection_{ds}.json ; mô hình cuối "ST-Adaptive-Ensemble-RL" = cấu hình được chọn.
2. Bảng tổng hợp (mean ± std qua các run, chỉ số chính + phụ, số run tốt hơn v3).
3. Kiểm định: t-test ghép cặp theo run và Diebold-Mariano trên từng run, mô hình cuối so với v3 và Hedge.

Đầu ra: results/gnn_rl/bang_tong_hop_gnn_rl.{csv,xlsx}, kiem_dinh_thong_ke_gnn_rl.csv, ablation_gnn_rl.csv
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.gnn_rl_common import (setup_utf8, parse_list, ALL_DATASETS, RESULTS_V4, LOGS_V4, load_cache,
                                    split_arrays, run_dir, load_json, save_json, jump_mask)
from evaluation.extra_metrics import extra_metrics
from evaluation.stats_tests import paired_ttest

METRICS = ['mse', 'mae', 'mse_in_range', 'mse_trim_jump', 'top1_se_share', 'rise_se_share']
MAIN_MODELS = [
    ('persistence', 'Persistence'), ('branch_xgboost', 'XGBoost'), ('branch_lightgbm_res', 'LightGBM-Residual'),
    ('branch_stwaveformer', 'ST-WaveFormer'), ('branch_odgraphformer', 'OD-GraphFormer'),
    ('v3_static', 'ST-Adaptive-Ensemble v3 (static)'), ('static_k4', 'Static K=4'), ('static_k3', 'Static K=3 (nhánh RL)'),
    ('hedge', 'Hedge'), ('hedge_floor', 'Hedge + sàn'), ('context_gate', 'Cổng MLP ngữ cảnh v2'),
    ('rl_sup', 'Gate giám sát (γ=0)'), ('rl_sac', 'RL-Gate (SAC)'), ('final', 'ST-Adaptive-Ensemble-RL'),
]
ABLATIONS = [
    ('odgraphformer_no_residual', 'OD-GraphFormer: bỏ mục tiêu phần dư'),
    ('odgraphformer_no_route', 'OD-GraphFormer: bỏ A_route'),
    ('odgraphformer_no_spatial_attn', 'OD-GraphFormer: bỏ attention giữa luồng'),
    ('rl_sac_no_gnn', 'RL-Gate: bỏ GNN'), ('rl_sac_no_floor', 'RL-Gate: bỏ ràng buộc sàn'),
    ('rl_sac_no_oof', 'RL-Gate: không dùng OOF'), ('rl_sac_no_temporal', 'RL-Gate: bỏ Transformer lịch sử'),
    ('rl_sac_no_kl', 'RL-Gate: bỏ phạt KL'), ('rl_sac_no_sigma', 'RL-Gate: bỏ σ̂'),
]


def dm_from_loss(d):
    """Diebold-Mariano từ chuỗi chênh lệch tổn thất theo bước (giống evaluation.stats_tests.diebold_mariano)."""
    d = np.asarray(d, dtype=np.float64)
    T = len(d)
    dc = d - d.mean()
    h = int(np.floor(T ** (1.0 / 3.0)))
    lrv = np.dot(dc, dc) / T
    for lag in range(1, h + 1):
        lrv += 2.0 * (1.0 - lag / (h + 1.0)) * np.dot(dc[lag:], dc[:-lag]) / T
    if lrv <= 0:
        return 0.0, 1.0
    dm = d.mean() / np.sqrt(lrv / T)
    return float(dm), float(2.0 * (1.0 - stats.norm.cdf(abs(dm))))


def _record(tag, ds, r, pred, y, last, keep):
    m = extra_metrics(pred, y, last, keep_mask=keep)
    se_t = ((np.asarray(pred, np.float64) - y) ** 2).mean(axis=1)
    return {'model': tag, 'dataset': ds, 'run': r, **{k: m[k] for k in METRICS}}, se_t


def collect(ds, run_ids):
    """Trả về (DataFrame chỉ số theo run, dict se_t[(model, run)])."""
    rows, se = [], {}
    for r in run_ids:
        c = load_cache(ds, r, 'v4')
        P, y, last = split_arrays(c, 'test')
        y64, l64 = y.astype(np.float64), last.astype(np.float64)
        keep = jump_mask(y64, l64)
        rec, s = _record('persistence', ds, r, last, y64, l64, keep)
        rows.append(rec)
        se[('persistence', r)] = s
        for k, b in enumerate(c['branches']):
            rec, s = _record(f'branch_{b}', ds, r, P[k], y64, l64, keep)
            rows.append(rec)
            se[(f'branch_{b}', r)] = s
        # Nhánh ablation của OD-GraphFormer: tính lại từ dự báo đã lưu
        for f in glob.glob(os.path.join(LOGS_V4, f'odgraphformer_*_data_{ds}_seq_*', f'run_{r}', 'y_pred_data.npy')):
            tag = os.path.basename(os.path.dirname(os.path.dirname(f))).split('_data_')[0]
            rec, s = _record(tag, ds, r, np.load(f), y64, l64, keep)
            rows.append(rec)
            se[(tag, r)] = s
        # Tầng kết hợp: chỉ số đã ghi lúc chạy (cùng mặt nạ dữ liệu) + se_t
        for f in glob.glob(os.path.join(LOGS_V4, f'*_data_{ds}', f'run_{r}', 'test_metrics.csv')):
            d = os.path.dirname(f)
            tag = os.path.basename(os.path.dirname(d)).split('_data_')[0]
            if tag.startswith('branch_') or tag == 'persistence' or not os.path.exists(os.path.join(d, 'se_t.npy')):
                continue
            m = pd.read_csv(f).iloc[0]
            rows.append({'model': tag, 'dataset': ds, 'run': r, **{k: float(m[k]) for k in METRICS}})
            se[(tag, r)] = np.load(os.path.join(d, 'se_t.npy'))
    return pd.DataFrame(rows), se


def select_final(ds):
    """Quy tắc chọn tầng kết hợp (spec 6.4.5), dùng điểm CV Val của run_0."""
    cv_f = os.path.join(run_dir('cv', ds, 0, seq=False), 'cv_scores.json')
    grid_f = os.path.join(run_dir('rl_sac', ds, None, seq=False), 'grid.json')
    if not os.path.exists(cv_f):
        return None
    cv = load_json(cv_f)['cv_mse']
    s_static, s_hedge = cv['static'], cv['hedge']
    s_rl = load_json(grid_f)['cv_mse'] if os.path.exists(grid_f) else float('inf')
    if s_rl < min(s_static, s_hedge) * 0.99:
        choice = 'rl_sac'
    elif s_hedge < s_static * 0.99:
        choice = 'hedge'
    else:
        choice = 'static_k3'
    out = {'dataset': ds, 'S_static': s_static, 'S_hedge': s_hedge, 'S_hedge_floor': cv.get('hedge_floor'),
           'S_rl': s_rl, 'choice': choice, 'rule': 'RL nếu S_rl < 0,99·min(S_static,S_hedge); Hedge nếu S_hedge < 0,99·S_static'}
    save_json(out, os.path.join(RESULTS_V4, f'p3_selection_{ds}.json'))
    return out


def summarize(df, names):
    g = df.groupby(['dataset', 'model'])
    out = g.agg(runs=('run', 'nunique'), **{f'{k}_mean': (k, 'mean') for k in METRICS},
                **{f'{k}_std': (k, 'std') for k in METRICS}).reset_index()
    for k in ('mse', 'mae', 'mse_in_range', 'mse_trim_jump'):
        out[f'{k}_mean'] *= 1e3
        out[f'{k}_std'] *= 1e3
    ref = df[df['model'] == 'v3_static'].set_index(['dataset', 'run'])['mse']
    nb = df.join(ref.rename('mse_v3'), on=['dataset', 'run'])
    nb = nb.assign(better=nb['mse'] < nb['mse_v3']).groupby(['dataset', 'model'])['better'].sum()
    out = out.join(nb.rename('n_better_than_v3'), on=['dataset', 'model'])
    out['name'] = out['model'].map(dict(names)).fillna(out['model'])
    order = {m: i for i, (m, _) in enumerate(names)}
    out['_o'] = out['model'].map(order).fillna(999)
    return out.sort_values(['dataset', '_o']).drop(columns='_o')


def tests(df, se, ds, run_ids, target='final', refs=('v3_static', 'hedge')):
    rows = []
    for ref in refs:
        a = df[(df['dataset'] == ds) & (df['model'] == target)].set_index('run')['mse']
        b = df[(df['dataset'] == ds) & (df['model'] == ref)].set_index('run')['mse']
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 2:
            continue
        t, p = paired_ttest(a[common].values, b[common].values)
        # Chỉ kiểm định DM khi hai chuỗi tổn thất cùng độ dài (ở chế độ quick_check một số tầng kết hợp bị cắt ngắn)
        dms = [dm_from_loss(se[(target, r)] - se[(ref, r)]) for r in common
               if (target, r) in se and (ref, r) in se and len(se[(target, r)]) == len(se[(ref, r)])]
        rows.append({'dataset': ds, 'model': target, 'vs': ref, 'runs': len(common),
                     'mse_model_e3': float(a[common].mean() * 1e3), 'mse_ref_e3': float(b[common].mean() * 1e3),
                     'rel_change_pct': float((a[common].mean() / b[common].mean() - 1) * 100),
                     't_stat': t, 't_p': p, 'n_runs_better': int((a[common] < b[common]).sum()),
                     'dm_mean': float(np.mean([d for d, _ in dms])) if dms else np.nan,
                     'dm_n_sig_better': int(sum(d < 0 and pv < 0.05 for d, pv in dms)),
                     'dm_n_sig_worse': int(sum(d > 0 and pv < 0.05 for d, pv in dms))})
    return rows


def report(datasets, run_ids):
    all_df, all_tests, sel = [], [], {}
    for ds in datasets:
        df, se = collect(ds, run_ids)
        s = select_final(ds)
        if s is not None:
            sel[ds] = s
            ch = df[df['model'] == s['choice']].copy()
            ch['model'] = 'final'
            df = pd.concat([df, ch], ignore_index=True)
            for r in run_ids:
                if (s['choice'], r) in se:
                    se[('final', r)] = se[(s['choice'], r)]
            print(f"[{ds.upper()}] chọn tầng kết hợp: {s['choice']} | CV e-3 static={s['S_static']*1e3:.4f} "
                  f"hedge={s['S_hedge']*1e3:.4f} rl={s['S_rl']*1e3:.4f}", flush=True)
        all_df.append(df)
        if s is not None:
            all_tests += tests(df, se, ds, run_ids)
        all_tests += tests(df, se, ds, run_ids, target='rl_sac', refs=('v3_static', 'hedge', 'rl_sup'))
    df = pd.concat(all_df, ignore_index=True)
    os.makedirs(RESULTS_V4, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_V4, 'chi_so_theo_run_gnn_rl.csv'), index=False)
    main_df = summarize(df[df['model'].isin([m for m, _ in MAIN_MODELS])], MAIN_MODELS)
    abl_models = [m for m, _ in ABLATIONS] + ['branch_odgraphformer', 'rl_sac']
    abl_df = summarize(df[df['model'].isin(abl_models)], ABLATIONS + [('branch_odgraphformer', 'OD-GraphFormer đầy đủ'),
                                                                     ('rl_sac', 'RL-Gate đầy đủ')])
    main_df.to_csv(os.path.join(RESULTS_V4, 'bang_tong_hop_gnn_rl.csv'), index=False, encoding='utf-8-sig')
    abl_df.to_csv(os.path.join(RESULTS_V4, 'ablation_gnn_rl.csv'), index=False, encoding='utf-8-sig')
    tdf = pd.DataFrame(all_tests)
    tdf.to_csv(os.path.join(RESULTS_V4, 'kiem_dinh_thong_ke_gnn_rl.csv'), index=False)
    try:
        with pd.ExcelWriter(os.path.join(RESULTS_V4, 'bang_tong_hop_gnn_rl.xlsx')) as w:
            main_df.to_excel(w, sheet_name='tong_hop', index=False)
            abl_df.to_excel(w, sheet_name='ablation', index=False)
            tdf.to_excel(w, sheet_name='kiem_dinh', index=False)
    except Exception as e:
        print(f"[CẢNH BÁO] không ghi được xlsx: {e}")
    pd.set_option('display.width', 220)
    cols = ['dataset', 'name', 'runs', 'mse_mean', 'mse_std', 'mse_trim_jump_mean', 'n_better_than_v3']
    print("\nBẢNG TỔNG HỢP (MSE x10^-3)")
    print(main_df[cols].round(4).to_string(index=False))
    if len(abl_df):
        print("\nABLATION")
        print(abl_df[cols].round(4).to_string(index=False))
    if len(tdf):
        print("\nKIỂM ĐỊNH")
        print(tdf.round(4).to_string(index=False))
    return main_df, abl_df, tdf


if __name__ == '__main__':
    setup_utf8()
    from baselines_ml.run_ml_baselines import parse_run_ids
    ap = argparse.ArgumentParser(description="P4: báo cáo cuối hướng V4")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_ids', default='0-9')
    a = ap.parse_args()
    report(parse_list(a.datasets, ALL_DATASETS), parse_run_ids(a.run_ids))
