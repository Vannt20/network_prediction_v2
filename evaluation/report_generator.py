import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import re
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

from evaluation.stats_tests import paired_ttest

ENSEMBLE_KEY = 'stadaptiveensemble'
BASE_NAMES = {
    'stadaptiveensemble': 'ST-Adaptive-Ensemble v3',
    'stwaveformer': 'ST-WaveFormer',
    'xgboost': 'XGBoost',
    'lightgbm_res': 'LightGBM-Residual',
    'lightgbm': 'LightGBM',
    'catboost': 'CatBoost',
    'extra_trees': 'Extra Trees',
    'bigru': 'BiGRU',
    'gwn': 'GWN',
    'dcrnn': 'DCRNN'
}


def _champion():
    import json
    f = os.path.join(parent_dir, 'results', 'champion_ml_model.json')
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            return json.load(fh).get('champion_model', 'xgboost')
    except Exception:
        return 'xgboost'


def model_label(mk, champion):
    base = BASE_NAMES.get(mk, mk)
    if mk == ENSEMBLE_KEY:
        return f"{base} (đề xuất)"
    if mk == 'stwaveformer':
        return f"{base} (nhánh học sâu)"
    if mk == champion:
        return f"{base} (nhánh học máy 1, champion)"
    if mk == 'lightgbm_res':
        return f"{base} (nhánh học máy 2)"
    return f"{base} (đối chứng)"


def model_order(champion):
    return [ENSEMBLE_KEY, 'stwaveformer', champion, 'lightgbm_res', 'xgboost', 'lightgbm', 'catboost']


DS_ORDER = {'SDN': 1, 'GEANT': 2, 'ABILENE': 3}


def calculate_ci95(values):
    n = len(values)
    if n < 2:
        return 0.0
    return stats.sem(values) * stats.t.ppf(0.975, n - 1)


LOG_DIR_TO_KEY = {'st_adaptive_ensemble': ENSEMBLE_KEY}


def load_results(logs_dir):
    """
    Đọc chỉ số theo run trực tiếp từ logs/<model>_data_<ds>_{seq_<L>|shared}/run_<r>/test_metrics.csv
    (nguồn gốc duy nhất; không phụ thuộc các file results_*.csv tổng hợp).
    Run DL bản cũ (thiếu cột seed) bị loại.
    Trả về {(DATASET, model_key): DataFrame theo run}.
    """
    from run_experiments import check_existing_run, expected_test_windows
    from features.feature_store import DATASET_CONFIGS
    n_win_cache = {}

    def n_win(ds):
        if ds not in n_win_cache:
            n_win_cache[ds] = expected_test_windows(ds, DATASET_CONFIGS[ds]['seq_len'])
        return n_win_cache[ds]

    out = {}
    pat = re.compile(r"^(?P<model>.+)_data_(?P<ds>[A-Za-z0-9]+)_(?P<kind>seq_\d+|shared)$", re.IGNORECASE)
    for mdir in glob.glob(os.path.join(logs_dir, '*_data_*')):
        m = pat.match(os.path.basename(mdir))
        if not m:
            continue
        key = LOG_DIR_TO_KEY.get(m.group('model').lower(), m.group('model').lower())
        ds_key = m.group('ds').lower()
        rows = []
        for rdir in glob.glob(os.path.join(mdir, 'run_*')):
            f = os.path.join(rdir, 'test_metrics.csv')
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            if d.empty or 'mse' not in d.columns:
                continue
            row = d.iloc[0].to_dict()
            # Loại log không đồng bộ với dữ liệu hiện tại
            if key == 'stwaveformer' and not check_existing_run(rdir, n_win(ds_key))[0]:
                print(f"[LOẠI] {rdir}: log DL không hợp lệ")
                continue
            if m.group('kind') == 'shared' and ds_key in DATASET_CONFIGS:
                yf = os.path.join(rdir, 'y_real_data.npy')
                if os.path.exists(yf) and np.load(yf, mmap_mode='r').size != n_win(ds_key) * DATASET_CONFIGS[ds_key]['flows']:
                    print(f"[LOẠI] {rdir}: số mẫu test khác dữ liệu hiện tại")
                    continue
            row['run'] = int(os.path.basename(rdir).split('_')[1])
            rows.append(row)
        if rows:
            out[(m.group('ds').upper(), key)] = pd.DataFrame(rows).sort_values('run').reset_index(drop=True)
    return out


def generate_thesis_report(results_dir=None, run_ids=None, logs_dir=None):
    """
    Bảng tổng hợp cho luận văn. Mọi mô hình chỉ được tính trên CÙNG tập run
    (mặc định: các run đã có kết quả ensemble), bảo đảm so sánh công bằng.
    """
    results_dir = results_dir or os.path.join(parent_dir, 'results')
    logs_dir = logs_dir or os.path.join(parent_dir, 'logs')
    os.makedirs(results_dir, exist_ok=True)
    res = load_results(logs_dir)
    champion = _champion()
    order = model_order(champion)
    records, tests = [], []

    for ds in sorted({k[0] for k in res}, key=lambda d: DS_ORDER.get(d, 99)):
        ens = res.get((ds, ENSEMBLE_KEY))
        if ens is None:
            print(f"[INFO] {ds}: chưa có kết quả ensemble -> bỏ qua dataset này trong bảng tổng hợp")
            continue
        runs = sorted(ens['run'].astype(int).unique()) if run_ids is None else run_ids
        models = sorted([k[1] for k in res if k[0] == ds],
                        key=lambda m: order.index(m) if m in order else 99)
        for mk in models:
            df = res[(ds, mk)]
            if runs is not None:
                df = df[df['run'].astype(int).isin(runs)]
            if df.empty:
                continue
            missing = sorted(set(runs) - set(df['run'].astype(int)))
            if missing:
                print(f"[CẢNH BÁO] {ds} {mk}: thiếu run {missing} so với ensemble -> số run không đồng nhất")
            mse, mae = df['mse'].values * 1e3, df['mae'].values * 1e3
            rmse = df['rmse'].values if 'rmse' in df else np.sqrt(df['mse'].values)
            if mk == ENSEMBLE_KEY and 'mse_in_range' in df:
                mir = df['mse_in_range'].values * 1e3
            elif ens is not None and f'branch_mse_in_range_{mk}' in ens:
                e = ens[ens['run'].isin(df['run'])]
                mir = e[f'branch_mse_in_range_{mk}'].values * 1e3
            else:
                mir = np.array([])
            t_ms = df['inference_time_ms'].values if 'inference_time_ms' in df else np.array([np.nan])
            ci = calculate_ci95(mse)
            records.append({
                'Dataset': ds,
                'Model': model_label(mk, champion),
                'Runs': len(df),
                'MSE (x10^-3)': f"{mse.mean():.3f} ± {mse.std(ddof=1) if len(mse) > 1 else 0:.3f}",
                'MSE 95% CI': f"[{mse.mean() - ci:.3f}, {mse.mean() + ci:.3f}]",
                'MAE (x10^-3)': f"{mae.mean():.3f} ± {mae.std(ddof=1) if len(mae) > 1 else 0:.3f}",
                'RMSE': f"{np.mean(rmse):.4f}",
                'MSE trong biên Train (x10^-3)': f"{mir.mean():.3f}" if len(mir) else '',
                'Inference Time (ms/lô 64)': f"{np.nanmean(t_ms):.2f}",
                '_mse': mse.mean(),
            })

            # Kiểm định: ensemble so với từng mô hình khác trên cùng các run
            if ens is not None and mk != ENSEMBLE_KEY:
                e = ens.set_index(ens['run'].astype(int))
                d = df.set_index(df['run'].astype(int))
                common = sorted(set(e.index) & set(d.index))
                t, p = paired_ttest(e.loc[common, 'mse'].values, d.loc[common, 'mse'].values)
                row = {'Dataset': ds, 'So sánh': f"Ensemble vs {model_label(mk, champion)}", 'Runs': len(common),
                       'ΔMSE TB (x10^-3)': float((e.loc[common, 'mse'] - d.loc[common, 'mse']).mean() * 1e3),
                       't (ghép cặp)': t, 'p (t-test)': p}
                if f'dm_stat_vs_{mk}' in e.columns:
                    dm = e.loc[common, f'dm_stat_vs_{mk}'].values
                    pv = e.loc[common, f'dm_p_vs_{mk}'].values
                    row['DM stat TB'] = float(np.mean(dm))
                    row['Số run DM<0 & p<0.05'] = f"{int(np.sum((dm < 0) & (pv < 0.05)))}/{len(common)}"
                tests.append(row)

    summary = pd.DataFrame(records)
    if summary.empty:
        print("Không có kết quả nào để xuất báo cáo.")
        return summary
    summary['Tốt nhất'] = ''
    for ds in summary['Dataset'].unique():
        idx = summary.loc[summary['Dataset'] == ds, '_mse'].idxmin()
        summary.loc[idx, 'Tốt nhất'] = '★'
    summary = summary.drop(columns=['_mse'])

    out_csv = os.path.join(results_dir, 'bang_tong_hop_luan_van.csv')
    summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
    tests_df = pd.DataFrame(tests)
    tests_csv = os.path.join(results_dir, 'kiem_dinh_thong_ke.csv')
    if not tests_df.empty:
        tests_df.to_csv(tests_csv, index=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(os.path.join(results_dir, 'bang_tong_hop_luan_van.xlsx')) as xw:
            summary.to_excel(xw, sheet_name='Ket qua', index=False)
            if not tests_df.empty:
                tests_df.to_excel(xw, sheet_name='Kiem dinh', index=False)
            abl = os.path.join(results_dir, 'ablation_summary.csv')
            if os.path.exists(abl):
                pd.read_csv(abl).to_excel(xw, sheet_name='Ablation', index=False)
    except Exception as e:
        print(f"(Không ghi được Excel: {e})")

    print("\n" + "=" * 110)
    print(" BẢNG TỔNG HỢP KẾT QUẢ (Mean ± Std qua các run; mọi mô hình dùng cùng tập run với ensemble)")
    print("=" * 110)
    print(summary.to_string(index=False))
    if not tests_df.empty:
        print("\nKIỂM ĐỊNH THỐNG KÊ:")
        print(tests_df.round(4).to_string(index=False))
    print(f"\n-> {out_csv}\n-> {tests_csv}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Xuất bảng tổng hợp và kiểm định thống kê cho luận văn")
    parser.add_argument('--run_ids', type=str, default=None, help="Ví dụ '5-9'; mặc định các run đã có kết quả ensemble")
    args = parser.parse_args()
    ids = None
    if args.run_ids:
        from baselines_ml.run_ml_baselines import parse_run_ids
        ids = parse_run_ids(args.run_ids)
    generate_thesis_report(run_ids=ids)
