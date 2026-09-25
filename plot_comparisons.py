import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

MODEL_MAP = {
    'lstm': 'LSTM',
    'bilstm': 'BiLSTM',
    'gru': 'GRU',
    'bigru': 'BiGRU',
    'gwn': 'GWN',
    'graphwavenet': 'GWN',
    'dcrnn': 'DCRNN',
    'stwaveformer': 'ST-WaveFormer',
    'stadaptiveensemble': 'ST-Adaptive-Ensemble',
    'st_adaptive_ensemble': 'ST-Adaptive-Ensemble',
    'xgboost': 'XGBoost',
    'lightgbm_res': 'LightGBM-Residual',
    'lightgbm': 'LightGBM',
    'catboost': 'CatBoost'
}

DATASET_MAP = {
    'sdn': 'SDN',
    'geant': 'GEANT',
    'abilene': 'ABILENE'
}

TARGET_SEQLEN = {
    'SDN': 60,
    'GEANT': 24,
    'ABILENE': 24
}

DATASET_ORDER = ['SDN', 'GEANT', 'ABILENE']
MODEL_ORDER = ['ST-Adaptive-Ensemble', 'ST-WaveFormer', 'XGBoost', 'LightGBM-Residual', 'LightGBM', 'CatBoost', 'GWN', 'BiGRU', 'BiLSTM', 'GRU', 'LSTM', 'DCRNN']


def recover_results_from_logs(logs_dir='logs', results_dir='results'):
    """
    Tự động khôi phục toàn bộ các file results_*_data_*.csv từ thư mục logs/
    nếu thư mục results/ bị xóa hoặc trống.
    """
    if not os.path.exists(logs_dir):
        return
    os.makedirs(results_dir, exist_ok=True)
    log_subdirs = glob.glob(os.path.join(logs_dir, "*_data_*_seq_*"))
    for ldir in log_subdirs:
        base = os.path.basename(ldir)
        m = re.match(r"^(?P<model>.+)_data_(?P<ds>[A-Za-z0-9]+)_seq_(?P<seq>\d+)$", base)
        if not m:
            continue
        raw_m = m.group('model')
        raw_ds = m.group('ds')
        seq_len = int(m.group('seq'))
        test_files = glob.glob(os.path.join(ldir, "run_*", "test_metrics.csv"))
        if not test_files:
            continue
        run_records = []
        for tf in sorted(test_files):
            rm = re.search(r"run_(\d+)", tf)
            run_id = int(rm.group(1)) if rm else 0
            try:
                tdf = pd.read_csv(tf)
                if not tdf.empty:
                    d = tdf.iloc[0].to_dict()
                    d['run'] = run_id
                    d['seq_len'] = seq_len
                    run_records.append(d)
            except Exception:
                continue
        if run_records:
            df_reconstructed = pd.DataFrame(run_records)
            # Chuẩn hóa tên mô hình để map đúng
            model_clean = MODEL_MAP.get(raw_m.lower(), raw_m).replace('-', '')
            out_name = f"results_{model_clean}_data_{raw_ds.lower()}.csv"
            out_path = os.path.join(results_dir, out_name)
            df_reconstructed.to_csv(out_path, index=False)
            print(f"  -> Đã khôi phục {len(run_records)} runs vào: {out_path}")


def collect_results_from_dir(results_dir='results'):
    """
    Quét toàn bộ các file kết quả chi tiết của từng mô hình và tập dữ liệu trong thư mục results/
    (ví dụ: results_BiGRU_data_geant.csv, results_STWaveFormer_data_sdn.csv, ...)
    để tổng hợp thành một DataFrame hoàn chỉnh.
    Tự động khôi phục từ logs/ nếu results/ bị xóa.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(results_dir, "results_*_data_*.csv"))
    if not csv_files and os.path.exists('logs'):
        print(f"Không tìm thấy file kết quả trong '{results_dir}'. Đang tự động khôi phục từ 'logs/'...")
        recover_results_from_logs(logs_dir='logs', results_dir=results_dir)
        csv_files = glob.glob(os.path.join(results_dir, "results_*_data_*.csv"))

    if not csv_files:
        print(f"Không tìm thấy file kết quả dạng results_*_data_*.csv nào trong '{results_dir}'.")
        return pd.DataFrame()

    records = []
    processed_pairs = set()

    csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        match = re.match(r"^results_(?P<model>[A-Za-z0-9]+)_data_(?P<dataset>[A-Za-z0-9]+)", fname, re.IGNORECASE)
        if not match:
            continue

        raw_model = match.group('model').lower()
        raw_dataset = match.group('dataset').lower()

        model_name = MODEL_MAP.get(raw_model, match.group('model'))
        dataset_name = DATASET_MAP.get(raw_dataset, match.group('dataset').upper())

        pair_key = (dataset_name, model_name)
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Lỗi khi đọc file {fpath}: {e}")
            continue

        if df.empty:
            continue

        df.columns = [c.strip().lower() for c in df.columns]

        target_seq = TARGET_SEQLEN.get(dataset_name, 24)
        if 'seq_len' in df.columns:
            if (df['seq_len'] == target_seq).any():
                df_target = df[df['seq_len'] == target_seq]
            else:
                max_seq = df['seq_len'].max()
                df_target = df[df['seq_len'] == max_seq]
                target_seq = int(max_seq)
        else:
            df_target = df

        mean_mse = df_target['mse'].mean() if 'mse' in df_target.columns else np.nan
        mean_mae = df_target['mae'].mean() if 'mae' in df_target.columns else np.nan

        if 'rmse' in df_target.columns:
            mean_rmse = df_target['rmse'].mean()
        elif not np.isnan(mean_mse):
            mean_rmse = np.sqrt(mean_mse)
        else:
            mean_rmse = np.nan

        mean_rse = df_target['rse'].mean() if 'rse' in df_target.columns else np.nan
        mean_mape = df_target['mape'].mean() if 'mape' in df_target.columns else np.nan

        mean_time = np.nan
        if 'inference_time_ms' in df_target.columns:
            mean_time = df_target['inference_time_ms'].mean()
        elif 'inference_time' in df_target.columns:
            mean_time = df_target['inference_time'].mean()

        records.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Seq_Len': target_seq,
            'MSE (x10^-3)': mean_mse * 1000.0 if not np.isnan(mean_mse) else np.nan,
            'MAE (x10^-3)': mean_mae * 1000.0 if not np.isnan(mean_mae) else np.nan,
            'RMSE': mean_rmse,
            'RSE': mean_rse,
            'MAPE': mean_mape,
            'Inference Time (ms)': mean_time
        })

    if not records:
        return pd.DataFrame()

    summary_df = pd.DataFrame(records)

    summary_df['ds_order'] = summary_df['Dataset'].apply(lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 99)
    summary_df['m_order'] = summary_df['Model'].apply(lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)
    summary_df = summary_df.sort_values(by=['ds_order', 'm_order']).drop(columns=['ds_order', 'm_order']).reset_index(drop=True)

    summary_csv = os.path.join(results_dir, 'bang_ket_qua_danh_gia_mo_hinh.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"-> Đã tổng hợp dữ liệu từ {len(csv_files)} file kết quả con vào: {summary_csv}")

    return summary_df


def generate_plots(results_dir='results', output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    df = collect_results_from_dir(results_dir=results_dir)

    if df.empty:
        summary_csv = os.path.join(results_dir, 'bang_ket_qua_danh_gia_mo_hinh.csv')
        if os.path.exists(summary_csv):
            print(f"Đọc dữ liệu từ file tổng hợp có sẵn: {summary_csv}")
            df = pd.read_csv(summary_csv)
        else:
            print("Không có dữ liệu thực nghiệm để vẽ biểu đồ.")
            return

    print("\n" + "=" * 90)
    print(" BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    unique_models = df['Model'].unique().tolist()
    palette = sns.color_palette("Set2", n_colors=max(len(unique_models), 6))
    model_palette = {m: palette[i % len(palette)] for i, m in enumerate(MODEL_ORDER if all(m in MODEL_ORDER for m in unique_models) else unique_models)}

    # 1. Vẽ biểu đồ so sánh MAE
    if 'MAE (x10^-3)' in df.columns and df['MAE (x10^-3)'].notna().any():
        plt.figure(figsize=(11, 6))
        ax = sns.barplot(data=df, x='Dataset', y='MAE (x10^-3)', hue='Model', palette=model_palette)
        plt.title('Comparison of MAE (x10^-3) across Models and Datasets', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Absolute Error (MAE x 10^-3)', fontsize=12, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')

        for p in ax.patches:
            val = p.get_height()
            if not np.isnan(val) and val > 0:
                ax.annotate(f'{val:.2f}', (p.get_x() + p.get_width() / 2., val),
                            ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                            textcoords='offset points')

        plt.tight_layout()
        mae_plot_path = os.path.join(output_dir, 'mae_comparison_reproduced.png')
        plt.savefig(mae_plot_path, dpi=300)
        plt.close()
        print(f"-> Đã lưu biểu đồ so sánh MAE: {mae_plot_path}")

    # 2. Vẽ biểu đồ so sánh MSE
    if 'MSE (x10^-3)' in df.columns and df['MSE (x10^-3)'].notna().any():
        plt.figure(figsize=(11, 6))
        ax = sns.barplot(data=df, x='Dataset', y='MSE (x10^-3)', hue='Model', palette=model_palette)
        plt.title('Comparison of MSE (x10^-3) across Models and Datasets', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Squared Error (MSE x 10^-3)', fontsize=12, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')

        for p in ax.patches:
            val = p.get_height()
            if not np.isnan(val) and val > 0:
                ax.annotate(f'{val:.2f}', (p.get_x() + p.get_width() / 2., val),
                            ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                            textcoords='offset points')

        plt.tight_layout()
        mse_plot_path = os.path.join(output_dir, 'mse_comparison_reproduced.png')
        plt.savefig(mse_plot_path, dpi=300)
        plt.close()
        print(f"-> Đã lưu biểu đồ so sánh MSE: {mse_plot_path}")

    # 3. Vẽ biểu đồ so sánh Inference Time
    if 'Inference Time (ms)' in df.columns and df['Inference Time (ms)'].notna().any():
        plt.figure(figsize=(11, 6))
        ax = sns.barplot(data=df, x='Dataset', y='Inference Time (ms)', hue='Model', palette=model_palette)
        plt.title('Comparison of Inference Time (ms/batch) across Models and Datasets', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')

        for p in ax.patches:
            val = p.get_height()
            if not np.isnan(val) and val > 0:
                ax.annotate(f'{val:.2f}', (p.get_x() + p.get_width() / 2., val),
                            ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                            textcoords='offset points')

        plt.tight_layout()
        time_plot_path = os.path.join(output_dir, 'inference_time_comparison_reproduced.png')
        plt.savefig(time_plot_path, dpi=300)
        plt.close()
        print(f"-> Đã lưu biểu đồ so sánh Inference Time: {time_plot_path}")


if __name__ == '__main__':
    generate_plots()
