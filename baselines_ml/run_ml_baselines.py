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
import time
import argparse
import numpy as np
import pandas as pd

# sys.path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.feature_store import prepare_feature_store
from baselines_ml.metrics import calc_metrics_numpy, measure_inference_time
from baselines_ml.lgbm_baseline import LGBMBaseline
from baselines_ml.catboost_baseline import CatBoostBaseline
from baselines_ml.xgboost_baseline import XGBoostBaseline
from baselines_ml.tree_baselines import ExtraTreesBaseline
from baselines_ml.lgbm_residual_baseline import LGBMResidualBaseline
from baselines_ml.model_selection import select_champion_ml_model


MODEL_CLASSES = {
    'lightgbm': LGBMBaseline,
    'catboost': CatBoostBaseline,
    'xgboost': XGBoostBaseline,
    'extra_trees': ExtraTreesBaseline,
    'lightgbm_res': LGBMResidualBaseline
}

# 3 GBDT chuẩn tham gia bầu Champion (nhánh 2) + LightGBM-Residual (nhánh 3, cố định theo thiết kế)
CHAMPION_CANDIDATES = ['lightgbm', 'catboost', 'xgboost']
ALL_MODELS = CHAMPION_CANDIDATES + ['lightgbm_res']
ALL_DATASETS = ['sdn', 'geant', 'abilene']


def parse_run_ids(run_ids=None, runs=5):
    """'5-9' | '0,1,2' | list | None (-> range(runs))."""
    if run_ids is None or run_ids == '':
        return list(range(runs))
    if isinstance(run_ids, (list, tuple)):
        return [int(r) for r in run_ids]
    out = []
    for part in str(run_ids).split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def get_model_instance(m_name, seed=42, quick_check=False, feature_names=None):
    cls = MODEL_CLASSES[m_name.lower()]
    if m_name == 'lightgbm_res':
        return cls(feature_names=feature_names, n_estimators=20 if quick_check else 1000, random_state=seed)
    if quick_check:
        if m_name in ['lightgbm', 'xgboost']:
            return cls(n_estimators=20, random_state=seed)
        elif m_name == 'catboost':
            return cls(iterations=20, random_seed=seed)
        elif m_name == 'extra_trees':
            return cls(n_estimators=10, max_depth=6, random_state=seed)
    else:
        if m_name in ['lightgbm', 'xgboost', 'extra_trees']:
            return cls(random_state=seed)
        elif m_name == 'catboost':
            return cls(random_seed=seed)
    return cls()


def run_ml_experiments(models=None, datasets=None, runs=5, quick_check=False, skip_existing=False, run_ids=None,
                       freeze_champion=False):
    if models is None or 'all' in models:
        models = ALL_MODELS
    if datasets is None or 'all' in datasets:
        datasets = ALL_DATASETS
    run_id_list = parse_run_ids(run_ids, runs)

    results_dir = os.path.join(parent_dir, 'results')
    logs_dir = os.path.join(parent_dir, 'logs')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 80)
    print(" MODULE A: TRADITIONAL ML BASELINE BENCHMARK (SHARED MODEL)")
    print(f" Datasets: {datasets} | Models: {models} | Run IDs: {run_id_list} | Quick Check: {quick_check}")
    print("=" * 80)

    val_records_for_champion = []

    for ds in datasets:
        print(f"\n---> Chuẩn bị dữ liệu cho Dataset: {ds.upper()}...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = prepare_feature_store(
            ds, data_dir=os.path.join(parent_dir, 'data')
        )
        print(f"     Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape} | Features: {len(metadata['feature_names'])}")

        if quick_check:
            # Lấy tập mẫu nhỏ để kiểm tra nhanh luồng chạy
            sample_tr = min(5000, len(X_train))
            sample_va = min(1000, len(X_val))
            sample_te = min(2000, len(X_test))
            X_tr, y_tr = X_train[:sample_tr], y_train[:sample_tr]
            X_va, y_va = X_val[:sample_va], y_val[:sample_va]
            X_te, y_te = X_test[:sample_te], y_test[:sample_te]
        else:
            X_tr, y_tr = X_train, y_train
            X_va, y_va = X_val, y_val
            X_te, y_te = X_test, y_test

        for m_name in models:
            run_metrics = []
            m_key = m_name.lower().replace('-', '_')
            log_model_name = f"{m_key}_data_{ds}_shared"

            for run_id in run_id_list:
                seed = 42 + run_id
                run_dir = os.path.join(logs_dir, log_model_name, f"run_{run_id}")
                os.makedirs(run_dir, exist_ok=True)
                test_csv = os.path.join(run_dir, 'test_metrics.csv')
                val_csv = os.path.join(run_dir, 'val_metrics.csv')
                model_file = os.path.join(run_dir, 'model.bin')

                if skip_existing and os.path.exists(test_csv):
                    try:
                        prev_df = pd.read_csv(test_csv)
                        if not prev_df.empty:
                            m_dict = prev_df.iloc[0].to_dict()
                            m_dict['run'] = run_id
                            run_metrics.append(m_dict)
                            print(f"  [SKIP] Đã có kết quả: {m_name} trên {ds.upper()} [run_{run_id}]")

                            # Tự động nạp hoặc tính nhanh chỉ số validation phục vụ Champion Selection
                            if os.path.exists(val_csv):
                                v_df = pd.read_csv(val_csv)
                                val_records_for_champion.append({
                                    'dataset': ds,
                                    'model': m_key,
                                    'run': run_id,
                                    'val_mse': float(v_df.iloc[0]['mse']),
                                    'inference_time_ms': float(v_df.iloc[0].get('inference_time_ms', 0.0))
                                })
                            elif os.path.exists(model_file):
                                model_inst = get_model_instance(m_key, seed=seed, quick_check=quick_check,
                                                                feature_names=metadata['feature_names'])
                                model_inst.load(model_file)
                                val_preds = model_inst.predict(X_va)
                                val_metrics = calc_metrics_numpy(val_preds, y_va)
                                val_inf_time = measure_inference_time(lambda b: model_inst.predict(b), X_va, batch_size=64)
                                val_metrics['inference_time_ms'] = val_inf_time
                                pd.DataFrame([val_metrics]).to_csv(val_csv, index=False)
                                val_records_for_champion.append({
                                    'dataset': ds,
                                    'model': m_key,
                                    'run': run_id,
                                    'val_mse': val_metrics['mse'],
                                    'inference_time_ms': val_inf_time
                                })
                            continue
                    except Exception as e:
                        print(f"    (Lỗi đọc skip cache: {e})")

                print(f"  [*] Huấn luyện {m_name.upper()} trên {ds.upper()} [run_{run_id}] (Seed {seed})...", flush=True)
                t0 = time.time()
                model_inst = get_model_instance(m_key, seed=seed, quick_check=quick_check,
                                                feature_names=metadata['feature_names'])
                try:
                    model_inst.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
                except (ImportError, ModuleNotFoundError) as e:
                    print(f"  [WARN] Thư viện cho {m_name.upper()} chưa được cài đặt ({e}), bỏ qua mô hình này.", flush=True)
                    break
                fit_time = time.time() - t0

                # Đánh giá trên tập Validation để tìm Champion
                val_preds = model_inst.predict(X_va)
                val_metrics = calc_metrics_numpy(val_preds, y_va)
                val_inf_time = measure_inference_time(lambda b: model_inst.predict(b), X_va, batch_size=64)
                val_metrics['inference_time_ms'] = val_inf_time

                val_records_for_champion.append({
                    'dataset': ds,
                    'model': m_key,
                    'run': run_id,
                    'val_mse': val_metrics['mse'],
                    'inference_time_ms': val_inf_time
                })

                # Đánh giá trên tập Test
                test_metrics, test_preds = model_inst.evaluate(X_te, y_te, batch_size=64)
                test_metrics['run'] = run_id
                test_metrics['fit_time_s'] = fit_time
                run_metrics.append(test_metrics)

                # Lưu metrics và checkpoints
                pd.DataFrame([test_metrics]).to_csv(test_csv, index=False)
                pd.DataFrame([val_metrics]).to_csv(val_csv, index=False)
                np.save(os.path.join(run_dir, 'y_pred_data.npy'), test_preds)
                np.save(os.path.join(run_dir, 'y_real_data.npy'), y_te)
                try:
                    model_inst.save(model_file)
                except Exception as e:
                    print(f"    (Cảnh báo lưu checkpoint model: {e})")

                print(f"[DONE] {m_name} {ds.upper()} run_{run_id} (seed {seed}) | Test MSE={test_metrics['mse']*1000.0:.3f}e-3 "
                      f"MAE={test_metrics['mae']*1000.0:.3f}e-3 | Inf Time={test_metrics['inference_time_ms']:.2f} ms | fit {fit_time/60:.1f} phút", flush=True)

            # Lưu kết quả tổng hợp của mô hình trên dataset
            # Gộp với các run đã có trong results CSV (chạy theo đợt: run 5-9 rồi 0-4)
            df_runs = pd.DataFrame(run_metrics)
            out_csv = os.path.join(results_dir, f"results_{m_key}_data_{ds}.csv")
            if os.path.exists(out_csv) and not df_runs.empty:
                try:
                    old = pd.read_csv(out_csv)
                    if 'run' in old.columns:
                        old = old[~old['run'].isin(df_runs['run'])]
                        df_runs = pd.concat([old, df_runs], ignore_index=True)
                except Exception:
                    pass
            if not df_runs.empty:
                df_runs = df_runs.sort_values('run').reset_index(drop=True)
                df_runs.to_csv(out_csv, index=False)

    # Tuyển chọn Champion trong nhóm GBDT chuẩn. Chỉ ghi champion_ml_model.json khi có đủ
    # 3 ứng viên x 3 dataset, tránh ghi đè kết quả khi chỉ chạy một phần mô hình/dataset.
    champ_records = [r for r in val_records_for_champion if r['model'] in CHAMPION_CANDIDATES]
    covered = {(r['model'], r['dataset']) for r in champ_records}
    full = all((m, d) in covered for m in CHAMPION_CANDIDATES for d in ALL_DATASETS)
    champ_file = os.path.join(results_dir, 'champion_ml_model.json')
    frozen_info = None
    if os.path.exists(champ_file):
        import json
        with open(champ_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        if info.get('frozen'):
            frozen_info = info
    if frozen_info is not None:
        print(f"\n[INFO] Champion đã được cố định: {frozen_info['champion_model'].upper()} "
              f"(bầu trên run {frozen_info.get('selected_on_runs')}) -> không bầu lại")
    elif champ_records and full:
        champion_name, rank_table = select_champion_ml_model(champ_records, results_dir=results_dir,
                                                             frozen=freeze_champion)
        print("\nBẢNG XẾP HẠNG GBDT CHUẨN (TẬP VALIDATION):")
        print(rank_table.to_string(index=False))
    elif champ_records:
        print("\n[INFO] Chưa đủ 3 GBDT x 3 dataset trong lần chạy này -> giữ nguyên results/champion_ml_model.json")

    return val_records_for_champion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Traditional ML Baselines (Module A)")
    parser.add_argument('--models', type=str, default='all', help="Comma-separated: lightgbm,catboost,xgboost,lightgbm_res (mặc định 'all'). Tùy chọn: extra_trees")
    parser.add_argument('--datasets', type=str, default='all', help="Comma-separated datasets: sdn,geant,abilene hoặc 'all'")
    parser.add_argument('--runs', type=int, default=5, help="Số run: run_0..run_{runs-1} (mặc định 5)")
    parser.add_argument('--run_ids', type=str, default=None, help="Chỉ định run cụ thể, ví dụ '5-9' hoặc '0,1,2' (ghi đè --runs)")
    parser.add_argument('--freeze_champion', action='store_true',
                        help="Cố định champion vừa bầu (các lần chạy sau không bầu lại)")
    parser.add_argument('--quick_check', action='store_true', help="Chạy kiểm tra nhanh logic hệ thống")
    parser.add_argument('--skip_existing', action='store_true', help="Bỏ qua các lần chạy đã có log")

    args = parser.parse_args()

    m_list = [m.strip() for m in args.models.split(',')] if args.models != 'all' else ALL_MODELS
    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ALL_DATASETS

    run_ml_experiments(
        models=m_list,
        datasets=d_list,
        runs=args.runs,
        quick_check=args.quick_check,
        skip_existing=args.skip_existing,
        run_ids=args.run_ids,
        freeze_champion=args.freeze_champion
    )
