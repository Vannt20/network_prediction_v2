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
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)
# run_experiments.py ghi log theo đường dẫn tương đối 'logs/' -> luôn chạy tại thư mục gốc dự án
os.chdir(parent_dir)

from baselines_ml.run_ml_baselines import run_ml_experiments, parse_run_ids
from training.precompute_cache import run_precompute, get_champion_name


def run_full_pipeline(datasets=None, runs=5, run_ids=None, epochs=200, patience=30, ml_models=None,
                      skip_dl=False, allow_cpu=False, skip_ablation=False, quick_check=False):
    if datasets is None or 'all' in datasets:
        datasets = ['sdn', 'geant', 'abilene']
    ids = parse_run_ids(run_ids, runs)
    id_str = ','.join(map(str, ids))
    champion = get_champion_name(os.path.join(parent_dir, 'results'))
    ml_models = ml_models or [champion, 'lightgbm_res']
    branches = ['stwaveformer', champion, 'lightgbm_res']

    print("=" * 90)
    print(" QUY TRÌNH ST-ADAPTIVE-ENSEMBLE v3")
    print(f" Datasets: {datasets} | Run IDs: {ids} | Nhánh: {branches}")
    print(f" DL: epochs={epochs}, patience={patience} | skip_dl={skip_dl} | quick_check={quick_check}")
    print("=" * 90)

    # BƯỚC 1: Nhánh học sâu ST-WaveFormer (GPU). Chỉ huấn luyện run còn thiếu hoặc là bản cũ.
    if skip_dl:
        print("\n[BƯỚC 1/6] Bỏ qua huấn luyện DL (--skip_dl): dùng checkpoint có sẵn, sẽ được kiểm tra ở Bước 3.")
    else:
        print("\n[BƯỚC 1/6] ST-WaveFormer (skip_existing, tự nhận diện run bản cũ)...")
        from run_experiments import run_all_experiments
        run_all_experiments(datasets=datasets, models=['STWaveFormer'],
                            epochs=2 if quick_check else epochs, patience=2 if quick_check else patience,
                            skip_existing=True, run_ids=ids, allow_cpu=allow_cpu or quick_check)

    # BƯỚC 2: Hai nhánh học máy (CPU)
    print(f"\n[BƯỚC 2/6] Nhánh học máy {ml_models}...")
    run_ml_experiments(models=ml_models, datasets=datasets, run_ids=ids,
                       quick_check=quick_check, skip_existing=True)

    # BƯỚC 3: Bộ đệm dự báo Val/Test (có kiểm tra khớp nhãn)
    print("\n[BƯỚC 3/6] Precompute cache v3...")
    run_precompute(datasets=datasets, run_ids=id_str, branches=branches, quick_check=quick_check,
                   skip_existing=not quick_check)

    # BƯỚC 4: Tầng kết hợp
    print("\n[BƯỚC 4/6] Stacking lồi bền vững theo luồng...")
    from training.train_stacking import train_all_stacking
    train_all_stacking(datasets=datasets, run_ids=id_str)

    # BƯỚC 5: Ablation
    if not skip_ablation:
        print("\n[BƯỚC 5/6] Ablation study...")
        from evaluation.ablation_study import run_ablation_experiments
        run_ablation_experiments(datasets=datasets, run_ids=id_str)
    else:
        print("\n[BƯỚC 5/6] Bỏ qua ablation (--skip_ablation).")

    # BƯỚC 6: Báo cáo và hình
    print("\n[BƯỚC 6/6] Báo cáo và hình vẽ...")
    from evaluation.report_generator import generate_thesis_report
    from evaluation.plot_gate_dynamics import plot_all_thesis_figures
    generate_thesis_report()
    plot_all_thesis_figures()

    print("\n" + "=" * 90)
    print(" HOÀN TẤT. Kết quả: results/bang_tong_hop_luan_van.csv, results/kiem_dinh_thong_ke.csv, results/ablation_summary.csv")
    print("=" * 90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chạy toàn bộ quy trình ST-Adaptive-Ensemble v3")
    parser.add_argument('--datasets', type=str, default='all', help="sdn,geant,abilene hoặc 'all'")
    parser.add_argument('--runs', type=int, default=5, help="Số run: run_0..run_{runs-1}")
    parser.add_argument('--run_ids', type=str, default=None, help="Ví dụ '5-9' hoặc '0-9' (ghi đè --runs)")
    parser.add_argument('--epochs', type=int, default=200, help="Epoch tối đa của ST-WaveFormer (mặc định 200)")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience (mặc định 30)")
    parser.add_argument('--ml_models', type=str, default=None,
                        help="Mô hình ML cần huấn luyện, mặc định '<champion>,lightgbm_res'; dùng 'all' để chạy cả LightGBM/CatBoost đối chứng")
    parser.add_argument('--skip_dl', action='store_true', help="Không huấn luyện DL, chỉ dùng checkpoint có sẵn (máy CPU)")
    parser.add_argument('--allow_cpu', action='store_true', help="Cho phép huấn luyện DL trên CPU")
    parser.add_argument('--skip_ablation', action='store_true')
    parser.add_argument('--quick_check', action='store_true', help="Kiểm tra nhanh pipeline (2 epoch, 20 cây, 10 mẫu)")
    args = parser.parse_args()

    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ['sdn', 'geant', 'abilene']
    if args.ml_models == 'all':
        m_list = ['lightgbm', 'catboost', 'xgboost', 'lightgbm_res']
    else:
        m_list = [m.strip() for m in args.ml_models.split(',')] if args.ml_models else None
    run_full_pipeline(datasets=d_list, runs=args.runs, run_ids=args.run_ids, epochs=args.epochs,
                      patience=args.patience, ml_models=m_list, skip_dl=args.skip_dl, allow_cpu=args.allow_cpu,
                      skip_ablation=args.skip_ablation, quick_check=args.quick_check)
