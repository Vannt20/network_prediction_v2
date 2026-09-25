import os
import json
import pandas as pd
import numpy as np


def select_champion_ml_model(val_records, results_dir='results', frozen=False):
    """
    Tuyển chọn Quán quân (Champion Model) duy nhất cho Nhánh ML theo đặc tả:
    - Scope: global_across_datasets (1 model duy nhất cho cả 3 dataset).
    - Criterion: xếp hạng theo MSE trên tập Validation.
    - Aggregation: mean_rank trên cả 3 datasets (tránh méo mó do chênh lệch biên độ giữa SDN và GEANT).
    - Tie-breaker: lowest_avg_inference_time.
    
    Tham số:
        val_records: List of dicts, mỗi dict chứa:
            {'dataset': 'sdn', 'model': 'lightgbm', 'val_mse': ..., 'inference_time_ms': ...}
        results_dir: thư mục lưu kết quả tuyển chọn.
        
    Trả về:
        champion_name: tên mô hình quán quân (ví dụ: 'lightgbm' hoặc 'catboost')
        summary_df: bảng xếp hạng chi tiết.
    """
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(val_records)
    if df.empty:
        raise ValueError("Không có bản ghi validation nào để tuyển chọn Champion model.")
        
    # Tính rank theo từng dataset
    df['rank'] = df.groupby('dataset')['val_mse'].rank(ascending=True, method='min')
    
    # Tổng hợp mean_rank và avg_inference_time theo mô hình
    summary = df.groupby('model').agg(
        mean_rank=('rank', 'mean'),
        avg_val_mse=('val_mse', 'mean'),
        avg_inference_time_ms=('inference_time_ms', 'mean')
    ).reset_index()
    
    # Sắp xếp: mean_rank tăng dần, tie-breaker: avg_inference_time_ms tăng dần
    summary = summary.sort_values(by=['mean_rank', 'avg_inference_time_ms'], ascending=[True, True]).reset_index(drop=True)
    
    champion_name = summary.iloc[0]['model']
    
    champion_info = {
        'champion_model': champion_name,
        'mean_rank': float(summary.iloc[0]['mean_rank']),
        'avg_val_mse': float(summary.iloc[0]['avg_val_mse']),
        'avg_inference_time_ms': float(summary.iloc[0]['avg_inference_time_ms']),
        'ranking_table': summary.to_dict(orient='records'),
        'selected_on_runs': sorted({int(r['run']) for r in val_records if 'run' in r}),
        'frozen': bool(frozen)
    }
    
    out_json = os.path.join(results_dir, 'champion_ml_model.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(champion_info, f, indent=4, ensure_ascii=False)
        
    print(f"\n=======================================================")
    print(f"[*] QUÁN QUÂN NHÁNH TRADITIONAL ML: {champion_name.upper()}")
    print(f"    Mean Rank: {champion_info['mean_rank']:.2f} | Avg Inf Time: {champion_info['avg_inference_time_ms']:.2f} ms")
    print(f"    Đã lưu thông tin quán quân vào: {out_json}")
    print(f"=======================================================\n")
    
    return champion_name, summary
