"""
Điều phối huấn luyện ST-Adaptive-Ensemble v3:
- precompute_cache.py : bộ đệm dự báo Val/Test của các nhánh (cache/v3/)
- train_stacking.py   : học tầng kết hợp stacking lồi theo luồng trên Validation
- run_ensemble.py     : chạy toàn bộ quy trình
Các module được import trực tiếp (ví dụ `from training.train_stacking import ...`) để tránh vòng import.
"""
