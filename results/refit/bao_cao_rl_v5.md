# ST-Adaptive-Ensemble-RL v5: refit trên Train+Val và RL-Gate — báo cáo tổng hợp (04/11/2026)

Mã: nhánh `rl_v5` (`training/refit_branches.py`, `training/kaggle_refit.py`, `rl_gate/eval_refit.py`,
`quytrinh/kaggle_refit.ipynb`). Quy trình theo `yeu_cau_claude_code_refit_kaggle.md`: scaler fit trên Train, số epoch / số cây
lấy từ lần huấn luyện gốc (không early stopping), trọng số stacking v3 dùng lại nguyên, y_test khớp `cache/v3` (sai lệch 0).
Phương án chính chốt trước khi xem Test: `rl_gate_refit_safe`. Chỉ ghi "cải thiện" khi TB 10 run thấp hơn v3 và p < 0,05
(t-test ghép cặp). Tái lập v3 từ `cache/v3` khớp luận văn: 4,019 / 0,723 / 1,844.

## 1. Kết luận phương án chính (Test MSE ×10⁻³, TB 10 run)

| Đợt | Dataset | v3 | `rl_gate_refit_safe` | so với v3 | run tốt hơn | p | Cải thiện có ý nghĩa |
|---|---|---|---|---|---|---|---|
| A. Chỉ refit 2 nhánh GBDT (ST-WaveFormer gốc) | SDN | 4,0187 | 3,8970 | −3,03% | 10/10 | 2,0e-08 | có |
| | GÉANT | 0,7231 | 0,7234 | +0,04% | 5/10 | 0,93 | không |
| | Abilene | 1,8442 | 1,6703 | −9,43% | 10/10 | 4,1e-13 | có |
| B. Refit đủ 3 nhánh (cả ST-WaveFormer) | GÉANT | 0,7231 | 0,7691 | +6,36% | 0/10 | 2,7e-05 | không (kém hơn v3) |

Đợt B chỉ chạy cho GÉANT (dataset duy nhất mà đợt A không cải thiện); SDN và Abilene không refit ST-WaveFormer.

## 2. Các biến thể phân tích

| Biến thể | SDN (A) | GÉANT (A) | Abilene (A) | GÉANT (B) |
|---|---|---|---|---|
| v3 | 4,0187 | 0,7231 | 1,8442 | 0,7231 |
| `v3_refit` (trọng số v3 × nhánh refit) | 3,9988 (−0,50%) | **0,7141 (−1,25%)** | 1,8331 (−0,60%) | 0,7498 (+3,69%) |
| `v3_refit_gbdt_only` | = v3_refit | = v3_refit | = v3_refit | 0,7149 (−1,13%) |
| `hedge_refit` | 3,9343 (−2,10%) | 0,7510 (+3,86%) | 1,6977 (−7,94%) | 0,7848 (+8,53%) |
| `rl_gate_refit` | 3,8967 (−3,04%) | 0,7505 (+3,79%) | 1,6655 (−9,69%) | 0,7853 (+8,60%) |
| **`rl_gate_refit_safe`** (chính) | **3,8970 (−3,03%)** | 0,7234 (+0,04%) | **1,6703 (−9,43%)** | 0,7691 (+6,36%) |
| tham chiếu `rl_gate_v3` (dự báo gốc) | 3,9107 (−2,69%) | 0,7529 (+4,12%) | 1,6531 (−10,36%) | 0,7529 |
| tham chiếu `rl_gate_v3_safe` | 3,9108 (−2,69%) | 0,7232 (+0,01%) | 1,6574 (−10,13%) | 0,7232 |

Từng nhánh (gốc → refit): XGBoost SDN 5,51 → 5,34, GÉANT 0,915 → 0,816, Abilene 2,764 → 2,610; LightGBM-Res SDN 4,73 → 4,70,
GÉANT 0,751 → 0,776, Abilene 1,855 → 1,843; ST-WaveFormer GÉANT 0,737 → 0,818.

## 3. Nhận xét

- Phần cải thiện ở SDN và Abilene đến chủ yếu từ RL-Gate: trên dự báo gốc RL-Gate đã cho 3,911 / 1,653. Refit GBDT thêm khoảng
  0,35% ở SDN và làm Abilene kém đi một chút (1,653 → 1,670).
- Refit GBDT giúp mô hình cây ngoại suy tốt hơn (XGBoost GÉANT −11%), nên `v3_refit` là biến thể tốt nhất ở GÉANT (−1,25%, 10/10 run,
  p = 4,3e-06). Nhưng RL-Gate làm hỏng phần lợi đó ở GÉANT: công tắc an toàn chỉ bật RL-Gate ở 1–1,4% số bước, vẫn kéo kết quả
  về ngang v3 (đợt A) hoặc kém hơn (đợt B), vì sai số GÉANT tập trung ở vài đỉnh lớn.
- Refit ST-WaveFormer trên Train+Val với E* cố định làm nhánh này kém đi rõ rệt ở GÉANT (0,737 → 0,818, +11%); không nên refit nhánh DL.
- Kiểm tra chỉ trên Val trước khi chạy: RL-Gate thắng v3 ở holdout Val cả 3 dataset (GÉANT 9/10 run), nên không có quy tắc chọn theo
  Val nào loại được RL-Gate ở GÉANT; phương án chính giữ nguyên như đã chốt.

## 4. Lưu ý khi đưa vào luận văn

- GÉANT run 0 (refit 2 nhánh GBDT, ST-WaveFormer gốc) đã được xem trước trên Test trước khi chạy đủ 10 run; người dùng quyết định
  giữ nguyên phương án chính.
- Đợt A chạy trên CPU máy local (xgboost 3.2.0, lightgbm 4.7.0, numpy 2.4); đợt B chạy trên Kaggle 2 tài khoản × 2 T4
  (torch 2.10 + CUDA 12.8, xgboost 3.2.0), ~77 s/epoch, mỗi run GÉANT dài 0,5–4,3 giờ. Nhánh GBDT GÉANT có ở cả hai đợt, khác nhau
  rất ít do môi trường (XGBoost 0,8163 so với 0,8175).

## 5. Vị trí kết quả

| Đường dẫn | Nội dung |
|---|---|
| `logs_refit/{stwaveformer,xgboost,lightgbm_res}_data_geant/run_*/` | Đợt B (Kaggle): dự báo Test, meta.json, model.pth / train_metrics.csv của ST-WaveFormer |
| `results/refit/geant_summary.md`, `geant_runs.json`, `summary_all.md` | Đánh giá đợt B |
| `logs_refit_gbdt/{xgboost,lightgbm_res}_data_{sdn,geant,abilene}/run_*/` | Đợt A (local): dự báo Test, meta.json |
| `results/refit/gbdt_only/` | Đánh giá đợt A (tái lập: `python rl_gate/eval_refit.py --gbdt_only --refit_root logs_refit_gbdt --out results/refit/gbdt_only`) |
| `status/refit_{a,b}_geant.json`, `status/refit_solo_gbdt.json` | Trạng thái cuối của từng tài khoản / đợt |
