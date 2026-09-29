# Báo cáo kết quả thực nghiệm (5 lần chạy)

Mô hình đề xuất: **ST-Adaptive-Ensemble v3**, gồm 3 nhánh (ST-WaveFormer, XGBoost, LightGBM-Residual), kết hợp bằng stacking bền vững. Cấu hình loss/scope được chọn bằng blocked CV trên tập Validation.
Mỗi mô hình chạy 5 lần (run 0–4) trên 3 bộ dữ liệu: SDN (seq 60), GEANT (seq 24), ABILENE (seq 24). Số liệu dưới đây là trung bình ± độ lệch chuẩn, MSE tính theo đơn vị ×10⁻³.

## 1. Kết quả chính

| Mô hình | SDN | GEANT | ABILENE |
|---|---|---|---|
| **Ensemble v3 (đề xuất)** | **4.022 ± 0.037** | **0.723 ± 0.004** | **1.846 ± 0.007** |
| ST-WaveFormer | 15.009 ± 0.305 | 0.736 ± 0.005 | 1.966 ± 0.016 |
| XGBoost (champion ML) | 5.511 ± 0.031 | 0.914 ± 0.003 | 2.764 ± 0.013 |
| LightGBM-Residual | 4.738 ± 0.036 | 0.752 ± 0.001 | 1.858 ± 0.006 |
| LightGBM (đối chứng) | 5.543 ± 0.796 | 0.919 ± 0.003 | 2.926 ± 0.040 |
| CatBoost (đối chứng) | 8.437 ± 0.022 | 0.914 ± 0.001 | 2.709 ± 0.039 |

Mức giảm MSE của Ensemble so với nhánh đơn tốt nhất:
- SDN: **−15.1%** so với LightGBM-Residual. So với XGBoost là −27.0%, so với ST-WaveFormer là −73.2%.
- GEANT: **−1.8%** so với ST-WaveFormer, −3.9% so với LightGBM-Residual.
- ABILENE: **−0.6%** so với LightGBM-Residual, −6.1% so với ST-WaveFormer.

Ensemble có MSE thấp nhất trên cả 3 bộ dữ liệu. Độ lệch chuẩn giữa các run nhỏ (dưới 1% giá trị trung bình), tức kết quả ổn định. Đổi lại, thời gian suy luận tăng: 10.8–14.7 ms/lô 64, so với 1–7 ms của từng nhánh đơn lẻ.

## 2. Kiểm định thống kê

- **t-test ghép cặp trên 5 run:** Ensemble tốt hơn có ý nghĩa ở cả 15/15 phép so sánh (p < 0.05). Riêng so với LightGBM-Residual, p ≈ 0.013 trên SDN và ABILENE, còn trên GEANT p ≈ 1.5×10⁻⁴.
- **Diebold–Mariano (theo từng mẫu, trong từng run):**
  - SDN và ABILENE: Ensemble tốt hơn có ý nghĩa ở **5/5 run** so với cả 3 nhánh.
  - GEANT: **không có ý nghĩa** (0–1/5 run, DM ≈ −1.4). Trên GEANT, lợi thế của Ensemble nhất quán giữa các run nhưng rất nhỏ ở mức từng mẫu.

## 3. Trọng số và cấu hình được chọn

| Dataset | Cấu hình (CV chọn) | w WaveFormer | w XGBoost | w LGBM-Res |
|---|---|---|---|---|
| SDN | mse / perflow | 0.006 | 0.170 | 0.823 |
| GEANT | huber / global | 0.449 | 0.286 | 0.265 |
| ABILENE | huber / global | 0.004 | 0.034 | 0.962 |

Mỗi bộ dữ liệu dựa vào nhánh khác nhau. GEANT chia trọng số khá đều giữa 3 nhánh, nhánh học sâu nặng nhất. SDN và ABILENE gần như chỉ dùng LightGBM-Residual. Điều này giải thích vì sao trên ABILENE, Ensemble chỉ hơn LightGBM-Residual 0.6%.

## 4. Ablation (MSE ×10⁻³, trung bình 5 run)

| Cấu hình | SDN | GEANT | ABILENE |
|---|---|---|---|
| Full (đề xuất) | **4.022** | 0.723 | 1.846 |
| Chỉ trọng số global | 4.697 | 0.723 | 1.846 |
| Chỉ trọng số perflow | 4.022 | 0.741 | 1.846 |
| Trung bình tĩnh 1/K | 6.323 | 0.723 | 1.865 |
| Chọn 1 nhánh tốt nhất (Val) | 4.738 | 0.914 | 1.858 |
| Cổng MLP ngữ cảnh (v2) | 4.701 | 0.851 | 1.860 |
| Bỏ ST-WaveFormer | 4.694 | 0.727 | 1.845 |
| Bỏ XGBoost | 4.157 | 0.722 | 1.857 |
| Bỏ LightGBM-Residual | 4.855 | 0.758 | 2.645 |

Nhận xét:
- **Trọng số theo luồng (perflow) là yếu tố quyết định trên SDN.** Nếu chỉ dùng trọng số global, MSE tăng 16.8%. Nếu lấy trung bình tĩnh, MSE tăng 57%. Trên GEANT thì ngược lại, perflow kém hơn global. Như vậy việc cho CV chọn scope là hợp lý.
- **LightGBM-Residual là nhánh quan trọng nhất.** Bỏ nhánh này, MSE tăng mạnh nhất: +43% trên ABILENE, +21% trên SDN.
- **Stacking v3 tốt hơn cổng MLP ngữ cảnh (v2)** trên cả 3 bộ dữ liệu, rõ nhất ở GEANT (0.723 so với 0.851).
- Trên GEANT và ABILENE, một số biến thể (bỏ XGBoost, bỏ WaveFormer, trung bình tĩnh) cho MSE ngang Full, chênh dưới 0.1%. Ở hai bộ này, cấu trúc đầy đủ không tạo thêm lợi ích đáng kể.
- *Lưu ý:* ở biến thể "Bỏ ST-WaveFormer" trên SDN, MSE tăng 16.7% dù nhánh này chỉ có trọng số 0.006. Nguyên nhân là khi còn 2 nhánh, CV chuyển sang chọn scope global (4.694 ≈ 4.697 của "chỉ global"). Mức tăng này đến từ việc đổi scope, không phản ánh đóng góp của nhánh học sâu. Cần ghi rõ điểm này khi trình bày.

## 5. Kết luận

1. Ensemble v3 đạt MSE thấp nhất trên cả 3 bộ dữ liệu, ổn định qua 5 run, và có ý nghĩa thống kê theo t-test ghép cặp.
2. Ensemble cải thiện mạnh trên SDN (−15%) nhờ trọng số theo luồng. Trên GEANT (−1.8%) và ABILENE (−0.6%), mức cải thiện nhỏ. Trên GEANT, cải thiện không có ý nghĩa theo DM ở mức từng mẫu.
3. Cái giá phải trả là thời gian suy luận tăng khoảng 2–3 lần so với nhánh nhanh nhất, nhưng vẫn ở mức ~15 ms/lô 64.
