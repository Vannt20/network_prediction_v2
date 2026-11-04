# ST-Adaptive-Ensemble-RL trên nhánh refit (Train+Val) — tổng hợp 3 dataset

Phương án chính chốt trước khi xem Test: `rl_gate_refit_safe`. Các biến thể còn lại chỉ để phân tích. Chỉ tuyên bố cải thiện khi TB 10 run thấp hơn v3 và p (t ghép cặp) < 0,05.

## Kết luận phương án chính

| Dataset | v3 | rl_gate_refit_safe | so với v3 | run tốt hơn | p | Cải thiện có ý nghĩa |
|---|---|---|---|---|---|---|
| SDN | 4.0187 | 3.8970 | -3.03% | 10/10 | 2.0e-08 | có |
| GEANT | 0.7231 | 0.7234 | +0.04% | 5/10 | 9.3e-01 | không |
| ABILENE | 1.8442 | 1.6703 | -9.43% | 10/10 | 4.1e-13 | có |

### SDN (10 run: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

Hedge (η, decay) chọn trên Val: (30, 0.9). Phương án chính, chốt trước khi xem Test: `rl_gate_refit_safe`. Nhánh refit trên Train+Val: xgboost, lightgbm_res (ST-WaveFormer giữ dự báo gốc của v3 -> v3_refit trùng v3_refit_gbdt_only).

| Biến thể | Mô tả | Test MSE ×10⁻³ (TB ± std) | so với v3 | run tốt hơn v3 | p (t ghép cặp, vs v3) | MAE ×10⁻³ | MSE bỏ 0,1% đỉnh v3 |
|---|---|---|---|---|---|---|---|
| v3 | Như luận văn (tái lập từ cache/v3) | 4.0187 ± 0.0293 | +0.00% | 0/10 | nan | 38.120 | 3.6075 |
| branch_refit[stwaveformer] | Không refit: nhánh stwaveformer gốc của v3 | 15.0733 ± 0.2677 | +275.08% | 0/10 | 4.1e-16 | 64.492 | 14.5963 |
| branch_refit[xgboost] | Nhánh xgboost refit trên Train+Val | 5.3377 ± 0.0304 | +32.82% | 0/10 | 3.6e-15 | 43.071 | 4.9559 |
| branch_refit[lightgbm_res] | Nhánh lightgbm_res refit trên Train+Val | 4.6966 ± 0.0311 | +16.87% | 0/10 | 7.9e-14 | 39.797 | 4.2749 |
| v3_refit | Trọng số stacking v3 × nhánh refit (chỉ GBDT, ST-WaveFormer gốc) | 3.9988 ± 0.0330 | -0.50% | 10/10 | 2.7e-03 | 37.892 | 3.5889 |
| v3_refit_gbdt_only | Chỉ thay 2 nhánh GBDT bằng bản refit, giữ ST-WaveFormer gốc | 3.9988 ± 0.0330 | -0.50% | 10/10 | 2.7e-03 | 37.892 | 3.5889 |
| hedge_refit | Hedge theo luồng trên nhánh refit, (η, decay) chọn trên Val | 3.9343 ± 0.0231 | -2.10% | 10/10 | 7.5e-08 | 37.443 | 3.5424 |
| rl_gate_refit | Policy RL-Gate học trên Val (dự báo gốc), áp dụng cho Test với dự báo refit | 3.8967 ± 0.0185 | -3.04% | 10/10 | 1.9e-08 | 37.223 | 3.5059 |
| **rl_gate_refit_safe** | Như trên + safety_switch(v3_refit, rl_gate_refit, margin=0.01) — phương án chính | 3.8970 ± 0.0184 | -3.03% | 10/10 | 2.0e-08 | 37.225 | 3.5062 |
| _branch_v3[stwaveformer]_ | Tham chiếu: nhánh stwaveformer gốc (chỉ học trên Train) | 15.0733 ± 0.2677 | +275.08% | 0/10 | 4.1e-16 | 64.492 | 14.5963 |
| _branch_v3[xgboost]_ | Tham chiếu: nhánh xgboost gốc (chỉ học trên Train) | 5.5092 ± 0.0305 | +37.09% | 0/10 | 1.3e-16 | 43.449 | 5.1346 |
| _branch_v3[lightgbm_res]_ | Tham chiếu: nhánh lightgbm_res gốc (chỉ học trên Train) | 4.7338 ± 0.0240 | +17.79% | 0/10 | 4.6e-15 | 40.108 | 4.3085 |
| _rl_gate_v3_ | Tham chiếu: cùng policy áp dụng cho dự báo gốc (như results/rl_gate) | 3.9107 ± 0.0179 | -2.69% | 10/10 | 1.1e-08 | 37.387 | 3.5198 |
| _rl_gate_v3_safe_ | Tham chiếu: rl_gate_v3 + safety_switch(v3, ·, 0.01) | 3.9108 ± 0.0179 | -2.69% | 10/10 | 1.1e-08 | 37.388 | 3.5199 |

- Tái lập v3 từ cache/v3: 4.0187 (mốc luận văn 4.019) -> khớp.
- Phương án chính `rl_gate_refit_safe`: 3.8970 so với v3 4.0187 (-3.03%, 10/10 run tốt hơn, p = 2.0e-08) -> cải thiện có ý nghĩa thống kê (TB thấp hơn v3 và p < 0,05).
- Tỷ lệ bước Test mà công tắc an toàn dùng RL-Gate: 99.7% (refit), 99.9% (tham chiếu, dự báo gốc).

### GEANT (10 run: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

Hedge (η, decay) chọn trên Val: (100, 0.98). Phương án chính, chốt trước khi xem Test: `rl_gate_refit_safe`. Nhánh refit trên Train+Val: xgboost, lightgbm_res (ST-WaveFormer giữ dự báo gốc của v3 -> v3_refit trùng v3_refit_gbdt_only).

| Biến thể | Mô tả | Test MSE ×10⁻³ (TB ± std) | so với v3 | run tốt hơn v3 | p (t ghép cặp, vs v3) | MAE ×10⁻³ | MSE bỏ 0,1% đỉnh v3 |
|---|---|---|---|---|---|---|---|
| v3 | Như luận văn (tái lập từ cache/v3) | 0.7231 ± 0.0032 | +0.00% | 0/10 | nan | 5.450 | 0.2855 |
| branch_refit[stwaveformer] | Không refit: nhánh stwaveformer gốc của v3 | 0.7369 ± 0.0036 | +1.90% | 0/10 | 3.2e-08 | 5.398 | 0.2981 |
| branch_refit[xgboost] | Nhánh xgboost refit trên Train+Val | 0.8163 ± 0.0095 | +12.89% | 0/10 | 3.0e-10 | 5.849 | 0.3263 |
| branch_refit[lightgbm_res] | Nhánh lightgbm_res refit trên Train+Val | 0.7758 ± 0.0054 | +7.29% | 0/10 | 1.5e-09 | 5.736 | 0.3159 |
| v3_refit | Trọng số stacking v3 × nhánh refit (chỉ GBDT, ST-WaveFormer gốc) | 0.7141 ± 0.0040 | -1.25% | 10/10 | 4.3e-06 | 5.444 | 0.2872 |
| v3_refit_gbdt_only | Chỉ thay 2 nhánh GBDT bằng bản refit, giữ ST-WaveFormer gốc | 0.7141 ± 0.0040 | -1.25% | 10/10 | 4.3e-06 | 5.444 | 0.2872 |
| hedge_refit | Hedge theo luồng trên nhánh refit, (η, decay) chọn trên Val | 0.7510 ± 0.0208 | +3.86% | 0/10 | 2.4e-03 | 5.455 | 0.3029 |
| rl_gate_refit | Policy RL-Gate học trên Val (dự báo gốc), áp dụng cho Test với dự báo refit | 0.7505 ± 0.0222 | +3.79% | 0/10 | 4.3e-03 | 5.516 | 0.3034 |
| **rl_gate_refit_safe** | Như trên + safety_switch(v3_refit, rl_gate_refit, margin=0.01) — phương án chính | 0.7234 ± 0.0081 | +0.04% | 5/10 | 9.3e-01 | 5.448 | 0.2904 |
| _branch_v3[stwaveformer]_ | Tham chiếu: nhánh stwaveformer gốc (chỉ học trên Train) | 0.7369 ± 0.0036 | +1.90% | 0/10 | 3.2e-08 | 5.398 | 0.2981 |
| _branch_v3[xgboost]_ | Tham chiếu: nhánh xgboost gốc (chỉ học trên Train) | 0.9153 ± 0.0031 | +26.58% | 0/10 | 9.4e-17 | 5.865 | 0.3331 |
| _branch_v3[lightgbm_res]_ | Tham chiếu: nhánh lightgbm_res gốc (chỉ học trên Train) | 0.7514 ± 0.0007 | +3.91% | 0/10 | 2.0e-09 | 5.758 | 0.3099 |
| _rl_gate_v3_ | Tham chiếu: cùng policy áp dụng cho dự báo gốc (như results/rl_gate) | 0.7529 ± 0.0046 | +4.12% | 0/10 | 5.9e-10 | 5.521 | 0.3083 |
| _rl_gate_v3_safe_ | Tham chiếu: rl_gate_v3 + safety_switch(v3, ·, 0.01) | 0.7232 ± 0.0032 | +0.01% | 0/10 | 2.4e-02 | 5.451 | 0.2856 |

- Tái lập v3 từ cache/v3: 0.7231 (mốc luận văn 0.723) -> khớp.
- Phương án chính `rl_gate_refit_safe`: 0.7234 so với v3 0.7231 (+0.04%, 5/10 run tốt hơn, p = 9.3e-01) -> KHÔNG đủ điều kiện tuyên bố cải thiện (cần TB thấp hơn v3 và p < 0,05).
- Tỷ lệ bước Test mà công tắc an toàn dùng RL-Gate: 1.2% (refit), 0.7% (tham chiếu, dự báo gốc).

### ABILENE (10 run: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

Hedge (η, decay) chọn trên Val: (100, 0.9). Phương án chính, chốt trước khi xem Test: `rl_gate_refit_safe`. Nhánh refit trên Train+Val: xgboost, lightgbm_res (ST-WaveFormer giữ dự báo gốc của v3 -> v3_refit trùng v3_refit_gbdt_only).

| Biến thể | Mô tả | Test MSE ×10⁻³ (TB ± std) | so với v3 | run tốt hơn v3 | p (t ghép cặp, vs v3) | MAE ×10⁻³ | MSE bỏ 0,1% đỉnh v3 |
|---|---|---|---|---|---|---|---|
| v3 | Như luận văn (tái lập từ cache/v3) | 1.8442 ± 0.0067 | +0.00% | 0/10 | nan | 16.273 | 0.9781 |
| branch_refit[stwaveformer] | Không refit: nhánh stwaveformer gốc của v3 | 1.9720 ± 0.0133 | +6.93% | 0/10 | 8.7e-10 | 16.933 | 1.1612 |
| branch_refit[xgboost] | Nhánh xgboost refit trên Train+Val | 2.6097 ± 0.0196 | +41.51% | 0/10 | 7.9e-15 | 17.336 | 1.7636 |
| branch_refit[lightgbm_res] | Nhánh lightgbm_res refit trên Train+Val | 1.8427 ± 0.0089 | -0.08% | 6/10 | 7.2e-01 | 16.219 | 0.9697 |
| v3_refit | Trọng số stacking v3 × nhánh refit (chỉ GBDT, ST-WaveFormer gốc) | 1.8331 ± 0.0078 | -0.60% | 9/10 | 5.2e-03 | 16.219 | 0.9713 |
| v3_refit_gbdt_only | Chỉ thay 2 nhánh GBDT bằng bản refit, giữ ST-WaveFormer gốc | 1.8331 ± 0.0078 | -0.60% | 9/10 | 5.2e-03 | 16.219 | 0.9713 |
| hedge_refit | Hedge theo luồng trên nhánh refit, (η, decay) chọn trên Val | 1.6977 ± 0.0045 | -7.94% | 10/10 | 2.8e-13 | 16.079 | 0.9621 |
| rl_gate_refit | Policy RL-Gate học trên Val (dự báo gốc), áp dụng cho Test với dự báo refit | 1.6655 ± 0.0057 | -9.69% | 10/10 | 2.6e-13 | 15.986 | 0.9537 |
| **rl_gate_refit_safe** | Như trên + safety_switch(v3_refit, rl_gate_refit, margin=0.01) — phương án chính | 1.6703 ± 0.0059 | -9.43% | 10/10 | 4.1e-13 | 15.989 | 0.9537 |
| _branch_v3[stwaveformer]_ | Tham chiếu: nhánh stwaveformer gốc (chỉ học trên Train) | 1.9720 ± 0.0133 | +6.93% | 0/10 | 8.7e-10 | 16.933 | 1.1612 |
| _branch_v3[xgboost]_ | Tham chiếu: nhánh xgboost gốc (chỉ học trên Train) | 2.7636 ± 0.0129 | +49.85% | 0/10 | 3.2e-18 | 17.470 | 1.9158 |
| _branch_v3[lightgbm_res]_ | Tham chiếu: nhánh lightgbm_res gốc (chỉ học trên Train) | 1.8554 ± 0.0054 | +0.61% | 0/10 | 1.4e-04 | 16.271 | 0.9756 |
| _rl_gate_v3_ | Tham chiếu: cùng policy áp dụng cho dự báo gốc (như results/rl_gate) | 1.6531 ± 0.0047 | -10.36% | 10/10 | 1.7e-13 | 15.993 | 0.9581 |
| _rl_gate_v3_safe_ | Tham chiếu: rl_gate_v3 + safety_switch(v3, ·, 0.01) | 1.6574 ± 0.0051 | -10.13% | 10/10 | 1.7e-13 | 16.001 | 0.9584 |

- Tái lập v3 từ cache/v3: 1.8442 (mốc luận văn 1.844) -> khớp.
- Phương án chính `rl_gate_refit_safe`: 1.6703 so với v3 1.8442 (-9.43%, 10/10 run tốt hơn, p = 4.1e-13) -> cải thiện có ý nghĩa thống kê (TB thấp hơn v3 và p < 0,05).
- Tỷ lệ bước Test mà công tắc an toàn dùng RL-Gate: 91.1% (refit), 87.6% (tham chiếu, dự báo gốc).

