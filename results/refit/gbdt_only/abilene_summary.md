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
