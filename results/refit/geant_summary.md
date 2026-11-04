### GEANT (10 run: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

Hedge (η, decay) chọn trên Val: (100, 0.98). Phương án chính, chốt trước khi xem Test: `rl_gate_refit_safe`. Nhánh refit trên Train+Val: stwaveformer, xgboost, lightgbm_res.

| Biến thể | Mô tả | Test MSE ×10⁻³ (TB ± std) | so với v3 | run tốt hơn v3 | p (t ghép cặp, vs v3) | MAE ×10⁻³ | MSE bỏ 0,1% đỉnh v3 |
|---|---|---|---|---|---|---|---|
| v3 | Như luận văn (tái lập từ cache/v3) | 0.7231 ± 0.0032 | +0.00% | 0/10 | nan | 5.450 | 0.2855 |
| branch_refit[stwaveformer] | Nhánh stwaveformer refit trên Train+Val | 0.8181 ± 0.0285 | +13.14% | 0/10 | 2.7e-06 | 5.463 | 0.3130 |
| branch_refit[xgboost] | Nhánh xgboost refit trên Train+Val | 0.8175 ± 0.0086 | +13.05% | 0/10 | 1.4e-10 | 5.852 | 0.3248 |
| branch_refit[lightgbm_res] | Nhánh lightgbm_res refit trên Train+Val | 0.7758 ± 0.0054 | +7.29% | 0/10 | 1.5e-09 | 5.736 | 0.3159 |
| v3_refit | Trọng số stacking v3 × 3 nhánh refit | 0.7498 ± 0.0130 | +3.69% | 0/10 | 7.2e-05 | 5.460 | 0.2896 |
| v3_refit_gbdt_only | Chỉ thay 2 nhánh GBDT bằng bản refit, giữ ST-WaveFormer gốc | 0.7149 ± 0.0033 | -1.13% | 10/10 | 1.1e-05 | 5.444 | 0.2871 |
| hedge_refit | Hedge theo luồng trên nhánh refit, (η, decay) chọn trên Val | 0.7848 ± 0.0106 | +8.53% | 0/10 | 4.7e-08 | 5.472 | 0.3053 |
| rl_gate_refit | Policy RL-Gate học trên Val (dự báo gốc), áp dụng cho Test với dự báo refit | 0.7853 ± 0.0104 | +8.60% | 0/10 | 3.3e-08 | 5.536 | 0.3059 |
| **rl_gate_refit_safe** | Như trên + safety_switch(v3_refit, rl_gate_refit, margin=0.01) — phương án chính | 0.7691 ± 0.0188 | +6.36% | 0/10 | 2.7e-05 | 5.465 | 0.2996 |
| _branch_v3[stwaveformer]_ | Tham chiếu: nhánh stwaveformer gốc (chỉ học trên Train) | 0.7369 ± 0.0036 | +1.90% | 0/10 | 3.2e-08 | 5.398 | 0.2981 |
| _branch_v3[xgboost]_ | Tham chiếu: nhánh xgboost gốc (chỉ học trên Train) | 0.9153 ± 0.0031 | +26.58% | 0/10 | 9.4e-17 | 5.865 | 0.3331 |
| _branch_v3[lightgbm_res]_ | Tham chiếu: nhánh lightgbm_res gốc (chỉ học trên Train) | 0.7514 ± 0.0007 | +3.91% | 0/10 | 2.0e-09 | 5.758 | 0.3099 |
| _rl_gate_v3_ | Tham chiếu: cùng policy áp dụng cho dự báo gốc (như results/rl_gate) | 0.7529 ± 0.0046 | +4.12% | 0/10 | 6.0e-10 | 5.527 | 0.3084 |
| _rl_gate_v3_safe_ | Tham chiếu: rl_gate_v3 + safety_switch(v3, ·, 0.01) | 0.7232 ± 0.0032 | +0.01% | 0/10 | 8.9e-03 | 5.451 | 0.2856 |

- Tái lập v3 từ cache/v3: 0.7231 (mốc luận văn 0.723) -> khớp.
- Phương án chính `rl_gate_refit_safe`: 0.7691 so với v3 0.7231 (+6.36%, 0/10 run tốt hơn, p = 2.7e-05) -> KHÔNG đủ điều kiện tuyên bố cải thiện (cần TB thấp hơn v3 và p < 0,05).
- Tỷ lệ bước Test mà công tắc an toàn dùng RL-Gate: 1.4% (refit), 1.0% (tham chiếu, dự báo gốc).
