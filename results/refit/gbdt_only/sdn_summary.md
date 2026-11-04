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
