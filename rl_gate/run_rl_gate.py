"""
Pilot ST-Adaptive-Ensemble-RL trên dự báo đã lưu của 3 nhánh (cache/v3), không huấn luyện lại nhánh nào.

Quy trình (giữ nguyên split và 10 seed của v3, Test chỉ dùng để chấm điểm):
  1. Hedge baseline: chọn (eta, decay) theo MSE trung bình trên Val của 10 run.
  2. RL-Gate: học trên 70% đầu Val, early stopping trên 30% cuối Val; chuẩn hóa trạng thái bằng thống kê Val.
  3. Trên Test: bộ nhớ Hedge khởi động từ Val, policy cố định; công tắc an toàn online với margin 1%.

Ví dụ:  python rl_gate/run_rl_gate.py --datasets sdn,geant,abilene --ablation
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_io import load_run
from rl_gate import (CHANNELS, blend, hedge_memories, hedge_predict, safety_switch, state_features,
                     train_policy)

HEDGE_GRID = [(eta, decay) for eta in [30, 100, 300, 1000, 3000] for decay in [0.9, 0.98]]
ABLATIONS = {  # tên: (kênh bộ nhớ, mask (U, b, V, d), thuật toán)
    'rl_gate': (CHANNELS, (1, 1, 1, 1), 'dpg'),
    'reinforce': (CHANNELS, (1, 1, 1, 1), 'reinforce'),
    'no_memory': ([], (0, 0, 1, 1), 'dpg'),
    'const_beta': (CHANNELS, (0, 1, 0, 1), 'dpg'),
    'no_delta': (CHANNELS, (1, 1, 0, 0), 'dpg'),
    'one_channel': ([('se', 100, 0.98)], (1, 1, 1, 1), 'dpg'),
}


def mse(a, b):
    return float(np.mean((a - b) ** 2) * 1e3)


def mae(a, b):
    return float(np.mean(np.abs(a - b)) * 1e3)


def hedge_val_scores(args):
    repo, ds, r = args
    d = load_run(repo, ds, r)
    return ds, r, [mse(hedge_predict(d['val_P'], d['val_y'], d['W'], *g), d['val_y']) for g in HEDGE_GRID]


def run_one(args):
    repo, ds, r, hedge_cfg, variants, margin, out_dir = args
    d = load_run(repo, ds, r)
    Pv, yv, Pt, yt = d['val_P'], d['val_y'], d['test_P'], d['test_y']
    prior = np.log(d['W'] + 1e-6)
    Sv = state_features(Pv, yv, d['val_last'], d['val_context'])
    St = state_features(Pt, yt, d['test_last'], d['test_context'])
    mu, sd = Sv.reshape(-1, Sv.shape[-1]).mean(0), Sv.reshape(-1, Sv.shape[-1]).std(0) + 1e-6
    Sv, St = (Sv - mu) / sd, (St - mu) / sd
    cut = int(yv.shape[0] * 0.7)
    A, B = slice(0, cut), slice(cut, yv.shape[0])

    static = blend(d['W'], Pt)
    hedge = hedge_predict(Pt, yt, d['W'], *hedge_cfg)
    q = np.quantile((static - yt) ** 2, 0.999)
    keep = (static - yt) ** 2 <= q          # bỏ 0,1% điểm sai số lớn nhất của v3 (chỉ số phụ)
    row = {'dataset': ds, 'run': r}

    def record(name, pred):
        row[f'{name}_mse'] = mse(pred, yt)
        row[f'{name}_mae'] = mae(pred, yt)
        row[f'{name}_mse_trim'] = float(np.mean(((pred - yt) ** 2)[keep]) * 1e3)

    record('v3', static)
    record('hedge', hedge)
    for name in variants:
        channels, mask, mode = ABLATIONS[name]
        if channels:
            Hv = hedge_memories(Pv, yv, channels)
            Ht = hedge_memories(Pt, yt, channels, warmup=(Pv, yv))
        else:
            Hv = np.zeros((yv.shape[0], 1) + Pv.shape[:1] + yv.shape[1:])
            Ht = np.zeros((yt.shape[0], 1) + Pt.shape[:1] + yt.shape[1:])
        policy, _ = train_policy(Sv[A], Hv[A], prior, Pv[:, A], yv[A], holdout=(Sv[B], Hv[B], Pv[:, B], yv[B]),
                                 mode=mode, mask=mask)
        pred = policy.predict(St, Ht, prior, Pt)
        record(name, pred)
        if name == 'rl_gate':
            weights, beta = policy.act(St, Ht, prior)
            switched, use_rl = safety_switch(static, pred, yt, margin)
            record('rl_gate_safe', switched)
            row['rl_gate_safe_use_rl'] = float(use_rl.mean())
            row['beta_mean'] = beta.mean((0, 2)).round(4).tolist()
            row['weight_mean'] = weights.mean((0, 2)).round(4).tolist()
            row['weight_std_over_time'] = float(weights.std(0).mean())
            np.save(os.path.join(out_dir, f'weights_flow_{ds}_run_{r}.npy'), weights.mean(0).astype(np.float32))
    return row


def summarize(rows, variants):
    from scipy.stats import ttest_rel
    lines = ['| Dataset | Mô hình | Test MSE ×10⁻³ (TB ± std) | so với v3 | run tốt hơn v3 | p (t ghép cặp, vs v3) | run tốt hơn Hedge | MAE ×10⁻³ | MSE bỏ 0,1% đỉnh |',
             '|---|---|---|---|---|---|---|---|---|']
    for ds in ['sdn', 'geant', 'abilene']:
        R = sorted([x for x in rows if x['dataset'] == ds], key=lambda x: x['run'])
        if not R:
            continue
        v3 = np.array([x['v3_mse'] for x in R])
        hd = np.array([x['hedge_mse'] for x in R])
        for name in ['v3', 'hedge'] + list(variants) + (['rl_gate_safe'] if 'rl_gate' in variants else []):
            v = np.array([x[f'{name}_mse'] for x in R])
            p = ttest_rel(v, v3).pvalue if name != 'v3' else float('nan')
            lines.append(f'| {ds} | {name} | {v.mean():.4f} ± {v.std():.4f} | {100 * (v.mean() / v3.mean() - 1):+.2f}% | '
                         f'{int((v < v3).sum())}/10 | {p:.1e} | {int((v < hd).sum())}/10 | '
                         f'{np.mean([x[f"{name}_mae"] for x in R]):.3f} | {np.mean([x[f"{name}_mse_trim"] for x in R]):.4f} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--repo', default=root)
    ap.add_argument('--datasets', default='sdn,geant,abilene')
    ap.add_argument('--runs', default='0-9')
    ap.add_argument('--ablation', action='store_true', help='chạy thêm các biến thể ablation')
    ap.add_argument('--margin', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(root, 'results', 'rl_gate'))
    args = ap.parse_args()
    a, b = map(int, args.runs.split('-'))
    runs, dss = list(range(a, b + 1)), args.datasets.split(',')
    os.makedirs(args.out, exist_ok=True)
    variants = list(ABLATIONS) if args.ablation else ['rl_gate']

    with Pool(args.workers) as pool:
        scores = {}
        for ds, r, s in pool.imap_unordered(hedge_val_scores, [(args.repo, ds, r) for ds in dss for r in runs]):
            scores.setdefault(ds, []).append(s)
        hedge_cfg = {ds: HEDGE_GRID[int(np.argmin(np.mean(scores[ds], 0)))] for ds in dss}
        print('Hedge (eta, decay) chọn trên Val:', hedge_cfg, flush=True)
        jobs = [(args.repo, ds, r, hedge_cfg[ds], variants, args.margin, args.out) for ds in dss for r in runs]
        rows = []
        for row in pool.imap_unordered(run_one, jobs):
            rows.append(row)
            print(f"{row['dataset']} run {row['run']}: v3 {row['v3_mse']:.4f} hedge {row['hedge_mse']:.4f} "
                  f"rl_gate {row['rl_gate_mse']:.4f} rl_gate_safe {row['rl_gate_safe_mse']:.4f}", flush=True)
    with open(os.path.join(args.out, 'rl_gate_runs.json'), 'w') as f:
        json.dump({'hedge_cfg': {k: list(v) for k, v in hedge_cfg.items()}, 'rows': rows}, f, indent=1)
    table = summarize(rows, variants)
    with open(os.path.join(args.out, 'rl_gate_summary.md'), 'w', encoding='utf-8') as f:
        f.write(table + '\n')
    print(table)


if __name__ == '__main__':
    main()
