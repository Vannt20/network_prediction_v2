"""
Đánh giá v3-refit và RL-Gate trên dự báo của 3 nhánh huấn luyện lại trên Train+Val (logs_refit/), chỉ cần numpy, scipy.

Biến thể (Test chỉ dùng để chấm điểm, không biến thể nào được chọn theo Test):
  v3                   trọng số stacking v3 × 3 nhánh gốc (tái lập từ cache/v3)
  branch_refit[k]      từng nhánh refit
  v3_refit             trọng số stacking v3 × 3 nhánh refit
  v3_refit_gbdt_only   chỉ thay 2 nhánh GBDT bằng bản refit, giữ ST-WaveFormer gốc
  hedge_refit          Hedge theo luồng trên nhánh refit, (eta, decay) chọn trên Val như run_rl_gate.py
  rl_gate_refit        policy RL-Gate học trên Val với dự báo gốc, áp dụng cho Test với dự báo refit;
                       bộ nhớ Hedge khởi động từ Val gốc
  rl_gate_refit_safe   như trên + safety_switch(v3_refit, rl_gate_refit, margin=0.01)  <- PHƯƠNG ÁN CHÍNH
Hàng tham chiếu (không thuộc so sánh chính): rl_gate_v3 / rl_gate_v3_safe = cùng policy áp dụng cho dự báo gốc,
dùng để đối chiếu với results/rl_gate.

Ví dụ:  python rl_gate/eval_refit.py --datasets geant
        python rl_gate/eval_refit.py --summary_all
        python rl_gate/eval_refit.py --datasets sdn --runs 0 --allow_partial --out <thư mục tạm>   (kiểm tra nhanh)
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_io import LOG_DIRS, load_run
from rl_gate import CHANNELS, blend, hedge_memories, hedge_predict, safety_switch, state_features, train_policy
from run_rl_gate import HEDGE_GRID, mae, mse

DATASETS = ['sdn', 'geant', 'abilene']
BRANCHES = ['stwaveformer', 'xgboost', 'lightgbm_res']
V3_THESIS = {'sdn': 4.019, 'geant': 0.723, 'abilene': 1.844}     # Test MSE ×10⁻³ của v3 trong luận văn
PRIMARY = 'rl_gate_refit_safe'
VARIANTS = (['v3'] + [f'branch_refit[{b}]' for b in BRANCHES] +
            ['v3_refit', 'v3_refit_gbdt_only', 'hedge_refit', 'rl_gate_refit', 'rl_gate_refit_safe'])
REFERENCE = [f'branch_v3[{b}]' for b in BRANCHES] + ['rl_gate_v3', 'rl_gate_v3_safe']
DESCRIPTION = {
    'v3': 'Như luận văn (tái lập từ cache/v3)',
    'v3_refit': 'Trọng số stacking v3 × 3 nhánh refit',
    'v3_refit_gbdt_only': 'Chỉ thay 2 nhánh GBDT bằng bản refit, giữ ST-WaveFormer gốc',
    'hedge_refit': 'Hedge theo luồng trên nhánh refit, (η, decay) chọn trên Val',
    'rl_gate_refit': 'Policy RL-Gate học trên Val (dự báo gốc), áp dụng cho Test với dự báo refit',
    'rl_gate_refit_safe': 'Như trên + safety_switch(v3_refit, rl_gate_refit, margin=0.01) — phương án chính',
    'rl_gate_v3': 'Tham chiếu: cùng policy áp dụng cho dự báo gốc (như results/rl_gate)',
    'rl_gate_v3_safe': 'Tham chiếu: rl_gate_v3 + safety_switch(v3, ·, 0.01)',
    **{f'branch_refit[{b}]': f'Nhánh {b} refit trên Train+Val' for b in BRANCHES},
    **{f'branch_v3[{b}]': f'Tham chiếu: nhánh {b} gốc (chỉ học trên Train)' for b in BRANCHES},
}


def parse_runs(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def load_refit(root, ds, r, shape):
    """P_refit [K,T,N] float64 theo thứ tự BRANCHES, kèm meta.json của từng nhánh."""
    P, metas = [], {}
    for b in BRANCHES:
        d = os.path.join(root, f'{b}_data_{ds}', f'run_{r}')
        with open(os.path.join(d, 'meta.json'), encoding='utf-8') as f:
            metas[b] = json.load(f)
        p = np.load(os.path.join(d, 'test_pred.npy'))
        if p.shape != shape or not np.isfinite(p).all():
            raise RuntimeError(f'{d}/test_pred.npy: kích thước {p.shape} (cần {shape}) hoặc có NaN')
        P.append(p.astype(np.float64))
    return np.stack(P), metas


def available_runs(root, ds, runs):
    return [r for r in runs if all(os.path.exists(os.path.join(root, f'{b}_data_{ds}', f'run_{r}', 'meta.json'))
                                   for b in BRANCHES)]


def hedge_val_mse(args):
    repo, ds, r = args
    d = load_run(repo, ds, r)
    return r, [mse(hedge_predict(d['val_P'], d['val_y'], d['W'], *g), d['val_y']) for g in HEDGE_GRID]


def select_hedge(repo, ds, runs, pool):
    """(eta, decay) có MSE trung bình trên Val nhỏ nhất, đúng quy tắc của run_rl_gate.py (dùng 10 run của dataset)."""
    scores = [s for _, s in pool.imap_unordered(hedge_val_mse, [(repo, ds, r) for r in runs])]
    return HEDGE_GRID[int(np.argmin(np.mean(scores, 0)))]


def eval_run(args):
    repo, root, ds, r, hedge_cfg, margin = args
    d = load_run(repo, ds, r)
    with open(os.path.join(repo, 'logs', LOG_DIRS[ds], f'run_{r}', 'stacking.json')) as f:
        assert json.load(f)['branch_names'] == BRANCHES, 'Thứ tự nhánh của stacking.json khác BRANCHES'
    Pv, yv, Pt, yt, W = d['val_P'], d['val_y'], d['test_P'], d['test_y'], d['W']
    Pr, metas = load_refit(root, ds, r, yt.shape)
    quick = any(m.get('quick_check') for m in metas.values())
    if Pr.shape != Pt.shape:
        raise RuntimeError(f'{ds} run_{r}: P_refit {Pr.shape} khác cache {Pt.shape}')
    prior = np.log(W + 1e-6)

    # RL-Gate: học trên Val với dự báo gốc, đúng như run_rl_gate.run_one
    Sv = state_features(Pv, yv, d['val_last'], d['val_context'])
    mu, sd = Sv.reshape(-1, Sv.shape[-1]).mean(0), Sv.reshape(-1, Sv.shape[-1]).std(0) + 1e-6
    Sv = (Sv - mu) / sd
    cut = int(yv.shape[0] * 0.7)
    A, B = slice(0, cut), slice(cut, yv.shape[0])
    Hv = hedge_memories(Pv, yv, CHANNELS)
    policy, best_epoch = train_policy(Sv[A], Hv[A], prior, Pv[:, A], yv[A], holdout=(Sv[B], Hv[B], Pv[:, B], yv[B]))

    def apply_policy(P):
        S = (state_features(P, yt, d['test_last'], d['test_context']) - mu) / sd
        H = hedge_memories(P, yt, CHANNELS, warmup=(Pv, yv))           # bộ nhớ Hedge khởi động từ Val gốc
        return policy.predict(S, H, prior, P)

    static = blend(W, Pt)
    v3_refit = blend(W, Pr)
    rl_refit = apply_policy(Pr)
    rl_v3 = apply_policy(Pt)
    preds = {
        'v3': static,
        **{f'branch_refit[{b}]': Pr[k] for k, b in enumerate(BRANCHES)},
        **{f'branch_v3[{b}]': Pt[k] for k, b in enumerate(BRANCHES)},
        'v3_refit': v3_refit,
        'v3_refit_gbdt_only': blend(W, np.stack([Pt[0], Pr[1], Pr[2]])),
        'hedge_refit': hedge_predict(Pr, yt, W, *hedge_cfg),
        'rl_gate_refit': rl_refit,
        'rl_gate_v3': rl_v3,
    }
    preds['rl_gate_refit_safe'], use_refit = safety_switch(v3_refit, rl_refit, yt, margin)
    preds['rl_gate_v3_safe'], use_v3 = safety_switch(static, rl_v3, yt, margin)

    keep = (static - yt) ** 2 <= np.quantile((static - yt) ** 2, 0.999)    # bỏ 0,1% điểm sai số lớn nhất của v3
    row = {'dataset': ds, 'run': r, 'quick_check': quick, 'policy_best_epoch': best_epoch,
           'rl_gate_refit_safe_use_rl': float(use_refit.mean()), 'rl_gate_v3_safe_use_rl': float(use_v3.mean()),
           'refit_meta': {b: {k: m.get(k) for k in ('n_epochs', 'n_trees', 'test_mse', 'test_mse_orig_branch',
                                                     'y_test_max_abs_diff', 'time_s')}
                          for b, m in metas.items()}}
    for name, p in preds.items():
        row[f'{name}_mse'] = mse(p, yt)
        row[f'{name}_mae'] = mae(p, yt)
        row[f'{name}_mse_trim'] = float(np.mean(((p - yt) ** 2)[keep]) * 1e3)
    return row


def _fmt_p(p):
    return 'nan' if not np.isfinite(p) else f'{p:.1e}'


def summarize_dataset(ds, rows, hedge_cfg):
    from scipy.stats import ttest_rel
    R = sorted(rows, key=lambda x: x['run'])
    n = len(R)
    v3 = np.array([x['v3_mse'] for x in R])
    lines = [f'### {ds.upper()} ({n} run: {", ".join(str(x["run"]) for x in R)})', '',
             f'Hedge (η, decay) chọn trên Val: {tuple(hedge_cfg)}. '
             f'Phương án chính, chốt trước khi xem Test: `{PRIMARY}`.', '',
             '| Biến thể | Mô tả | Test MSE ×10⁻³ (TB ± std) | so với v3 | run tốt hơn v3 | p (t ghép cặp, vs v3) | MAE ×10⁻³ | MSE bỏ 0,1% đỉnh v3 |',
             '|---|---|---|---|---|---|---|---|']
    stats = {}
    for name in VARIANTS + REFERENCE:
        v = np.array([x[f'{name}_mse'] for x in R])
        p = float(ttest_rel(v, v3).pvalue) if name != 'v3' and n > 1 and np.any(v != v3) else float('nan')
        stats[name] = {'mean': float(v.mean()), 'std': float(v.std()), 'rel': float(v.mean() / v3.mean() - 1),
                       'n_better': int((v < v3).sum()), 'p': p}
        label = f'**{name}**' if name == PRIMARY else (f'_{name}_' if name in REFERENCE else name)
        lines.append(f'| {label} | {DESCRIPTION[name]} | {v.mean():.4f} ± {v.std():.4f} | {100 * stats[name]["rel"]:+.2f}% | '
                     f'{stats[name]["n_better"]}/{n} | {_fmt_p(p)} | '
                     f'{np.mean([x[f"{name}_mae"] for x in R]):.3f} | {np.mean([x[f"{name}_mse_trim"] for x in R]):.4f} |')
    lines.append('')
    ref = V3_THESIS[ds]
    ok = round(stats['v3']['mean'], 3) == ref
    lines.append(f'- Tái lập v3 từ cache/v3: {stats["v3"]["mean"]:.4f} (mốc luận văn {ref:.3f}) -> '
                 f'{"khớp" if ok else "KHÔNG KHỚP" + (" (chưa đủ 10 run)" if n < 10 else "")}.')
    s = stats[PRIMARY]
    better = s['mean'] < stats['v3']['mean'] and np.isfinite(s['p']) and s['p'] < 0.05
    lines.append(f'- Phương án chính `{PRIMARY}`: {s["mean"]:.4f} so với v3 {stats["v3"]["mean"]:.4f} '
                 f'({100 * s["rel"]:+.2f}%, {s["n_better"]}/{n} run tốt hơn, p = {_fmt_p(s["p"])}) -> '
                 + ('cải thiện có ý nghĩa thống kê (TB thấp hơn v3 và p < 0,05).' if better else
                    'KHÔNG đủ điều kiện tuyên bố cải thiện (cần TB thấp hơn v3 và p < 0,05).'))
    lines.append(f'- Tỷ lệ bước Test mà công tắc an toàn dùng RL-Gate: '
                 f'{np.mean([x["rl_gate_refit_safe_use_rl"] for x in R]):.1%} (refit), '
                 f'{np.mean([x["rl_gate_v3_safe_use_rl"] for x in R]):.1%} (tham chiếu, dự báo gốc).')
    if any(x['quick_check'] for x in R):
        lines.append('- **CẢNH BÁO: có kết quả quick_check (2 epoch / 20 cây), chỉ để kiểm tra pipeline.**')
    return '\n'.join(lines), stats, {'v3_reproduced': ok, 'primary_improved': bool(better)}


def write_summary_all(out_dir):
    parts, verdict = ['# ST-Adaptive-Ensemble-RL trên nhánh refit (Train+Val) — tổng hợp 3 dataset', '',
                      f'Phương án chính chốt trước khi xem Test: `{PRIMARY}`. Các biến thể còn lại chỉ để phân tích. '
                      'Chỉ tuyên bố cải thiện khi TB 10 run thấp hơn v3 và p (t ghép cặp) < 0,05.', ''], []
    for ds in DATASETS:
        f = os.path.join(out_dir, f'{ds}_runs.json')
        if not os.path.exists(f):
            parts += [f'### {ds.upper()}: chưa có kết quả', '']
            continue
        with open(f, encoding='utf-8') as fh:
            J = json.load(fh)
        table, stats, flags = summarize_dataset(ds, J['rows'], J['hedge_cfg'])
        parts += [table, '']
        verdict.append(f'| {ds.upper()} | {stats["v3"]["mean"]:.4f} | {stats[PRIMARY]["mean"]:.4f} | '
                       f'{100 * stats[PRIMARY]["rel"]:+.2f}% | {stats[PRIMARY]["n_better"]}/{len(J["rows"])} | '
                       f'{_fmt_p(stats[PRIMARY]["p"])} | {"có" if flags["primary_improved"] else "không"} |')
    head = ['## Kết luận phương án chính', '',
            f'| Dataset | v3 | {PRIMARY} | so với v3 | run tốt hơn | p | Cải thiện có ý nghĩa |', '|---|---|---|---|---|---|---|']
    text = '\n'.join(parts[:4] + head + verdict + [''] + parts[4:])
    with open(os.path.join(out_dir, 'summary_all.md'), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    return text


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=root)
    ap.add_argument('--refit_root', default=os.path.join(root, 'logs_refit'))
    ap.add_argument('--datasets', default='sdn,geant,abilene')
    ap.add_argument('--runs', default='0-9')
    ap.add_argument('--margin', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--allow_partial', action='store_true', help='đánh giá cả khi chưa đủ mọi run (kiểm tra nhanh)')
    ap.add_argument('--summary_all', action='store_true', help='chỉ gộp {ds}_runs.json đã có thành summary_all.md')
    ap.add_argument('--out', default=os.path.join(root, 'results', 'refit'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.summary_all:
        print(write_summary_all(args.out))
        return

    runs = parse_runs(args.runs)
    with Pool(args.workers) as pool:
        for ds in args.datasets.split(','):
            have = available_runs(args.refit_root, ds, runs)
            if have != runs and not args.allow_partial:
                raise SystemExit(f'{ds}: thiếu kết quả refit của run {sorted(set(runs) - set(have))}')
            if not have:
                print(f'{ds}: chưa có run nào')
                continue
            # (eta, decay) của Hedge chọn trên Val của đủ 10 run như run_rl_gate.py, không phụ thuộc số run đã refit
            hedge_cfg = select_hedge(args.repo, ds, list(range(10)), pool)
            rows = sorted(pool.map(eval_run, [(args.repo, args.refit_root, ds, r, hedge_cfg, args.margin) for r in have]),
                          key=lambda x: x['run'])
            table, stats, flags = summarize_dataset(ds, rows, hedge_cfg)
            with open(os.path.join(args.out, f'{ds}_runs.json'), 'w', encoding='utf-8') as f:
                json.dump({'dataset': ds, 'primary': PRIMARY, 'hedge_cfg': list(hedge_cfg), 'margin': args.margin,
                           'stats': stats, **flags, 'rows': rows}, f, indent=1)
            with open(os.path.join(args.out, f'{ds}_summary.md'), 'w', encoding='utf-8') as f:
                f.write(table + '\n')
            print(table, flush=True)


if __name__ == '__main__':
    main()
