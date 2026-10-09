"""
Huấn luyện OD-GraphFormer (spec Mục 4.3) theo số bước cố định - ngân sách 1 ngày.

  Loss = MSE(ŷ, y) + λ_nll * GaussianNLL(y; stopgrad(ŷ), σ̂²)
  AdamW, warmup + cosine theo max_steps, AMP, grad clip 5, batch 64 lấy mẫu ngẫu nhiên các cửa sổ Train.
  Đánh giá Val mỗi eval_every bước, giữ checkpoint tốt nhất, dừng khi patience_evals lần không cải thiện
  hoặc chạm --max_minutes.

Artifact: logs/gnn_rl/{tag}_data_{ds}_seq_{L}/run_{r}/
  best_model.pth, config.json, train_log.csv, test_metrics.csv,
  pred_val.npy, sigma_val.npy, y_pred_data.npy (Test), sigma_test.npy, y_real_data.npy

Chế độ OOF (--train_frac 0.6, do build_oof.py gọi): học trên 60% đầu Train (10% cuối phần này để dừng sớm),
dự báo 40% cuối Train -> pred_oof.npy, sigma_oof.npy, y_oof.npy.

Đo tốc độ: --benchmark 50 chạy 50 bước, ghi results/gnn_rl/benchmark_odgf_{ds}.json (s/bước, max_steps đề xuất).
"""
import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.gnn_rl_common import (setup_utf8, parse_list, ALL_DATASETS, RESULTS_V4, ODGF_MAX_STEPS,
                                    ODGF_MAX_MINUTES, load_splits, comb_array, gather_windows, run_dir, save_json,
                                    load_json, set_seed, get_device, oof_ranges, huber)
from features.feature_store import DATASET_CONFIGS
from features.graph_builder import build_flow_graphs, resolve_topology, node_ids
from Graph_models.od_graphformer import ODGraphFormer, build_odgf_from_config
from baselines_ml.metrics import calc_metrics_numpy
from baselines_ml.run_ml_baselines import parse_run_ids

TAG = 'odgraphformer'
PATCH_LEN = {'sdn': 10, 'geant': 6, 'abilene': 6}
TARGET_MINUTES = {'sdn': 12, 'geant': 35, 'abilene': 18}  # mục tiêu thiết kế / run (vòng 2, spec Mục 15)


def default_config(ds_key, args):
    cfg = DATASET_CONFIGS[ds_key]
    return {
        'dataset': ds_key, 'seq_len': cfg['seq_len'], 'num_flows': cfg['flows'],
        'd_model': args.d_model, 'n_temporal': args.layers, 'n_spatial': args.layers, 'nhead': 4,
        'dropout': 0.1, 'patch_len': args.patch_len or PATCH_LEN[ds_key], 'patch_stride': None, 'eps_s': 1e-3,
        'anchor': 'mean' if args.no_residual else args.anchor,
        'use_identity': not args.no_identity, 'spatial_attn': not args.no_spatial_attn,
        'use_adp': 'adp' in args.graphs, 'fixed_graphs': [g for g in ('route', 'od') if g in args.graphs],
        'lambda_nll': args.lambda_nll, 'lr': args.lr, 'weight_decay': 1e-4, 'batch_size': args.batch_size,
        'max_steps': args.max_steps or ODGF_MAX_STEPS[ds_key], 'warmup': args.warmup,
        'eval_every': args.eval_every, 'patience_evals': args.patience_evals,
        'max_minutes': args.max_minutes or ODGF_MAX_MINUTES[ds_key], 'amp': not args.no_amp,
        'topology': resolve_topology(ds_key, args.topology), 'train_frac': args.train_frac,
    }


def choose_anchor(comb_val, L):
    """
    Mốc neo của OD-GraphFormer theo Val (spec Mục 15): so Huber (delta 0,05) của hai dự báo không cần học trên Val,
    'last' = x_{t-1} (Persistence) và 'mean' = trung bình L bước của cửa sổ. Mốc nào sai số nhỏ hơn được chọn.
    Dùng Huber thay MSE vì MSE trên Val của GÉANT bị vài đỉnh chi phối (MSE chọn 'mean', Huber chọn 'last');
    với Huber, kết quả trùng với Val của các mô hình OD-GraphFormer đã huấn luyện ở vòng 1 trên cả 3 dataset.
    """
    x = comb_val[..., 0].double().cpu().numpy()
    T = x.shape[0]
    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)], axis=0)
    t = np.arange(L, T)
    y = x[t]
    e_last, e_mean = x[t - 1] - y, (cs[t] - cs[t - L]) / L - y
    info = {'mse_last': float(np.mean(e_last ** 2)), 'mse_mean': float(np.mean(e_mean ** 2)),
            'huber_last': huber(e_last), 'huber_mean': huber(e_mean), 'criterion': 'huber'}
    info['auto'] = 'last' if info['huber_last'] <= info['huber_mean'] else 'mean'
    return info


def resolve_anchor(ds_key, cfg, comb_val):
    """'auto' -> mốc tốt hơn trên Val; 'other' -> mốc còn lại (ablation); 'last' / 'mean' giữ nguyên."""
    info = choose_anchor(comb_val, cfg['seq_len'])
    a = cfg.get('anchor', 'auto')
    chosen = info['auto'] if a == 'auto' else ({'last': 'mean', 'mean': 'last'}[info['auto']] if a == 'other' else a)
    cfg.update({'anchor': chosen, 'anchor_info': info, 'residual': chosen == 'last'})
    save_json({'dataset': ds_key, **info}, os.path.join(RESULTS_V4, f'odgf_anchor_{ds_key}.json'))
    return chosen


def prepare(ds_key, cfg, device):
    sp = load_splits(ds_key)
    pairs, label2idx = node_ids(sp['columns'])
    g = build_flow_graphs(ds_key, sp['columns'], cfg['topology'])
    cfg.update({'src_idx': [label2idx[s] for s, _ in pairs], 'dst_idx': [label2idx[d] for _, d in pairs],
                'num_nodes': len(label2idx), 'steps_per_day': sp['steps_per_day'], 'graph_meta': g['meta']})
    graphs = {'route': g['A_route'], 'od': g['A_od']}
    combs = {k: torch.from_numpy(comb_array(sp[k])).to(device) for k in ('train', 'val', 'test')}
    resolve_anchor(ds_key, cfg, combs['val'])
    return combs, graphs


def predict(model, comb, targets, L, device, amp, bs=128):
    model.eval()
    ys, lvs, times = [], [], []
    with torch.no_grad():
        for i in range(0, len(targets), bs):
            idx = torch.as_tensor(targets[i:i + bs], device=device)
            x, _ = gather_windows(comb, idx, L)
            t0 = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda'):
                y, lv = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0 * 64.0 / len(idx))
            ys.append(y.float().cpu())
            lvs.append(lv.float().cpu())
    y = torch.cat(ys).clamp_min(0.0).numpy()
    sigma = torch.exp(0.5 * torch.cat(lvs).clamp(-30, 10)).numpy()
    return y.astype(np.float32), sigma.astype(np.float32), float(np.mean(times)) if times else float('nan')


def lr_lambda(warmup, max_steps):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, max_steps - warmup))))
    return f


def train_one(ds_key, run_id, cfg, combs, graphs, out_dir, device, quick_check=False, benchmark=0):
    L = cfg['seq_len']
    set_seed(42 + run_id)
    model = build_odgf_from_config(cfg, graphs).to(device)
    T_tr = combs['train'].shape[0]
    if cfg['train_frac'] < 1.0:
        fit_t, es_t, oof_t = oof_ranges(T_tr, L, cfg['train_frac'])
        es_comb = combs['train']
    else:
        fit_t, es_t, es_comb = np.arange(L, T_tr), np.arange(L, combs['val'].shape[0]), combs['val']
        oof_t = None
    max_steps = cfg['max_steps']
    if quick_check:
        fit_t, es_t, max_steps = fit_t[:256], es_t[:64], 20
        cfg = {**cfg, 'eval_every': 10, 'max_steps': max_steps}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg['warmup'] if not quick_check else 2, max_steps))
    amp = cfg['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    gen = torch.Generator().manual_seed(42 + run_id)
    fit_t_dev = torch.as_tensor(fit_t, device=device)

    def step_fn():
        model.train()
        pick = torch.randint(0, len(fit_t), (cfg['batch_size'],), generator=gen).to(device)
        x, y = gather_windows(combs['train'], fit_t_dev[pick], L)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            y_hat, lv = model(x)
        y_hat, lv = y_hat.float(), lv.float().clamp(-30, 10)
        mse = ((y_hat - y) ** 2).mean()
        nll = 0.5 * (lv + (y - y_hat.detach()) ** 2 / torch.exp(lv)).mean()
        loss = mse + cfg['lambda_nll'] * nll
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        return float(mse.detach()), float(nll.detach())

    if benchmark:
        for _ in range(5):
            step_fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(benchmark):
            step_fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        sps = (time.perf_counter() - t0) / benchmark
        t1 = time.perf_counter()
        predict(model, es_comb, es_t, L, device, amp)
        eval_s = time.perf_counter() - t1
        n_eval = max_steps // cfg['eval_every']
        proj = (sps * max_steps + eval_s * n_eval) / 60.0
        target = TARGET_MINUTES[ds_key]
        rec = int(max(250, min(max_steps, (target * 60 - eval_s * n_eval) / sps)) // 250 * 250)
        info = {'dataset': ds_key, 'device': str(device), 's_per_step': sps, 'eval_s': eval_s,
                'max_steps': max_steps, 'projected_minutes': proj, 'target_minutes': target,
                'recommended_max_steps': rec, 'n_params': sum(p.numel() for p in model.parameters())}
        save_json(info, os.path.join(RESULTS_V4, f'benchmark_odgf_{ds_key}.json'))
        print(f"[BENCHMARK {ds_key.upper()}] {sps*1000:.0f} ms/bước, eval {eval_s:.1f}s, dự kiến {proj:.1f} phút "
              f"cho {max_steps} bước (mục tiêu {target}) -> max_steps đề xuất {rec}", flush=True)
        return info

    best, best_state, wait, hist = float('inf'), None, 0, []
    t_start, capped, step = time.time(), False, 0
    run_mse = run_nll = 0.0
    for step in range(1, max_steps + 1):
        mse, nll = step_fn()
        run_mse += mse
        run_nll += nll
        if step % cfg['eval_every'] == 0 or step == max_steps:
            pv, _, _ = predict(model, es_comb, es_t, L, device, amp)
            yv = es_comb[torch.as_tensor(es_t, device=device), :, 0].cpu().numpy()
            v = float(np.mean((pv - yv) ** 2))
            n = cfg['eval_every'] if step % cfg['eval_every'] == 0 else step % cfg['eval_every']
            hist.append({'step': step, 'train_mse': run_mse / n, 'train_nll': run_nll / n, 'val_mse': v,
                         'lr': sched.get_last_lr()[0], 'minutes': (time.time() - t_start) / 60})
            run_mse = run_nll = 0.0
            mark = ''
            if v < best:
                best, wait, mark = v, 0, ' *'
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
            print(f"    step {step:5d}/{max_steps} | train MSE {hist[-1]['train_mse']*1e3:.4f}e-3 | "
                  f"val MSE {v*1e3:.4f}e-3{mark} | {hist[-1]['minutes']:.1f} phút", flush=True)
            if wait >= cfg['patience_evals']:
                break
        if (time.time() - t_start) / 60 > cfg['max_minutes']:
            capped = True
            print(f"    [CẮT THỜI GIAN] dừng ở bước {step} sau {cfg['max_minutes']} phút", flush=True)
            break
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, 'best_model.pth'))
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, 'train_log.csv'), index=False)
    best_step = int(min(hist, key=lambda h: h['val_mse'])['step']) if hist else 0
    meta = {'seed': 42 + run_id, 'run': run_id, 'steps_done': step, 'best_step': best_step,
            'best_es_mse': best, 'minutes': (time.time() - t_start) / 60, 'time_capped': capped}
    save_json({**cfg, **meta}, os.path.join(out_dir, 'config.json'))

    if oof_t is not None:
        po, so, _ = predict(model, combs['train'], oof_t, L, device, amp)
        np.save(os.path.join(out_dir, 'pred_oof.npy'), po)
        np.save(os.path.join(out_dir, 'sigma_oof.npy'), so)
        np.save(os.path.join(out_dir, 'y_oof.npy'), combs['train'][torch.as_tensor(oof_t, device=device), :, 0].cpu().numpy())
        print(f"    [OOF] {len(oof_t)} bước, MSE={np.mean((po - np.load(os.path.join(out_dir, 'y_oof.npy')))**2)*1e3:.4f}e-3", flush=True)
        return meta

    val_t = np.arange(L, combs['val'].shape[0])
    test_t = np.arange(L, combs['test'].shape[0])
    pv, sv, _ = predict(model, combs['val'], val_t, L, device, amp)
    pt, stt, inf_ms = predict(model, combs['test'], test_t, L, device, amp)
    yv = combs['val'][torch.as_tensor(val_t, device=device), :, 0].cpu().numpy()
    yt = combs['test'][torch.as_tensor(test_t, device=device), :, 0].cpu().numpy()
    for name, arr in (('pred_val', pv), ('sigma_val', sv), ('y_pred_data', pt), ('sigma_test', stt),
                      ('y_real_data', yt)):
        np.save(os.path.join(out_dir, f'{name}.npy'), arr)
    m = calc_metrics_numpy(pt, yt)
    m.update({**meta, 'val_mse': float(np.mean((pv - yv) ** 2)), 'inference_time_ms': inf_ms})
    pd.DataFrame([m]).to_csv(os.path.join(out_dir, 'test_metrics.csv'), index=False)
    print(f"[DONE] {cfg.get('tag', TAG)} {ds_key.upper()} run_{run_id} | {step} bước (tốt nhất {best_step}) | "
          f"Val MSE={m['val_mse']*1e3:.3f}e-3 | Test MSE={m['mse']*1e3:.3f}e-3 | {meta['minutes']:.1f} phút", flush=True)
    return m


def run(args):
    device = get_device()
    datasets = parse_list(args.datasets, ALL_DATASETS)
    for ds in datasets:
        cfg = default_config(ds, args)
        cfg['tag'] = args.tag
        combs, graphs = prepare(ds, cfg, device)
        print(f"\n[{args.tag} {ds.upper()}] device={device} | topology={cfg['topology']} "
              f"({cfg['graph_meta']['note']}) | đồ thị {cfg['fixed_graphs'] + (['adp'] if cfg['use_adp'] else [])} | "
              f"max_steps={cfg['max_steps']} | max_minutes={cfg['max_minutes']} | mốc neo {cfg['anchor']} "
              f"(Val Huber last={cfg['anchor_info']['huber_last']*1e3:.4f}e-3, "
              f"mean={cfg['anchor_info']['huber_mean']*1e3:.4f}e-3)", flush=True)
        if args.benchmark:
            train_one(ds, 0, cfg, combs, graphs, None, device, benchmark=args.benchmark)
            continue
        for r in parse_run_ids(args.run_ids):
            out = args.out_dir or run_dir(args.tag, ds, r)
            done = os.path.join(out, 'pred_oof.npy' if args.train_frac < 1 else 'test_metrics.csv')
            if args.skip_existing and os.path.exists(done):
                print(f"  [SKIP] {out}", flush=True)
                continue
            train_one(ds, r, cfg, combs, graphs, out, device, quick_check=args.quick_check)
        del combs
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def load_trained(out_dir, device=None):
    """Nạp lại mô hình đã huấn luyện từ thư mục log (config.json + best_model.pth)."""
    from features.graph_builder import build_flow_graphs
    from features.feature_store import load_raw_dataset
    device = device or get_device()
    cfg = load_json(os.path.join(out_dir, 'config.json'))
    g = build_flow_graphs(cfg['dataset'], list(load_raw_dataset(cfg['dataset']).columns), cfg['topology'])
    model = build_odgf_from_config(cfg, {'route': g['A_route'], 'od': g['A_od']})
    with open(os.path.join(out_dir, 'best_model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    return model.to(device).eval(), cfg


def build_parser():
    ap = argparse.ArgumentParser(description="Huấn luyện OD-GraphFormer (V4)")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_ids', default='0-2')
    ap.add_argument('--tag', default=TAG, help="Tên thư mục log, ví dụ odgraphformer_anchor_other cho ablation")
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--max_minutes', type=float, default=None)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--patience_evals', type=int, default=6)
    ap.add_argument('--lambda_nll', type=float, default=0.1)
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--patch_len', type=int, default=None)
    ap.add_argument('--topology', default='auto', choices=['auto', 'physical', 'knn'])
    ap.add_argument('--graphs', default='route,od,adp', help="Tập đồ thị dùng, ví dụ 'od,adp' (bỏ A_route)")
    ap.add_argument('--anchor', default='auto', choices=['auto', 'last', 'mean', 'other'],
                    help="Mốc neo: auto = chọn theo Val; other = mốc còn lại (ablation)")
    ap.add_argument('--no_residual', action='store_true', help="Tương đương --anchor mean (giữ để tương thích)")
    ap.add_argument('--no_identity', action='store_true')
    ap.add_argument('--no_spatial_attn', action='store_true')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--train_frac', type=float, default=1.0, help="< 1: chế độ OOF")
    ap.add_argument('--out_dir', default=None, help="Ghi đè thư mục log (dùng cho OOF)")
    ap.add_argument('--benchmark', type=int, default=0, help="Số bước đo tốc độ (0 = huấn luyện bình thường)")
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--quick_check', action='store_true')
    return ap


if __name__ == '__main__':
    setup_utf8()
    a = build_parser().parse_args()
    a.graphs = parse_list(a.graphs, ['route', 'od', 'adp'])
    run(a)
