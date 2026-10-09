"""
Điều phối toàn bộ hướng V4 (P0 -> P5) cho một nhóm dataset, theo hàng đợi ưu tiên (spec Mục 10.2).

  1. Benchmark tốc độ OD-GraphFormer -> chỉnh max_steps cho vừa mục tiêu thời gian
  2. OD-GraphFormer theo --run_ids (giai đoạn hiện tại: 3 run, seed 42-44)
  3. OOF (OD-GraphFormer trên GPU, GBDT trên CPU song song) -> cache v4 -> quyết định nhánh
  4. RL: tiền huấn luyện + lưới run_0 -> SAC từng run, gate giám sát; static/Hedge/cổng MLP (CPU)
  5. Bảng chính P4
  6. Ablation (tùy chọn, bị bỏ khi đã dùng quá 75% ngân sách thời gian), P5, báo cáo đầy đủ

Mỗi GPU có 2 "đơn vị": job OD-GraphFormer chiếm 2, job RL chiếm 1 (2 tiến trình RL / GPU).
Không có GPU (chạy thử trên CPU): mọi job chạy tuần tự trên CPU.

Ví dụ:
  python training/run_gnn_rl.py --datasets geant --time_budget_h 10
  python training/run_gnn_rl.py --datasets sdn --run_ids 0 --quick_check      # chạy thử trên CPU
"""
import os
import sys
import time
import json
import argparse
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.gnn_rl_common import setup_utf8, parse_list, ALL_DATASETS, RESULTS_V4, ODGF_MAX_STEPS

V4_BRANCHES = 'stwaveformer,xgboost,lightgbm_res,odgraphformer'
ODGF_ABLATIONS = {'odgraphformer_no_residual': ['--no_residual'],
                  'odgraphformer_no_route': ['--graphs', 'od,adp'],
                  'odgraphformer_no_spatial_attn': ['--no_spatial_attn']}
RL_ABLATIONS = ['no_gnn', 'no_floor', 'no_oof']
UNITS = {'gpu': 2, 'gpu_small': 1, 'cpu': 0}


class Job:
    def __init__(self, name, cmd, kind='cpu', deps=(), prio=0, optional=False):
        self.name, self.cmd, self.kind, self.deps, self.prio, self.optional = name, cmd, kind, list(deps), prio, optional
        self.state, self.proc, self.t0, self.t1, self.gpu, self.tries, self.log = 'wait', None, None, None, None, 0, None


def _hm(s):
    m = int(s // 60)
    return f"{m // 60}h{m % 60:02d}m"


class Scheduler:
    def __init__(self, jobs, n_gpus, cpu_slots=2, time_budget_h=10.0, log_dir=None, on_tick=None, tick_s=20,
                 status_every_s=60, max_retry=1):
        self.jobs = {j.name: j for j in jobs}
        self.n_gpus, self.cpu_slots = n_gpus, cpu_slots
        self.budget_s = time_budget_h * 3600
        self.log_dir = log_dir or os.path.join(parent_dir, 'logs', 'gnn_rl', '_jobs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.on_tick, self.tick_s, self.status_every_s, self.max_retry = on_tick, tick_s, status_every_s, max_retry
        self.t_start = time.time()
        self.events = []

    def _gpu_free(self):
        used = {g: 0 for g in range(self.n_gpus)}
        for j in self.jobs.values():
            if j.state == 'run' and j.gpu is not None:
                used[j.gpu] += UNITS[j.kind]
        return {g: 2 - u for g, u in used.items()}

    def _ready(self, j):
        for d in j.deps:
            dj = self.jobs.get(d)
            if dj is None:
                continue
            if dj.state in ('wait', 'run'):
                return False
            if dj.state == 'fail' and not dj.optional:
                return None                                   # phụ thuộc bắt buộc lỗi -> bỏ job này
        return True

    def _launch(self, j, gpu):
        cmd = j.cmd() if callable(j.cmd) else j.cmd
        j.log = os.path.join(self.log_dir, f"{j.name}.log")
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8',
               'CUDA_VISIBLE_DEVICES': '' if gpu is None else str(gpu)}
        if j.kind == 'cpu':
            env['OMP_NUM_THREADS'] = env.get('OMP_NUM_THREADS', '4')
        f = open(j.log, 'a', encoding='utf-8')
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        j.proc = subprocess.Popen(cmd, cwd=parent_dir, env=env, stdout=f, stderr=subprocess.STDOUT)
        j.state, j.t0, j.gpu, j.tries = 'run', time.time(), gpu, j.tries + 1
        self.events.append(f"{time.strftime('%H:%M')} bắt đầu {j.name}" + (f" (GPU{gpu})" if gpu is not None else ''))

    def _schedule(self):
        elapsed = time.time() - self.t_start
        running_cpu = sum(1 for j in self.jobs.values() if j.state == 'run' and (j.kind == 'cpu' or self.n_gpus == 0))
        cand = sorted((j for j in self.jobs.values() if j.state == 'wait'), key=lambda j: (j.prio, j.name))
        for j in cand:
            r = self._ready(j)
            if r is None:
                j.state = 'skip'
                self.events.append(f"{time.strftime('%H:%M')} bỏ {j.name} (phụ thuộc lỗi)")
                continue
            if not r:
                continue
            if j.optional and elapsed > 0.75 * self.budget_s:
                j.state = 'skip'
                self.events.append(f"{time.strftime('%H:%M')} bỏ {j.name} (đã dùng > 75% ngân sách)")
                continue
            if self.n_gpus == 0 or j.kind == 'cpu':
                if running_cpu < (self.cpu_slots if self.n_gpus else 1):
                    self._launch(j, None)
                    running_cpu += 1
                continue
            free = self._gpu_free()
            g = max(free, key=lambda k: free[k])
            if free[g] >= UNITS[j.kind]:
                self._launch(j, g)

    def _poll(self):
        for j in self.jobs.values():
            if j.state != 'run':
                continue
            rc = j.proc.poll()
            if rc is None:
                continue
            j.t1 = time.time()
            if rc == 0:
                j.state = 'done'
                self.events.append(f"{time.strftime('%H:%M')} xong {j.name} ({_hm(j.t1 - j.t0)})")
            elif j.tries <= self.max_retry:
                j.state = 'wait'
                self.events.append(f"{time.strftime('%H:%M')} LỖI {j.name} (exit {rc}), thử lại")
            else:
                j.state = 'fail'
                self.events.append(f"{time.strftime('%H:%M')} LỖI {j.name} (exit {rc}), xem {j.log}")

    def status(self):
        el = time.time() - self.t_start
        cnt = {}
        for j in self.jobs.values():
            cnt[j.state] = cnt.get(j.state, 0) + 1
        lines = [f"[{_hm(el)} / {_hm(self.budget_s)}] " + " | ".join(f"{k} {v}" for k, v in sorted(cnt.items()))]
        for j in self.jobs.values():
            if j.state == 'run':
                lines.append(f"  chạy  {j.name:32s} {'GPU' + str(j.gpu) if j.gpu is not None else 'CPU':5s} "
                             f"{_hm(time.time() - j.t0)}")
        lines += [f"  {e}" for e in self.events[-8:]]
        print("\n".join(lines), flush=True)

    def run(self):
        last_status = 0.0
        while True:
            self._poll()
            self._schedule()
            if self.on_tick:
                try:
                    self.on_tick(self)
                except Exception as e:
                    print(f"[CẢNH BÁO] on_tick lỗi: {e}", flush=True)
            if time.time() - last_status >= self.status_every_s:
                self.status()
                last_status = time.time()
            if all(j.state in ('done', 'fail', 'skip') for j in self.jobs.values()):
                break
            time.sleep(self.tick_s)
        self.status()
        summary = {n: {'state': j.state, 'minutes': round((j.t1 - j.t0) / 60, 1) if j.t1 and j.t0 else None}
                   for n, j in self.jobs.items()}
        failed = [n for n, j in self.jobs.items() if j.state == 'fail' and not j.optional]
        return summary, failed


def _bench_steps(ds, frac=1.0):
    f = os.path.join(RESULTS_V4, f'benchmark_odgf_{ds}.json')
    steps = ODGF_MAX_STEPS[ds]
    if os.path.exists(f):
        steps = min(steps, json.load(open(f, encoding='utf-8'))['recommended_max_steps'])
    return max(100, int(steps * frac))


def build_jobs(datasets, run_ids, abl_run_ids, quick_check=False, with_p0=True, with_ablation=True):
    py = sys.executable
    qc = ['--quick_check'] if quick_check else []
    R = run_ids
    jobs, main_names = [], []

    def add(*a, **k):
        j = Job(*a, **k)
        jobs.append(j)
        if not j.optional:
            main_names.append(j.name)
        return j

    rid = _parse(run_ids)
    for ds in datasets:
        T = lambda s: f"{s}_{ds}"
        if with_p0:
            add(T('p0_v3'), [py, 'training/reproduce_v3.py', '--datasets', ds, '--run_ids', R], 'cpu', prio=3)
            add(T('p0_hedge'), [py, 'training/run_hedge.py', '--cache_version', 'v3', '--tag_suffix', '_v3',
                                '--methods', 'hedge,hedge_floor', '--datasets', ds, '--run_ids', R] + qc, 'cpu', prio=3)
            add(T('p0_headroom'), [py, 'evaluation/headroom.py', '--datasets', ds, '--run_ids', R], 'cpu', prio=3)
        # P1
        odgf_deps = []
        if not quick_check:
            add(T('bench'), [py, 'training/train_od_graphformer.py', '--datasets', ds, '--benchmark', '50'], 'gpu', prio=0)
            odgf_deps = [T('bench')]
        for r in rid:
            add(T(f'odgf_r{r}'), (lambda ds=ds, r=r: [py, 'training/train_od_graphformer.py', '--datasets', ds,
                                                      '--run_ids', str(r), '--max_steps', str(_bench_steps(ds)),
                                                      '--skip_existing'] + qc), 'gpu', odgf_deps, prio=1)
        add(T('cache'), [py, 'training/precompute_cache.py', '--cache_version', 'v4', '--datasets', ds, '--run_ids', R,
                         '--branches', V4_BRANCHES, '--skip_existing'], 'cpu', [T(f'odgf_r{r}') for r in rid], prio=1)
        add(T('decision'), [py, 'training/p1_decision.py', '--datasets', ds, '--run_ids', R]
            + (['--force', 'odgraphformer'] if quick_check else []), 'cpu', [T('cache')], prio=1)
        add(T('static_k4'), [py, 'training/run_hedge.py', '--branches', V4_BRANCHES, '--methods', 'static',
                             '--static_tag', 'static_k4', '--datasets', ds, '--run_ids', R] + qc, 'cpu', [T('cache')], prio=2)
        # P2
        add(T('oof_odgf'), (lambda ds=ds: [py, 'training/build_oof.py', '--datasets', ds, '--branches', 'odgraphformer',
                                           '--max_steps', str(_bench_steps(ds, 0.6)), '--skip_existing'] + qc),
            'gpu', odgf_deps, prio=2)
        add(T('oof_gbdt'), [py, 'training/build_oof.py', '--datasets', ds, '--branches', 'xgboost,lightgbm_res',
                            '--skip_existing'] + qc, 'cpu', prio=1)
        add(T('oof'), [py, 'training/build_oof.py', '--datasets', ds, '--skip_existing'] + qc, 'cpu',
            [T('oof_odgf'), T('oof_gbdt'), T('cache')], prio=1)
        # P3
        add(T('combiners'), [py, 'training/run_hedge.py', '--static_tag', 'static_k3', '--datasets', ds, '--run_ids', R]
            + qc, 'cpu', [T('decision')], prio=2)
        add(T('rl_grid'), [py, 'training/train_rl_gate.py', '--algo', 'sac', '--phase', 'grid', '--datasets', ds,
                           '--skip_existing'] + qc, 'gpu_small', [T('decision'), T('oof')], prio=1)
        chunks = [rid[i:i + 5] for i in range(0, len(rid), 5)]
        for i, ch in enumerate(chunks):
            add(T(f'rl_runs{i}'), [py, 'training/train_rl_gate.py', '--algo', 'sac', '--phase', 'runs', '--datasets', ds,
                                   '--run_ids', ','.join(map(str, ch)), '--skip_existing'] + qc, 'gpu_small',
                [T('rl_grid')], prio=1)
        add(T('rl_sup'), [py, 'training/train_rl_gate.py', '--algo', 'sup', '--datasets', ds, '--run_ids', R,
                          '--skip_existing'] + qc, 'gpu_small', [T('rl_grid')], prio=2)
        # Ablation (tùy chọn)
        if with_ablation:
            for tag, flags in ODGF_ABLATIONS.items():
                for r in _parse(abl_run_ids):
                    add(T(f'abl_{tag}_r{r}'), (lambda ds=ds, r=r, tag=tag, flags=flags: [
                        py, 'training/train_od_graphformer.py', '--datasets', ds, '--run_ids', str(r), '--tag', tag,
                        '--max_steps', str(_bench_steps(ds)), '--skip_existing'] + flags + qc),
                        'gpu', odgf_deps, prio=5, optional=True)
            for v in RL_ABLATIONS:
                add(T(f'abl_rl_{v}'), [py, 'training/train_rl_gate.py', '--algo', 'sac', '--variant', v, '--phase', 'runs',
                                       '--datasets', ds, '--run_ids', abl_run_ids, '--skip_existing'] + qc,
                    'gpu_small', [T('rl_grid'), T('oof')], prio=6, optional=True)
            add(T('p5'), [py, 'evaluation/plot_weight_trajectories.py', '--datasets', ds], 'cpu',
                [T(f'rl_runs{i}') for i in range(len(chunks))], prio=4, optional=True)
    ds_arg = ','.join(datasets)
    add('report', [py, 'evaluation/report_gnn_rl.py', '--datasets', ds_arg, '--run_ids', R], 'cpu',
        list(main_names), prio=4)
    add('report_full', [py, 'evaluation/report_gnn_rl.py', '--datasets', ds_arg, '--run_ids', R], 'cpu',
        [j.name for j in jobs], prio=9, optional=True)
    return jobs


def _parse(s):
    from baselines_ml.run_ml_baselines import parse_run_ids
    return parse_run_ids(s)


def main(args):
    import torch
    datasets = parse_list(args.datasets, ALL_DATASETS)
    n_gpus = 0 if args.cpu_only else torch.cuda.device_count()
    jobs = build_jobs(datasets, args.run_ids, args.ablation_run_ids, args.quick_check,
                      with_p0=not args.no_p0, with_ablation=not args.no_ablation)
    print(f"[V4] datasets={datasets} | run_ids={args.run_ids} | ablation run_ids={args.ablation_run_ids} | "
          f"GPU={n_gpus} | {len(jobs)} job | ngân sách {args.time_budget_h} giờ", flush=True)
    sch = Scheduler(jobs, n_gpus, args.cpu_slots, args.time_budget_h, status_every_s=args.status_every_s,
                    tick_s=2 if args.quick_check else 20)
    summary, failed = sch.run()
    os.makedirs(RESULTS_V4, exist_ok=True)
    with open(os.path.join(RESULTS_V4, f"jobs_{'_'.join(datasets)}.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if failed:
        print(f"[LỖI] job bắt buộc thất bại: {failed}", flush=True)
        sys.exit(1)


def build_parser():
    ap = argparse.ArgumentParser(description="Điều phối hướng V4 (OD-GraphFormer + RL-Gate)")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--run_ids', default='0-2', help="Giai đoạn hiện tại: 3 run; '0-9' để chạy đủ 10 run")
    ap.add_argument('--ablation_run_ids', default='0-2')
    ap.add_argument('--time_budget_h', type=float, default=10.0)
    ap.add_argument('--cpu_slots', type=int, default=2)
    ap.add_argument('--status_every_s', type=int, default=60)
    ap.add_argument('--no_p0', action='store_true')
    ap.add_argument('--no_ablation', action='store_true')
    ap.add_argument('--cpu_only', action='store_true')
    ap.add_argument('--quick_check', action='store_true')
    return ap


if __name__ == '__main__':
    setup_utf8()
    main(build_parser().parse_args())
