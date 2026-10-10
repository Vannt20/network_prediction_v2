"""
Chạy hướng V4 trên Kaggle với 2 tài khoản song song (spec Mục 0.3, 10.3), không đụng tới main.

  Tài khoản A: GÉANT            -> push lên nhánh GNN_RL_{vòng}_a   (vòng 1: GNN_RL_a)
  Tài khoản B: SDN + Abilene    -> push lên nhánh GNN_RL_{vòng}_b
Cả hai clone từ nhánh mã GNN_RL. Tài khoản xong sau gộp kết quả của tài khoản kia, dựng lại cache V4
cho các dataset đó (chỉ đọc log, không huấn luyện) rồi lập báo cáo chung và push lên GNN_RL_{vòng}_results.
Mỗi vòng có nhánh, file trạng thái và thư mục log riêng (GNN_RL_ROUND, mặc định r2) để không lẫn với vòng trước.

Push ngay khi mỗi job xong (mỗi run OD-GraphFormer, mỗi nhóm run RL-Gate, OOF, báo cáo...), commit kèm MSE
nếu có; thêm push định kỳ 15 phút làm dự phòng. Thư mục push: logs/gnn_rl, results/gnn_rl, data/graphs, status.
Không push cache/v4 (dựng lại được từ cache/v3 + logs/gnn_rl) để tránh làm repo phình thêm vài GB.

Dùng trong notebook (sau khi clone và cd vào repo, `git` là hàm chạy git có xác thực như notebook phase 2):
    from training.kaggle_gnn_rl import KaggleV4
    KaggleV4('A', git).run()
"""
import os
import re
import sys
import glob
import json
import time
import threading
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from training.gnn_rl_common import run_dir, ROUND, LOGS_V4, RESULTS_V4, CACHE_VERSION_V4, V4_BRANCHES

PLAN = {'A': ['geant'], 'B': ['sdn', 'abilene']}
# Giai đoạn hiện tại chỉ chạy 3 run (seed 42-44); đổi thành '0-9' khi chạy đủ 10 run
DEFAULT_RUN_IDS = '0-2'
CODE_BRANCH = 'GNN_RL'
_RB = CODE_BRANCH if ROUND == 'r1' else f"{CODE_BRANCH}_{ROUND}"          # tiền tố nhánh kết quả của vòng
FINAL_BRANCH = f"{_RB}_results"
LOCK_BRANCH = f"{_RB}_final_lock"
_REL = lambda p: os.path.relpath(p, parent_dir).replace(os.sep, '/')
PUSH_PATHS = [_REL(LOGS_V4), _REL(RESULTS_V4), 'data/graphs', 'status']
STATUS = lambda acc: f"status/gnn_rl_{acc}.json" if ROUND == 'r1' else f"status/gnn_rl_{ROUND}_{acc}.json"
MAX_MB = 95


def _read_mse(f):
    try:
        import pandas as pd
        return float(pd.read_csv(f).iloc[0]['mse']) * 1e3
    except Exception:
        return None


class KaggleV4:
    def __init__(self, account, git, run_ids=DEFAULT_RUN_IDS, ablation_run_ids='0-2', time_budget_h=10.0,
                 push_every_s=900):
        assert account in PLAN
        self.account, self.git = account, git
        self.other = 'B' if account == 'A' else 'A'
        self.branch, self.other_branch = f"{_RB}_{account.lower()}", f"{_RB}_{self.other.lower()}"
        self.datasets = PLAN[account]
        self.run_ids, self.ablation_run_ids, self.budget = run_ids, ablation_run_ids, time_budget_h
        self.push_every_s, self.last_push = push_every_s, time.time()
        self.lock = threading.Lock()
        self.wd = parent_dir
        self.sch, self.pushed = None, 0
        self.pushed_jobs = set()
        cur = self.git('rev-parse', '--abbrev-ref', 'HEAD', cwd=self.wd, quiet=True).stdout.strip()
        assert cur != 'main', "Không chạy V4 trên nhánh main"
        self.git('checkout', '-B', self.branch, cwd=self.wd, quiet=True)

    # ---------- git ----------
    def _note(self, msg):
        """Ghi vào danh sách sự kiện của bảng tiến độ (khi đang chạy) thay vì in xen vào output."""
        if self.sch is not None:
            self.sch.events.append(f"{time.strftime('%H:%M')} {msg}")
        else:
            print(msg, flush=True)

    def push(self, msg):
        with self.lock:
            paths = [p for p in PUSH_PATHS if os.path.exists(os.path.join(self.wd, p))]
            big = [os.path.join(a, f) for p in paths for a, _, fs in os.walk(os.path.join(self.wd, p)) for f in fs
                   if os.path.getsize(os.path.join(a, f)) > MAX_MB * 1024 ** 2]
            for f in big:
                self._note(f"bỏ qua file > {MAX_MB} MB: {os.path.relpath(f, self.wd)}")
            self.git('add', '-A', *paths, cwd=self.wd, quiet=True)
            for f in big:
                self.git('reset', '-q', '--', os.path.relpath(f, self.wd), cwd=self.wd, quiet=True, check=False)
            if self.git('diff', '--cached', '--quiet', cwd=self.wd, check=False, quiet=True).returncode == 0:
                return False
            self.git('commit', '-q', '-m', f"[V4 {self.account}] {msg}", cwd=self.wd, quiet=True)
            for attempt in range(3):
                if self.git('push', '-u', 'origin', self.branch, cwd=self.wd, auth=True, check=False,
                            quiet=True).returncode == 0:
                    self.pushed += 1
                    self._note(f"push {self.branch} ({self.pushed}): {msg}")
                    return True
                time.sleep(15 * (attempt + 1))
            self._note("push lỗi, thử lại ở lần sau")
            return False

    def _job_msg(self, name):
        """Mô tả job cho commit, kèm MSE Test (x10^-3) khi đọc được."""
        m = re.match(r'(?:abl_(?P<tag>odgraphformer_\w+?)_|odgf_)r(?P<r>\d+)_(?P<ds>[a-z]+)$', name)
        if m:
            tag = m.group('tag') or 'odgraphformer'
            f = os.path.join(run_dir(tag, m.group('ds'), int(m.group('r'))), 'test_metrics.csv')
            mse = _read_mse(f)
            return f"{tag} {m.group('ds').upper()} run_{m.group('r')}" + (f" MSE={mse:.3f}e-3" if mse else '')
        m = re.match(r'rl_runs\d+_(?P<ds>[a-z]+)$', name)
        if m:
            ds = m.group('ds')
            got = []
            for d in sorted(glob.glob(os.path.join(run_dir('rl_sac', ds, None, seq=False), 'run_*'))):
                mse = _read_mse(os.path.join(d, 'test_metrics.csv'))
                if mse:
                    got.append(f"{os.path.basename(d)}={mse:.3f}")
            return f"RL-Gate {ds.upper()} " + (' '.join(got) + 'e-3' if got else 'xong')
        return f"xong {name}"

    def _tick(self, sch):
        new = sorted(n for n, j in sch.jobs.items() if j.state == 'done' and n not in self.pushed_jobs)
        periodic = time.time() - self.last_push >= self.push_every_s
        if not new and not periodic:
            return
        self.pushed_jobs.update(new)
        self.last_push = time.time()
        if new:
            msg = '; '.join(self._job_msg(n) for n in new)
        else:
            done = sum(1 for j in sch.jobs.values() if j.state == 'done')
            msg = f"{','.join(self.datasets)}: {done}/{len(sch.jobs)} job xong (định kỳ)"
        threading.Thread(target=self.push, args=(msg,), daemon=True).start()

    def _remote_has(self, branch, path):
        if not self.git('ls-remote', '--heads', 'origin', branch, cwd=self.wd, auth=True, quiet=True).stdout.strip():
            return False
        self.git('fetch', '-q', '--depth', '1', 'origin', f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                 cwd=self.wd, auth=True, quiet=True)
        return self.git('cat-file', '-e', f"origin/{branch}:{path}", cwd=self.wd, check=False, quiet=True).returncode == 0

    # ---------- chạy ----------
    def run(self):
        from training.run_gnn_rl import build_jobs, Scheduler
        import torch
        jobs = build_jobs(self.datasets, self.run_ids, self.ablation_run_ids)
        sch = Scheduler(jobs, torch.cuda.device_count(), cpu_slots=2, time_budget_h=self.budget, on_tick=self._tick,
                        live=True, title=f"V4 tài khoản {self.account} | {','.join(d.upper() for d in self.datasets)} | "
                                         f"run {self.run_ids}")
        self.sch = sch
        t0 = time.time()
        summary, failed = sch.run()
        self.sch = None
        os.makedirs(os.path.join(self.wd, 'status'), exist_ok=True)
        with open(os.path.join(self.wd, STATUS(self.account)), 'w', encoding='utf-8') as f:
            json.dump({'account': self.account, 'datasets': self.datasets, 'run_ids': self.run_ids,
                       'finished_utc': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'session_hours': round((time.time() - t0) / 3600, 2), 'failed': failed, 'jobs': summary},
                      f, indent=2, ensure_ascii=False)
        self.push(f"hoàn tất {','.join(self.datasets)} ({len(failed)} job lỗi)")
        if failed:
            print(f"[LỖI] job bắt buộc thất bại: {failed}. Chạy lại notebook để tiếp tục (skip_existing).")
            return False
        return self.final()

    def final(self):
        """Tài khoản xong sau: gộp kết quả tài khoản kia, dựng lại cache v4, báo cáo chung."""
        if not self._remote_has(self.other_branch, STATUS(self.other)):
            print(f"Tài khoản {self.other} chưa xong -> dừng (tài khoản {self.other} sẽ làm phần cuối).")
            return True
        if self.git('push', 'origin', f"HEAD:refs/heads/{LOCK_BRANCH}", cwd=self.wd, auth=True, check=False,
                    quiet=True).returncode != 0:
            print("Tài khoản kia đang làm phần cuối -> dừng.")
            return True
        other_ds = PLAN[self.other]
        files = self.git('ls-tree', '-r', '--name-only', f"origin/{self.other_branch}", '--', *PUSH_PATHS,
                         cwd=self.wd, quiet=True).stdout.split()
        files = [f for f in files if any(f"_{d}" in f or f.startswith('status/') for d in other_ds)]
        for i in range(0, len(files), 200):
            self.git('checkout', f"origin/{self.other_branch}", '--', *files[i:i + 200], cwd=self.wd, quiet=True)
        # File P0 dùng chung tên cho mọi dataset: ghép dòng của tài khoản kia thay vì bỏ qua (vòng 1, 2 bị thiếu)
        import io
        import pandas as pd
        from training.gnn_rl_common import upsert_csv
        for name in ('p0_reproduce_v3.csv', 'p0_headroom_v3.csv'):
            rel = f"{_REL(RESULTS_V4)}/{name}"
            r = self.git('show', f"origin/{self.other_branch}:{rel}", cwd=self.wd, check=False, quiet=True)
            if r.returncode == 0 and r.stdout.strip():
                upsert_csv(pd.read_csv(io.StringIO(r.stdout)), os.path.join(self.wd, rel))
        print(f"Gộp {len(files)} file từ {self.other_branch}", flush=True)
        py = sys.executable
        all_ds = PLAN['A'] + PLAN['B']
        for ds in other_ds:
            subprocess.run([py, 'training/precompute_cache.py', '--cache_version', CACHE_VERSION_V4, '--datasets', ds,
                            '--run_ids', self.run_ids, '--branches', ','.join(V4_BRANCHES)],
                           cwd=self.wd, check=True)
        subprocess.run([py, 'evaluation/report_gnn_rl.py', '--datasets', ','.join(all_ds), '--run_ids', self.run_ids],
                       cwd=self.wd, check=True)
        self.push("gộp A + B, báo cáo chung")
        self.git('push', 'origin', f"HEAD:refs/heads/{FINAL_BRANCH}", cwd=self.wd, auth=True)
        print(f"Đã push kết quả chung lên {FINAL_BRANCH}.")
        return True
