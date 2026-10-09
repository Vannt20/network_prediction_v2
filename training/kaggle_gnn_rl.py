"""
Chạy hướng V4 trên Kaggle với 2 tài khoản song song (spec Mục 0.3, 10.3), không đụng tới main.

  Tài khoản A: GÉANT            -> push lên nhánh GNN_RL_a
  Tài khoản B: SDN + Abilene    -> push lên nhánh GNN_RL_b
Cả hai clone từ nhánh mã GNN_RL. Tài khoản xong sau gộp kết quả của tài khoản kia, dựng lại cache v4
cho các dataset đó (chỉ đọc log, không huấn luyện) rồi lập báo cáo chung và push lên GNN_RL_results.

Push định kỳ: logs/gnn_rl, results/gnn_rl, data/graphs, status. Không push cache/v4 (dựng lại được
từ cache/v3 + logs/gnn_rl) để tránh làm repo phình thêm vài GB.

Dùng trong notebook (sau khi clone và cd vào repo, `git` là hàm chạy git có xác thực như notebook phase 2):
    from training.kaggle_gnn_rl import KaggleV4
    KaggleV4('A', git).run()
"""
import os
import sys
import json
import time
import threading
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

PLAN = {'A': ['geant'], 'B': ['sdn', 'abilene']}
CODE_BRANCH = 'GNN_RL'
FINAL_BRANCH = 'GNN_RL_results'
LOCK_BRANCH = 'GNN_RL_final_lock'
PUSH_PATHS = ['logs/gnn_rl', 'results/gnn_rl', 'data/graphs', 'status']
MAX_MB = 95


class KaggleV4:
    def __init__(self, account, git, run_ids='0-9', ablation_run_ids='0-2', time_budget_h=10.0, push_every_s=900):
        assert account in PLAN
        self.account, self.git = account, git
        self.other = 'B' if account == 'A' else 'A'
        self.branch, self.other_branch = f"{CODE_BRANCH}_{account.lower()}", f"{CODE_BRANCH}_{self.other.lower()}"
        self.datasets = PLAN[account]
        self.run_ids, self.ablation_run_ids, self.budget = run_ids, ablation_run_ids, time_budget_h
        self.push_every_s, self.last_push = push_every_s, time.time()
        self.lock = threading.Lock()
        self.wd = parent_dir
        cur = self.git('rev-parse', '--abbrev-ref', 'HEAD', cwd=self.wd, quiet=True).stdout.strip()
        assert cur != 'main', "Không chạy V4 trên nhánh main"
        self.git('checkout', '-B', self.branch, cwd=self.wd, quiet=True)

    # ---------- git ----------
    def push(self, msg):
        with self.lock:
            paths = [p for p in PUSH_PATHS if os.path.exists(os.path.join(self.wd, p))]
            big = [os.path.join(a, f) for p in paths for a, _, fs in os.walk(os.path.join(self.wd, p)) for f in fs
                   if os.path.getsize(os.path.join(a, f)) > MAX_MB * 1024 ** 2]
            for f in big:
                print(f"[CẢNH BÁO] bỏ qua file > {MAX_MB} MB: {f}", flush=True)
            self.git('add', '-A', *paths, cwd=self.wd, quiet=True)
            for f in big:
                self.git('reset', '-q', '--', os.path.relpath(f, self.wd), cwd=self.wd, quiet=True, check=False)
            if self.git('diff', '--cached', '--quiet', cwd=self.wd, check=False, quiet=True).returncode == 0:
                return False
            self.git('commit', '-q', '-m', f"[V4 {self.account}] {msg}", cwd=self.wd, quiet=True)
            for attempt in range(3):
                if self.git('push', '-u', 'origin', self.branch, cwd=self.wd, auth=True, check=False,
                            quiet=True).returncode == 0:
                    print(f"[push] {self.branch}: {msg}", flush=True)
                    return True
                time.sleep(15 * (attempt + 1))
            print("[CẢNH BÁO] push lỗi, thử lại ở lần sau", flush=True)
            return False

    def _tick(self, sch):
        if time.time() - self.last_push >= self.push_every_s:
            self.last_push = time.time()
            done = sum(1 for j in sch.jobs.values() if j.state == 'done')
            threading.Thread(target=self.push, args=(f"{','.join(self.datasets)}: {done}/{len(sch.jobs)} job xong",),
                             daemon=True).start()

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
        sch = Scheduler(jobs, torch.cuda.device_count(), cpu_slots=2, time_budget_h=self.budget, on_tick=self._tick)
        t0 = time.time()
        summary, failed = sch.run()
        os.makedirs(os.path.join(self.wd, 'status'), exist_ok=True)
        with open(os.path.join(self.wd, 'status', f'gnn_rl_{self.account}.json'), 'w', encoding='utf-8') as f:
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
        if not self._remote_has(self.other_branch, f"status/gnn_rl_{self.other}.json"):
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
        print(f"Gộp {len(files)} file từ {self.other_branch}", flush=True)
        py = sys.executable
        all_ds = PLAN['A'] + PLAN['B']
        for ds in other_ds:
            subprocess.run([py, 'training/precompute_cache.py', '--cache_version', 'v4', '--datasets', ds,
                            '--run_ids', self.run_ids, '--branches', 'stwaveformer,xgboost,lightgbm_res,odgraphformer'],
                           cwd=self.wd, check=True)
        subprocess.run([py, 'evaluation/report_gnn_rl.py', '--datasets', ','.join(all_ds), '--run_ids', self.run_ids],
                       cwd=self.wd, check=True)
        self.push("gộp A + B, báo cáo chung")
        self.git('push', 'origin', f"HEAD:refs/heads/{FINAL_BRANCH}", cwd=self.wd, auth=True)
        print(f"Đã push kết quả chung lên {FINAL_BRANCH}.")
        return True
