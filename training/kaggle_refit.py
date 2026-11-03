"""
Điều phối refit 3 nhánh của v3 trên Train+Val (training/refit_branches.py) trên Kaggle và đánh giá RL-Gate.

- Đơn vị công việc = (nhánh mô hình, dataset, run). Xong một đơn vị (file tồn tại, đúng shape, không NaN) thì
  commit + push NGAY chỉ các file của đơn vị đó lên nhánh của tài khoản (rl_v5_a / rl_v5_b / rl_v5_solo).
- GPU: mỗi run ST-WaveFormer là một tiến trình, 1 run / GPU, GPU rảnh nhận run kế tiếp (run dài nhất theo E* × s/epoch trước để
  cân tải 2 GPU), giãn cách 60 giây giữa hai lần khởi tạo. CPU, song song: XGBoost rồi LightGBM-Res.
- Dataset đủ 3 nhánh × mọi run -> rl_gate/eval_refit.py -> results/refit/{ds}_summary.md, {ds}_runs.json.
- SOLO tạo summary_all.md ngay trên nhánh của mình. Với A/B: tài khoản xong sau giữ khóa rl_v5_final_lock,
  gộp nhánh của tài khoản kia vào rl_v5_results (merge --no-ff) và tạo results/refit/summary_all.md.
- gbdt_only=True: chỉ refit XGBoost và LightGBM-Res (CPU, không cần GPU), ST-WaveFormer giữ dự báo gốc của v3;
  mọi nhánh git / file trạng thái / thư mục kết quả có hậu tố riêng (rl_v5_a_gbdt, status/refit_a_gbdt.json,
  results/refit/gbdt_only/) nên không lẫn với đợt refit đủ 3 nhánh.
- datasets / tag: ghi đè PLAN và thêm hậu tố đợt chạy, ví dụ chia 10 run GÉANT cho 2 tài khoản (A: run 4-7,
  B: run 0-3,8,9, tag "geant"). Tài khoản chỉ chạy một phần run thì không đánh giá riêng; bước tổng hợp đánh giá
  mọi dataset đã đủ 10 run sau khi gộp.
- Output chỉ là một bảng trạng thái vẽ lại mỗi 60 giây; log của từng tiến trình con ở
  {log_dir}/{model}_{ds}_run{r}.log, mở bằng mode "w" (ghi đè khi chạy lại).

Quy tắc git: chỉ `git add` đường dẫn cụ thể, không force push, mỗi lệnh push đẩy đúng một nhánh
(HEAD:refs/heads/<nhánh>), trước mỗi lần push luôn fetch + pull --rebase nhánh của mình; rebase xung đột thì
abort, dừng push và báo trên bảng trạng thái.
"""
import collections
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

from training.refit_branches import MODELS, OUT_ROOT, n_rounds, unit_dir

PLAN = {"A": ["geant"], "B": ["sdn", "abilene"], "SOLO": ["sdn", "geant", "abilene"]}
BASE_BRANCH = "rl_v5"
LOCK_BRANCH = "rl_v5_final_lock"
RESULTS_BRANCH = "rl_v5_results"
SMOKE_BRANCH = "rl_v5_smoke"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
MAX_MB = 95                       # GitHub từ chối file > 100 MB
RETRY_WAITS = (2, 4, 8, 16)       # lỗi mạng khi push: thử lại tối đa 4 lần
GBDT = ["xgboost", "lightgbm_res"]
# s/epoch ước lượng trên T4 cho dữ liệu Train+Val (suy từ thời gian các run gốc), chỉ dùng để xếp thứ tự run dài trước
SEC_PER_EPOCH = {"sdn": 37, "geant": 77, "abilene": 83}
EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+) \|.*\| ([\d.]+) s/epoch \| VRAM ([\d.]+) GB")


def branch_of(account, suffix=""):
    return f"rl_v5_{account.lower()}{suffix}"


def _now():
    return time.strftime("%H:%M")


def _utc():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def _hm(seconds):
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}m"


def _last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 4096))
            lines = [x for x in f.read().decode("utf-8", "ignore").splitlines() if x.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def _clear():
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except Exception:
        pass


def _parse_ids(s):
    out = []
    for part in str(s).split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


class _Proc:
    def __init__(self, key, popen, log, gpu=None):
        self.key, self.p, self.log, self.gpu, self.t0 = key, popen, log, gpu, time.time()


class Refit:
    def __init__(self, account, git, workdir, run_ids="0-9", quick_check=False, status_every=60, dl_gap_s=60,
                 max_retry=2, eval_workers=4, log_dir="/kaggle/working/logs_refit", gbdt_only=False, datasets=None, tag=""):
        assert account in PLAN, f"ACCOUNT phải là một trong {list(PLAN)}"
        self.account, self.git, self.wd = account, git, workdir
        self.quick = quick_check
        self.datasets = ["sdn"] if quick_check else list(datasets or PLAN[account])
        self.ids = [0] if quick_check else _parse_ids(run_ids)
        self.run_ids = "0" if quick_check else run_ids
        self.gbdt_only = gbdt_only
        self.models = GBDT if gbdt_only else MODELS
        # hậu tố nhánh / trạng thái / khóa: chế độ chỉ GBDT và nhãn đợt chạy (ví dụ "geant" khi chia run GÉANT cho 2 tài khoản)
        self.sfx = ("_gbdt" if gbdt_only else "") + (f"_{tag}" if tag else "")
        # tài khoản chỉ chạy một phần run: không đánh giá riêng, bước tổng hợp đánh giá trên đủ 10 run sau khi gộp
        self.split = not quick_check and self.ids != list(range(10))
        self.branch = SMOKE_BRANCH + self.sfx if quick_check else branch_of(account, self.sfx)
        self.others = [] if quick_check or account == "SOLO" else [a for a in ("A", "B") if a != account]
        self.status_file = f"status/refit_{account.lower() if not quick_check else 'smoke'}{self.sfx}.json"
        self.eval_dir = "results/refit/gbdt_only" if gbdt_only else "results/refit"
        self.lock_branch, self.results_branch = LOCK_BRANCH + self.sfx, RESULTS_BRANCH + self.sfx
        self.status_every, self.dl_gap_s, self.max_retry = status_every, dl_gap_s, max_retry
        self.eval_workers, self.log_dir = eval_workers, log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.py = sys.executable
        try:
            import torch
            self.gpus = list(range(torch.cuda.device_count()))
        except ImportError:
            self.gpus = []
        self.slots = self.gpus or [None]              # không có GPU (kiểm tra nhanh trên CPU): 1 làn DL trên CPU

        self.units = [(m, ds, r) for ds in self.datasets for r in self.ids for m in self.models]
        self.e_star = ({} if gbdt_only else
                       {(ds, r): n_rounds("stwaveformer", ds, r)[0] for ds in self.datasets for r in self.ids})
        self.events = collections.deque(maxlen=10)
        self.git_lock = threading.Lock()
        self.commit_q = queue.Queue()
        self.n_commits = self.n_pushed = 0
        self.push_blocked = None                       # lý do dừng push (rebase xung đột / push lỗi)
        self.state, self.do_final = "running", False
        self.running_desc, self.failed = [], []
        self.t_start = time.time()
        self.started_utc = _utc()
        threading.Thread(target=self._commit_worker, daemon=True).start()

    # ---------- trạng thái của đơn vị ----------
    def done(self, m, ds, r):
        d = unit_dir(m, ds, r, os.path.join(self.wd, OUT_ROOT))
        try:
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            import numpy as np
            p = np.load(os.path.join(d, "test_pred.npy"), mmap_mode="r")
            return (bool(meta.get("quick_check")) == self.quick and p.shape == tuple(meta["shape"])
                    and p.dtype == np.float32 and bool(np.isfinite(p).all()))
        except (OSError, ValueError, KeyError):
            return False

    def unit_files(self, m, ds, r):
        d = os.path.relpath(unit_dir(m, ds, r, os.path.join(self.wd, OUT_ROOT)), self.wd).replace("\\", "/")
        names = ["test_pred.npy", "meta.json"] + (["train_metrics.csv", "model.pth"] if m == "stwaveformer" else [])
        return [f"{d}/{n}" for n in names]

    def _meta(self, m, ds, r):
        with open(os.path.join(self.wd, self.unit_files(m, ds, r)[1]), encoding="utf-8") as f:
            return json.load(f)

    def eval_files(self, ds):
        return [f"{self.eval_dir}/{ds}_summary.md", f"{self.eval_dir}/{ds}_runs.json"]

    def evaluated(self, ds, ids=None):
        try:
            with open(os.path.join(self.wd, self.eval_files(ds)[1]), encoding="utf-8") as f:
                rows = json.load(f)["rows"]
            return sorted(x["run"] for x in rows) == (ids or self.ids) and all(x["quick_check"] == self.quick for x in rows)
        except (OSError, ValueError, KeyError):
            return False

    def ds_complete(self, ds):
        return all(self.done(m, ds, r) for m in self.models for r in self.ids)

    # ---------- tiến trình con ----------
    def _spawn(self, key, cmd, log_name, gpu=None, one_thread=False):
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore",
                   CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        if one_thread:                                 # tiến trình DL: phần tính chính nằm trên GPU
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        log = os.path.join(self.log_dir, log_name)
        with open(log, "w", encoding="utf-8") as f:    # "w": chạy lại thì ghi đè, không nối dài
            p = subprocess.Popen(cmd, cwd=self.wd, env=env, stdout=f, stderr=subprocess.STDOUT)
        return _Proc(key, p, log, gpu)

    def _launch_unit(self, m, ds, r, gpu=None):
        cmd = [self.py, "training/refit_branches.py", "--model", m, "--dataset", ds, "--run", str(r)]
        if self.quick:
            cmd.append("--quick_check")
        return self._spawn((m, ds, r), cmd, f"{m}_{ds}_run{r}.log", gpu=gpu,
                           one_thread=m == "stwaveformer" and gpu is not None)

    def _eval_cmd(self, ds, run_ids):
        return ([self.py, "rl_gate/eval_refit.py", "--datasets", ds, "--runs", run_ids, "--workers", str(self.eval_workers),
                 "--out", self.eval_dir] + (["--allow_partial"] if self.quick else []) + (["--gbdt_only"] if self.gbdt_only else []))

    def _launch_eval(self, ds):
        return self._spawn(ds, self._eval_cmd(ds, self.run_ids), f"eval_{ds}.log")

    def _reap_unit(self, pr, requeue, retries):
        rc = pr.p.poll()
        if rc is None:
            return False
        m, ds, r = pr.key
        if rc == 0 and self.done(m, ds, r):
            meta = self._meta(m, ds, r)
            rounds = f"E*={meta['n_epochs']}" if m == "stwaveformer" else f"{meta['n_trees']} cây"
            self.events.append(f"{_now()}  xong {m} {ds.upper()} run_{r}: {rounds}, MSE {meta['test_mse'] * 1e3:.4f}e-3 "
                               f"(gốc {meta['test_mse_orig_branch'] * 1e3:.4f}e-3), {_hm(time.time() - pr.t0)}")
            self.commit_q.put((self.unit_files(m, ds, r),
                               f"[refit {self.account}] {m} {ds} run {r}: {rounds}, test MSE={meta['test_mse'] * 1e3:.4f}e-3"))
            return True
        retries[pr.key] += 1
        tail = _last_line(pr.log)[:90]
        if retries[pr.key] <= self.max_retry:
            requeue.appendleft(pr.key)
            self.events.append(f"{_now()}  lỗi {m} {ds.upper()} run_{r} (exit {rc}): {tail} -> chạy lại")
        else:
            self.failed.append(f"{m} {ds} run_{r}")
            self.events.append(f"{_now()}  THẤT BẠI {m} {ds.upper()} run_{r} (exit {rc}), xem {os.path.basename(pr.log)}")
        return True

    # ---------- vòng điều phối ----------
    def train(self):
        # đơn vị đã xong trong phiên này nhưng chưa commit (ví dụ notebook bị dừng giữa chừng) -> commit lại
        for m, ds, r in self.units:
            if self.done(m, ds, r):
                files = [f for f in self.unit_files(m, ds, r) if os.path.exists(os.path.join(self.wd, f))]
                if self.git("status", "--porcelain", "--", *files, cwd=self.wd, quiet=True).stdout.strip():
                    self.commit_q.put((files, f"[refit {self.account}] {m} {ds} run {r} (commit lại)"))
        dl_q = collections.deque(sorted(((m, ds, r) for m, ds, r in self.units if m == "stwaveformer" and not self.done(m, ds, r)),
                                        key=lambda u: -self.e_star[(u[1], u[2])] * SEC_PER_EPOCH[u[1]]))
        cpu_q = collections.deque(u for u in self.units if u[0] in GBDT and not self.done(*u))
        dl_run, cpu_run, eval_run = {}, None, None
        retries = collections.Counter()
        last_launch, last_status = 0.0, 0.0
        while True:
            for slot, pr in list(dl_run.items()):
                if self._reap_unit(pr, dl_q, retries):
                    del dl_run[slot]
            if cpu_run is not None and self._reap_unit(cpu_run, cpu_q, retries):
                cpu_run = None
            if eval_run is not None and eval_run.p.poll() is not None:
                ds = eval_run.key
                if eval_run.p.returncode == 0 and self.evaluated(ds):
                    with open(os.path.join(self.wd, self.eval_files(ds)[1]), encoding="utf-8") as f:
                        st = json.load(f)["stats"]
                    v3, pm = st["v3"]["mean"], st["rl_gate_refit_safe"]
                    self.events.append(f"{_now()}  đánh giá {ds.upper()}: v3 {v3:.4f}, rl_gate_refit_safe {pm['mean']:.4f} "
                                       f"({100 * pm['rel']:+.2f}%, p={pm['p']:.1e})")
                    self.commit_q.put((self.eval_files(ds), f"[refit {self.account}] đánh giá {ds}: v3 {v3:.4f}e-3, "
                                                            f"rl_gate_refit_safe {pm['mean']:.4f}e-3 ({100 * pm['rel']:+.2f}%)"))
                else:
                    retries[("eval", ds)] += 1
                    self.events.append(f"{_now()}  lỗi đánh giá {ds.upper()}: {_last_line(eval_run.log)[:90]}")
                    if retries[("eval", ds)] > self.max_retry:
                        self.failed.append(f"eval {ds}")
                eval_run = None

            free = [s for s in self.slots if s not in dl_run]
            while free and dl_q and time.time() - last_launch >= self.dl_gap_s:
                u = dl_q.popleft()
                if self.done(*u):
                    continue
                s = free.pop(0)
                dl_run[s] = self._launch_unit(*u, gpu=s)
                last_launch = time.time()
                self.events.append(f"{_now()}  bắt đầu stwaveformer {u[1].upper()} run_{u[2]} "
                                   f"(E*={self.e_star[(u[1], u[2])]}) trên {'GPU' + str(s) if s is not None else 'CPU'}")
            if cpu_run is None and cpu_q:
                u = cpu_q.popleft()
                if not self.done(*u):
                    cpu_run = self._launch_unit(*u)
            if eval_run is None:
                ready = [ds for ds in self.datasets if not self.split and not self.evaluated(ds)
                         and retries[("eval", ds)] <= self.max_retry
                         and self.ds_complete(ds)]
                if ready:
                    eval_run = self._launch_eval(ready[0])

            self.running_desc = ([f"{u[0]} {u[1]} run_{u[2]}" for u in (p.key for p in dl_run.values())] +
                                 ([f"{cpu_run.key[0]} {cpu_run.key[1]} run_{cpu_run.key[2]}"] if cpu_run else []) +
                                 ([f"eval {eval_run.key}"] if eval_run else []))
            eval_wait = not self.split and any(not self.evaluated(ds) and retries[("eval", ds)] <= self.max_retry and self.ds_complete(ds)
                            for ds in self.datasets)
            finished = not dl_run and not dl_q and cpu_run is None and not cpu_q and eval_run is None and not eval_wait
            if time.time() - last_status >= self.status_every or finished:
                self.render(dl_run, cpu_run, eval_run)
                last_status = time.time()
            if finished:
                break
            time.sleep(5)
        self.commit_q.join()
        self.render({}, None, None)
        if self.failed:
            raise RuntimeError(f"Có đơn vị thất bại: {self.failed}. Chạy lại notebook để tiếp tục (bỏ qua phần đã xong).")

    # ---------- hiển thị ----------
    def _gpu_stats(self):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            out = {}
            for line in r.stdout.strip().splitlines():
                i, u, mem = [x.strip() for x in line.split(",")]
                out[int(i)] = f"GPU {u:>3}%  {int(mem) / 1024:4.1f} GB"
            return out
        except Exception:
            return {}

    @staticmethod
    def _ram_free():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return f"{int(line.split()[1]) / 1024 ** 2:.1f} GB"
        except OSError:
            pass
        return "?"

    def render(self, dl_run, cpu_run, eval_run):
        _clear()
        n = len(self.units)
        n_dl = sum(self.done("stwaveformer", ds, r) for ds in self.datasets for r in self.ids)
        n_ok = sum(self.done(*u) for u in self.units)
        gs = self._gpu_stats()
        load = f"{os.getloadavg()[0]:.1f}" if hasattr(os, "getloadavg") else "?"
        lines = [f"Refit {self.account}{' (quick_check)' if self.quick else ''} | {','.join(d.upper() for d in self.datasets)} | "
                 f"run {self.run_ids} | nhánh {self.branch} | {_hm(time.time() - self.t_start)} | {time.strftime('%H:%M:%S')} | "
                 f"RAM trống {self._ram_free()} | CPU load {load}/{os.cpu_count()}"]
        for s in self.slots:
            name = f"GPU{s}" if s is not None else "DL  "
            pr = dl_run.get(s)
            if pr is None:
                lines.append(f"{name}  rảnh  {gs.get(s, '')}")
                continue
            _, ds, r = pr.key
            hit = EPOCH_RE.search(_last_line(pr.log))
            prog = (f"epoch {int(hit.group(1)):3d}/{hit.group(2)}  {float(hit.group(3)):6.1f} s/epoch  VRAM {hit.group(4)} GB"
                    if hit else "đang nạp dữ liệu")
            lines.append(f"{name}  {ds.upper():7s} run_{r}  {prog}  {gs.get(s, '')}  ({_hm(time.time() - pr.t0)})")
        lines.append(f"CPU   {cpu_run.key[0]} {cpu_run.key[1].upper()} run_{cpu_run.key[2]} ({_hm(time.time() - cpu_run.t0)})"
                     if cpu_run else "CPU   rảnh")
        lines.append(f"EVAL  {eval_run.key.upper()} ({_hm(time.time() - eval_run.t0)})" if eval_run else "EVAL  rảnh")
        dl_txt = "chỉ GBDT" if self.gbdt_only else f"ST-WaveFormer {n_dl}/{len(self.datasets) * len(self.ids)}"
        lines.append(f"Xong  {n_ok}/{n} đơn vị ({dl_txt}) | đánh giá "
                     f"{sum(self.evaluated(d) for d in self.datasets)}/{len(self.datasets)} dataset | commit {self.n_commits}, "
                     f"đã push {self.n_pushed} | lỗi {len(self.failed)} | push: {'DỪNG - ' + self.push_blocked if self.push_blocked else 'OK'}")
        events = list(self.events)                     # luồng commit có thể đang ghi thêm sự kiện
        if events:
            lines.append("Gần đây")
            lines += [f"  {e}" for e in events]
        print("\n".join(lines), flush=True)

    # ---------- git ----------
    def _write_status(self):
        done = [f"{m} {ds} run_{r}" for m, ds, r in self.units if self.done(m, ds, r)]
        st = {"account": self.account, "branch": self.branch, "datasets": self.datasets, "run_ids": self.run_ids,
              "quick_check": self.quick, "state": self.state, "started_utc": self.started_utc, "updated_utc": _utc(),
              "session_hours": round((time.time() - self.t_start) / 3600, 2), "n_done": len(done),
              "n_total": len(self.units), "running": self.running_desc, "done_units": done,
              "evaluated": [d for d in self.datasets if self.evaluated(d)], "failed": self.failed,
              "commits": self.n_commits + 1, "push_blocked": self.push_blocked}
        path = os.path.join(self.wd, self.status_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:            # ghi đè: chỉ giữ trạng thái mới nhất
            json.dump(st, f, ensure_ascii=False, indent=1)

    def _remote_has(self, branch):
        return bool(self.git("ls-remote", "--heads", "origin", branch, cwd=self.wd, auth=True, quiet=True).stdout.strip())

    def _push(self, branch=None):
        """fetch + pull --rebase nhánh của mình rồi push đúng một nhánh; trả về True nếu đã push."""
        branch = branch or self.branch
        if self.push_blocked:
            return False
        if self._remote_has(branch):
            self.git("fetch", "-q", "origin", branch, cwd=self.wd, auth=True, quiet=True)
            r = self.git("pull", "-q", "--rebase", "--autostash", "origin", branch, cwd=self.wd, auth=True,
                         check=False, quiet=True)
            if r.returncode:
                self.git("rebase", "--abort", cwd=self.wd, check=False, quiet=True)
                self.push_blocked = f"rebase xung đột với origin/{branch}"
                self.events.append(f"{_now()}  {self.push_blocked}: đã abort, dừng push (không ghi đè)")
                return False
        for wait in RETRY_WAITS + (None,):
            if self.git("push", "-q", "origin", f"HEAD:refs/heads/{branch}", cwd=self.wd, auth=True,
                        check=False, quiet=True).returncode == 0:
                return True
            if wait is None:
                break
            time.sleep(wait)
        self.events.append(f"{_now()}  push lỗi sau {len(RETRY_WAITS)} lần thử lại, sẽ thử ở commit sau")
        return False

    def _commit(self, paths, msg):
        with self.git_lock:
            self._write_status()
            files = []
            for p in [p for p in paths if os.path.exists(os.path.join(self.wd, p))] + [self.status_file]:
                if os.path.getsize(os.path.join(self.wd, p)) > MAX_MB * 1024 ** 2:
                    self.events.append(f"{_now()}  bỏ qua {p}: > {MAX_MB} MB")
                else:
                    files.append(p)
            self.git("add", "-f", "--", *files, cwd=self.wd, quiet=True)
            if not self.git("diff", "--cached", "--quiet", cwd=self.wd, check=False, quiet=True).returncode:
                return
            self.git("commit", "-q", "-m", msg, cwd=self.wd, quiet=True)
            self.n_commits += 1
            if self._push():
                self.n_pushed = self.n_commits

    def _commit_worker(self):
        while True:
            paths, msg = self.commit_q.get()
            try:
                self._commit(paths, msg)
            except Exception as e:
                self.events.append(f"{_now()}  commit lỗi ({msg[:40]}): {str(e)[:80]}")
            finally:
                self.commit_q.task_done()

    # ---------- kết thúc phần của tài khoản ----------
    def missing(self):
        return ([f"{m} {ds} run_{r}" for m, ds, r in self.units if not self.done(m, ds, r)] +
                [f"đánh giá {ds}" for ds in self.datasets if not self.split and not self.evaluated(ds)])

    def _remote_status(self, account):
        """Đọc status/refit_{x}.json trên nhánh của tài khoản kia (chỉ tải các commit sau rl_v5)."""
        br = branch_of(account, self.sfx)
        if not self._remote_has(br):
            return None
        r = self.git("fetch", "-q", f"--shallow-exclude={BASE_BRANCH}", "origin", f"+refs/heads/{br}:refs/remotes/origin/{br}",
                     cwd=self.wd, auth=True, check=False, quiet=True)
        if r.returncode:
            return None
        r = self.git("show", f"origin/{br}:status/refit_{account.lower()}{self.sfx}.json", cwd=self.wd, check=False, quiet=True)
        try:
            return json.loads(r.stdout) if r.returncode == 0 else None
        except ValueError:
            return None

    def finish(self):
        miss = self.missing()
        assert not miss, f"Chưa đủ kết quả: {miss}. Chạy lại notebook để tiếp tục."
        self.state = "done"
        self.commit_q.put(([], f"[refit {self.account}] hoàn tất {','.join(self.datasets)} run {self.run_ids}"))
        self.commit_q.join()
        if self.push_blocked or self.n_pushed < self.n_commits:
            raise RuntimeError(f"Chưa push được hết commit lên {self.branch}: {self.push_blocked or 'lỗi mạng'}")
        print(f"Tài khoản {self.account}: đủ kết quả {','.join(self.datasets)} run {self.run_ids}, đã push lên {self.branch}.")
        if self.quick:
            return False
        if self.account == "SOLO":                     # không có tài khoản khác: tổng hợp ngay trên nhánh của mình
            self.do_final = True
            return True
        waiting = [o for o in self.others if (self._remote_status(o) or {}).get("state") != "done"]
        if waiting:
            print(f"Tài khoản {', '.join(waiting)} chưa xong -> dừng tại đây; tài khoản xong sau sẽ làm bước tổng hợp.")
            return False
        with self.git_lock:
            if self._remote_has(self.lock_branch):
                print(f"Nhánh {self.lock_branch} đã có (tài khoản khác đang/đã tổng hợp) -> bỏ qua bước tổng hợp.")
                return False
            # khóa = một commit rỗng không có cha; nếu nhánh vừa được tạo bởi tài khoản kia thì push bị từ chối
            sha = self.git("commit-tree", EMPTY_TREE, "-m", f"khóa tổng hợp refit: tài khoản {self.account} {_utc()}",
                           cwd=self.wd, quiet=True).stdout.strip()
            if self.git("push", "-q", "origin", f"{sha}:refs/heads/{self.lock_branch}", cwd=self.wd, auth=True,
                        check=False, quiet=True).returncode:
                print(f"Không tạo được {self.lock_branch} (tài khoản khác vừa giữ khóa) -> bỏ qua bước tổng hợp.")
                return False
        self.do_final = True
        print(f"Đã giữ khóa {self.lock_branch}: tài khoản {self.account} làm bước tổng hợp.")
        return True

    # ---------- tổng hợp cuối: rl_v5_results = rl_v5 + merge --no-ff các nhánh tài khoản ----------
    def final(self):
        g = self.git
        if self.account == "SOLO":
            self._summary_all()
            with self.git_lock:
                self._commit_summary_all()
                if not self._push():
                    raise RuntimeError(f"Push {self.branch} lỗi: {self.push_blocked or 'lỗi mạng'}")
            print(f"Đã push {self.eval_dir}/summary_all.md lên {self.branch}.")
            return
        with self.git_lock:
            g("fetch", "-q", "--depth", "1", "origin", f"+refs/heads/{BASE_BRANCH}:refs/remotes/origin/{BASE_BRANCH}",
              cwd=self.wd, auth=True, quiet=True)
            own = g("rev-parse", "HEAD", cwd=self.wd, quiet=True).stdout.strip()
            sources = [(self.branch, own)] + [(branch_of(o, self.sfx), f"origin/{branch_of(o, self.sfx)}") for o in self.others]
            for o in self.others:
                assert (self._remote_status(o) or {}).get("state") == "done", f"Nhánh {branch_of(o, self.sfx)} chưa hoàn tất"
            g("checkout", "-q", "-B", self.results_branch, f"origin/{BASE_BRANCH}", cwd=self.wd, quiet=True)
            for name, ref in sources:
                # clone nông: lịch sử chung bị cắt nên cần --allow-unrelated-histories; đường dẫn của các tài khoản
                # tách biệt, file chung giống hệt nhau nên không có xung đột
                r = g("merge", "--no-ff", "--allow-unrelated-histories", "-q", "-m", f"Gộp {name} vào {self.results_branch}",
                      ref, cwd=self.wd, check=False, quiet=True)
                if r.returncode:
                    g("merge", "--abort", cwd=self.wd, check=False, quiet=True)
                    raise RuntimeError(f"Gộp {name} bị xung đột -> đã abort, không push. Kiểm tra trên máy.")
        self._summary_all()
        with self.git_lock:
            self._commit_summary_all()
            self.push_blocked = None
            if not self._push(self.results_branch):
                raise RuntimeError(f"Push {self.results_branch} lỗi: {self.push_blocked or 'lỗi mạng'}")
        print(f"Đã push {self.results_branch}.")

    def _summary_all(self):
        # dataset đủ 10 run sau khi gộp (ví dụ GÉANT chia cho 2 tài khoản) nhưng chưa có đánh giá đủ 10 run -> đánh giá ngay
        full = list(range(10))
        for ds in ["sdn", "geant", "abilene"]:
            if self.evaluated(ds, full) or not all(self.done(m, ds, r) for m in self.models for r in full):
                continue
            print(f"Đánh giá {ds.upper()} trên đủ 10 run", flush=True)
            r = subprocess.run(self._eval_cmd(ds, "0-9"), cwd=self.wd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"})
            if r.returncode or not self.evaluated(ds, full):
                print((r.stdout + r.stderr)[-3000:])
                raise RuntimeError(f"eval_refit.py {ds} lỗi")
        r = subprocess.run([self.py, "rl_gate/eval_refit.py", "--summary_all", "--out", self.eval_dir], cwd=self.wd,
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode:
            print((r.stdout + r.stderr)[-3000:])
            raise RuntimeError("eval_refit.py --summary_all lỗi")

    def _commit_summary_all(self):
        g = self.git
        files = [f"{self.eval_dir}/summary_all.md"] + [f for ds in ["sdn", "geant", "abilene"] for f in self.eval_files(ds)
                                                       if os.path.exists(os.path.join(self.wd, f))]
        g("add", "--", *files, cwd=self.wd, quiet=True)
        if g("diff", "--cached", "--quiet", cwd=self.wd, check=False, quiet=True).returncode:
            g("commit", "-q", "-m", f"[refit {self.account}] tổng hợp: {self.eval_dir}/summary_all.md",
              cwd=self.wd, quiet=True)

    def summary(self):
        """Bảng tóm tắt cuối (thay cho bảng trạng thái)."""
        _clear()
        f = os.path.join(self.wd, f"{self.eval_dir}/summary_all.md")
        if self.do_final and os.path.exists(f):
            print(open(f, encoding="utf-8").read())
            return
        for ds in self.datasets:
            p = os.path.join(self.wd, self.eval_files(ds)[0])
            if os.path.exists(p):
                print(open(p, encoding="utf-8").read())
        print(f"commit {self.n_commits}, đã push {self.n_pushed} lên {self.branch} | "
              f"thời gian {_hm(time.time() - self.t_start)}")
