"""
Điều phối huấn luyện trên Kaggle khi chia việc cho 2 account (A, B) chạy song song.

Mỗi account huấn luyện một nhóm dataset và push lên nhánh riêng (run_a / run_b):
  - GPU: mỗi run ST-WaveFormer là một tiến trình, 1 run / GPU; GPU rảnh nhận run kế tiếp.
  - CPU, làn ML: 4 mô hình học máy (champion + lightgbm_res trước), độ ưu tiên thường như đợt 1.
  - CPU, làn Ensemble: khi một (dataset, run) đủ 3 nhánh -> cache, stacking, ablation của run đó.
Account xong trước thoát ngay (không tốn quota). Account xong sau gộp nhánh của account kia,
dựng lại bảng kết quả, lập báo cáo cho toàn bộ run và push lên nhánh kết quả chung.
"""
import collections
import functools
import json
import os
import re
import subprocess
import sys
import threading
import time

import pandas as pd

SEQ = {"sdn": 60, "geant": 24, "abilene": 24}
ML_ALL = ["xgboost", "lightgbm", "catboost", "lightgbm_res"]
ENS_TAG = "st_adaptive_ensemble"
MAX_MB = 95          # GitHub từ chối file > 100 MB
STABLE_S = 15        # file đánh dấu phải đứng yên ít nhất chừng này giây mới coi là ghi xong

ENS_DRIVER = """
import sys
ds, r = sys.argv[1], sys.argv[2]
from training.precompute_cache import run_precompute
from training.train_stacking import train_all_stacking
from evaluation.ablation_study import run_ablation_experiments
run_precompute(datasets=[ds], run_ids=r, skip_existing=True)
train_all_stacking(datasets=[ds], run_ids=r)
run_ablation_experiments(datasets=[ds], run_ids=r)
"""

EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+).*?Patience: (\d+)/(\d+) \| ([\d.]+)s/epoch")
ML_RE = re.compile(r"Huấn luyện (\S+) trên (\S+) \[run_(\d+)\]")


def _now():
    return time.strftime("%H:%M")


def _tail(path, n=16384):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - n))
        return f.read().decode("utf-8", "ignore")


def _stable(path):
    return os.path.exists(path) and time.time() - os.path.getmtime(path) > STABLE_S


def _mse(csv_path):
    try:
        return float(pd.read_csv(csv_path).iloc[0]["mse"]) * 1e3
    except Exception:
        return float("nan")


def _hm(seconds):
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}m"


def _clear():
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except Exception:
        pass


class _Proc:
    def __init__(self, key, popen, log, gpu=None):
        self.key, self.p, self.log, self.gpu, self.t0 = key, popen, log, gpu, time.time()


class Phase2:
    def __init__(self, account, plan, run_ids, git, workdir, final_run_ids,
                 final_branch="run", lock_branch="phase2_final_lock",
                 epochs=200, patience=30, max_retry=2, status_every=60, push_every=60, dl_gap_s=60,
                 log_dir="/kaggle/working/logs_kaggle"):
        import torch
        from run_experiments import parse_run_ids

        self.account = account
        self.other = next(a for a in plan if a != account)
        self.plan = plan
        self.datasets = plan[account]
        self.run_ids = run_ids
        self.ids = parse_run_ids(run_ids)
        self.final_run_ids = final_run_ids
        self.git, self.wd = git, workdir
        self.branch = f"run_{account.lower()}"
        self.other_branch = f"run_{self.other.lower()}"
        self.final_branch, self.lock_branch = final_branch, lock_branch
        self.epochs, self.patience, self.max_retry = epochs, patience, max_retry
        self.status_every, self.push_every = status_every, push_every
        self.dl_gap_s = dl_gap_s          # giãn cách giữa 2 lần mở tiến trình DL (tránh cùng lúc nạp dữ liệu)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.py = sys.executable
        self.gpus = list(range(torch.cuda.device_count()))

        champ = json.load(open("results/champion_ml_model.json", encoding="utf-8"))
        assert champ.get("frozen"), "Champion chưa được cố định (cần kết quả đợt 1 trên nhánh run)"
        self.champion = champ["champion_model"]
        self.ml_core = [self.champion, "lightgbm_res"]
        self.ml_control = [m for m in ML_ALL if m not in self.ml_core]

        self.events = collections.deque(maxlen=10)
        self.git_lock = threading.Lock()
        self.pushed = 0
        self.do_final = False
        self.t_start = time.time()

    # ---------- đường dẫn và trạng thái ----------
    @staticmethod
    def dl_dir(ds, r):
        return f"logs/stwaveformer_data_{ds}_seq_{SEQ[ds]}/run_{r}"

    @staticmethod
    def ml_dir(m, ds, r):
        return f"logs/{m}_data_{ds}_shared/run_{r}"

    @staticmethod
    def ens_dir(ds, r):
        return f"logs/{ENS_TAG}_data_{ds}_seq_{SEQ[ds]}/run_{r}"

    @staticmethod
    def cache_file(ds, r):
        return f"cache/v3/{ds}_run_{r}.pt"

    @staticmethod
    @functools.lru_cache(None)
    def _n_win(ds):
        from run_experiments import expected_test_windows
        return expected_test_windows(ds, SEQ[ds])

    def dl_done(self, ds, r):
        from run_experiments import check_existing_run
        return check_existing_run(self.dl_dir(ds, r), self._n_win(ds))[0]

    def ml_done(self, m, ds, r):
        return os.path.exists(os.path.join(self.ml_dir(m, ds, r), "model.bin"))

    def ens_done(self, ds, r):
        return os.path.exists(os.path.join(self.ens_dir(ds, r), "ablation.csv"))

    def ens_pending(self, ds, r):
        """Đủ 3 nhánh nhưng chưa có ensemble."""
        return (not self.ens_done(ds, r) and self.dl_done(ds, r)
                and all(self.ml_done(m, ds, r) for m in self.ml_core))

    def ens_ready(self, ds, r):
        # chờ file ghi cuối cùng của run DL đứng yên (tiến trình DL đã ghi xong)
        return self.ens_pending(ds, r) and _stable(self.dl_dir(ds, r) + "/y_pred_data_raw.npy")

    def missing(self, datasets=None, ids=None, need_ablation=True):
        """need_ablation=False: run đợt 1 không có ablation.csv theo run (nằm trong results/ablation_results.csv)."""
        out = []
        for ds in datasets or self.datasets:
            for r in ids or self.ids:
                if not self.dl_done(ds, r):
                    out.append(f"{ds} run_{r} ST-WaveFormer")
                out += [f"{ds} run_{r} {m}" for m in ML_ALL if not self.ml_done(m, ds, r)]
                ens_ok = self.ens_done(ds, r) if need_ablation else os.path.exists(self.ens_dir(ds, r) + "/test_metrics.csv")
                if not ens_ok or not os.path.exists(self.cache_file(ds, r)):
                    out.append(f"{ds} run_{r} ensemble")
        return out

    # ---------- khởi chạy tiến trình ----------
    def _spawn(self, key, cmd, log_name, gpu=None, one_thread=False):
        # Không dùng nice cho ML/Ensemble: LightGBM (OpenMP) và torch đồng bộ luồng ở mỗi bước, luồng nào bị
        # tiến trình DL chiếm lõi sẽ kéo cả vòng lặp chậm hàng chục lần (đo được ở đợt 2: 5 phút -> 2-3 giờ / run).
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        if one_thread:                         # tiến trình DL: 1 luồng CPU, phần tính chính nằm trên GPU
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        log = os.path.join(self.log_dir, log_name)
        with open(log, "a") as f:
            p = subprocess.Popen(cmd, cwd=self.wd, env=env, stdout=f, stderr=subprocess.STDOUT)
        return _Proc(key, p, log, gpu)

    def _launch_dl(self, ds, r, gpu):
        cmd = [self.py, "run_experiments.py", "--model", "STWaveFormer", "--dataset", ds, "--run_ids", str(r),
               "--epochs", str(self.epochs), "--patience", str(self.patience), "--skip_existing"]
        return self._spawn((ds, r), cmd, f"dl_{ds}_run{r}.log", gpu=gpu, one_thread=True)

    def _launch_ml(self, ds, models):
        cmd = [self.py, "baselines_ml/run_ml_baselines.py", "--models", ",".join(models), "--datasets", ds,
               "--run_ids", self.run_ids, "--skip_existing"]
        return self._spawn((ds, tuple(models)), cmd, f"ml_{ds}_{'_'.join(models)}.log")

    def _launch_ens(self, ds, r):
        cmd = [self.py, "-c", ENS_DRIVER, ds, str(r)]
        return self._spawn((ds, r), cmd, f"ens_{ds}_run{r}.log", gpu=self.gpus[0] if self.gpus else None)

    # ---------- vòng điều phối ----------
    def train(self):
        dl_queue = collections.deque((ds, r) for ds in self.datasets for r in self.ids if not self.dl_done(ds, r))
        ml_queue = collections.deque((ds, ms) for ms in (self.ml_core, self.ml_control) for ds in self.datasets
                                     if not all(self.ml_done(m, ds, r) for m in ms for r in self.ids))
        dl_run, ml_run, ens_run = {}, None, None
        retries, failed = collections.Counter(), []
        last_dl_launch, last_status, last_push = 0.0, 0.0, time.time()
        self._push_thread = None

        while True:
            # thu hồi tiến trình đã kết thúc
            for gpu, pr in list(dl_run.items()):
                rc = pr.p.poll()
                if rc is None:
                    continue
                del dl_run[gpu]
                ds, r = pr.key
                if rc == 0 and self.dl_done(ds, r):
                    n_ep = len(pd.read_csv(os.path.join(self.dl_dir(ds, r), "train_metrics.csv")))
                    self.events.append(f"{_now()}  DL xong {ds.upper()} run_{r}: MSE {_mse(self.dl_dir(ds, r) + '/test_metrics.csv'):.3f}e-3, "
                                       f"{n_ep} epoch, {(time.time() - pr.t0) / 60:.0f} phút")
                    continue
                retries[("dl", ds, r)] += 1
                oom = "out of memory" in _tail(pr.log, 4000).lower() or rc in (-9, 137)
                if retries[("dl", ds, r)] <= self.max_retry:
                    dl_queue.appendleft((ds, r))
                    self.events.append(f"{_now()}  DL lỗi {ds.upper()} run_{r} (exit {rc}{', hết bộ nhớ' if oom else ''}), chạy lại")
                else:
                    failed.append(f"DL {ds} run_{r}")
                    self.events.append(f"{_now()}  DL THẤT BẠI {ds.upper()} run_{r} (exit {rc}), xem {os.path.basename(pr.log)}")

            if ml_run is not None and ml_run.p.poll() is not None:
                rc, (ds, ms) = ml_run.p.returncode, ml_run.key
                ok = rc == 0 and all(self.ml_done(m, ds, r) for m in ms for r in self.ids)
                if ok:
                    self.events.append(f"{_now()}  ML xong {ds.upper()} {','.join(ms)} ({(time.time() - ml_run.t0) / 60:.0f} phút)")
                else:
                    retries[("ml", ds, ms)] += 1
                    if retries[("ml", ds, ms)] <= self.max_retry:
                        ml_queue.appendleft((ds, ms))
                        self.events.append(f"{_now()}  ML lỗi {ds.upper()} {','.join(ms)} (exit {rc}), chạy lại")
                    else:
                        failed.append(f"ML {ds} {','.join(ms)}")
                        self.events.append(f"{_now()}  ML THẤT BẠI {ds.upper()} {','.join(ms)}, xem {os.path.basename(ml_run.log)}")
                ml_run = None

            if ens_run is not None and ens_run.p.poll() is not None:
                rc, (ds, r) = ens_run.p.returncode, ens_run.key
                if rc == 0 and self.ens_done(ds, r):
                    self.events.append(f"{_now()}  Ensemble xong {ds.upper()} run_{r}: MSE "
                                       f"{_mse(self.ens_dir(ds, r) + '/test_metrics.csv'):.3f}e-3")
                else:
                    retries[("ens", ds, r)] += 1
                    msg = "chạy lại" if retries[("ens", ds, r)] <= self.max_retry else f"THẤT BẠI, xem {os.path.basename(ens_run.log)}"
                    if retries[("ens", ds, r)] > self.max_retry:
                        failed.append(f"Ensemble {ds} run_{r}")
                    self.events.append(f"{_now()}  Ensemble lỗi {ds.upper()} run_{r} (exit {rc}), {msg}")
                ens_run = None

            # khởi chạy việc mới
            free = [g for g in self.gpus if g not in dl_run]
            while free and dl_queue and time.time() - last_dl_launch >= self.dl_gap_s:
                ds, r = dl_queue.popleft()
                if self.dl_done(ds, r):
                    continue
                g = free.pop(0)
                dl_run[g] = self._launch_dl(ds, r, g)
                last_dl_launch = time.time()
                self.events.append(f"{_now()}  DL bắt đầu {ds.upper()} run_{r} trên GPU{g}")
            if ml_run is None and ml_queue:
                ds, ms = ml_queue.popleft()
                ml_run = self._launch_ml(ds, ms)
            if ens_run is None:
                ready = [(ds, r) for ds in self.datasets for r in self.ids
                         if retries[("ens", ds, r)] <= self.max_retry and self.ens_ready(ds, r)]
                if ready:
                    ens_run = self._launch_ens(*ready[0])

            # push định kỳ ở luồng nền (không chặn vòng điều phối)
            if time.time() - last_push >= self.push_every and not (self._push_thread and self._push_thread.is_alive()):
                self._push_thread = threading.Thread(target=self._safe_push, daemon=True)
                self._push_thread.start()
                last_push = time.time()

            ens_wait = any(retries[("ens", ds, r)] <= self.max_retry and self.ens_pending(ds, r)
                           for ds in self.datasets for r in self.ids)
            done = not dl_run and not dl_queue and ml_run is None and not ml_queue and ens_run is None and not ens_wait
            if time.time() - last_status >= self.status_every or done:
                self._render(dl_run, ml_run, ens_run, len(dl_queue), failed)
                last_status = time.time()
            if done:
                break
            time.sleep(5)

        if self._push_thread:
            self._push_thread.join()
        self.push_completed(wait_stable=False)
        self._render({}, None, None, 0, failed)
        if failed:
            raise RuntimeError(f"Có việc thất bại: {failed}. Chạy lại notebook để tiếp tục (bỏ qua phần đã xong).")

    # ---------- hiển thị ----------
    def _gpu_stats(self):
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        out = {}
        for line in r.stdout.strip().splitlines():
            i, u, m = [x.strip() for x in line.split(",")]
            out[int(i)] = f"{u:>3}%  {int(m) / 1024:4.1f} GB"
        return out

    @staticmethod
    def _ram_free():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return int(line.split()[1]) / 1024 ** 2
        except OSError:
            pass
        return float("nan")

    def _render(self, dl_run, ml_run, ens_run, n_wait, failed):
        _clear()
        ids = self.ids
        n = len(self.datasets) * len(ids)
        dl_ok = sum(self.dl_done(ds, r) for ds in self.datasets for r in ids)
        ml_ok = sum(self.ml_done(m, ds, r) for m in ML_ALL for ds in self.datasets for r in ids)
        ens_ok = sum(self.ens_done(ds, r) for ds in self.datasets for r in ids)
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
        gs = self._gpu_stats() if self.gpus else {}
        lines = [f"Account {self.account} | {','.join(d.upper() for d in self.datasets)} | run {self.run_ids} | "
                 f"{_hm(time.time() - self.t_start)} | {time.strftime('%H:%M:%S')} UTC | "
                 f"RAM trống {self._ram_free():.1f} GB | CPU load {load:.1f}/{os.cpu_count()}"]
        for g in self.gpus:
            pr = dl_run.get(g)
            if pr is None:
                lines.append(f"GPU{g}  rảnh{'' if not n_wait else ' (chờ giãn cách khởi động)'}  {gs.get(g, '')}")
                continue
            ds, r = pr.key
            m = EPOCH_RE.findall(_tail(pr.log))
            if m:
                ep, tot, pat, ptot, spe = m[-1]
                prog = (f"epoch {int(ep):3d}/{tot}  best {int(ep) - int(pat):3d}  "
                        f"patience {int(pat):2d}/{ptot}  {float(spe):5.1f} s/epoch")
            else:
                prog = "đang nạp dữ liệu"
            lines.append(f"GPU{g}  {ds.upper():7s} run_{r}  {prog}  {gs.get(g, '')}  ({_hm(time.time() - pr.t0)})")
        if ml_run is not None:
            hit = ML_RE.findall(_tail(ml_run.log))
            cur = f"{hit[-1][0].lower()} {hit[-1][1]} run_{hit[-1][2]}" if hit else "đang nạp dữ liệu"
            lines.append(f"ML    {ml_run.key[0].upper()} {','.join(ml_run.key[1])}: {cur}")
        else:
            lines.append("ML    rảnh")
        if ens_run is not None:
            t = _tail(ens_run.log)
            stage = "ablation" if "ABLATION" in t else "stacking" if "TẦNG KẾT HỢP" in t else "cache"
            lines.append(f"ENS   {ens_run.key[0].upper()} run_{ens_run.key[1]}: {stage}")
        else:
            lines.append("ENS   rảnh")
        lines.append(f"Xong  DL {dl_ok}/{n} | ML {ml_ok}/{n * len(ML_ALL)} | Ensemble {ens_ok}/{n} | "
                     f"đã push {self.pushed} commit | lỗi {len(failed)}")
        if self.events:
            lines.append("Gần đây")
            lines += [f"  {e}" for e in self.events]
        print("\n".join(lines), flush=True)

    # ---------- git ----------
    def _items(self, wait_stable=True):
        """(đường dẫn, thông điệp commit) của mọi phần đã hoàn tất."""
        ok = _stable if wait_stable else os.path.exists
        out = []
        for ds in self.datasets:
            for r in self.ids:
                d = self.dl_dir(ds, r)
                if ok(d + "/y_pred_data_raw.npy"):
                    out.append((d, f"[{self.account}] {ds.upper()} run_{r} ST-WaveFormer MSE={_mse(d + '/test_metrics.csv'):.3f}e-3"))
                for m in ML_ALL:
                    d = self.ml_dir(m, ds, r)
                    if ok(d + "/model.bin"):
                        out.append((d, f"[{self.account}] {ds.upper()} run_{r} {m} MSE={_mse(d + '/test_metrics.csv'):.3f}e-3"))
                d = self.ens_dir(ds, r)
                if ok(d + "/ablation.csv"):
                    out.append((self.cache_file(ds, r), f"[{self.account}] {ds.upper()} run_{r} cache v3"))
                    out.append((d, f"[{self.account}] {ds.upper()} run_{r} ensemble MSE={_mse(d + '/test_metrics.csv'):.3f}e-3"))
        return out

    def push_completed(self, wait_stable=True):
        with self.git_lock:
            new = 0
            for path, msg in self._items(wait_stable):
                if not os.path.exists(path):
                    continue
                files = [path] if os.path.isfile(path) else [os.path.join(a, f) for a, _, fs in os.walk(path) for f in fs]
                if any(os.path.getsize(f) > MAX_MB * 1024 ** 2 for f in files):
                    self.events.append(f"{_now()}  bỏ qua {path}: có file > {MAX_MB} MB")
                    continue
                self.git("add", "-A", path, cwd=self.wd, quiet=True)
                if self.git("diff", "--cached", "--quiet", cwd=self.wd, check=False, quiet=True).returncode:
                    self.git("commit", "-q", "-m", msg, cwd=self.wd, quiet=True)
                    new += 1
            if new:
                for attempt in range(3):
                    if self.git("push", "-u", "origin", self.branch, cwd=self.wd, auth=True,
                                check=False, quiet=True).returncode == 0:
                        self.pushed += new
                        self.events.append(f"{_now()}  push {new} commit lên {self.branch}")
                        return new
                    time.sleep(15 * (attempt + 1))
                self.events.append(f"{_now()}  push lỗi, thử lại ở lần sau")
            return 0

    def _safe_push(self):
        try:
            self.push_completed()
        except Exception as e:
            self.events.append(f"{_now()}  push lỗi: {str(e)[:80]}")

    def _fetch(self, branch):
        if not self.git("ls-remote", "--heads", "origin", branch, cwd=self.wd, auth=True, quiet=True).stdout.strip():
            return False
        self.git("fetch", "-q", "--depth", "1", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                 cwd=self.wd, auth=True, quiet=True)
        return True

    def _remote_has(self, branch, path):
        return self._fetch(branch) and self.git("cat-file", "-e", f"origin/{branch}:{path}", cwd=self.wd,
                                                check=False, quiet=True).returncode == 0

    # ---------- kết thúc phần của account ----------
    def finish(self):
        miss = self.missing()
        assert not miss, f"Chưa đủ kết quả: {miss}. Chạy lại notebook để tiếp tục."
        self.push_completed(wait_stable=False)
        marker = f"status/phase2_{self.account}.json"
        os.makedirs("status", exist_ok=True)
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({"account": self.account, "datasets": self.datasets, "run_ids": self.run_ids,
                       "finished_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "session_hours": round((time.time() - self.t_start) / 3600, 2)}, f, indent=2)
        with self.git_lock:
            self.git("add", marker, cwd=self.wd, quiet=True)
            if self.git("diff", "--cached", "--quiet", cwd=self.wd, check=False, quiet=True).returncode:
                self.git("commit", "-q", "-m", f"[{self.account}] hoàn tất {','.join(self.datasets)} run {self.run_ids}",
                         cwd=self.wd, quiet=True)
            self.git("push", "-u", "origin", self.branch, cwd=self.wd, auth=True, quiet=True)
        print(f"Account {self.account}: đủ kết quả {','.join(self.datasets)} run {self.run_ids}, đã push lên {self.branch}.")

        if not self._remote_has(self.other_branch, f"status/phase2_{self.other}.json"):
            print(f"Account {self.other} chưa xong -> dừng tại đây để tiết kiệm quota. "
                  f"Account {self.other} sẽ làm phần cuối khi xong.")
            return False
        r = self.git("push", "origin", f"HEAD:refs/heads/{self.lock_branch}", cwd=self.wd, auth=True,
                     check=False, quiet=True)
        if r.returncode != 0:
            print(f"Account {self.other} đang làm phần cuối (nhánh {self.lock_branch} đã có) -> dừng tại đây.")
            return False
        self.do_final = True
        print(f"Account {self.other} đã xong -> account {self.account} làm phần cuối.")
        return True

    # ---------- phần cuối: gộp 2 nhánh, dựng lại bảng, báo cáo ----------
    def _step(self, title, cmd, gpu=None):
        t0 = time.time()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)}
        r = subprocess.run(cmd, cwd=self.wd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        log = os.path.join(self.log_dir, f"final_{len(os.listdir(self.log_dir))}.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write(r.stdout + r.stderr)
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-3000:])
            raise RuntimeError(f"{title} lỗi (exit {r.returncode}), log: {log}")
        print(f"  {title}: xong ({time.time() - t0:.0f} s)")

    def final(self):
        from run_experiments import parse_run_ids
        all_ds = [d for acc in self.plan.values() for d in acc]
        print(f"Gộp {self.other_branch} vào {self.branch}")
        with self.git_lock:
            self._fetch(self.final_branch)
            self._fetch(self.other_branch)
            files = self.git("diff", "--name-only", f"origin/{self.final_branch}", f"origin/{self.other_branch}",
                             "--", "logs", "cache", "status", cwd=self.wd, quiet=True).stdout.split()
            for i in range(0, len(files), 200):
                self.git("checkout", f"origin/{self.other_branch}", "--", *files[i:i + 200], cwd=self.wd, quiet=True)
        print(f"  lấy {len(files)} file từ {self.other_branch}")
        final_ids = parse_run_ids(self.final_run_ids)
        miss = self.missing(all_ds, final_ids, need_ablation=False)
        assert not miss, f"Thiếu kết quả sau khi gộp: {miss}"

        print(f"Dựng lại bảng kết quả cho run {self.final_run_ids}")
        gpu = self.gpus[0] if self.gpus else None
        for ds in all_ds:
            self._step(f"ST-WaveFormer {ds.upper()}",
                       [self.py, "run_experiments.py", "--model", "STWaveFormer", "--dataset", ds,
                        "--run_ids", self.final_run_ids, "--skip_existing"], gpu=gpu)
        self._step("4 mô hình học máy",
                   [self.py, "baselines_ml/run_ml_baselines.py", "--models", ",".join(ML_ALL),
                    "--datasets", ",".join(all_ds), "--run_ids", self.final_run_ids, "--skip_existing"])
        for ds in all_ds:
            rows = []
            for r in final_ids:
                d = pd.read_csv(os.path.join(self.ens_dir(ds, r), "test_metrics.csv"))
                d["run"] = r
                rows.append(d)
            pd.concat(rows, ignore_index=True).to_csv(f"results/results_STAdaptiveEnsemble_data_{ds}.csv", index=False)
        print("  Ensemble: xong")
        self._step("Ablation (tổng hợp)",
                   [self.py, "evaluation/ablation_study.py", "--summary_only", "--run_ids", self.final_run_ids])
        ab = pd.read_csv("results/ablation_summary.csv")
        assert (ab["runs"] == len(final_ids)).all() and set(ab["dataset"]) == {d.upper() for d in all_ds},             f"Ablation chưa đủ {len(final_ids)} run cho mọi dataset/cấu hình"
        self._step("Báo cáo và hình vẽ",
                   [self.py, "-c", "from evaluation.report_generator import generate_thesis_report; "
                                   "from evaluation.plot_gate_dynamics import plot_all_thesis_figures; "
                                   "generate_thesis_report(); plot_all_thesis_figures()"])

        with self.git_lock:
            self.git("add", "-A", "logs", "cache", "results", "status", cwd=self.wd, quiet=True)
            if self.git("diff", "--cached", "--quiet", cwd=self.wd, check=False, quiet=True).returncode:
                self.git("commit", "-q", "-m", f"Gộp {self.branch} + {self.other_branch}, báo cáo run {self.final_run_ids}",
                         cwd=self.wd, quiet=True)
            self.git("push", "origin", self.branch, cwd=self.wd, auth=True, quiet=True)
            self.git("push", "origin", f"HEAD:{self.final_branch}", cwd=self.wd, auth=True)
        print(f"Đã push kết quả cuối lên nhánh {self.final_branch}.")

    def summary(self):
        pd.set_option("display.width", 200)
        t = pd.read_csv("results/bang_tong_hop_luan_van.csv")
        print(t[["Dataset", "Model", "Runs", "MSE (x10^-3)", "MAE (x10^-3)"]].to_string(index=False))
        k = pd.read_csv("results/kiem_dinh_thong_ke.csv")
        print()
        print(k.round(4).to_string(index=False))
        a = pd.read_csv("results/ablation_summary.csv")
        print()
        print(a.pivot(index="config", columns="dataset", values="mean_mse").round(4).to_string())
