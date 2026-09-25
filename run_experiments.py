import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import time
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange

# Cấu hình sys.path tương thích đa nền tảng (Colab, Linux, Windows, Notebooks)
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
for p in [current_dir, os.path.join(current_dir, 'Graph_models')]:
    abs_p = os.path.abspath(p)
    if abs_p not in sys.path:
        sys.path.insert(0, abs_p)

try:
    from Graph_models.gwn import GWNet
    from Graph_models.dcrnn import DCRNNModel
    from Graph_models.st_waveformer import STWaveFormer
except ImportError:
    from gwn import GWNet
    from dcrnn import DCRNNModel
    from st_waveformer import STWaveFormer

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)" if torch.cuda.is_available() else "CPU"

# ==============================================================================
# Model Definitions
# ==============================================================================


class LSTM_TM(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, seq_len=24):
        super(LSTM_TM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, :, :, 0]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class BiLSTM_TM(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, seq_len=24):
        super(BiLSTM_TM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, :, :, 0]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class GRU_TM(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, seq_len=24):
        super(GRU_TM, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, :, :, 0]
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


class BiGRU_TM(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, seq_len=24):
        super(BiGRU_TM, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, :, :, 0]
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


# ==============================================================================
# Metric Calculations & Seed Control
# ==============================================================================
EPS = 1e-8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calc_metrics(preds, labels):
    preds = preds.float()
    labels = labels.float()
    rse = torch.sum((preds - labels) ** 2) / (torch.sum((labels - torch.mean(labels)) ** 2) + EPS)
    mae = torch.mean(torch.abs(preds - labels))
    mse = torch.mean((preds - labels) ** 2)
    rmse = torch.sqrt(mse)
    # WAPE & sMAPE triệt tiêu hiện tượng chia cho số cận 0
    wape = torch.sum(torch.abs(preds - labels)) / (torch.sum(torch.abs(labels)) + EPS)
    smape = torch.mean(2.0 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels) + EPS)) * 100.0
    # Masked MAPE loại trừ các mẫu cận 0
    mask = torch.abs(labels) > 1e-4
    if torch.any(mask):
        mape = torch.mean(torch.abs((preds[mask] - labels[mask]) / labels[mask]))
    else:
        mape = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    return rse, mae, mse, mape, rmse, wape, smape


# ==============================================================================
# Dataset and Data Loading
# ==============================================================================

class TrafficDataset(Dataset):
    def __init__(self, model_name, x, y, device):
        self.model_name = model_name.lower()
        self.device = device
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.nsample = self.x.shape[0]

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.model_name in ['stwaveformer', 'st-waveformer']:
            # STWaveFormer nhận đầu vào đa kênh [seq_len, num_flows, channels]
            pass
        else:
            if x.dim() == 3:
                x = x[:, :, 0]
            if self.model_name == 'gwn':
                x = torch.unsqueeze(x, -1)

        if len(y.shape) > 1 and y.shape[0] == 1:
            y = torch.squeeze(y, dim=0)

        return {'x': x, 'y': y}


def prepare_dataset(dataset_name, in_seq_len, out_seq_len=1, batch_size=64, model_name='lstm'):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    fpath = os.path.join(data_dir, f'{dataset_name}.csv')
    if not os.path.exists(fpath):
        fpath = os.path.join(data_dir, f'{dataset_name.upper()}.csv')

    df = pd.read_csv(fpath, parse_dates=['time'])
    df = df.set_index(['time'])

    # Đảm bảo tính đơn điệu của chuỗi thời gian và khử trùng lặp (ví dụ 288 mốc trùng ở Abilene)
    if not df.index.is_monotonic_increasing:
        n_before = len(df)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        print(f"[{dataset_name.upper()}] Cảnh báo: Đã loại bỏ {n_before - len(df)} timestamp trùng lặp, đảm bảo tính đơn điệu.")

    total_steps = len(df)
    train_size = int(total_steps * 0.7)                 # 70% Train
    val_size = int(total_steps * 0.1)                   # 10% Validation

    train_df = df.iloc[0:train_size]                    # 70% đầu tiên theo trục thời gian
    val_df = df.iloc[train_size:train_size + val_size]  # 10% tiếp theo
    test_df = df.iloc[train_size + val_size:]           # 20% cuối cùng (tương lai)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_norm = scaler.fit_transform(train_df)
    val_norm = scaler.transform(val_df)
    test_norm = scaler.transform(test_df)

    # TOD và DOW chuẩn hóa: Time-of-day (0-1) and Day-of-week (0-1)
    time_idx = df.index
    tod = (time_idx.hour * 60.0 + time_idx.minute) / 1440.0
    dow = time_idx.dayofweek / 7.0

    num_flows = train_norm.shape[1]
    tod_arr = np.tile(tod.values[:, None], (1, num_flows))
    dow_arr = np.tile(dow.values[:, None], (1, num_flows))

    # Stack channels: [0]=traffic, [1]=tod, [2]=dow
    comb_train = np.stack([train_norm, tod_arr[0:train_size], dow_arr[0:train_size]], axis=-1).astype(np.float32)
    comb_val = np.stack([val_norm, tod_arr[train_size:train_size + val_size],
                        dow_arr[train_size:train_size + val_size]], axis=-1).astype(np.float32)
    comb_test = np.stack([test_norm, tod_arr[train_size + val_size:],
                         dow_arr[train_size + val_size:]], axis=-1).astype(np.float32)

    def create_sliding_window(arr, seq_in, seq_out):
        xs, ys = [], []
        for i in range(len(arr) - seq_in - seq_out + 1):
            xs.append(arr[i: i + seq_in])
            ys.append(arr[i + seq_in: i + seq_in + seq_out, :, 0])  # target is traffic volume
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    x_train, y_train = create_sliding_window(comb_train, in_seq_len, out_seq_len)
    x_val, y_val = create_sliding_window(comb_val, in_seq_len, out_seq_len)
    x_test, y_test = create_sliding_window(comb_test, in_seq_len, out_seq_len)

    train_dataset = TrafficDataset(model_name, x_train, y_train, device)
    val_dataset = TrafficDataset(model_name, x_val, y_val, device)
    test_dataset = TrafficDataset(model_name, x_test, y_test, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, num_flows, list(df.columns)


# ==============================================================================
# Model Factory
# ==============================================================================

def build_model(model_name, dataset_name, in_seq_len, num_flows, num_nodes):
    m_name = model_name.lower().replace('-', '')
    if m_name == 'lstm':
        return LSTM_TM(input_dim=num_flows, hidden_dim=100, seq_len=in_seq_len)
    elif m_name == 'bilstm':
        return BiLSTM_TM(input_dim=num_flows, hidden_dim=100, seq_len=in_seq_len)
    elif m_name == 'gru':
        return GRU_TM(input_dim=num_flows, hidden_dim=100, seq_len=in_seq_len)
    elif m_name == 'bigru':
        return BiGRU_TM(input_dim=num_flows, hidden_dim=100, seq_len=in_seq_len)
    elif m_name == 'gwn':
        return GWNet.from_args(
            supports=None, aptinit=None, num_nodes=num_nodes, num_flows=num_flows, in_dim=1,
            apt_size=10, out_seq_len=1, in_seq_len=in_seq_len, hidden=32,
            stride=2, kernel_size=2, blocks=2, layers=2,
            cat_feat_gc=False, do_graph_conv=True, addaptadj=True,
            dropout=0.5, batch_size=64, device=str(device), verbose=False
        )
    elif m_name == 'dcrnn':
        adj_path = os.path.join(os.path.dirname(__file__), 'data', f'{dataset_name}_adj.npy')
        if not os.path.exists(adj_path):
            adj_mx = np.eye(num_nodes, dtype=np.float32)
        else:
            adj_mx = np.load(adj_path)
        return DCRNNModel(adj_mx=adj_mx, seq_len=in_seq_len, nodes=num_nodes, pre_len=1, device=device, num_rnn_layers=2, rnn_units=32)
    elif m_name in ['stwaveformer', 'st_waveformer']:
        return STWaveFormer(input_dim=num_flows, num_nodes=num_nodes, seq_len=in_seq_len, d_model=64, num_layers=2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# ==============================================================================
# Training & Testing Functions
# ==============================================================================

def train_and_eval_model(model, train_loader, val_loader, test_loader, scaler=None, columns=None, epochs=200, patience=30, lr=1e-3, weight_decay=1e-4, logdir='logs', model_name='lstm', dataset_name='', run_id=0, total_runs=1, seed=42):
    os.makedirs(logdir, exist_ok=True)
    m_name = model_name.lower().replace('-', '')

    if m_name in ['stwaveformer', 'st_waveformer']:
        lossfn = nn.SmoothL1Loss(beta=0.01)
    else:
        lossfn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if m_name in ['stwaveformer', 'st_waveformer']:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: (0.97) ** ep)

    best_val_loss = float('inf')
    best_model_path = os.path.join(logdir, 'best_model.pth')
    patience_counter = 0
    history = []

    model.to(device)
    print(f"Device: {device_name}", flush=True)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            optimizer.zero_grad()
            out = model(x)
            if out.dim() == 4:
                out = out[:, :, :, -1]
            if out.dim() == 3 and out.size(1) == 1:
                out = out.squeeze(1)
            loss = lossfn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch['x'].to(device), batch['y'].to(device)
                out = model(x)
                if out.dim() == 4:
                    out = out[:, :, :, -1]
                if out.dim() == 3 and out.size(1) == 1:
                    out = out.squeeze(1)
                v_loss = lossfn(out, y)
                val_losses.append(v_loss.item())

        mean_tr_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_losses)

        history.append({
            'epoch': epoch + 1,
            'seed': seed,
            'lr': current_lr,
            'train_loss': mean_tr_loss,
            'val_loss': mean_val_loss
        })

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0
            with open(best_model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            saved_str = "*"
        else:
            patience_counter += 1
            saved_str = " "

        # in ra màn hình history kèm thông tin run
        run_info = f" | Run {run_id + 1}/{total_runs} (Seed: {seed})" if total_runs > 1 else f" (Seed: {seed})"
        tag = f"[{model_name}/{dataset_name.upper()}{run_info}]" if dataset_name else f"[{model_name}{run_info}]"
        print(f"  {tag} Epoch {epoch+1:03d}/{epochs} | LR: {current_lr:.6f} | Train Loss: {mean_tr_loss:.6f} | Val Loss: {mean_val_loss:.6f} (Best: {best_val_loss:.6f}){saved_str} | Patience: {patience_counter}/{patience}", flush=True)

        if patience_counter >= patience:
            print(f"  --> Early stopping triggered at epoch {epoch+1}", flush=True)
            break

    # Save history at the end of training
    pd.DataFrame(history).to_csv(os.path.join(logdir, 'train_metrics.csv'), index=False)

    # Load best model for testing
    if os.path.exists(best_model_path):
        with open(best_model_path, 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=device))

    # Test evaluation & inference timing
    model.eval()
    all_preds, all_reals = [], []
    inference_times = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            start_t = time.perf_counter()
            out = model(x)
            end_t = time.perf_counter()
            inference_times.append((end_t - start_t) * 1000.0)  # in ms

            if out.dim() == 4:
                out = out[:, :, :, -1]
            if out.dim() == 3 and out.size(1) == 1:
                out = out.squeeze(1)

            # NOTE: Bỏ clamp trong không gian normalized
            all_preds.append(out.cpu())
            all_reals.append(y.cpu())

    y_hat = torch.cat(all_preds, dim=0)
    y_real = torch.cat(all_reals, dim=0)

    rse, mae, mse, mape, rmse, wape, smape = calc_metrics(y_hat, y_real)
    avg_inference_time = np.mean(inference_times)  # ms

    # Đánh giá trên thang đo Raw sau inverse_transform
    y_hat_np = y_hat.numpy()
    y_real_np = y_real.numpy()
    if scaler is not None:
        y_hat_raw = scaler.inverse_transform(y_hat_np)
        y_real_raw = scaler.inverse_transform(y_real_np)
    else:
        y_hat_raw = y_hat_np.copy()
        y_real_raw = y_real_np.copy()

    # Chỉ clamp về >= 0 sau khi đã chuyển prediction về traffic raw thực tế
    y_hat_raw = np.clip(y_hat_raw, a_min=0.0, a_max=None)
    y_real_raw = np.clip(y_real_raw, a_min=0.0, a_max=None)

    raw_mae = float(np.mean(np.abs(y_hat_raw - y_real_raw)))
    raw_mse = float(np.mean((y_hat_raw - y_real_raw) ** 2))
    raw_rmse = float(np.sqrt(raw_mse))
    raw_wape = float(np.sum(np.abs(y_hat_raw - y_real_raw)) / (np.sum(np.abs(y_real_raw)) + EPS))
    raw_smape = float(np.mean(2.0 * np.abs(y_hat_raw - y_real_raw) / (np.abs(y_hat_raw) + np.abs(y_real_raw) + EPS)) * 100.0)

    test_metrics = {
        'seed': seed,
        'lr': lr,
        'mse': float(mse.item()),
        'mae': float(mae.item()),
        'rmse': float(rmse.item()),
        'rse': float(rse.item()),
        'mape': float(mape.item()),
        'wape': float(wape.item()),
        'smape': float(smape.item()),
        'raw_mae': raw_mae,
        'raw_rmse': raw_rmse,
        'raw_wape': raw_wape,
        'raw_smape': raw_smape,
        'inference_time_ms': float(avg_inference_time)
    }

    # Giữ riêng flow OD_10-8 trong báo cáo như một trường hợp Distribution Shift
    if columns is not None:
        od_10_8_idx = None
        for idx, col in enumerate(columns):
            if '10-8' in str(col) or '10_8' in str(col):
                od_10_8_idx = idx
                break
        if od_10_8_idx is not None:
            f_pred = y_hat_raw[:, od_10_8_idx]
            f_real = y_real_raw[:, od_10_8_idx]
            test_metrics['od_10_8_mae_raw'] = float(np.mean(np.abs(f_pred - f_real)))
            test_metrics['od_10_8_rmse_raw'] = float(np.sqrt(np.mean((f_pred - f_real) ** 2)))
            test_metrics['od_10_8_wape_raw'] = float(np.sum(np.abs(f_pred - f_real)) / (np.sum(np.abs(f_real)) + EPS))

    # Save test logs
    test_df = pd.DataFrame([test_metrics])
    test_df.to_csv(os.path.join(logdir, 'test_metrics.csv'), index=False)
    np.save(os.path.join(logdir, 'y_real_data.npy'), y_real_np)
    np.save(os.path.join(logdir, 'y_pred_data.npy'), y_hat_np)
    np.save(os.path.join(logdir, 'y_real_data_raw.npy'), y_real_raw)
    np.save(os.path.join(logdir, 'y_pred_data_raw.npy'), y_hat_raw)

    # Dọn dẹp bộ nhớ RAM / VRAM
    model.to('cpu')
    del all_preds, all_reals, y_hat, y_real
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return test_metrics


# ==============================================================================
# Main Experiment Runner
# ==============================================================================

DATASET_CONFIGS = {
    'sdn': {'nodes': 14, 'flows': 196, 'seq_len': 60},
    'geant': {'nodes': 23, 'flows': 529, 'seq_len': 24},
    'abilene': {'nodes': 12, 'flows': 144, 'seq_len': 24}
}

MODELS_LIST = ['BiGRU', 'GWN', 'STWaveFormer']


def parse_run_ids(run_ids=None, runs=1):
    """'5-9' | '0,1,2' | list | None (-> range(runs))."""
    if run_ids is None or run_ids == '':
        return list(range(runs))
    if isinstance(run_ids, (list, tuple)):
        return [int(r) for r in run_ids]
    out = []
    for part in str(run_ids).split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def expected_test_windows(dataset_name, seq_len):
    """Số cửa sổ Test của dữ liệu hiện tại (sau khử trùng lặp) - dùng để phát hiện log cũ."""
    from features.feature_store import load_raw_dataset
    total = len(load_raw_dataset(dataset_name))
    test_len = total - int(total * 0.7) - int(total * 0.1)
    return test_len - seq_len


def check_existing_run(logdir, n_test_windows):
    """
    Trả về (hợp lệ, lý do). Một run cũ bị coi là KHÔNG hợp lệ khi:
    - test_metrics.csv không có cột seed (phiên bản mã cũ), hoặc
    - số bước test trong y_real_data.npy khác dữ liệu hiện tại (ví dụ log sinh trước khi khử trùng lặp).
    """
    f = os.path.join(logdir, 'test_metrics.csv')
    if not os.path.exists(f):
        return False, 'chưa có kết quả'
    df = pd.read_csv(f)
    if df.empty:
        return False, 'test_metrics.csv rỗng'
    if 'seed' not in df.columns or pd.isna(df.iloc[0]['seed']):
        return False, 'log phiên bản cũ (thiếu seed)'
    yf = os.path.join(logdir, 'y_real_data.npy')
    if os.path.exists(yf):
        n = np.load(yf, mmap_mode='r').shape[0]
        if n != n_test_windows:
            return False, f'số bước test {n} khác dữ liệu hiện tại {n_test_windows}'
    if not os.path.exists(os.path.join(logdir, 'best_model.pth')):
        return False, 'thiếu best_model.pth'
    return True, 'ok'


def run_all_experiments(datasets=None, models=None, epochs=200, patience=30, runs=1, skip_existing=False,
                        run_ids=None, allow_cpu=False):
    if datasets is None:
        datasets = ['sdn', 'geant', 'abilene']
    if models is None:
        models = MODELS_LIST
    run_id_list = parse_run_ids(run_ids, runs)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    summary_records = []

    print("=" * 80, flush=True)
    print(" VANNT NETWORK TRAFFIC PREDICTION (ST-WAVEFORMER)", flush=True)
    print(f"Device: {device_name}", flush=True)
    print(f"Datasets: {datasets} | Models: {models} | Run IDs: {run_id_list} | Epochs: {epochs} | Patience: {patience} | Skip existing: {skip_existing}", flush=True)
    print("=" * 80, flush=True)

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        seq_len = cfg['seq_len']
        num_nodes = cfg['nodes']
        num_flows = cfg['flows']

        print(
            f"\nDATASET: {ds.upper()} (nodes={num_nodes}, flows={num_flows}, seq_len={seq_len})", flush=True)

        n_test_windows = expected_test_windows(ds, seq_len)

        for m_name in models:
            run_metrics = []

            for run_id in run_id_list:
                seed = SEEDS[run_id % len(SEEDS)]
                set_seed(seed)

                logdir = os.path.join(
                    'logs', f"{m_name.lower().replace('-','')}_data_{ds}_seq_{seq_len}", f"run_{run_id}")
                test_metrics_path = os.path.join(logdir, 'test_metrics.csv')
                run_str = f" [run_{run_id} | Seed: {seed}]"

                # Kiểm tra cờ --skip_existing: chỉ tái sử dụng run hợp lệ với dữ liệu và mã hiện tại
                if skip_existing and os.path.exists(test_metrics_path):
                    valid, reason = check_existing_run(logdir, n_test_windows)
                    if valid:
                        metrics = pd.read_csv(test_metrics_path).iloc[0].to_dict()
                        print(f"\n[SKIP] Đã có kết quả hợp lệ: {m_name} trên {ds.upper()}{run_str}", flush=True)
                        metrics['run'] = run_id
                        metrics['seq_len'] = seq_len
                        run_metrics.append(metrics)
                        continue
                    print(f"\n[RETRAIN] {m_name} trên {ds.upper()}{run_str}: {reason} -> huấn luyện lại.", flush=True)

                if not torch.cuda.is_available() and not allow_cpu:
                    raise RuntimeError(
                        f"Cần huấn luyện {m_name} {ds.upper()} run_{run_id} nhưng máy không có GPU. "
                        f"Hãy chạy bước này trên Kaggle/Colab (GPU), hoặc thêm --allow_cpu nếu thực sự muốn chạy CPU.")

                print(f"\n---> Training: {m_name} on {ds.upper()} dataset{run_str}...", flush=True)

                train_loader, val_loader, test_loader, scaler, _, columns = prepare_dataset(
                    ds, in_seq_len=seq_len, out_seq_len=1, batch_size=64, model_name=m_name
                )
                model = build_model(m_name, ds, seq_len, num_flows, num_nodes)
                metrics = train_and_eval_model(
                    model, train_loader, val_loader, test_loader, scaler=scaler, columns=columns,
                    epochs=epochs, patience=patience, logdir=logdir, model_name=m_name, dataset_name=ds,
                    run_id=run_id, total_runs=len(run_id_list), seed=seed
                )

                metrics['run'] = run_id
                metrics['seq_len'] = seq_len
                run_metrics.append(metrics)

                # Dọn dẹp bộ nhớ sau mỗi lần chạy
                if 'model' in locals():
                    del model
                if 'train_loader' in locals():
                    del train_loader, val_loader, test_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Gộp với các run đã có (chạy theo đợt), bỏ các dòng của run không còn hợp lệ
            df_runs = pd.DataFrame(run_metrics)
            out_csv = os.path.join(results_dir, f"results_{m_name.replace('-', '')}_data_{ds}.csv")
            if os.path.exists(out_csv):
                try:
                    old = pd.read_csv(out_csv)
                    if 'run' in old.columns:
                        old = old[~old['run'].isin(df_runs['run'])]
                        keep = [r for r in old['run'].unique()
                                if check_existing_run(os.path.join('logs', f"{m_name.lower().replace('-','')}_data_{ds}_seq_{seq_len}", f"run_{int(r)}"), n_test_windows)[0]]
                        df_runs = pd.concat([old[old['run'].isin(keep)], df_runs], ignore_index=True)
                except Exception:
                    pass
            df_runs = df_runs.sort_values('run').reset_index(drop=True)
            df_runs.to_csv(out_csv, index=False)

            mean_mse = df_runs['mse'].mean()
            mean_mae = df_runs['mae'].mean()
            mean_rmse = df_runs['rmse'].mean()
            mean_rse = df_runs['rse'].mean()
            mean_mape = df_runs['mape'].mean()
            mean_wape = df_runs['wape'].mean() if 'wape' in df_runs else 0.0
            mean_raw_mae = df_runs['raw_mae'].mean() if 'raw_mae' in df_runs else mean_mae
            mean_raw_rmse = df_runs['raw_rmse'].mean() if 'raw_rmse' in df_runs else mean_rmse
            mean_raw_wape = df_runs['raw_wape'].mean() if 'raw_wape' in df_runs else 0.0
            mean_time = df_runs['inference_time_ms'].mean()

            record = {
                'Dataset': ds.upper(),
                'Model': m_name,
                'Seq_Len': seq_len,
                'MSE (x10^-3)': mean_mse * 1000.0,
                'MAE (x10^-3)': mean_mae * 1000.0,
                'RMSE': mean_rmse,
                'RSE': mean_rse,
                'WAPE (%)': mean_wape * 100.0,
                'Raw MAE': mean_raw_mae,
                'Raw RMSE': mean_raw_rmse,
                'Raw WAPE (%)': mean_raw_wape * 100.0,
                'Inference Time (ms)': mean_time
            }
            if 'od_10_8_mae_raw' in df_runs:
                record['OD_10-8 Raw MAE'] = df_runs['od_10_8_mae_raw'].mean()
            summary_records.append(record)

            print(
                f"[*] Kết quả {m_name} trên {ds.upper()}: MSE={mean_mse*1000.0:.3f}e-3 | MAE={mean_mae*1000.0:.3f}e-3 | Raw MAE={mean_raw_mae:.3f} | WAPE={mean_wape*100.0:.2f}% | Time={mean_time:.3f} ms", flush=True)

    # Tổng hợp bảng kết quả danh gia mo hinh
    try:
        from plot_comparisons import collect_results_from_dir
        summary_df = collect_results_from_dir(results_dir=results_dir)
        summary_csv = os.path.join(results_dir, 'bang_ket_qua_danh_gia_mo_hinh.csv')
    except Exception as e:
        summary_df = pd.DataFrame(summary_records)
        summary_csv = os.path.join(results_dir, 'bang_ket_qua_danh_gia_mo_hinh.csv')
        summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 90, flush=True)
    print(" BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (THỦ CÔNG / CỦA LUẬN VĂN) ", flush=True)
    print("=" * 90, flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print("=" * 90, flush=True)
    print(f"-> Đã lưu bảng kết quả tổng hợp tích lũy tại: {summary_csv}", flush=True)

    return summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Network Traffic Prediction & Model Evaluation")
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'sdn', 'geant', 'abilene'])
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'LSTM', 'BiLSTM', 'GRU', 'BiGRU', 'GWN', 'DCRNN', 'STWaveFormer'])
    parser.add_argument('--epochs', type=int, default=200, help='Max training epochs per model')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--runs', type=int, default=1, help='Number of repeated runs (run_0..run_{runs-1})')
    parser.add_argument('--run_ids', type=str, default=None, help="Chỉ định run, ví dụ '0-4' hoặc '5,6' (ghi đè --runs)")
    parser.add_argument('--allow_cpu', action='store_true', help='Cho phép huấn luyện DL trên CPU (mặc định bị chặn)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip already completed runs (load test_metrics.csv)')
    parser.add_argument('--quick_check', action='store_true', help='Run 2 epochs for quick pipeline verification')

    args = parser.parse_args()

    ds_list = [args.dataset] if args.dataset != 'all' else ['sdn', 'geant', 'abilene']
    m_list = [args.model] if args.model != 'all' else MODELS_LIST

    if args.quick_check:
        print("=== CHẾ ĐỘ KIỂM TRA NHANH (QUICK CHECK) ===")
        run_all_experiments(datasets=ds_list, models=m_list, epochs=2, patience=2,
                            runs=args.runs, skip_existing=args.skip_existing,
                            run_ids=args.run_ids, allow_cpu=True)
    else:
        run_all_experiments(datasets=ds_list, models=m_list, epochs=args.epochs,
                            patience=args.patience, runs=args.runs, skip_existing=args.skip_existing,
                            run_ids=args.run_ids, allow_cpu=args.allow_cpu)
