import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import zipfile
import gzip
import glob
import pandas as pd
import numpy as np
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

print("=== BẮT ĐẦU CHUẨN BỊ BỘ DỮ LIỆU ĐẦU VÀO ===")

# 1. Chuẩn bị SDN.csv / sdn.csv
sdn_upper = os.path.join(DATA_DIR, 'SDN.csv')
sdn_lower = os.path.join(DATA_DIR, 'sdn.csv')
if os.path.exists(sdn_upper):
    df_sdn = pd.read_csv(sdn_upper, header=None)
    if df_sdn.shape[0] == 6258:
        # Bỏ dòng header lỗi đầu tiên nếu có
        df_sdn = df_sdn.iloc[1:].copy()
    col_names = ['time'] + [f'OD_{i}-{j}' for i in range(1, 15) for j in range(1, 15)]
    df_sdn.columns = col_names
    df_sdn.to_csv(sdn_lower, index=False)
    print("-> Bộ dữ liệu SDN đã sẵn sàng tại:", sdn_lower, "với kích thước:", df_sdn.shape)

# 2. Giải nén và chuẩn bị geant.csv
geant_zip = os.path.join(RESULTS_DIR, 'GEANT-OD_pair_time_convert.csv.zip')
geant_csv = os.path.join(DATA_DIR, 'geant.csv')

if not os.path.exists(geant_csv):
    print("-> Đang giải nén geant.csv từ zip...")
    with zipfile.ZipFile(geant_zip, 'r') as z:
        with z.open('GEANT-OD_pair_time_convert.csv') as source, open(geant_csv, 'wb') as target:
            shutil.copyfileobj(source, target)
    print("-> Đã tạo thành công:", geant_csv)
else:
    print("-> File geant.csv đã tồn tại tại:", geant_csv)

# 3. Tạo abilene.csv từ các file X01.gz ... X24.gz
abilene_csv = os.path.join(DATA_DIR, 'abilene.csv')
if not os.path.exists(abilene_csv):
    print("-> Đang trích xuất và tổng hợp abilene.csv từ data/AbileneTM-all/...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'AbileneTM-all/X*.gz')))
    
    dates_map = {
        'X01': '2004-03-01', 'X02': '2004-03-08', 'X03': '2004-04-02', 'X04': '2004-04-09',
        'X05': '2004-04-22', 'X06': '2004-05-01', 'X07': '2004-05-08', 'X08': '2004-05-15',
        'X09': '2004-05-22', 'X10': '2004-05-29', 'X11': '2004-06-05', 'X12': '2004-06-12',
        'X13': '2004-06-19', 'X14': '2004-06-26', 'X15': '2004-07-03', 'X16': '2004-07-10',
        'X17': '2004-07-17', 'X18': '2004-07-24', 'X19': '2004-07-31', 'X20': '2004-08-07',
        'X21': '2004-08-13', 'X22': '2004-08-21', 'X23': '2004-08-28', 'X24': '2004-09-04'
    }
    
    all_dfs = []
    cols = [f'OD_{i}-{j}' for i in range(1, 13) for j in range(1, 13)]
    
    for f in files:
        name = os.path.basename(f).split('.')[0]
        with gzip.open(f, 'rt') as gz:
            arr = np.array([l.strip().split() for l in gz if l.strip()], dtype=float)
        real_od = arr[:, 0::5]
        start_time = dates_map.get(name, '2004-03-01')
        time_idx = pd.date_range(start=start_time, periods=len(real_od), freq='5min')
        df_week = pd.DataFrame(real_od, columns=cols)
        df_week.insert(0, 'time', time_idx)
        all_dfs.append(df_week)
        
    total_df = pd.concat(all_dfs, ignore_index=True)
    # Lấy 48096 bản ghi chuẩn như mô tả Table 12 bài báo
    if len(total_df) > 48096:
        total_df = total_df.iloc[:48096]
    total_df.to_csv(abilene_csv, index=False)
    print("-> Đã tạo thành công:", abilene_csv, "với kích thước:", total_df.shape)
else:
    print("-> File abilene.csv đã tồn tại tại:", abilene_csv)

print("=== HOÀN TẤT CHUẨN BỊ CẢ 3 BỘ DỮ LIỆU ===")
