import re
import numpy as np
import scipy.sparse as sp


def parse_od_columns(columns):
    """
    Phân tích danh sách tên cột để trích xuất cặp (src, dst) cho từng luồng OD.
    Định dạng mong đợi: 'OD_{src}-{dst}' hoặc tương đương.
    """
    od_pairs = []
    pattern = re.compile(r"OD_([A-Za-z0-9]+)[-_]([A-Za-z0-9]+)", re.IGNORECASE)
    
    for idx, col in enumerate(columns):
        m = pattern.search(col)
        if m:
            od_pairs.append((m.group(1), m.group(2)))
        else:
            # Fallback nếu tên cột không theo mẫu OD_x-y
            od_pairs.append((str(idx), str(idx)))
            
    return od_pairs


def build_od_topology_matrices(columns, as_sparse=False):
    """
    Xây dựng 2 ma trận ánh xạ nguồn/đích M_in và M_out kích thước [N, N].
    - M_in[i, j] = 1 nếu luồng j != i có cùng node đích với luồng i (cùng đổ về 1 đích).
    - M_out[i, j] = 1 nếu luồng j != i có cùng node nguồn với luồng i (cùng xuất phát từ 1 nguồn).
    
    Khi nhân với vector lưu lượng X[t-1] kích thước [T, N]:
    - neighbor_in = X @ M_in.T  ([T, N])
    - neighbor_out = X @ M_out.T ([T, N])
    Phép tính hoàn tất trong vài mili-giây mà không cần duyệt vòng lặp.
    """
    od_pairs = parse_od_columns(columns)
    N = len(od_pairs)
    
    M_in = np.zeros((N, N), dtype=np.float32)
    M_out = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        src_i, dst_i = od_pairs[i]
        for j in range(N):
            if i == j:
                continue
            src_j, dst_j = od_pairs[j]
            if dst_i == dst_j:
                M_in[i, j] = 1.0
            if src_i == src_j:
                M_out[i, j] = 1.0
                
    if as_sparse:
        return sp.csr_matrix(M_in), sp.csr_matrix(M_out)
    return M_in, M_out


def compute_spatial_neighbors(traffic_arr, M_in, M_out):
    """
    Tính nhanh neighbor_sum_in và neighbor_sum_out cho toàn bộ chuỗi thời gian.
    - traffic_arr: mảng numpy [T, N] (ví dụ giá trị tại t-1)
    - M_in, M_out: ma trận [N, N]
    
    Trả về:
    - neighbor_in: [T, N]
    - neighbor_out: [T, N]
    """
    if sp.issparse(M_in):
        neighbor_in = traffic_arr @ M_in.T.toarray()
        neighbor_out = traffic_arr @ M_out.T.toarray()
    else:
        neighbor_in = np.matmul(traffic_arr, M_in.T)
        neighbor_out = np.matmul(traffic_arr, M_out.T)
        
    return neighbor_in, neighbor_out
