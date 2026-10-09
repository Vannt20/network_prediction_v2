"""
Đồ thị luồng cho OD-GraphFormer và RL-Gate (spec Mục 3).

  A_od    : hai luồng chung nút nguồn hoặc chung nút đích (từ build_od_topology_matrices)
  A_route : chuẩn hóa(R^T R) + I, R [E, N] là ma trận định tuyến liên kết x luồng theo đường đi
            ngắn nhất (số hop), chia đều khi có nhiều đường bằng nhau (ECMP). Luồng tự thân có cột 0.

Nguồn liên kết:
  topology='physical': data/topology/{ds}_links.csv (cột src,dst; chỉ số nút trùng tên cột OD_{src}-{dst})
  topology='knn'     : ĐỒ THỊ SUY DIỄN từ data/{ds}_adj.npy (cây khung trọng số lớn nhất + k cạnh mạnh nhất mỗi nút).
                       *_adj.npy là ma trận trọng số liên tục, KHÔNG phải topo vật lý (spec F1).
"""
import os
import sys
import argparse
from collections import deque
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.spatial_features import parse_od_columns, build_od_topology_matrices

DATA_DIR = os.path.join(parent_dir, 'data')


def sym_norm(A, add_self=True):
    A = np.asarray(A, dtype=np.float64)
    if add_self:
        A = A + np.eye(len(A))
    d = A.sum(axis=1)
    d_inv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    return (d_inv[:, None] * A * d_inv[None, :]).astype(np.float32)


def node_ids(columns):
    """Danh sách nút (giữ thứ tự xuất hiện của nút nguồn) và ánh xạ nhãn -> chỉ số 0..n-1."""
    pairs = parse_od_columns(columns)
    labels = []
    for s, d in pairs:
        for v in (s, d):
            if v not in labels:
                labels.append(v)
    try:
        labels = sorted(labels, key=lambda v: int(v))
    except ValueError:
        pass
    return pairs, {v: i for i, v in enumerate(labels)}


def links_physical(ds_key, label2idx):
    f = os.path.join(DATA_DIR, 'topology', f'{ds_key}_links.csv')
    if not os.path.exists(f):
        raise FileNotFoundError(f"Chưa có {f} (topo vật lý). Dùng --topology knn hoặc bổ sung file (spec Mục 3.2).")
    import pandas as pd
    df = pd.read_csv(f)
    E = set()
    for s, d in zip(df['src'].astype(str), df['dst'].astype(str)):
        a, b = label2idx[s], label2idx[d]
        if a != b:
            E.add((min(a, b), max(a, b)))
    return sorted(E)


def links_knn(ds_key, n, k=2):
    """Cây khung trọng số lớn nhất (bảo đảm liên thông) hợp với k cạnh mạnh nhất của mỗi nút."""
    A = np.load(os.path.join(DATA_DIR, f'{ds_key}_adj.npy')).astype(np.float64)
    assert A.shape == (n, n), f"{ds_key}_adj.npy có kích thước {A.shape}, cần ({n},{n})"
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, -np.inf)
    E = set()
    parent = list(range(n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    cand = sorted(((A[i, j], i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] > 0), reverse=True)
    for _, i, j in cand:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            E.add((i, j))
    for i in range(n):
        for j in np.argsort(-A[i])[:k]:
            if np.isfinite(A[i, j]) and A[i, j] > 0:
                E.add((min(i, int(j)), max(i, int(j))))
    if not _connected(n, E):
        raise RuntimeError(f"{ds_key}_adj.npy không cho đồ thị liên thông")
    return sorted(E), k


def _adj_list(n, E):
    nb = [[] for _ in range(n)]
    for a, b in E:
        nb[a].append(b)
        nb[b].append(a)
    return nb


def _connected(n, E):
    nb = _adj_list(n, E)
    seen, q = {0}, deque([0])
    while q:
        u = q.popleft()
        for v in nb[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == n


def routing_matrix(n, E, pairs_idx):
    """
    R [E, F]: R[e, f] = phần lưu lượng của luồng f đi qua liên kết e (đường ngắn nhất theo hop, ECMP chia đều).
    pairs_idx: danh sách (src_idx, dst_idx) theo thứ tự cột.
    """
    nb = _adj_list(n, E)
    eid = {e: i for i, e in enumerate(E)}
    # BFS từ mọi nút: khoảng cách + số đường ngắn nhất
    dist = np.full((n, n), np.inf)
    cnt = np.zeros((n, n))
    for s in range(n):
        dist[s, s], cnt[s, s] = 0, 1
        q = deque([s])
        while q:
            u = q.popleft()
            for v in nb[u]:
                if dist[s, v] == np.inf:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
                if dist[s, v] == dist[s, u] + 1:
                    cnt[s, v] += cnt[s, u]
    R = np.zeros((len(E), len(pairs_idx)), dtype=np.float64)
    n_ecmp = 0
    for f, (s, d) in enumerate(pairs_idx):
        if s == d or not np.isfinite(dist[s, d]):
            continue
        if cnt[s, d] > 1:
            n_ecmp += 1
        # Phần lưu lượng qua cạnh (u,v) = cnt[s,u] * cnt[v,d] / cnt[s,d] khi u->v nằm trên một đường ngắn nhất
        for (a, b), e in eid.items():
            for u, v in ((a, b), (b, a)):
                if dist[s, u] + 1 + dist[v, d] == dist[s, d]:
                    R[e, f] += cnt[s, u] * cnt[v, d] / cnt[s, d]
    return R.astype(np.float32), dist, n_ecmp


def build_flow_graphs(ds_key, columns, topology='knn', knn_k=2, cache=True, rebuild=False):
    """Trả về {'A_od': [N,N], 'A_route': [N,N], 'R': [E,N], 'links': [(a,b)], 'meta': {...}} (đã chuẩn hóa)."""
    out_f = os.path.join(DATA_DIR, 'graphs', f'{ds_key}_{topology}.npz')
    if cache and not rebuild and os.path.exists(out_f):
        z = np.load(out_f, allow_pickle=True)
        return {'A_od': z['A_od'], 'A_route': z['A_route'], 'R': z['R'], 'links': [tuple(x) for x in z['links']],
                'meta': z['meta'].item()}
    pairs, label2idx = node_ids(columns)
    n = len(label2idx)
    pairs_idx = [(label2idx[s], label2idx[d]) for s, d in pairs]
    if topology == 'physical':
        E, k_used = links_physical(ds_key, label2idx), None
    elif topology == 'knn':
        E, k_used = links_knn(ds_key, n, knn_k)
    else:
        raise ValueError(f"topology không hỗ trợ: {topology}")
    R, dist, n_ecmp = routing_matrix(n, E, pairs_idx)
    M_in, M_out = build_od_topology_matrices(columns)
    A_od = sym_norm(np.maximum(M_in, M_out))
    A_route = sym_norm(R.T.astype(np.float64) @ R.astype(np.float64))
    finite = dist[np.isfinite(dist) & (dist > 0)]
    meta = {
        'dataset': ds_key, 'topology': topology, 'knn_k': k_used, 'nodes': n, 'flows': len(pairs),
        'links': len(E), 'connected': bool(_connected(n, E)), 'mean_hops': float(finite.mean()) if finite.size else 0.0,
        'n_ecmp_flows': int(n_ecmp),
        'density_A_route': float((A_route > 0).mean()), 'density_A_od': float((A_od > 0).mean()),
        'note': 'đồ thị suy diễn từ *_adj.npy, không phải topo vật lý' if topology == 'knn' else 'topo vật lý',
    }
    if cache:
        os.makedirs(os.path.dirname(out_f), exist_ok=True)
        np.savez_compressed(out_f, A_od=A_od, A_route=A_route, R=R, links=np.asarray(E, dtype=np.int32).reshape(-1, 2),
                            meta=np.asarray(meta, dtype=object))
    return {'A_od': A_od, 'A_route': A_route, 'R': R, 'links': E, 'meta': meta}


def resolve_topology(ds_key, topology='auto'):
    """'auto': dùng 'physical' nếu có data/topology/{ds}_links.csv, ngược lại 'knn'."""
    if topology != 'auto':
        return topology
    return 'physical' if os.path.exists(os.path.join(DATA_DIR, 'topology', f'{ds_key}_links.csv')) else 'knn'


if __name__ == '__main__':
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, 'reconfigure'):
            s.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description="Dựng và kiểm tra đồ thị luồng (A_od, A_route)")
    ap.add_argument('--datasets', default='all')
    ap.add_argument('--topology', default='auto', choices=['auto', 'physical', 'knn'])
    ap.add_argument('--knn_k', type=int, default=2)
    a = ap.parse_args()
    from features.feature_store import load_raw_dataset
    ds_list = ['sdn', 'geant', 'abilene'] if a.datasets == 'all' else a.datasets.split(',')
    for ds in ds_list:
        cols = list(load_raw_dataset(ds).columns)
        topo = resolve_topology(ds, a.topology)
        g = build_flow_graphs(ds, cols, topo, a.knn_k, cache=True, rebuild=True)
        print(f"[{ds.upper()}] " + ", ".join(f"{k}={v}" for k, v in g['meta'].items()))
