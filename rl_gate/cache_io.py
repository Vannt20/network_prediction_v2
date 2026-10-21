"""Đọc cache/v3/{ds}_run_{r}.pt (dự báo Val/Test của 3 nhánh) và trọng số stacking v3, không bắt buộc có torch."""
import json
import os
import pickle
import zipfile

import numpy as np

LOG_DIRS = {
    'sdn': 'st_adaptive_ensemble_data_sdn_seq_60',
    'geant': 'st_adaptive_ensemble_data_geant_seq_24',
    'abilene': 'st_adaptive_ensemble_data_abilene_seq_24',
}
_DTYPES = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64,
           'IntStorage': np.int32, 'HalfStorage': np.float16, 'BoolStorage': np.bool_, 'ByteStorage': np.uint8}


class _Storage:
    def __init__(self, name):
        self.name = name


def _load_pt_without_torch(path):
    """Giải nén file torch.save (định dạng zip) thành dict numpy; chỉ dùng cho cache do repo này tạo ra."""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    root = names[0].split('/')[0]
    pkl = [n for n in names if n.endswith('data.pkl')][0]
    storages = {}

    def rebuild(storage, offset, size, stride, *args):
        if len(size) == 0:
            return storage[offset]
        return np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                               strides=[s * storage.itemsize for s in stride]).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2':
                return rebuild
            if mod == 'torch' and name.endswith('Storage'):
                return _Storage(name)
            if mod == 'collections' and name == 'OrderedDict':
                import collections
                return collections.OrderedDict
            if mod.startswith('torch'):
                return lambda *a, **k: None
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, st, key, _, _ = pid
            if key not in storages:
                dt = _DTYPES.get(st.name if isinstance(st, _Storage) else 'FloatStorage', np.float32)
                storages[key] = np.frombuffer(z.read(f'{root}/data/{key}'), dtype=dt)
            return storages[key]

    return Unpickler(z.open(pkl)).load()


def _to_numpy(x):
    return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)


def load_run(repo_root, dataset, run_id):
    """
    Trả về dict float64: {split}_P [K,T,N], {split}_y [T,N], {split}_last [T,N], {split}_context [T,N,4]
    với split in {val, test}, và W [K,N] (trọng số stacking v3, global được lặp ra theo luồng).
    """
    path = os.path.join(repo_root, 'cache', 'v3', f'{dataset}_run_{run_id}.pt')
    try:
        import torch
        cache = torch.load(path, weights_only=False)
    except ImportError:
        cache = _load_pt_without_torch(path)
    out = {f'{s}_{k}': _to_numpy(cache[s][k]).astype(np.float64)
           for s in ('val', 'test') for k in ('P', 'y', 'last', 'context')}
    with open(os.path.join(repo_root, 'logs', LOG_DIRS[dataset], f'run_{run_id}', 'stacking.json')) as f:
        W = np.array(json.load(f)['weights'], dtype=np.float64)
    out['W'] = W if W.ndim == 2 else np.repeat(W[:, None], out['val_P'].shape[2], axis=1)
    return out
