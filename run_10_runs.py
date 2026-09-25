import os
import sys
import time
import argparse
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description="Chạy ST-Adaptive-Ensemble v3 cho một đợt run (mặc định đủ 10 run: 0-9)")
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'sdn', 'geant', 'abilene'])
    parser.add_argument('--run_ids', type=str, default='0-9', help="Ví dụ '5-9' (đợt 1) hoặc '0-9' (đủ 10 run)")
    parser.add_argument('--epochs', type=int, default=200, help="Epoch tối đa ST-WaveFormer (mặc định 200)")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience (mặc định 30)")
    parser.add_argument('--skip_dl', action='store_true', help="Không huấn luyện DL (máy chỉ có CPU)")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(root, 'training', 'run_ensemble.py'),
           '--datasets', args.dataset, '--run_ids', args.run_ids,
           '--epochs', str(args.epochs), '--patience', str(args.patience)]
    if args.skip_dl:
        cmd.append('--skip_dl')

    print("#" * 80)
    print(f"# ST-ADAPTIVE-ENSEMBLE v3 | dataset={args.dataset} | run_ids={args.run_ids} | "
          f"epochs={args.epochs} | patience={args.patience}")
    print("#" * 80)
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=root)
    print(f"\nThời gian: {(time.time() - t0) / 60:.1f} phút | exit code: {ret.returncode}")
    sys.exit(ret.returncode)


if __name__ == '__main__':
    main()
