"""Small validation sweep for MADE capacity/regularisation on 28x28 binarised MNIST.

Trains on 55,000 images and scores 5,000 held-out training images, so the choice of
hyperparameters does not touch the 10,000-image test set (which is only read at the end
of each run for reference). Writes sweep_made.csv.

Usage:
    & 'D:\\HKStudy\\.venv\\Scripts\\python.exe' sweep_made.py
"""
import json, io, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SUITES = {
    1: [
        ('h1024_l2_drop00', ['--hidden', '1024', '--layers', '2']),
        ('h1024_l2_drop10', ['--hidden', '1024', '--layers', '2', '--dropout', '0.1']),
        ('h1024_l2_drop20', ['--hidden', '1024', '--layers', '2', '--dropout', '0.2']),
        ('h512_l1_drop00', ['--hidden', '512', '--layers', '1']),
        ('h512_l1_drop10', ['--hidden', '512', '--layers', '1', '--dropout', '0.1']),
        ('h1024_l2_drop10_wd1e4', ['--hidden', '1024', '--layers', '2', '--dropout', '0.1', '--weight-decay', '1e-4']),
        ('h512_l2_drop10', ['--hidden', '512', '--layers', '2', '--dropout', '0.1']),
    ],
    # Round 2: pure L2 on the best capacity, plus cheaper 2-layer models.
    2: [
        ('h1024_l2_wd1e3', ['--hidden', '1024', '--layers', '2', '--weight-decay', '1e-3']),
        ('h1024_l2_wd1e2', ['--hidden', '1024', '--layers', '2', '--weight-decay', '1e-2']),
        ('h512_l2_drop00', ['--hidden', '512', '--layers', '2']),
        ('h1024_l1_drop00', ['--hidden', '1024', '--layers', '1']),
    ],
    # Round 3: does the single-layer winner just need more epochs?
    3: [('h1024_l1', ['--hidden', '1024', '--layers', '1'])],
    # Round 4: wider single-layer models.
    4: [('h1536_l1', ['--hidden', '1536', '--layers', '1']),
        ('h2048_l1', ['--hidden', '2048', '--layers', '1'])],
}
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--suite', type=int, default=1, choices=sorted(SUITES))
_ap.add_argument('--epochs', type=int, default=400)
_a = _ap.parse_args()
CONFIGS = SUITES[_a.suite]
OUT = ROOT/('outputs_sweep' if _a.suite == 1 else f'outputs_sweep{_a.suite}')
rows = []
for tag, extra in CONFIGS:
    target = OUT/tag
    cmd = [PY, '-m', 'cs5494.train', '--models', 'made', '--samples', '60000', '--test-samples', '10000',
           '--val', '5000', '--side', '28', '--epochs', str(_a.epochs), '--batch', '1024', '--lr', '0.001',
           '--scheduler', 'cosine', '--device', 'cuda', '--workers', '4', '--output', str(target)] + extra
    print('\n$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    m = json.loads(io.open(target/'metrics.json', encoding='utf-8').read())['made']
    rows.append((tag, m['validation_bits_per_pixel'], m['test_bits_per_pixel'],
                 m['training_and_preview_seconds']))
    print(f'{tag:24s} val {m["validation_bits_per_pixel"]:.4f} bpp  test {m["test_bits_per_pixel"]:.4f} bpp', flush=True)

rows.sort(key=lambda r: r[1])
lines = ['config,val_bpp,test_bpp,train_seconds'] + [f'{t},{v:.4f},{te:.4f},{s:.0f}' for t, v, te, s in rows]
(ROOT/('sweep_made.csv' if _a.suite == 1 else f'sweep_made{_a.suite}.csv')).write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines))
