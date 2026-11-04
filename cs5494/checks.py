"""Repository checks.

    units      causal-gradient / shape assertions for all three model families
    nll        recompute held-out NLL from saved checkpoints and compare with metrics.json
    report     extract the numbers printed in report/main.pdf and compare with the JSON files
    manifest   create or verify a SHA256 manifest of a folder tree

Examples
    python -m cs5494.checks units
    python -m cs5494.checks nll --runs results/autoregressive
    python -m cs5494.checks report
    python -m cs5494.checks manifest . --verify
"""
import argparse, hashlib, io, json, sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST

from cs5494.train import Autoregressive, Generator, Discriminator


# --------------------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------------------
def cmd_units(a):
    for hidden in (0, 32):
        model = Autoregressive(16, hidden)
        x = torch.randn(3, 16, requires_grad=True)
        for i in range(16):
            g = torch.autograd.grad(model(x)[:, i].sum(), x, retain_graph=True)[0]
            assert torch.count_nonzero(g[:, i:]) == 0, 'causality violated'
        sample = model.sample(5, 'cpu')
        assert sample.shape == (5, 16) and ((sample == 0) | (sample == 1)).all()
    fake = Generator()(torch.randn(5, 64))
    assert fake.shape == (5, 784) and fake.abs().max() <= 1
    assert Discriminator()(fake).shape == (5,)
    print('PASS: causal gradients, binary sampling, GAN shapes and range')


# --------------------------------------------------------------------------------------
# nll
# --------------------------------------------------------------------------------------
def cmd_nll(a):
    test = MNIST(a.data, train=False, download=True)
    bad = 0
    for run in a.runs:
        run = Path(run)
        if not (run/'metrics.json').exists():
            print('skip (no metrics.json):', run)
            continue
        cfg = json.loads(io.open(run/'config.json', encoding='utf-8').read())
        metrics = json.loads(io.open(run/'metrics.json', encoding='utf-8').read())
        if 'gan' in metrics:
            continue
        side = cfg['side']
        raw = test.data[:cfg['test_samples']].float().unsqueeze(1)/255
        tx = (F.interpolate(raw, size=(side, side), mode='area') >= .5).float().flatten(1)
        for name in metrics:
            ckpt = run/name/'model.pt'
            if not ckpt.exists():
                continue
            m = Autoregressive(side*side, cfg['hidden'] if name == 'made' else 0,
                               cfg.get('layers', 1), cfg.get('dropout', 0.0))
            m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
            m.eval()
            with torch.no_grad():
                nll = sum(F.binary_cross_entropy_with_logits(m(b), b, reduction='sum').item()
                          for b in tx.split(128))/len(tx)
            saved = metrics[name]['test_nll_nats_per_image']
            ok = abs(nll-saved) <= a.tol
            bad += 0 if ok else 1
            print(f'{run.as_posix():34s} {name:6s} recomputed={nll:.5f} saved={saved:.5f} '
                  f'diff={abs(nll-saved):.1e} {"OK" if ok else "MISMATCH"}')
    print('all consistent' if bad == 0 else f'{bad} mismatches')
    return 1 if bad else 0


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def cmd_report(a):
    import pymupdf

    root = Path(__file__).resolve().parents[1]
    text = '\n'.join(p.get_text() for p in pymupdf.open(root/a.pdf))

    def load(p):
        return json.loads(io.open(root/p, encoding='utf-8').read())

    ar = load('results/autoregressive/metrics.json')
    ab = load('results/ablation_10k/metrics.json')
    ens = load('results/ensemble_metrics.json')
    ens10 = load('results/ensemble_10k_metrics.json')
    ev = load('comparisons/gan_eval.json')
    ev2 = load('comparisons/gan_eval_ema.json')
    ev3 = load('comparisons/gan_eval_ema_tmp.json')

    members = ens['members']
    mean_nats = sum(m['nats'] for m in members)/len(members)
    mean_bpp = sum(m['bpp'] for m in members)/len(members)
    m10 = ens10['members']
    mean10_nats = sum(m['nats'] for m in m10)/len(m10)
    k3 = ens['ensembles'][-1]
    k3_10 = ens10['ensembles'][-1]
    mlp_arch = ev['runs']['results/gan_initial']
    mlp_shared = ev['runs']['results/gan_mlp_shared_lr']
    mlp_ttur = ev['runs']['results/gan_mlp']
    dcgan_raw = ev2['runs']['results/gan_ema']
    dcgan_ema = ev3['runs']['results/gan_ema::ema']

    checks = [
        ('FVSBN 60k nats', f"{ar['fvsbn']['test_nll_nats_per_image']:.2f}", '75.72'),
        ('FVSBN 60k bpp', f"{ar['fvsbn']['test_bits_per_pixel']:.4f}", '0.1393'),
        ('FVSBN 60k train s', f"{ar['fvsbn']['training_and_preview_seconds']:.1f}", '47.4'),
        ('MADE mean nats', f'{mean_nats:.2f}', '68.45'),
        ('MADE mean bpp', f'{mean_bpp:.4f}', '0.1260'),
        ('MADE K=2 nats', f"{ens['ensembles'][0]['nats']:.2f}", '67.07'),
        ('MADE K=3 nats', f"{k3['nats']:.2f}", '66.46'),
        ('MADE K=3 bpp', f"{k3['bpp']:.4f}", '0.1223'),
        ('FVSBN 10k nats', f"{ab['fvsbn']['test_nll_nats_per_image']:.2f}", '120.17'),
        ('FVSBN 10k bpp', f"{ab['fvsbn']['test_bits_per_pixel']:.4f}", '0.2211'),
        ('MADE 10k mean nats', f'{mean10_nats:.2f}', '80.19'),
        ('MADE 10k K=3 bpp', f"{k3_10['bpp']:.4f}", '0.1415'),
        ('MLP initial FD', f"{mlp_arch['fd']:.2f}", '34.61'),
        ('MLP shared-lr FD', f"{mlp_shared['fd']:.2f}", '38.30'),
        ('MLP TTUR FD', f"{mlp_ttur['fd']:.2f}", '7.71'),
        ('DCGAN live FD', f"{dcgan_raw['fd']:.2f}", '7.15'),
        ('DCGAN EMA FD', f"{dcgan_ema['fd']:.2f}", '7.12'),
    ]

    bad = 0
    for name, computed, printed in checks:
        if computed != printed:
            print(f'ROUNDING DRIFT  {name}: computed {computed}, report {printed}')
            bad += 1
        if printed not in text:
            print(f'MISSING IN PDF  {name}: {printed}')
            bad += 1
    print(f'{len(checks)} numbers checked, {bad} problem(s)')
    return 1 if bad else 0


# --------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------
def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cmd_manifest(a):
    root = Path(a.root)
    out = Path(a.out) if a.out else root/'MANIFEST_SHA256.json'
    if a.verify:
        manifest = json.loads(io.open(out, encoding='utf-8').read())
        bad = [rel for rel, want in manifest.items()
               if not (root/rel).exists() or _sha(root/rel) != want]
        print(f'{len(manifest)} entries, {len(bad)} mismatched')
        for rel in bad[:20]:
            print('MISMATCH', rel)
        return 1 if bad else 0

    files = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if p.name in {out.name, '__pycache__'} or p.suffix == '.pyc':
            continue
        files[rel] = _sha(p)
    out.write_text(json.dumps(files, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out} with {len(files)} entries')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    u = sub.add_parser('units', help='causality and shape assertions')
    u.set_defaults(func=cmd_units)

    n = sub.add_parser('nll', help='recompute NLL from checkpoints')
    n.add_argument('--runs', nargs='+', default=['results/autoregressive', 'results/ablation_10k'])
    n.add_argument('--data', default='data')
    n.add_argument('--tol', type=float, default=1e-3,
                   help='GPU checkpoints differ from the CPU recomputation by ~2e-4 (TF32 + reduction order)')
    n.set_defaults(func=cmd_nll)

    r = sub.add_parser('report', help='compare report numbers with the JSON records')
    r.add_argument('--pdf', default='report/main.pdf')
    r.set_defaults(func=cmd_report)

    m = sub.add_parser('manifest', help='create or verify a SHA256 manifest')
    m.add_argument('root', nargs='?', default='.')
    m.add_argument('--out', default=None)
    m.add_argument('--verify', action='store_true')
    m.set_defaults(func=cmd_manifest)

    a = ap.parse_args(argv)
    rc = a.func(a)
    sys.exit(rc or 0)


if __name__ == '__main__':
    main()
