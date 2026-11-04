"""External evaluation for the generative models.

Two sub-commands, both standardising their output as JSON next to the report:

    gan       trains (or loads) an independent MNIST classifier and scores every GAN
              checkpoint with confidence, class entropy and a feature Frechet distance.
    mixture   computes the held-out likelihood of a mixture of autoregressive models,
              p(x) = 1/K * sum_k p_k(x), for K = 1..n and writes the K-vs-bpp curve.

Examples
    python -m cs5494.evaluate gan --runs results/gan_ema --weights ema
    python -m cs5494.evaluate mixture --runs results/made_seed42 results/made_seed43 \
        results/made_seed44 --out results/ensemble_metrics.json
"""
import argparse, io, json, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST

from cs5494.train import Autoregressive, build_gan


# --------------------------------------------------------------------------------------
# classifier used as an independent evaluator (also labels the per-digit figures)
# --------------------------------------------------------------------------------------
class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*7*7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)

    def features(self, x):
        """128-d penultimate activations, used by the Frechet-distance proxy."""
        return self.net[:-2](x)


def load_classifier(path='comparisons/gan_classifier.pt', device='cpu'):
    clf = Clf()
    clf.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return clf.to(device).eval()


def train_or_load(args, dev):
    """Return an evaluated classifier, training it once if the cache is missing."""
    tr = MNIST('data', train=True, download=True)
    te = MNIST('data', train=False, download=True)
    xs = (tr.data.float()/255).unsqueeze(1)
    ys = tr.targets.clone()
    xt = (te.data.float()/255).unsqueeze(1).to(dev)
    yt = te.targets.clone().to(dev)

    clf = Clf().to(dev)
    ckpt = Path(args.classifier)
    if ckpt.exists():
        clf.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
        print('loaded cached classifier', ckpt)
    else:
        opt = torch.optim.Adam(clf.parameters(), 1e-3)
        n = min(args.train_n, len(xs))
        for ep in range(args.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i+256]
                opt.zero_grad()
                F.cross_entropy(clf(xs[idx].to(dev)), ys[idx].to(dev)).backward()
                opt.step()
            with torch.no_grad():
                acc = (clf(xt).argmax(1) == yt).float().mean().item()
            print(f'classifier epoch {ep+1}: test acc {acc:.4f}')
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clf.state_dict(), ckpt)
    clf.eval()
    return clf, xt, yt


def frechet(a, b):
    """Frechet distance between two feature Gaussians (the FID formula)."""
    from scipy.linalg import sqrtm
    mu1, mu2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1-mu2)**2).sum() + np.trace(c1 + c2 - 2*covmean))


# --------------------------------------------------------------------------------------
# gan: classifier statistics
# --------------------------------------------------------------------------------------
def cmd_gan(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    clf, xt, yt = train_or_load(a, dev)

    with torch.no_grad():
        acc = (clf(xt).argmax(1) == yt).float().mean().item()
        print(f'evaluator accuracy on 10,000 real test images: {acc:.4f}')
        p = clf(xt[:5000]).softmax(1)
        hist = torch.bincount(p.argmax(1), minlength=10).float()/len(p)
        ref = {'conf': p.max(1).values.mean().item(),
               'entropy': float(-(hist*torch.log(hist.clamp_min(1e-12))).sum()),
               'acc': acc}
        real_feat = clf.features(xt[:5000]).double().cpu().numpy()
    print('real MNIST reference:', {k: round(v, 4) for k, v in ref.items()})
    print(f'real self-FD (sanity, should be ~0): {frechet(real_feat, real_feat[500:]):.3f}')

    res = {'classifier_test_acc': acc, 'reference_real_mnist': ref, 'runs': {}}
    ckpt_name = 'model_ema.pt' if a.weights == 'ema' else 'model.pt'
    for run in a.runs:
        if not (Path(run)/'gan'/ckpt_name).exists():
            print('skip (no checkpoint):', run, ckpt_name)
            continue
        cfg = json.loads(io.open(Path(run)/'config.json', encoding='utf-8').read())
        g = build_gan(cfg.get('arch', 'mlp'))[0].to(dev)
        g.load_state_dict(torch.load(Path(run)/'gan'/ckpt_name, map_location=dev, weights_only=True))
        g.eval()
        key = run if a.weights == 'raw' else f'{run}::{a.weights}'
        torch.manual_seed(123)
        with torch.no_grad():
            imgs = (g(torch.randn(a.n, 64, device=dev))+1)/2
            imgs = imgs.clamp(0, 1).reshape(-1, 1, 28, 28)
            p = clf(imgs).softmax(1)
        hist = torch.bincount(p.argmax(1), minlength=10).float()/len(p)
        ent = float(-(hist*torch.log(hist.clamp_min(1e-12))).sum())
        with torch.no_grad():
            gen_feat = clf.features(imgs).double().cpu().numpy()
        fd = frechet(real_feat, gen_feat)
        row = {'conf': p.max(1).values.mean().item(), 'entropy': ent, 'fd': fd, 'acc': None,
               'classes_used': int((hist > 0.005).sum()),
               'class_hist': [round(float(v), 4) for v in hist],
               'blank_rate': float(((imgs.mean((1, 2, 3)) < .02) | (imgs.mean((1, 2, 3)) > .98)).float().mean())}
        res['runs'][key] = row
        print(f"{key:56s} conf={row['conf']:.4f} entropy={row['entropy']:.4f} "
              f"FD={fd:8.2f} classes={row['classes_used']:2d} blank={row['blank_rate']:.3f}")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding='utf-8')
    print('written', out)


# --------------------------------------------------------------------------------------
# mixture: likelihood of K autoregressive members
# --------------------------------------------------------------------------------------
def _load_member(run, device):
    run = Path(run)
    cfg = json.loads(io.open(run/'config.json', encoding='utf-8').read())
    side = cfg['side']
    model = Autoregressive(side*side, cfg['hidden'], cfg.get('layers', 1), cfg.get('dropout', 0.0))
    model.load_state_dict(torch.load(run/'made/model.pt', map_location=device, weights_only=True))
    return model.to(device).eval(), cfg, side


@torch.no_grad()
def _per_image_nll(model, x, device, batch=128):
    out = []
    for b in x.split(batch):
        b = b.to(device)
        out.append(F.binary_cross_entropy_with_logits(model(b), b, reduction='none').sum(1).cpu())
    return torch.cat(out)


def cmd_mixture(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, cfgs, side = [], [], None
    for r in a.runs:
        m, cfg, s = _load_member(r, dev)
        models.append(m); cfgs.append(cfg); side = s

    test = MNIST('data', train=False, download=True)
    n = cfgs[0]['test_samples']
    raw = test.data[:n].float().unsqueeze(1)/255
    x = (F.interpolate(raw, size=(side, side), mode='area') >= .5).float().flatten(1)
    scale = side*side*np.log(2)

    members, nlls = [], []
    for run, m, cfg in zip(a.runs, models, cfgs):
        t0 = time.perf_counter()
        with torch.no_grad():
            m.sample(100, dev)
        infer = time.perf_counter()-t0
        p = _per_image_nll(m, x, dev)
        nlls.append(p)
        members.append({'run': str(run), 'seed': cfg.get('seed'),
                        'nats': float(p.mean()), 'bpp': float(p.mean()/scale),
                        'params': int(sum(q.numel() for q in m.parameters())),
                        'inference_100_seconds': infer})
        print(f"{str(run):34s} seed={cfg.get('seed')} nats={p.mean():.4f} bpp={p.mean()/scale:.4f}")

    metrics = {'side': side, 'test_samples': n, 'members': members, 'ensembles': []}
    plots = []
    for k in range(2, len(nlls)+1):
        mix = torch.logsumexp(-torch.stack(nlls[:k]), dim=0) - np.log(k)
        nats = float((-mix).mean())
        params = sum(m['params'] for m in members[:k])
        metrics['ensembles'].append({'k': k, 'members': [m['run'] for m in members[:k]],
                                     'nats': nats, 'bpp': nats/scale, 'params': params})
        print(f'mixture K={k}  nats={nats:.4f}  bpp={nats/scale:.4f}  params={params}')
        plots.append((k, nats/scale))
    plots.insert(0, (1, members[0]['bpp']))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('written', out)

    plt.figure(figsize=(5.2, 3.2))
    ks, bpps = zip(*plots)
    plt.plot(ks, bpps, 'o-')
    for k, b in plots:
        plt.annotate(f'{b:.4f}', (k, b), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)
    plt.xticks(ks); plt.xlabel('number of mixture members K')
    plt.ylabel('test bits/pixel'); plt.title('MADE mixture, test split')
    plt.tight_layout()
    Path(a.figure).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.figure, dpi=130); plt.close()
    print('written', a.figure)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    g = sub.add_parser('gan', help='classifier statistics for GAN checkpoints')
    g.add_argument('--runs', nargs='+', default=['results/gan_initial', 'results/gan_mlp'])
    g.add_argument('--n', type=int, default=2000)
    g.add_argument('--weights', choices=['raw', 'ema'], default='raw')
    g.add_argument('--train-n', type=int, default=20000)
    g.add_argument('--epochs', type=int, default=3)
    g.add_argument('--classifier', default='comparisons/gan_classifier.pt')
    g.add_argument('--out', default='comparisons/gan_eval.json')
    g.set_defaults(func=cmd_gan)

    m = sub.add_parser('mixture', help='held-out likelihood of a mixture of MADE models')
    m.add_argument('--runs', nargs='+', required=True)
    m.add_argument('--out', default='results/ensemble_metrics.json')
    m.add_argument('--figure', default='results/ensemble_ablation.png')
    m.set_defaults(func=cmd_mixture)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
