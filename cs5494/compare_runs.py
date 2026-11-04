"""Compare several results folders (archived package vs GPU runs) side by side.

Usage:
    & 'D:\\HKStudy\\.venv\\Scripts\\python.exe' compare_runs.py
    ... compare_runs.py --roots results
    ... compare_runs.py --roots results/ablation_14x14 results

Writes COMPARISON_5080_vs_ORIGINAL.md and comparisons/*.png. Only reads metrics.json,
history.json, config.json and the saved PNGs, so it also works with the archived folder
alone. Bits/pixel is used for the autoregressive models because it stays comparable when
the image resolution changes between runs.
"""
import argparse, io, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AR_RUNS = [('autoregressive', 'fvsbn', '60k'), ('autoregressive', 'made', '60k'),
           ('ablation_10k', 'fvsbn', '10k'), ('ablation_10k', 'made', '10k')]
LOSS_LABEL = {'fvsbn': 'train NLL (nats/image)', 'made': 'train NLL (nats/image)', 'gan': 'BCE'}


def load_json(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(io.open(p, encoding='utf-8').read())


def label_of(root):
    s = str(root).replace('\\', '/')
    if '_archive' in s:
        return 'archive'
    return Path(s).name


def arch_of(root):
    cfg = load_json(Path(root)/'autoregressive/config.json', {}) or {}
    if not cfg:
        return '?'
    return f"{cfg.get('side', '?')}x{cfg.get('side', '?')}, {cfg.get('layers', 1)}x{cfg.get('hidden', '?')}"


def grid_stats(png):
    png = Path(png)
    if not png.exists():
        return None
    a = np.asarray(Image.open(png).convert('L'), dtype=np.float32)/255.
    h, w = a.shape
    cell = h//10
    t = np.stack([a[r*cell:(r+1)*cell, c*cell:(c+1)*cell].ravel() for r in range(10) for c in range(10)])
    d = [float(np.linalg.norm(t[i]-t[j])) for i in range(len(t)) for j in range(i+1, len(t))]
    return {'mean_ink': float(a.mean()), 'pairwise': float(np.mean(d))}


def side_by_side(paths, titles, label, dst):
    paths = [Path(p) for p in paths]
    if not all(p.exists() for p in paths):
        return None
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    ims = [Image.open(p).convert('RGB') for p in paths]
    # Autoregressive grids are tiny at 14x14; scale everything to a comparable width
    # with nearest neighbour so strokes stay crisp.
    def _scale(im, target=560):
        k = max(1, int(round(target/max(im.width, 1))))
        return im.resize((im.width*k, im.height*k), Image.NEAREST)
    ims = [_scale(im) for im in ims]
    w = max(i.width for i in ims); pad = 54
    canvas = Image.new('RGB', (w, sum(i.height for i in ims)+pad*len(ims)+8), 'white')
    draw = ImageDraw.Draw(canvas)
    y = 0
    for title, im in zip(titles, ims):
        draw.text((6, y+6), f'{label}  -  {title}', fill='black', font=font)
        canvas.paste(im, ((w-im.width)//2, y+pad))
        y += im.height+pad+8
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return dst


def plot_losses(curves, label, dst, name):
    curves = [(t, h) for t, h in curves if h]
    if len(curves) < 2:
        return None
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 3.6))
    for title, h in curves:
        plt.plot([r['epoch'] for r in h], [r['loss'] for r in h], label=title)
        if name == 'gan':
            plt.plot([r['epoch'] for r in h], [r.get('discriminator_loss', np.nan) for r in h],
                     '--', alpha=.6, label=title+' D')
    plt.xlabel('Epoch'); plt.ylabel(LOSS_LABEL[name]); plt.legend(fontsize=8); plt.title(label)
    plt.tight_layout(); plt.savefig(dst, dpi=130); plt.close()
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+',
                    default=['results'])
    ap.add_argument('--out', default='COMPARISON_5080_vs_ORIGINAL.md')
    ap.add_argument('--figures', default='comparisons')
    ap.add_argument('--gan-eval', default='comparisons/gan_eval.json')
    a = ap.parse_args()
    roots = [Path(r) for r in a.roots if Path(r).exists()]
    fig = Path(a.figures)
    ev = load_json(a.gan_eval, {}) or {}
    L = ['# Archived package vs GPU runs', '']
    for r in roots:
        L.append(f"- `{r.as_posix()}` — autoregressive images {arch_of(r)}")
    L.append('')

    # 1. autoregressive likelihood
    L += ['## 1. Autoregressive test likelihood (bits/pixel, lower is better)', '',
          'Bits/pixel is comparable across resolutions; nats/image are not, so they are shown in brackets.', '',
          '| run | ' + ' | '.join(label_of(r) for r in roots) + ' |',
          '|---|' + '---:|'*len(roots)]
    for base, name, tag in AR_RUNS:
        cells = []
        for r in roots:
            m = (load_json(r/base/'metrics.json', {}) or {}).get(name)
            cells.append(f"{m['test_bits_per_pixel']:.4f} ({m['test_nll_nats_per_image']:.2f})" if m else 'n/a')
        L.append(f'| {name} @ {tag} | ' + ' | '.join(cells) + ' |')
    L.append('')

    # 2. GAN
    L += ['## 2. GAN variants', '',
          'Scored with an independent CNN (accuracy on real test digits is reported below). '
          'Real MNIST reference: confidence 0.939, class entropy 2.299 (max ln 10 = 2.303), FD 0. '
          'FD is the Frechet distance between 128-d classifier features of 2000 generated and 5000 real images.', '',
          '| run | arch | s/epoch | confidence | class entropy | FD (lower better) | classes |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in roots:
        for d in sorted(p for p in r.iterdir() if p.is_dir() and p.name.startswith('gan')):
            m = (load_json(d/'metrics.json', {}) or {}).get('gan')
            cfg = load_json(d/'config.json', {}) or {}
            if not m:
                continue
            st = (ev.get('runs', {}) or {}).get(d.as_posix(), {})
            ep = cfg.get('epochs', 1)
            L.append(f"| {label_of(r)}/{d.name} | {cfg.get('arch','mlp')} | "
                     f"{m['training_and_preview_seconds']/ep:.2f} | {st.get('conf', float('nan')):.4f} | "
                     f"{st.get('entropy', float('nan')):.4f} | {st.get('fd', float('nan')):.2f} | "
                     f"{st.get('classes_used','n/a')} |")
    if ev:
        L += ['', f"Classifier test accuracy: {ev.get('classifier_test_acc',0):.4f}."]
    L.append('')

    # 3. speed
    L += ['## 3. Speed per epoch (seconds, including preview generation)', '',
          '| run | ' + ' | '.join(label_of(r) for r in roots) + ' |', '|---|' + '---:|'*len(roots)]
    for base, name, tag in AR_RUNS[:2]:
        cells = []
        for r in roots:
            m = (load_json(r/base/'metrics.json', {}) or {}).get(name)
            cfg = load_json(r/base/'config.json', {}) or {}
            cells.append(f"{m['training_and_preview_seconds']/cfg['epochs']:.3f}" if m and cfg else 'n/a')
        L.append(f'| {name} training | ' + ' | '.join(cells) + ' |')
    cells = []
    for r in roots:
        m = (load_json(r/'gan/metrics.json', {}) or {}).get('gan')
        cfg = load_json(r/'gan/config.json', {}) or {}
        cells.append(f"{m['training_and_preview_seconds']/cfg['epochs']:.3f}" if m and cfg else 'n/a')
    L.append('| gan training | ' + ' | '.join(cells) + ' |')
    L.append('')

    # 4. figures
    L += ['## 4. Figures', '']
    for name in ('fvsbn', 'made'):
        p = side_by_side([r/f'autoregressive/{name}/samples.png' for r in roots],
                         [f'{label_of(r)} ({arch_of(r)})' for r in roots],
                         f'{name} samples', fig/f'samples_{name}.png')
        if p:
            L += [f'![{name} samples]({p.as_posix()})', '']
    gp = side_by_side([r/'gan/gan/samples.png' for r in roots],
                      [f'{label_of(r)} ({(load_json(r/"gan/config.json", {}) or {}).get("arch","mlp")})' for r in roots],
                      'gan samples', fig/'samples_gan.png')
    if gp:
        L += [f'![gan samples]({gp.as_posix()})', '']
    for base, name, tag in AR_RUNS:
        q = plot_losses([(label_of(r), load_json(r/base/name/'history.json', [])) for r in roots],
                        f'{name} @ {tag} training loss', fig/f'loss_{name}_{tag}.png', name)
        if q:
            L += [f'![{name} @ {tag} loss]({q.as_posix()})', '']
    q = plot_losses([(label_of(r), load_json(r/'gan/gan/history.json', [])) for r in roots],
                    'gan training loss', fig/'loss_gan.png', 'gan')
    if q:
        L += [f'![gan loss]({q.as_posix()})', '']

    # 5. verdict: best bits/pixel per model, and GAN scores
    L += ['## 5. Verdict', '']
    for base, name, tag in AR_RUNS:
        vals = {}
        for r in roots:
            m = (load_json(r/base/'metrics.json', {}) or {}).get(name)
            if m:
                vals[label_of(r)] = m['test_bits_per_pixel']
        if len(vals) > 1:
            best = min(vals, key=vals.get)
            first, last = list(vals)[0], list(vals)[-1]
            L.append(f'- {name} @ {tag}: best is **{best}** ({vals[best]:.4f} bpp); '
                     f'first -> last change {vals[last]-vals[first]:+.4f} bpp.')
    conf = {}
    for r in roots:
        st = (ev.get('runs', {}) or {}).get((r/'gan').as_posix(), {})
        if st:
            conf[label_of(r)] = (st.get('fd', float('nan')), st.get('conf', 0), st.get('entropy', 0))
    if len(conf) > 1:
        best = min(conf, key=lambda k: conf[k][0])
        L.append(f'- GAN: lowest FD is **{best}** (FD {conf[best][0]:.2f}); ' +
                 ', '.join(f'{k} FD {v[0]:.2f} / conf {v[1]:.4f}' for k, v in conf.items()) + '.')
    L += ['', 'Caveats: one seed per configuration; runs differ in training budget as well as architecture; '
              'timings come from different machines (archived numbers are CPU).']

    Path(a.out).write_text('\n'.join(L)+'\n', encoding='utf-8')
    print('\n'.join(L))
    print(f'\nwritten: {a.out}')


if __name__ == '__main__':
    main()
