"""Figures used by the report.

    digits     draws unconditional samples from every reported model, labels them with the
               independent classifier and keeps the first N per predicted digit, producing a
               grid with one column per digit (a selection, not conditional generation).
    variants   stacks the three MLP-GAN sample grids (shared lr / larger batch / TTUR) with
               the feature distances read from comparisons/gan_eval.json.

Examples
    python -m cs5494.figures digits --candidates 4000 --order 0123456789
    python -m cs5494.figures variants
"""
import argparse, io, json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from cs5494.train import Autoregressive, build_gan
from cs5494.evaluate import load_classifier

# label, run folder, base folder, model name, which checkpoint
DEFAULT_SPECS = [
    ('fvsbn', 'results', 'autoregressive', 'fvsbn', 'raw'),
    ('made', 'results', 'autoregressive', 'made', 'raw'),
    ('mlp', 'results', 'gan_mlp', 'gan', 'raw'),
    ('dcgan_raw', 'results', 'gan_ema', 'gan', 'raw'),
    ('dcgan_ema', 'results', 'gan_ema', 'gan', 'ema'),
]
COMPOSITE = ['fvsbn', 'made', 'dcgan_ema']
TITLES = {'fvsbn': 'FVSBN', 'made': 'MADE', 'mlp': 'MLP-GAN',
          'dcgan_raw': 'DCGAN (live)', 'dcgan_ema': 'DCGAN (EMA)'}


def _load_model(run, base, name, which, device):
    run = Path(run)
    cfg = json.loads(io.open(run/base/'config.json', encoding='utf-8').read())
    if name == 'gan':
        model = build_gan(cfg.get('arch', 'mlp'))[0]
        side = 28
        ckpt = run/base/name/('model_ema.pt' if which == 'ema' else 'model.pt')
    else:
        side = cfg['side']
        model = Autoregressive(side*side, cfg['hidden'] if name == 'made' else 0,
                               cfg.get('layers', 1), cfg.get('dropout', 0.0))
        ckpt = run/base/name/'model.pt'
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.to(device).eval(), cfg, side


@torch.no_grad()
def _draw(model, name, side, n, device):
    x = (model(torch.randn(n, 64, device=device))+1)/2 if name == 'gan' else model.sample(n, device)
    return x.clamp(0, 1).reshape(-1, 1, side, side)


@torch.no_grad()
def _classify(clf, imgs, batch=512):
    probs = torch.cat([clf(b).softmax(1) for b in imgs.split(batch)])
    return probs.argmax(1).cpu(), probs.max(1).values.cpu()


def _grid_png(imgs, path, order, cell=56, header=34, pad=6):
    rows = imgs.shape[0] // len(order)
    tiles = imgs.reshape(rows, len(order), *imgs.shape[1:])
    w = len(order)*cell + (len(order)+1)*pad
    h = rows*cell + (rows+1)*pad + header
    canvas = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    for c, digit in enumerate(order):
        d.text((pad + c*(cell+pad) + cell//2 - 6, 6), str(digit), fill='black', font=font)
    for r in range(rows):
        for c in range(len(order)):
            arr = (tiles[r, c].cpu().numpy().transpose(1, 2, 0)*255).astype('uint8')
            im = Image.fromarray(arr.squeeze()).resize((cell, cell), Image.NEAREST).convert('RGB')
            canvas.paste(im, (pad + c*(cell+pad), header + pad + r*(cell+pad)))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return canvas


def _composite(panels, path, gap=26):
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:
        font = ImageFont.load_default()
    w = sum(p.width for _, p in panels) + gap*(len(panels)+1)
    h = max(p.height for _, p in panels) + 52
    canvas = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(canvas)
    x = gap
    for title, p in panels:
        d.text((x, 12), title, fill='black', font=font)
        canvas.paste(p, (x, 46))
        x += p.width + gap
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return canvas


def cmd_digits(a):
    order = [int(c) for c in a.order]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = load_classifier(a.classifier, dev)
    stats = {'candidates': a.candidates, 'per_digit': a.per_digit, 'order': order, 'runs': {}}
    panels = {}

    for label, run, base, name, which in DEFAULT_SPECS:
        model, cfg, side = _load_model(run, base, name, which, dev)
        torch.manual_seed(a.seed)
        imgs = _draw(model, name, side, a.candidates, dev)
        pred, conf = _classify(clf, imgs)
        picks, counts = [], {}
        for digit in order:
            idx = (pred == digit).nonzero(as_tuple=True)[0]
            counts[str(digit)] = int(len(idx))
            picks.append(idx[:a.per_digit])
        if any(len(p) < a.per_digit for p in picks):
            short = {d: c for d, c in counts.items() if c < a.per_digit}
            print(f'  WARNING {label}: not enough samples for {short}; increase --candidates')
            continue
        sel = torch.stack(picks, 1).reshape(-1)      # rows = draws, columns = digits
        chosen = imgs[sel]
        out = Path(a.outdir)/f'digits_{label}.png'
        canvas = _grid_png(chosen, out, order)
        panels[label] = (TITLES.get(label, label), canvas)
        mconf = conf[sel].mean().item()
        stats['runs'][label] = {'run': run, 'base': base, 'model': name, 'which': which,
                                'side': side, 'candidates': a.candidates,
                                'selected': int(len(sel)), 'mean_confidence_selected': round(mconf, 4),
                                'min_count': min(counts.values()), 'png': out.as_posix()}
        print(f'{label:10s} candidates={a.candidates} selected={len(sel)} '
              f'mean_conf={mconf:.3f} min_count={min(counts.values()):4d} -> {out}')

        if label in ('fvsbn', 'made', 'mlp') or label == 'dcgan_ema':
            canon = Path(run)/base/name/('samples_by_digit_ema.png' if which == 'ema' else 'samples_by_digit.png')
            _grid_png(chosen, canon, order)
            stats['runs'][label]['canonical'] = canon.as_posix()

    if all(k in panels for k in COMPOSITE):
        _composite([panels[k] for k in COMPOSITE], a.composite)
        print('written', a.composite)

    Path(a.stats).write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print('written', a.stats)


MLP_ROWS = [
    ('results/gan_initial/gan/samples.png', 'shared lr, 60 epochs (batch 128)'),
    ('results/gan_mlp_shared_lr/gan/samples.png', 'shared lr, 150 epochs (batch 256)'),
    ('results/gan_mlp/gan/samples.png', 'two-timescale, 150 epochs'),
]
MLP_KEYS = ['results/gan_initial', 'results/gan_mlp_shared_lr', 'results/gan_mlp']


def cmd_variants(a):
    ev = json.loads(io.open(a.eval, encoding='utf-8').read()) if Path(a.eval).exists() else {}
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()

    panels = []
    for (path, title), key in zip(MLP_ROWS, MLP_KEYS):
        p = Path(path)
        if not p.exists():
            print('missing, skipped:', path)
            continue
        st = ev.get('runs', {}).get(key, {})
        cap = f"{title}  |  FD {st['fd']:.1f}" if 'fd' in st else title
        im = Image.open(p).convert('RGB').resize((a.width, a.width), Image.NEAREST)
        panels.append((cap, im))

    gap, header = 10, 34

    def text_w(s):
        try:
            return int(font.getlength(s))
        except Exception:
            return len(s)*11

    width = max([a.width + 2*gap] + [text_w(cap) + 2*gap + 8 for cap, _ in panels])
    canvas = Image.new('RGB', (width, sum(p.height + header + gap for _, p in panels) + gap), 'white')
    d = ImageDraw.Draw(canvas)
    y = gap
    for cap, im in panels:
        d.text((gap, y + 6), cap, fill='black', font=font)
        canvas.paste(im, (gap, y + header))
        y += im.height + header + gap
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print('written', out, canvas.size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    d = sub.add_parser('digits', help='per-digit sample grids for every reported model')
    d.add_argument('--classifier', default='comparisons/gan_classifier.pt')
    d.add_argument('--candidates', type=int, default=4000)
    d.add_argument('--per-digit', type=int, default=10)
    d.add_argument('--seed', type=int, default=123)
    d.add_argument('--order', default='0123456789')
    d.add_argument('--outdir', default='report/figures')
    d.add_argument('--composite', default='comparisons/digit_grid_comparison.png')
    d.add_argument('--stats', default='comparisons/digit_grid.json')
    d.set_defaults(func=cmd_digits)

    v = sub.add_parser('variants', help='MLP-GAN training-regime comparison figure')
    v.add_argument('--eval', default='comparisons/gan_eval.json')
    v.add_argument('--out', default='comparisons/mlp_gan_variants.png')
    v.add_argument('--width', type=int, default=302)
    v.set_defaults(func=cmd_variants)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
