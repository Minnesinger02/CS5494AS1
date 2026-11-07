# CS5494 Assignment — autoregressive and adversarial MNIST generation

**Repository:** <https://github.com/Minnesinger02/CS5494AS1> (code, data, checkpoints, report)

All experiments live in one sequence under `results/`. Code is a small package, `cs5494/`,
with one module per job. The final report is `submission/60777841_61083251_60369937.pdf`,
built from `report/main.tex`.

## Quick start

```powershell
# regenerate every metric, figure and the PDF from the saved checkpoints (~1 min, no training)
powershell -NoProfile -ExecutionPolicy Bypass -File .\reproduce.ps1

# full retraining (~12 min on an RTX 5080), writes into results/
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_windows.ps1
```

Details, timings and troubleshooting: **`REPRODUCE.md`**. A plain-text version of the run
instructions — the `readme.txt` the assignment asks for — is `readme.txt`.

## Code layout

| module | responsibility |
|---|---|
| `cs5494/train.py` | FVSBN / MADE / MLP-GAN / DCGAN definitions, training loop, previews, metrics |
| `cs5494/sample.py` | draw samples from a checkpoint (GAN: live or moving-average weights) |
| `cs5494/evaluate.py` | `gan` classifier scores, `mixture` likelihood for K MADE members |
| `cs5494/figures.py` | `digits` per-digit grids, `variants` MLP training-regime figure |
| `cs5494/checks.py` | `units` assertions, `nll` re-computation, `report` number check, `manifest` |
| `cs5494/compare_runs.py` | side-by-side comparison of result folders |
| `cs5494/sweep_made.py` | hyper-parameter sweep for MADE |

Every command runs as a module from the repository root, so no `PYTHONPATH` tweaking is needed:

```powershell
$py = 'D:\HKStudy\.venv\Scripts\python.exe'
& $py -m cs5494.checks units
& $py -m cs5494.train --models fvsbn made --samples 60000 --test-samples 10000 `
      --side 28 --hidden 1536 --layers 1 --epochs 400 --batch 1024 --lr 0.001 `
      --scheduler cosine --device cuda --workers 4 --seed 42 --output results/autoregressive
& $py -m cs5494.train --models gan --arch dcgan --samples 60000 --epochs 400 --batch 512 `
      --lr 0.0002 --lr-d 0.00002 --scheduler cosine --ema 0.999 `
      --device cuda --workers 4 --output results/gan_ema
& $py -m cs5494.evaluate mixture --runs results/made_seed42 results/made_seed43 results/made_seed44
& $py -m cs5494.evaluate gan --runs results/gan_ema --weights ema
& $py -m cs5494.figures digits
& $py -m cs5494.checks report
```

## Environment (the machine that produced `results/`)

| item | value |
|---|---|
| interpreter | `D:\HKStudy\.venv\Scripts\python.exe` (Python 3.14.5) |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| GPU | RTX 5080, 15.9 GiB usable, CUDA 12.8 |
| extra packages | numpy, matplotlib, pillow, scipy |
| report build | XeLaTeX + Biber (TeX Live 2026) |

`requirements.txt` lists the runtime packages; `requirements-lock.txt` is the version snapshot of
the machine that produced the original package and should **not** be installed as-is.

The PowerShell scripts take the interpreter from `PATH`; pass `-Python <path>` to override it.
They fall back to `D:\HKStudy\.venv\Scripts\python.exe` only when that file exists, so a fresh
clone runs on any machine with `python` and `torch` installed.

## Results

| model | test NLL | bits/pixel | note |
|---|---:|---:|---|
| FVSBN, 60k | 75.72 | 0.1393 | linear, no hidden units |
| MADE, 60k, single | 68.45 ± 0.08 | 0.1260 | mean of seeds 42/43/44 |
| **MADE mixture K=3, 60k** | **66.46** | **0.1223** | 7.23 M params, 3× evaluation cost |
| FVSBN, 10k | 120.17 | 0.2211 | |
| MADE, 10k, single | 80.19 ± 0.13 | 0.1476 | |
| MADE mixture K=3, 10k | 76.92 | 0.1415 | |

Nats per image are not comparable across resolutions; bits per pixel is. A 14×14 ablation
(`results/ablation_14x14`) reaches 33.66 nats, which looks better but is worse per pixel.

GAN has no likelihood, so an independent classifier scores every variant
(96.4% on real test digits; real reference: confidence 0.939 / entropy 2.299 / FD 0):

| variant | confidence | entropy | FD |
|---|---:|---:|---:|
| MLP, initial run (shared lr, 60 ep) | 0.8802 | 2.1737 | 34.61 |
| MLP, shared lr at batch 256 | 0.8787 | 2.1655 | 38.30 |
| MLP, two-timescale | 0.8852 | 2.2808 | 7.71 |
| DCGAN, live weights | 0.8477 | 2.2735 | 7.15 |
| DCGAN, moving average | 0.8476 | 2.2740 | 7.12 |

The last two differ by 0.03 while two identical runs differ by 0.14, so the moving average is
reported as a free stability option, not as a proven gain.

Per-epoch wall clock on the RTX 5080: FVSBN 0.43 → 0.12 s, MADE 1.07 → 0.14 s, GAN 5.29 → 0.80 s
against the CPU baseline. Hyperparameter evidence: `HYPERPARAM_SWEEP.md`; integrated results:
`RESULTS.md`.

## Layout

```
cs5494/            code package (see table above)
report/            LaTeX source, figures, build.ps1, main.pdf
submission/        final PDF
results/           every checkpoint, metric and training curve
comparisons/       GAN scores, comparison figures, per-digit JSON
data/              MNIST (auto-download if absent)
logs/              run logs
run_windows.ps1    full retraining pipeline
reproduce.ps1      fast reproduction from saved checkpoints
package.ps1        rebuild the submission zip
```

## Notes

- Every sample figure groups unconditional samples by the digit predicted by the classifier
  (columns 0–9, ten draws per column); this is a selection, not conditional generation.
- Bits per pixel is the comparable unit across resolutions; a mixture is evaluated with
  `logsumexp − log K`, never by averaging logits.
- Single seed per configuration except the autoregressive mixture; FD is a proxy and
  differences below about one unit are within evaluation noise.
- `results/*/config.json` records the command-line arguments used at the time; folders may have
  been moved afterwards.

## Google Colab

Open https://colab.research.google.com/github/Minnesinger02/CS5494AS1/blob/main/assignment.ipynb .
The first code cell clones the repository into /content/CS5494AS1, changes to that
directory and installs requirements.txt. Choose a GPU runtime for full retraining.
Running all cells trains the main models again and can take substantial time.
Local training is valid; Colab retraining is not needed to submit the saved results.
For quick verification from saved weights, after the setup cell run:
    !python -m cs5494.checks units
    !python -m cs5494.sample results/gan_ema gan --which ema
For a clean local checkout, install requirements.txt before running commands.

