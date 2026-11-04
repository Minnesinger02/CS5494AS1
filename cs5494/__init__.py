"""CS5494 assignment code: models, training, evaluation and figure generation.

Modules
    train.py          model definitions (FVSBN, MADE, MLP/DCGAN) and the training loop
    sample.py         draw samples from a saved checkpoint (GAN: live or EMA weights)
    evaluate.py       external scoring: classifier statistics for GANs, mixture likelihood
    figures.py        report figures: per-digit grids, MLP variant comparison
    checks.py         unit tests, NLL re-computation, report/JSON consistency, hashes
    compare_runs.py   side-by-side comparison of result folders
    sweep_made.py     hyper-parameter sweep for the masked autoregressive model

Run everything from the repository root, e.g.

    python -m cs5494.train --models fvsbn made --samples 60000 --side 28 --hidden 1536 ...
    python -m cs5494.evaluate gan --runs results/gan_ema --weights ema
    python -m cs5494.checks report
"""
