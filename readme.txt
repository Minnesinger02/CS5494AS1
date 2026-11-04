CS5494 Topics in Generative AI -- Assignment
Autoregressive and adversarial generative models for MNIST
================================================================================

Student : LIU Naduo (刘纳多)
Student ID : 60777841
Repository : https://github.com/Minnesinger02/CS5494AS1
Report : submission/60777841.pdf   (16 pages, in this folder)

This file is the plain-text "how to run the code" readme required by the assignment.
The same instructions, with more detail, are in README.md and REPRODUCE.md, and in
Appendix A (Run instructions) of the report.

--------------------------------------------------------------------------------
1. WHAT IS IN THIS FOLDER
--------------------------------------------------------------------------------
cs5494/          the code package, one module per job
    train.py             FVSBN, MADE, MLP-GAN and DCGAN definitions, training loop,
                         previews and metrics
    sample.py            draw samples from a saved checkpoint (GAN: live or EMA weights)
    evaluate.py          classifier scores for the GANs, mixture likelihood for MADE
    figures.py           per-digit sample grids, MLP training-regime figure
    checks.py            unit checks, NLL recomputation, report/JSON check, manifest
    compare_runs.py      side-by-side comparison of result folders
    sweep_made.py        MADE hyper-parameter sweep
report/          LaTeX source (main.tex, refs.bib), figures, build.ps1, main.pdf
submission/      final report PDF, named by student ID
results/         every checkpoint, metric and training curve used in the report
comparisons/     GAN evaluation JSONs and comparison figures
data/MNIST/      the MNIST data, bundled, so nothing is downloaded at run time
logs/            run logs
assignment.ipynb notebook version of the pipeline (works on Google Colab)
reproduce.ps1    rebuild every metric, figure and the PDF from the saved checkpoints
run_windows.ps1  full retraining pipeline
readme.txt       this file

--------------------------------------------------------------------------------
2. REQUIREMENTS
--------------------------------------------------------------------------------
Python 3.10 or newer with the packages in requirements.txt:

    pip install -r requirements.txt

i.e. torch, torchvision, numpy, matplotlib, pillow, scipy (+ pymupdf for the report
number check). A CUDA GPU is optional: every command accepts --device cpu, which is
about 7x slower. requirements-lock.txt is the exact version snapshot of the machine
that produced results/ and should NOT be installed as-is.

--------------------------------------------------------------------------------
3. QUICK START -- reproduce every number and figure from the saved checkpoints
--------------------------------------------------------------------------------
No training, about one minute on a laptop:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\reproduce.ps1

The script takes the Python interpreter from PATH; pass -Python <path> to override:

    powershell ... -File .\reproduce.ps1 -Python C:\path\to\python.exe

It runs, in order: unit checks -> resample every model -> classifier scores for all
GAN variants -> mixture likelihood for K=1..3 -> per-digit grids -> NLL recomputation
-> LaTeX report -> check that the 17 numbers quoted in the PDF match the JSON files.
Add -SkipReport if you have no TeX installation (the compiled PDF is already in
submission/).

Everything can also be run one command at a time, from the repository root:

    python -m cs5494.checks units
    python -m cs5494.sample results/gan_ema gan --which ema
    python -m cs5494.evaluate mixture --runs results/made_seed42 results/made_seed43 ^
           results/made_seed44
    python -m cs5494.evaluate gan --runs results/gan_ema --weights ema
    python -m cs5494.figures digits
    python -m cs5494.checks report
    python -m cs5494.checks manifest . --verify

(^ is the Windows line continuation; use \ in bash.)

--------------------------------------------------------------------------------
4. FULL RETRAINING (~12 minutes on an RTX 5080)
--------------------------------------------------------------------------------
    powershell -NoProfile -ExecutionPolicy Bypass -File .\run_windows.ps1

Three MADE seeds on 60k images -> three on 10k -> mixture evaluation -> DCGAN with
exponential moving average -> sampling -> scoring -> figures -> report. Results are
written to results/ with the same seeds, so they overwrite the bundled folders.

Useful switches: -SkipTests, -Seeds 42 (single seed, quick check), -Python <path>.

Single runs, for reference:

    python -m cs5494.train --models fvsbn made --samples 60000 --test-samples 10000 ^
        --side 28 --hidden 1536 --layers 1 --epochs 400 --batch 1024 --lr 0.001 ^
        --scheduler cosine --device cuda --workers 4 --seed 42 ^
        --output results/autoregressive

    python -m cs5494.train --models gan --arch dcgan --samples 60000 --epochs 400 ^
        --batch 512 --lr 0.0002 --lr-d 0.00002 --scheduler cosine --ema 0.999 ^
        --device cuda --workers 4 --output results/gan_ema

--------------------------------------------------------------------------------
5. GOOGLE COLAB
--------------------------------------------------------------------------------
Clone the repository, open assignment.ipynb and run the cells top to bottom; the
notebook calls the same modules with --device auto. On a CPU-only Colab runtime the
autoregressive models still finish, but expect roughly 7x the times in the report.

--------------------------------------------------------------------------------
6. WHERE THE NUMBERS COME FROM
--------------------------------------------------------------------------------
Every table entry in the report is traceable:

    results/<run>/config.json    the exact command-line arguments of that run
    results/<run>/metrics.json   its test NLL, bits per pixel and timings
    results/<run>/<model>/history.json   per-epoch loss and wall clock
    results/ensemble_metrics.json        mixture likelihood for K = 1, 2, 3
    comparisons/gan_eval*.json           classifier scores for each GAN variant
    MANIFEST_SHA256.json                 SHA-256 of every file in the package

python -m cs5494.checks report re-extracts the PDF text and compares the 17 numbers
quoted in the report with the JSON files above; python -m cs5494.checks nll
recomputes the held-out likelihood from the saved weights.

--------------------------------------------------------------------------------
7. TROUBLESHOOTING
--------------------------------------------------------------------------------
"python not found"        install Python, or pass -Python <path> to the script.
"No module named cs5494"  run the command from the repository root.
CUDA not available        add --device cpu (or --device auto).
"xelatex not found"       only the report rebuild needs LaTeX; the compiled PDF is
                          already in submission/, and reproduce.ps1 -SkipReport
                          skips that step.
Large weight files        each MADE checkpoint is ~18 MB because the causal masks are
                          stored as buffers next to the weights; they load normally.
