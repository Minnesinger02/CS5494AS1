# 复现说明（CS5494 Assignment）

代码是一个小包 `cs5494/`，每个模块负责一件事；全部实验产物按顺序放在 `results/` 下。
本文件给出两条复现路径：**零训练快速复现**（约 1 分钟）和**完整重训**（约 12 分钟）。

## 0. 先看什么

| 想看 | 打开 |
|---|---|
| 最终报告（16 页） | `submission/60777841_61083251_60369937.pdf` |
| 报告源码 | `report/main.tex`、`report/refs.bib` |
| 五个模型的样本（按数字分列） | `report/figures/digits_*.png` |
| 三模型并排对照 | `comparisons/digit_grid_comparison.png` |
| 自回归主表数据 | `results/autoregressive/metrics.json` |
| 集成结果 | `results/ensemble_metrics.json`、`results/ensemble_10k_metrics.json` |
| GAN 结果 | `comparisons/gan_eval*.json` |
| 调参依据 | `HYPERPARAM_SWEEP.md`、`sweep_made*.csv` |
| 集成与 EMA 明细 | `RESULTS.md` |
| 实验顺序与结论 | `00_组内对比说明.md` |

## 1. 环境

| 项 | 值 |
|---|---|
| Python | 3.14.5（本机 `D:\HKStudy\.venv\Scripts\python.exe`） |
| 深度学习 | torch 2.11.0+cu128、torchvision 0.26.0+cu128 |
| 其他 | numpy、matplotlib、pillow、scipy |
| 报告编译（可选） | XeLaTeX + Biber（TeX Live 2026，`D:\texlive\2026\bin\windows`） |
| GPU | RTX 5080 16 GB（CPU 可跑，约慢 7 倍） |

换机器时把 `$py` 换成自己的解释器：

```powershell
$py = (Get-Command python).Source
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` 是运行依赖；`requirements-lock.txt` 是原包环境的版本快照，不要直接照装。

## 2. 零训练快速复现（推荐先跑，约 1 分钟）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\reproduce.ps1
```

它依次执行（全部通过 `python -m` 调用 `cs5494/` 里的模块）：

| 步骤 | 命令 | 作用 |
|---|---|---|
| 1 | `-m cs5494.checks units` | 因果梯度、二值采样、GAN 形状自检 |
| 2 | `-m cs5494.sample <run> <model>` | 从权重重新采样（含 DCGAN 的 raw/ema） |
| 3 | `-m cs5494.evaluate gan --runs ...` | 独立分类器给 6 个 GAN 版本打分 |
| 4 | `-m cs5494.evaluate mixture --runs ...` | 重算 K=1/2/3 的混合似然 |
| 5 | `-m cs5494.figures digits` | 生成 5 个模型的按数字分列图 |
| 6 | `-m cs5494.figures variants` | 生成 MLP 三版本对照图 |
| 7 | `-m cs5494.checks nll --runs ...` | 从 checkpoint 重算 NLL 并比对 |
| 8 | `report\build.ps1` | XeLaTeX + Biber 编译 |
| 9 | `-m cs5494.checks report` | 抽取 PDF 文本，核对 17 个数值 |

换解释器：`powershell -File .\reproduce.ps1 -Python "C:\path\python.exe"`；
不想编译报告可加 `-SkipReport`。

## 3. 完整重训（约 12 分钟，RTX 5080）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_windows.ps1
```

依次：3 个 MADE 种子（60k）→ 3 个 10k 种子 → 集成评估 → DCGAN+EMA → 采样 → 评分 → 出图 →
编报告 → 数值核对。结果写入 `results/`（与随包结果同种子，会覆盖同名目录）。

单步示例：

```powershell
& $py -m cs5494.train --models made --samples 60000 --test-samples 10000 --side 28 `
    --hidden 1536 --layers 1 --epochs 400 --batch 1024 --lr 0.001 --scheduler cosine `
    --device cuda --workers 4 --seed 42 --output results/made_seed42

& $py -m cs5494.train --models gan --arch dcgan --samples 60000 --epochs 400 --batch 512 `
    --lr 0.0002 --lr-d 0.00002 --scheduler cosine --ema 0.999 `
    --device cuda --workers 4 --output results/gan_ema
```

`run_windows.ps1 -SkipTests` 跳过自检；`-Seeds 42` 只训一个种子用于快速验证。

## 4. 只重建报告

```powershell
cd report
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
cd ..
& $py -m cs5494.checks report
```

报告中的每个数字都来自 `results/` 与 `comparisons/` 的 JSON；`checks report` 会把 PDF 文本与
这些文件逐项比对，不一致即报错。

## 5. 完整性校验

```powershell
& $py -m cs5494.checks manifest . --verify
```

应输出 `N entries, 0 mismatched`。

## 6. 目录说明

```
.
├─ cs5494/                    代码包
│   ├─ train.py               模型（FVSBN / MADE / MLP-GAN / DCGAN）与训练循环
│   ├─ sample.py              从权重重采样（GAN 可选 raw / ema）
│   ├─ evaluate.py            gan：分类器打分；mixture：K 个 MADE 的混合似然
│   ├─ figures.py             digits：按数字分列；variants：MLP 三版本对照
│   ├─ checks.py              units / nll / report / manifest 四类检查
│   ├─ compare_runs.py        多套结果目录的并排对比
│   └─ sweep_made.py          MADE 容量/正则化扫描
├─ report/                    LaTeX 源文件、图、build.ps1、main.pdf
├─ submission/                最终 PDF
├─ results/                   全部实验产物（按模型/设置平铺）
│   ├─ autoregressive/        FVSBN 与单模型 MADE（同进程训练）
│   ├─ made_seed42|43|44/     三个 MADE 种子（60k）
│   ├─ made10k_seed42|43|44/  三个 MADE 种子（10k）
│   ├─ ensemble_metrics.json / ensemble_10k_metrics.json / *_ablation.png
│   ├─ gan_initial/           最早的 MLP-GAN（单学习率，60 轮）
│   ├─ gan_mlp_shared_lr/     同学习率 + 大 batch
│   ├─ gan_mlp/               MLP + TTUR
│   ├─ gan_dcgan_repeat/      同配置重复运行（估计噪声）
│   ├─ gan_ema/               DCGAN + EMA（model.pt 与 model_ema.pt）
│   ├─ ablation_10k/          10,000 张训练的子集实验
│   └─ ablation_14x14/        14×14 分辨率对照
├─ comparisons/               对比图、GAN 评分 JSON、逐数字展示 JSON
├─ data/                      MNIST（离线可用；缺失时自动下载）
├─ logs/                      运行日志
├─ run_windows.ps1            完整重训
├─ reproduce.ps1              零训练复现
└─ package.ps1                重新打包
```

## 7. 常见问题

**CUDA 不可用**：把 `--device cuda` 换成 `cpu` 或 `auto`；所有命令都接受 `--device`。

**`xelatex: not found`**：只影响重建报告，把 `D:\texlive\2026\bin\windows` 换成自己的路径，
或直接用 `submission/` 里已编译的 PDF。

**`报告里的姓名/学号`**：已填写三人小组（刘纳多 60777841、崔兆楠 61083251、王偲铭 60369937），
标题页同时给出代码与数据的 GitHub 链接 <https://github.com/Minnesinger02/CS5494AS1>；
提交件按学号命名为 `submission/60777841_61083251_60369937.pdf`。组员如有变动，需要重编译报告并
按 `学号_学号_学号.pdf` 重新命名。

**`No module named cs5494`**：请从仓库根目录运行（`python -m cs5494.xxx` 需要根目录在搜索路径上，
在根目录执行即满足）。

**权重文件较大**：MADE 的 `model.pt` 约 18 MB，掩码被注册成 buffer 一起保存，不影响使用。
