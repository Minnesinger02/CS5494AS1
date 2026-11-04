# 集成与 EMA 结果

日期：2026-11-04。硬件：RTX 5080 15.9 GiB。数据：28×28 二值化 MNIST（阈值 0.5），训练 60,000 张 / 测试 10,000 张。
产物：`results/made_seed42|43|44/`、`results/ensemble_metrics.json`、`results/gan_ema/`。

## 一、自回归：3 模型集成

单模型（1×1536，400 轮，batch 1024，lr 1e-3 + cosine）：

| 种子 | 测试 NLL | bpp |
|---|---:|---:|
| 42 | 68.365 | 0.1258 |
| 43 | 68.513 | 0.1261 |
| 44 | 68.486 | 0.1260 |
| 均值 ± 标准差 | **68.45 ± 0.08** | 0.1260 |

混合分布 p(x)=1/K·Σp_k(x)（对数域 `logsumexp − log K`）：

| K | 测试 NLL | bpp | 参数量 | 相对 K=1 |
|---:|---:|---:|---:|---:|
| 1 | 68.45 | 0.1260 | 2.41 M | — |
| 2 | 67.07 | 0.1234 | 4.82 M | −2.0% |
| 3 | **66.46** | **0.1223** | 7.23 M | **−2.9%** |

10,000 张同口径：

| K | 测试 NLL | bpp |
|---:|---:|---:|
| 1 | 80.19 | 0.1476 |
| 2 | 77.86 | 0.1433 |
| 3 | **76.92** | **0.1415** |

采样说明：混合分布没有闭式自回归采样解，展示用样本来自单个成员；表中 bpp 是混合分布的似然。
曲线：`results/ensemble_ablation.png`、`results/ensemble_10k_ablation.png`。

## 二、GAN：TTUR + EMA

| 版本 | 结构 | 轮数 | 置信度 | 类别熵 | FD |
|---|---|---|---:|---:|---:|
| 初始（单学习率，batch 128） | MLP | 60 | 0.8802 | 2.1737 | 34.61 |
| 同学习率 + 大 batch | MLP | 150 | 0.8787 | 2.1655 | 38.30 |
| TTUR（D=1e-4） | MLP | 150 | 0.8852 | 2.2808 | 7.71 |
| TTUR（D=2e-5）实时权重 | DCGAN | 400 | 0.8477 | 2.2735 | 7.15 |
| **同上 + EMA(d=0.999)** | DCGAN | 400 | 0.8476 | 2.2740 | **7.12** |

关于 EMA 的结论要如实写：FD 从 7.15 降到 7.12，差 0.03；同一配置两次独立运行的 FD 差 0.14
（`results/gan_dcgan_repeat` 与 `results/gan_ema` 的实时权重，cuDNN 卷积不可位级复现），
因此 **0.03 在噪声范围内，不能宣称显著提升**。EMA 的价值在于"推理不增加成本、无需改动训练"的稳定性选项。

## 三、与前一次实现的对比

| 指标 | 14×14 实现（`results/ablation_14x14`） | 当前 28×28 | 变化 |
|---|---:|---:|---:|
| FVSBN NLL | 33.66 | 75.72 | 像素数 196→784，只比 bpp |
| FVSBN bpp | 0.2478 | **0.1393** | −43.8% |
| MADE NLL | 28.04 | 68.45 | 同上 |
| MADE bpp | 0.2064 | **0.1260** | −38.9% |
| GAN 最佳 FD | 34.61 | **7.12** | −79.4% |

注：14×14 与 28×28 的 nats/image 不可直接比较；表中 bpp 才是同口径。
MADE 的单模型数字（68.45）与 `results/autoregressive/made`（68.59）差异来自随机数流——
后者与 FVSBN 在同一进程里训练，初始化与批次顺序不同，两者都是 seed 42 的合法结果。

## 四、命令

```powershell
$py = 'D:\HKStudy\.venv\Scripts\python.exe'
foreach ($s in 42,43,44) {
  & $py -m cs5494.train --models made --samples 60000 --test-samples 10000 --side 28 --hidden 1536 --layers 1 `
    --epochs 400 --batch 1024 --lr 0.001 --scheduler cosine --device cuda --workers 4 --seed $s `
    --output "results/made_seed$s"
}
& $py -m cs5494.evaluate mixture --runs results/made_seed42 results/made_seed43 results/made_seed44
& $py -m cs5494.train --models gan --arch dcgan --samples 60000 --epochs 400 --batch 512 --lr 0.0002 `
  --lr-d 0.00002 --scheduler cosine --ema 0.999 --device cuda --workers 4 --output results/gan_ema
& $py -m cs5494.evaluate gan --runs results/gan_ema --weights raw --out comparisons/gan_eval_ema.json
& $py -m cs5494.evaluate gan --runs results/gan_ema --weights ema --out comparisons/gan_eval_ema_tmp.json
```

完整流程见 `run_windows.ps1`；零训练复现见 `reproduce.ps1`。
