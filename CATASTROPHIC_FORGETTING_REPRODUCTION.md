# 灾难性遗忘复现状态记录

最后更新：2026-07-01

本次补充：服务器完整 catastrophic baseline 已跑完；结果已整理到 `published_results/baseline/paper_ch4_catastrophic_source_seed0.json`，并作为当前 baseline 参考。


## 2026-07-01 正式服务器 baseline 结果

运行信息：

- 运行目录：`/root/autodl-tmp/NGSG-spyketorch-dev-2928f7e`
- run name：`paper_ch4_catastrophic_source_seed0`
- 配置：`configs/baseline/catastrophic_mnist_emnist.yaml`
- 代码版本：`2928f7e fix: reuse feature checkpoints across cache paths`
- 本机原始副本：`experiments/server_paper_ch4_catastrophic_source_seed0/`
- GitHub 精简结果：`published_results/baseline/paper_ch4_catastrophic_source_seed0.json`

关键结果：

| 指标 | 本次结果 | 论文 catastrophic forgetting 参考 |
| --- | ---: | ---: |
| Initial MNIST / Task1 after Task1 | 93.14% | 90.8 ± 0.9% |
| Subsequent MNIST / Task1 after Task2 | 47.54% | 48.1 ± 4.8% |
| Subsequent EMNIST / Task2 after Task2 | 75.43% | 78.4 ± 1.2% |
| Forgetting | 45.60% | 约 42.7% |
| Avg Acc | 61.48% | - |

运行细节：

- Task1 MNIST：24,000 train / 10,000 test。
- Task2 EMNIST ABDEGHNQRS：24,000 train / 8,000 test。
- Task1 S1/S2 checkpoint：`checkpoints/features/paper_task1_s1e2_s2e4_f2d9040e64b69b5e.pt`，exact match 加载。
- Task2 S1/S2 checkpoint：`checkpoints/features/paper_task2_s1e2_s2e4_60c0a06b55746fb6.pt`，fallback match 加载。
- Task1 C2 cache：`data/features/c2/126b5223a233559d`。
- 训练过程已使用 `c2_feature_cache.batch_size: 1024`，避免 cached C2 tensor 一次性 24,000 batch 造成 CUDA OOM。

判断：

catastrophic forgetting 趋势已经复现出来。Task2 后 MNIST 保留率 47.54%，和论文 48.1% 非常接近；Initial MNIST 高约 2.3 个点，EMNIST 低约 3 个点。当前可以把该结果作为后续 winner-frequency logging 和 NGSG 的 baseline 参考，但如果目标是严格追论文表格数字，还需要继续核对 EMNIST 数据处理、作者 notebook 的张量保存格式、随机种子和评估协议。
## 2026-06-30 晚更新：active baseline YAML 收敛

`configs/baseline/` 当前只保留 3 个 active YAML：

- `catastrophic_mnist_emnist.yaml`：服务器正式完整 catastrophic baseline。
- `catastrophic_mnist_emnist_feature_checkpoint.yaml`：仅用于重建 S1/S2 checkpoint 和 C2 cache。
- `catastrophic_mnist_emnist_paper_medium.yaml`：本地中等规模诊断。

旧的 toy/probe/stabilizer/frozen/Langevin/joint-training YAML 已删除；本文件后面的旧命令只作为历史实验记录，不再代表当前推荐入口。当前不复现 joint training。
## 2026-06-30 晚更新：S1/S2 checkpoint 已生成并推送

当前 `dev` 已包含 feature-cache implementation、正式 S1/S2 checkpoint 和 active config 收敛说明。

已经完成的本地 feature-only 运行：

- 运行名：`paper_feature_checkpoint_full`
- 配置：`configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml`
- 设备：CUDA
- Task1 MNIST：24,000 个训练样本，S1 2 epoch，S2 4 epoch
- Task2 EMNIST：24,000 个训练样本，S1 2 epoch，S2 4 epoch
- S3：跳过，`output_training.skipped = feature_only`

已随 git 跟踪并推送的小 checkpoint：

```text
checkpoints/features/paper_task1_s1e2_s2e4_f26edcfb75b5d681.pt
checkpoints/features/paper_task2_s1e2_s2e4_60c0a06b55746fb6.pt
```

本地已生成但不进 git 的大缓存：

```text
data/preprocessed/paper_source/e40948d119942523 -> 24000 个 .pt
data/preprocessed/paper_source/a390cd0731f5a594 -> 24000 个 .pt
data/features/c2/7ea7511c03cbf772 -> 24000 个 .pt
data/features/c2/8d3b69701aacd0b0 -> 24000 个 .pt
```

`data/features/c2` 两个正式目录合计约 46GB，因此不通过 git 同步。服务器端拉取 `dev` 后可以直接获得 S1/S2 checkpoint；首次跑完整 baseline 时会在服务器本地重建 C2 cache，之后可重复使用。

服务器当前推荐命令：

```bash
git fetch origin
git checkout dev
git pull origin dev
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist.yaml --device cuda --run-name paper_ch4_catastrophic_source_seed0
```

只有当 checkpoint 缺失或需要重建时，才运行 feature-only 配置：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml --device cuda --run-name paper_feature_checkpoint_full
```
## 2026-06-30 更新：缓存与服务器分支整理

`origin/codex/server-preprocess-cache` 上的共享 infra 已合并进 `dev`。之后本地和服务器都应优先跟 `dev` 跑，不再把 `codex/server-preprocess-cache` 当作长期实验分支维护。

本次进入 `dev` 的关键改动：

- paper-source 路线增加离线预处理缓存，缓存目录为 `data/preprocessed/paper_source/<hash>/`。
- 训练日志增加 epoch 级进度输出，长跑时更容易判断是否仍在推进。
- EMNIST letters 在服务器上如果经由 `torchvision.datasets.EMNIST(split="letters")` 初始化失败，会从 `data/emnist/EMNIST/raw/gzip/` 下的 raw idx / idx.gz 文件直接读取。
- `SERVER_LATEST_STATUS.md` 被明确视为服务器运行时快照，不再进入 git；需要长期保留的信息应整理进本文档或 README。

服务器更新代码时建议：

```bash
git fetch origin
git checkout dev
git pull origin dev
pip install -r requirements.txt
```

缓存进度检查示例：

```bash
find data/preprocessed/paper_source -name '*.pt' | wc -l
tail -n 50 logs/preprocess_*.log
```

## 2026-06-29 最新更新：暂停运行并移植论文源码

按你的要求，已经停止正在运行的 `paper_medium_source_port_seed0` 中等规模实验。停止前它没有报错，stderr 为空；进度到 task1 的 S3 R-STDP 第 24/50 个 epoch 左右，训练 proxy 约 45%-47%。这次停止是主动暂停，不是程序崩溃。

这次整理后的主线不再继续修旧的近似实现，而是把论文作者源码仓库 `dmitryanton68/continuous_learning` 中的 `MozafariMNIST2018` 路线移植到当前项目：

- 新增模型文件：`src/models/paper_mozafari.py`。
- 新增模型入口：`PaperMozafariMNIST2018` / `build_paper_mozafari_network`。
- trainer 中新增 paper-source 分支：当配置里 `model.architecture: paper_spyketorch` 时，直接走论文源码兼容训练流程。
- 主完整配置 `configs/baseline/catastrophic_mnist_emnist.yaml` 已切换为 `architecture: paper_spyketorch` 和 `learning_rule: paper_source_rstdp`。
- 中等规模调试配置保留在 `configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml`。

论文源码兼容实现目前对齐的关键点：

| 模块 | 现在的实现 |
| --- | --- |
| 输入预处理 | 6 个 DoG kernel，`utils.Filter(..., padding=6, thresholds=50)`，`sf.local_normalization(..., 8)`，`utils.Intensity2Latency(15)` |
| S1 | `snn.Convolution(6, 30, 5, 0.8, 0.05)`，阈值 15，`k=5` |
| S2 | `snn.Convolution(30, 250, 3, 0.8, 0.05)`，阈值 10，`k=8` |
| S3 | `snn.Convolution(250, 200, 5, 0.8, 0.05)` |
| S1/S2 STDP | `snn.STDP(..., (0.004, -0.003))` |
| S3 reward STDP | `snn.STDP(conv3, (0.004, -0.003), False, 0.2, 0.8)` |
| S3 anti-STDP | `snn.STDP(conv3, (-0.004, 0.0005), False, 0.2, 0.8)` |
| 输出映射 | 200 个 S3 feature map，每类 20 个，`decision_map = [0]*20 + ... + [9]*20` |
| R-STDP 自适应学习率 | 按论文源码的 batch correct/wrong 比例更新 reward / punish 学习率 |

关于 `use_weight_stabilizer`：这个字段不是论文实验中的一个显式超参数名。它来自我们旧 trainer 对 SpykeTorch `snn.STDP` 的包装。论文作者源码里 S3 reward/anti-STDP 是直接写死为 `use_stabilizer=False, lower_bound=0.2, upper_bound=0.8`；S1/S2 则使用 `snn.STDP` 默认行为。因此新的 `paper_spyketorch` 路线不再通过全局 `use_weight_stabilizer` 控制论文实验，避免把旧诊断参数误当成论文设置。

当前推荐命令：

```powershell
# 只检查配置和任务规模，不训练
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist.yaml --device auto --dry-run --run-name paper_source_strict_dryrun

# 中等规模源码移植版，用来先看学习曲线
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist_paper_medium.yaml --device auto --run-name paper_medium_source_port_seed0

# 完整论文规模，极慢，确认中等规模趋势正常后再跑
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist.yaml --device auto --run-name paper_ch4_catastrophic_source_seed0
```

仍需注意：当前移植已经按作者源码结构对齐，并已加入 EMNIST raw idx fallback 来绕过服务器上 `torchvision.datasets.EMNIST(split="letters")` 初始化不稳定的问题。不过 EMNIST 方向修正、作者仓库里预处理好的张量文件来源、以及 notebook 中实际加载的 `.pt` 文件格式仍可能带来细微差异。后续如果要追到论文表格数值，需要继续核对这些数据保存与加载细节。

## 2026-06-29 中等规模论文源码移植版运行结果

运行名：`paper_medium_source_port_seed0_live`

运行命令：

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe -u scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist_paper_medium.yaml --device auto --run-name paper_medium_source_port_seed0_live
```

实时日志：

```powershell
Get-Content -Path F:\paper_code\NGSG-spyketorch\experiments\_logs\paper_medium_source_port_seed0_live.out.log -Tail 30 -Wait
```

结果文件：`experiments/paper_medium_source_port_seed0_live/result.json`

配置规模：MNIST 每类 100 个训练样本，EMNIST 每类 100 个训练样本；task1 的 S3 R-STDP 50 epoch，task2 的 S3 R-STDP 10 epoch。

训练过程摘要：

| 阶段 | 最后一个 S3 epoch 的 train_acc_proxy |
| --- | --- |
| task1 / MNIST | 54.3% |
| task2 / EMNIST | 45.0% |

最终测试指标：

| 指标 | 数值 |
| --- | --- |
| Task1 after Task1 / MNIST 初训后 | 46.2% |
| Task1 after Task2 / EMNIST 后 MNIST 保持 | 22.6% |
| Task2 after Task2 / EMNIST 后 EMNIST | 41.7% |
| Forgetting | 23.6 个百分点 |
| Avg Acc | 32.15% |

判断：这次中等规模结果没有回到随机水平，也没有 silent network；它已经出现论文 baseline 想观察的方向，即 MNIST 先学到一部分，随后训练 EMNIST 后 MNIST 明显下降。它和论文 Table 1 的 90.8% / 48.1% / 78.4% 仍然差距很大，但当前不是同等实验规模：这里是每类 100 个样本、S3 初训 50 epoch，而论文初训使用每类约 2400 个样本且 S3 初训 600 epoch。因此这次结果更适合作为“源码移植版能学习并能产生遗忘现象”的中等规模 smoke/diagnostic，而不是最终论文数值复现。
## 当前结论

当前仓库的主 catastrophic forgetting 路线已经从“仓库内近似版 STDP/R-STDP”和“tutorial 风格近似复现”进一步切换为**论文作者源码 dmitryanton68/continuous_learning 的 SpykeTorch 移植版**。

也就是说：

- `src/` 下面的主训练路径现在使用 `SpykeTorch.snn.Convolution`、`SpykeTorch.snn.Pooling`、`SpykeTorch.snn.STDP` 和 `SpykeTorch.functional.fire`。
- 之前写的 dense tensor 近似 STDP/R-STDP、近似网络和 supervised sanity baseline 已经移动到 `approx/legacy_approx/`。
- `requirements.txt` 已加入官方安装源：`git+https://github.com/miladmozafari/SpykeTorch.git`。
- 已在 `C:\Users\pw\.conda\envs\Spyketorch\python.exe` 环境中安装官方 SpykeTorch，并跑通了一个小规模 probe。

## 目标

复现论文 **Continuous Learning of Spiking Networks Trained with Local Rules** 中 Table 1 的 `Catastrophic forgetting` baseline。

论文目标结果：

| 阶段 | 数据集 | 论文结果 |
| --- | --- | --- |
| 初始训练 | MNIST | 90.8% +/- 0.9 |
| 后续训练后的旧任务保持 | MNIST | 48.1% +/- 4.8 |
| 后续训练的新任务 | EMNIST | 78.4% +/- 1.2 |

论文协议：

1. 先训练 MNIST 数字任务。  
2. 假设 MNIST 旧数据不可再用。  
3. 再用同一个网络训练 EMNIST 字母任务。  
4. 后续训练中 S1 和 S2 分别用 STDP 训练 2 和 4 个 epoch。  
5. S3 用 R-STDP 训练 100 个 epoch。  
6. 表格脚注说明初始 MNIST 训练使用了 600 个 epoch。  

协议核对来源：论文 arXiv 页面/PDF：`https://arxiv.org/abs/2111.09553`。

## 重要修正

论文 Table 1 不是使用全部 26 个 EMNIST 字母，也不是 36 类输出。

论文使用 10 个 EMNIST 大写字母：

`A, B, D, E, G, H, N, Q, R, S`

在 torchvision 的 `EMNIST(split="letters")` 中，letters 标签从 1 开始，因此对应：

`[1, 2, 4, 5, 7, 8, 14, 17, 18, 19]`

当前实现把它们映射到 10 个输出类：

`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

因此当前主配置仍然是：10 类输出，每类 20 个 S3 神经元，总共 200 个 S3 神经元。

## 当前主实现

### 数据协议

文件：`src/utils/data.py`

当前支持：

- `tasks.task_specs` 多数据集任务配置。
- MNIST + EMNIST letters 混合任务。
- EMNIST 图像方向修正。
- 标签重映射。
- 每类固定数量的平衡采样。

完整论文配置中：

- MNIST 训练：每类 2400，共 24000。
- EMNIST 训练：每类 2400，共 24000。
- MNIST 测试：完整 10000。
- EMNIST 所选 10 个字母测试：约 8000。

### 模型

文件：`src/models/network.py`

当前模型使用官方 SpykeTorch：

- DoG 预处理使用 `SpykeTorch.utils.DoGKernel` 生成卷积核。
- Spike latency 编码使用 `SpykeTorch.utils.Intensity2Latency`。
- S1/S2/S3 使用 `SpykeTorch.snn.Convolution`。
- C1/C2 使用 `SpykeTorch.snn.Pooling`。
- firing 使用 `SpykeTorch.functional.fire`。
- point-wise inhibition 使用 `SpykeTorch.functional.pointwise_inhibition`。

网络结构：

- DoG 输入通道：6。
- S1 maps：30。
- S2 maps：250。
- S3 neurons：200。
- classes：10。
- neurons per class：20。

### S1/S2 STDP

文件：`src/trainers/baseline_trainer.py`

当前 S1/S2 的 STDP 直接使用官方：

`SpykeTorch.snn.STDP`

训练时逐样本生成 spike-wave，然后调用：

```python
stdp(input_spikes, potentials, output_spikes, kwta=..., inhibition_radius=...)
```

注意：官方 SpykeTorch 的 `snn.STDP` 对象内部学习率参数默认在 CPU。为了支持 CUDA，训练器中创建 STDP 后会执行：

```python
stdp.to(device)
```

### S3 R-STDP

文件：`src/plasticity/rstdp.py`

已按 SpykeTorch 官方 `tutorial.ipynb` 的 reinforcement learning 写法改为 `STDP + anti-STDP`：

```python
stdp = snn.STDP(conv_layer=s3, learning_rate=(a_plus, a_minus))
anti_stdp = snn.STDP(conv_layer=s3, learning_rate=(-a_plus, anti_a_minus))
```

训练时先用 `SpykeTorch.functional.get_k_winners(...)` 选择 S3 winner，再用 `winner_feature // neurons_per_class` 得到预测类别：

- 如果预测正确：调用官方 `stdp(...)`。
- 如果预测错误：调用官方 `anti_stdp(...)`。

因此当前 S3 不再是仓库内手写 delta 公式；它是 tutorial 风格的 reward-modulated STDP，底层两条分支都调用官方 `SpykeTorch.snn.STDP`。
## 近似实现隔离

旧近似实现已经移动到：

`approx/legacy_approx/`

包含：

- `approx/legacy_approx/models/layers.py`
- `approx/legacy_approx/models/network.py`
- `approx/legacy_approx/models/supervised.py`
- `approx/legacy_approx/plasticity/stdp.py`
- `approx/legacy_approx/plasticity/rstdp.py`
- `approx/legacy_approx/configs/catastrophic_supervised.yaml`

这些文件不再是主路径，只作为历史参考和对照。

## 当前配置

### 完整论文协议配置

`configs/baseline/catastrophic_mnist_emnist.yaml`

用途：按论文规模尝试复现。

主要设置：

- `learning_rule: spyketorch_stdp_rstdp`
- MNIST 每类 2400。
- EMNIST 每类 2400。
- S1 STDP：task1 2 epoch，task2 2 epoch。
- S2 STDP：task1 4 epoch，task2 4 epoch。
- S3 R-STDP：task1 600 epoch，task2 100 epoch。

### 快速 probe 配置

`configs/baseline/catastrophic_mnist_emnist_probe.yaml`

用途：快速确认官方 SpykeTorch 路线能跑通。

主要设置：

- 每类训练样本：20。
- 每类测试样本：20。
- batch size：16。
- S1/S2/S3 每个阶段 1 epoch。

## 已完成验证

### 安装官方 SpykeTorch

已在用户指定环境安装：

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe -m pip install git+https://github.com/miladmozafari/SpykeTorch.git
```

`requirements.txt` 也已加入同一个安装源。

### 语法检查

已通过：

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe -m py_compile src\models\network.py src\models\layers.py src\models\__init__.py src\plasticity\rstdp.py src\plasticity\__init__.py src\trainers\baseline_trainer.py src\utils\data.py scripts\run_baseline.py
```

### 官方 SpykeTorch probe

运行命令：

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist_probe.yaml --device auto --run-name official_spyketorch_probe_seed0
```

结果：

| 指标 | 数值 |
| --- | --- |
| Task1 after Task1 | 5.0% |
| Task1 after Task2 | 12.0% |
| Task2 after Task2 | 11.5% |
| Forgetting | -7.0% |
| Avg Acc | 11.75% |

解释：这个结果只是 probe，不代表论文复现结果。它的数据量和 epoch 都极小，目的只是验证官方 SpykeTorch 主路径可以端到端执行。它已经确认：

- 数据构建正确。
- 官方 `snn.Convolution` 能运行。
- 官方 `snn.STDP` 能运行。
- S3 reward update 能运行。
- 训练和评估结果能写入 JSON/CSV。

### 中等规模 SpykeTorch tutorial R-STDP 实验

运行命令：
```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist_medium.yaml --device auto --run-name medium_tutorial_rstdp_seed0
```

配置：每类 100 个训练样本、100 个测试样本；S1/S2 分别为 2/4 epoch；S3 R-STDP 为 task1 10 epoch、task2 5 epoch。

结果：
| 指标 | 数值 |
| --- | --- |
| Task1 after Task1 | 10.2% |
| Task1 after Task2 | 10.0% |
| Task2 after Task2 | 10.0% |
| Forgetting | 0.2% |
| Avg Acc | 10.0% |

S3 训练 proxy：task1 在第 6-7 个 epoch 最高约 15.8%，随后回落到 10.7%；task2 基本维持在 10%。这说明当前实现虽然已经按 SpykeTorch tutorial 调用 `stdp + anti_stdp`，但中等规模下仍未形成有效分类能力，问题更可能在特征层参数、输入预处理、阈值、winner/class 映射或论文原始超参数对齐上。
### 2026-06-29 诊断结论：准确率卡在随机水平的原因

已停止完整严格实验 `paper_ch4_catastrophic_strict_seed0`。停止前进度为 task1 的 S3 第 6/600 个 epoch 左右，训练 proxy 仍约 10%，说明问题不是灾难性遗忘，而是 MNIST 初始任务尚未学起来。

诊断结果：

- CUDA 正常，训练进程确实在 RTX 3060 Laptop GPU 上运行，但 GPU 利用率较低，主要瓶颈是单样本 STDP/R-STDP Python 循环。
- 随机初始化时 S3 winner 分布大致均匀，各层 spike 不为 0，因此不是完全 silent network。
- 去掉 `pointwise_inhibition` 没有改善，反而使预测更偏向少数类，因此它不是主要原因。
- 严格配置下 `weight_mean=0.8` 且 `weight_clip_max=0.8`，初始约 49%-50% 权重已经大于等于 0.8。使用官方 STDP stabilizer 时，更新项包含 `(weight - lower_bound) * (upper_bound - weight)`，权重靠近上界时更新接近 0，导致大量突触几乎被冻结。

短诊断对照：

| 设置 | S3 5 epoch 后趋势 |
| --- | --- |
| `use_weight_stabilizer: true`，`wmin=0.2,wmax=0.8` | 约 10%-11%，基本不动 |
| `use_weight_stabilizer: false`，`wmin=0.2,wmax=0.8` | 训练 proxy 从 9% 升到约 30.5%，短诊断集约 37.5% |

因此当前修正版先关闭 stabilizer，保留论文权重边界和 SpykeTorch tutorial 的 `stdp + anti_stdp` 训练形式，用中等规模实验验证 MNIST 是否能先学起来。

### 修正版诊断配置

文件：`configs/baseline/catastrophic_mnist_emnist_medium_stabilizer_off.yaml`

主要设置：

- 每类训练样本：100。
- 每类测试样本：100。
- `weight_clip_min: 0.2`。
- `weight_clip_max: 0.8`。
- `use_weight_stabilizer: false`。
- S1/S2：2/4 epoch。
- S3：task1 50 epoch，task2 10 epoch。
- 每 1000 个样本输出一次进度日志。

运行命令：

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist_medium_stabilizer_off.yaml --device auto --run-name medium_stabilizer_off_seed0
```
### 修正版运行中记录

运行名：`medium_stabilizer_off_seed0`

当前进度日志显示修正版已经进入 task1 的 S3 训练，早期训练 proxy 明显高于之前 stabilizer 开启时的随机水平：

| S3 epoch | train_acc_proxy |
| --- | --- |
| 1/50 | 8.5% |
| 2/50 | 15.5% |
| 3/50 | 22.1% |
| 4/50 | 26.7% |

这说明关闭 stabilizer 后，S3 R-STDP 至少在中等规模诊断中已经开始学习。后续需要等 `result.json` 生成后再判断 MNIST test accuracy、EMNIST subsequent accuracy 和 forgetting。

查看实时进度：

```powershell
Get-Content -Path F:\paper_code\NGSG-spyketorch\experiments\_logs\medium_stabilizer_off_seed0.out.log -Tail 30 -Wait
```
## 当前仍需注意的问题

S3 R-STDP 已经改成 SpykeTorch tutorial 的 `stdp + anti_stdp` 形式。后续重点不再是“是否使用官方 STDP”，而是复现实验细节是否足够接近论文：

1. 继续核对论文与作者代码中的阈值、学习率和 winner/class 映射。
2. 增加中等规模实验，先确认 MNIST 初训能高于随机水平。
3. 调整 firing threshold、STDP 学习率和 anti-STDP 奖惩参数。
4. 加每个 epoch 的 eval snapshot，观察学习曲线。
5. 完整跑 600/100 epoch 前，先用小规模和中等规模配置确认训练曲线方向正确。
## 如何运行

### 快速验证官方 SpykeTorch 路线

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist_probe.yaml --device auto
```

### 完整论文规模尝试

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist.yaml --device auto
```

完整配置非常耗时，建议先跑中等规模配置再跑完整 600/100 epoch。
