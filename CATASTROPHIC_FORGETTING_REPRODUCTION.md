# 灾难性遗忘复现状态记录

最后更新：2026-06-29

## 当前结论

当前仓库的主实现已经从“仓库内近似版 STDP/R-STDP”切换为**调用官方 SpykeTorch 包**的实现。

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

官方 SpykeTorch 包中没有现成的 reward-modulated STDP 类，因此当前 S3 使用的是：

- S3 层本身：`SpykeTorch.snn.Convolution`
- spike/potential 表示：SpykeTorch 网络产生的 spike-wave 和 potentials
- 更新器：仓库内 `SpykeTorchRewardSTDP`

这不是旧的 dense tensor 近似模型；它是在官方 SpykeTorch S3 卷积权重上做 reward/punishment 更新。

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

## 当前仍需注意的问题

虽然主路径已经切换到官方 SpykeTorch，但仍有一个关键差距：官方包本身没有现成 R-STDP 类。因此 S3 的 reward-modulated 更新仍然是本仓库实现，但它操作的是官方 SpykeTorch 的 `snn.Convolution` 权重和 spike-wave，不再是之前的 dense-tensor 近似网络。

下一步要想更贴近论文，需要继续做：

1. 对照论文或原作者代码确认 S3 R-STDP 的精确公式和更新时机。
2. 增加中等规模实验，先确认 MNIST 初训能高于随机水平。
3. 调整 firing threshold、STDP 学习率和 R-STDP 奖惩参数。
4. 加每个 epoch 的 eval snapshot，观察学习曲线。
5. 如果需要更严格复现，继续寻找论文原始实验脚本，而不仅仅使用 SpykeTorch 基础包。

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