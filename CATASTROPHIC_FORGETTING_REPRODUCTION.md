# 灾难性遗忘复现状态记录

最后更新：2026-06-28

## 目标

当前目标是先复现论文 **Continuous Learning of Spiking Networks Trained with Local Rules** 里的 `Catastrophic forgetting` baseline，再在这个基础上继续做 NGSG。

论文 Table 1 中 `Catastrophic forgetting` 这一行的目标结果是：

| 阶段 | 数据集 | 论文结果 |
| --- | --- | --- |
| 初始训练 | MNIST | 90.8% +/- 0.9 |
| 后续训练后的旧任务保持 | MNIST | 48.1% +/- 4.8 |
| 后续训练的新任务 | EMNIST | 78.4% +/- 1.2 |

论文协议是：先在 MNIST 数字上训练网络；之后假设 MNIST 旧数据不可再用，把同一个网络继续训练到 EMNIST 字母任务上。论文描述中，后续 EMNIST 训练时恢复原始学习率，S1 和 S2 层分别用 STDP 再训练 2 和 4 个 epoch，最后 S3 层用 R-STDP 训练 100 个 epoch。表格脚注说明初始 MNIST 训练使用了 600 个 epoch。

协议核对来源：论文 arXiv 页面/PDF：`https://arxiv.org/abs/2111.09553`。

## 重要修正

最开始的本地草稿曾经把 EMNIST 的 26 个字母全部纳入，并使用 36 类输出空间。这不符合论文 Table 1 的设置。

论文使用的是 10 个 EMNIST 大写字母：

`A, B, D, E, G, H, N, Q, R, S`

在 torchvision 的 `EMNIST(split="letters")` 中，letters 标签是从 1 开始的，因此对应标签为：

`[1, 2, 4, 5, 7, 8, 14, 17, 18, 19]`

当前实现把这些字母映射到和 MNIST 数字相同的 10 个输出组：

`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

也就是说，模型仍然是 10 类输出，每一类有 20 个 S3 神经元，总共 200 个 S3 神经元。

## 当前项目状态

当前仓库里有三层 baseline：

1. 旧版 MNIST split 近似 baseline：`configs/baseline/*.yaml` 中原有配置。
2. 监督学习 sanity baseline：`configs/baseline/catastrophic_supervised.yaml`，只用于验证持续学习评估流程和结果记录是否正确。
3. 论文协议 catastrophic forgetting baseline：`configs/baseline/catastrophic_mnist_emnist.yaml`。

论文协议 baseline 当前使用：

- Task 1：MNIST 数字。
- Task 2：EMNIST 中的 `A/B/D/E/G/H/N/Q/R/S` 十个大写字母。
- 输出类别数：10。
- 每类 S3 神经元：20。
- S3 总神经元数：200。
- MNIST 训练集：每类 2400 张，共 24000 张。
- EMNIST 训练集：每类 2400 张，共 24000 张。
- MNIST 测试集：完整 10000 张。
- EMNIST 测试集：所选 10 个字母对应测试样本，共 8000 张。
- S1/S2：仓库内实现的局部 STDP 近似。
- S3：仓库内实现的 R-STDP 近似。

## 新增和修改的文件

### 新增文件

- `configs/baseline/catastrophic_mnist_emnist.yaml`
  - 论文协议完整配置，包含 600/100 epoch 的长训练设定。
- `configs/baseline/catastrophic_mnist_emnist_probe.yaml`
  - 快速冒烟测试配置，用很小的数据量和 epoch 验证流程能跑通。
- `configs/baseline/catastrophic_supervised.yaml`
  - 监督学习 sanity baseline，用于验证 catastrophic forgetting 指标和日志流程。
- `src/models/supervised.py`
  - 给 sanity baseline 使用的小 MLP。
- `src/plasticity/stdp.py`
  - S1/S2 卷积层局部 STDP 更新器。
- `CATASTROPHIC_FORGETTING_REPRODUCTION.md`
  - 当前这份中文复现状态和操作记录。

### 修改文件

- `src/utils/data.py`
  - 增加 `task_specs` 配置方式。
  - 支持 MNIST/EMNIST 混合任务。
  - 支持 EMNIST 图像方向修正。
  - 支持标签重映射。
  - 支持每类固定数量的平衡采样。
- `src/plasticity/rstdp.py`
  - 将原来的简单输出层加减更新，改为带 active/inactive 分支的奖惩更新。
  - 加入权重边界和稳定项。
- `src/trainers/baseline_trainer.py`
  - 增加 `learning_rule: paper_stdp_rstdp`。
  - 加入阶段式训练流程：S1 STDP -> S2 STDP -> S3 R-STDP。
  - 保留旧的 `rstdp` 和 `backprop` 路径，方便对照。
- `src/models/__init__.py`
  - 导出 supervised sanity model。

## 操作记录

### 2026-06-28：检查仓库和运行环境

- 确认用户指定的运行环境是：`C:\Users\pw\.conda\envs\Spyketorch\python.exe`。
- 确认该环境中 PyTorch 和 CUDA 可用。
- 确认该环境没有安装原始 `SpykeTorch` 包。
- 因此当前路线是先在本仓库内实现近似版 STDP/R-STDP，而不是直接调用原始 SpykeTorch。

### 2026-06-28：监督学习 sanity baseline

- 增加了一个监督学习 MLP baseline，用来确认持续学习指标、任务顺序、输出记录是否正常。
- 跑通了 `catastrophic_supervised.yaml`。
- 该 sanity baseline 的结果是：
  - Task1 after Task1：99.32%
  - Task1 after Task2：0.00%
  - Task2 after Task2：97.92%
- 这个结果只能证明评估流程能体现 catastrophic forgetting，不能算论文复现。

### 2026-06-28：论文协议对齐

- 增加了 MNIST -> EMNIST 的任务协议。
- 一开始误用了全部 26 个 EMNIST 字母和 36 类输出。
- 对照论文后修正为 10 个 EMNIST 大写字母：`A, B, D, E, G, H, N, Q, R, S`。
- 增加每类平衡采样。
- 完整配置 dry-run 后确认：
  - MNIST train size：24000
  - MNIST test size：10000
  - EMNIST train size：24000
  - EMNIST selected-letter test size：8000

### 2026-06-28：实现本仓库内 STDP/R-STDP

- 新增 `src/plasticity/stdp.py`，用于 S1/S2 层局部 STDP。
- 在 `src/trainers/baseline_trainer.py` 中新增阶段式训练：
  - S1 STDP
  - S2 STDP
  - S3 R-STDP
- 重写增强 `src/plasticity/rstdp.py`：
  - active/inactive 前突触分支
  - reward/punishment 更新
  - 权重边界裁剪
  - 权重稳定项
- 新增 `paper_stdp_rstdp` 学习规则。

### 2026-06-28：probe 冒烟测试

运行命令：

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist_probe.yaml --device auto --run-name paper_catastrophic_probe_seed0
```

probe 设置：

- 每类训练样本：200。
- 每类测试样本：100。
- 每个任务 S1 STDP：1 epoch。
- 每个任务 S2 STDP：1 epoch。
- 每个任务 S3 R-STDP：1 epoch。

probe 结果：

| 指标 | 数值 |
| --- | --- |
| Task1 after Task1 | 10.0% |
| Task1 after Task2 | 10.0% |
| Task2 after Task2 | 10.0% |
| Forgetting | 0.0% |
| Avg Acc | 10.0% |

解释：这个 probe 只是冒烟测试。它证明新的论文协议训练流程可以端到端执行，可以记录 S1/S2 特征层更新，也可以记录 S3 输出层更新，并能写出结果。它不应该被当作论文数值复现结果。

## 如何运行

### 快速冒烟测试

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist_probe.yaml --device auto
```

### 完整论文协议尝试

```powershell
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist.yaml --device auto
```

完整配置很长：MNIST S3 是 600 epoch，后续 EMNIST S3 是 100 epoch，运行时间会比较久。

## 当前与论文仍然不一致的地方

当前版本比之前更接近论文，但仍不能保证数值能直接对齐 Table 1。

已知差距：

- 没有安装和调用原始 SpykeTorch 包。
- 当前 spike timing 仍然是 dense tensor 代理，不是原始事件/排序脉冲实现。
- S1/S2 STDP 是仓库内基于 winner patch 的局部近似。
- S3 R-STDP 是仓库内基于 active/inactive 特征掩码和 class-group winner 的近似。
- Lateral inhibition 仍然是轻量代理实现。
- 论文中的学习率恢复/递增细节目前通过配置近似表达。
- 论文报告的是多次运行的 mean +/- std；当前只会按单个 seed 记录结果，除非手动多 seed 重复运行。

## 下一步工程建议

1. 不要直接先跑 600/100 的完整长实验；先跑中等 schedule：
   - 每类 2400 样本。
   - S1/S2 按论文 epoch。
   - MNIST S3 先试 10-50 epoch。
   - EMNIST S3 先试 10-20 epoch。
2. 先确认 MNIST 在后续训练前能否明显高于随机水平。
3. 如果 MNIST 一直停在 10%，优先调 S3 R-STDP，而不是盲目加 epoch。
4. 增加每个 epoch 的 MNIST/EMNIST eval snapshot，观察学习曲线。
5. 如果本仓库近似实现仍然难以学习，需要安装或移植原始 SpykeTorch，并对比每层输出和更新幅度。

## 当前判断

当前项目已经具备论文 catastrophic forgetting baseline 的正确实验协议，并且有一套可运行的本仓库内 STDP/R-STDP 近似实现。现在还不能称为成功复现 Table 1 的数值结果。下一步的核心问题已经不是数据和流程，而是当前 STDP/R-STDP 近似规则能不能先把 MNIST 学到高于随机水平。