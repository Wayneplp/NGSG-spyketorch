# Reproduction Plan for "Continuous Learning of Spiking Networks Trained with Local Rules"

## 0. 这份文档是干什么的

这份文档用于指导我们先复现原论文，再在复现成功的基础上加入 NGSG。

核心原则只有一句话：

> 先严格复现原论文 baseline，不要一开始就改模型；先证明原方法能跑通，再开始加自己的东西。

## 1. 当前代码状态（2026-06-30）

当前主线已经从早期近似实现切到论文作者源码风格的 SpykeTorch 移植路线：

- 主集成分支是 `dev`，服务器也应跟 `dev` 或 `dev` 上的 release tag。
- `baseline/continuous-learning` 用于 baseline 复现，`ngsg/novelty-gated-growth` 用于 NGSG 创新实现。
- `configs/baseline/catastrophic_mnist_emnist.yaml` 是完整规模 paper-source catastrophic baseline 配置。
- `configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml` 是中等规模诊断配置。
- `src/trainers/baseline_trainer.py` 中的 `paper_spyketorch` / `paper_source_rstdp` 路线是当前主复现路径。
- `src/utils/data.py` 已包含 paper-source 预处理缓存和 EMNIST raw idx fallback。
- `SERVER_LATEST_STATUS.md`、`data/`、`logs/`、`checkpoints/`、`experiments/`、预处理 `.pt` 缓存都不进 git。

当前可以直接使用的命令：

```powershell
# 只检查配置和任务规模，不训练
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist.yaml --device auto --dry-run --run-name paper_source_strict_dryrun

# 中等规模源码移植版，用来先看学习曲线
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist_paper_medium.yaml --device auto --run-name paper_medium_source_port_seed0

# 完整论文规模，极慢，确认中等规模趋势正常后再跑
C:\Users\pw\.conda\envs\Spyketorch\python.exe scripts\run_baseline.py --config configs\baseline\catastrophic_mnist_emnist.yaml --device auto --run-name paper_ch4_catastrophic_source_seed0
```

## 2. 复现目标

当前复现目标不是“立刻做出 NGSG”，而是分成两个阶段。

### 阶段 A：原论文复现

先尽可能还原原论文中的持续学习实验流程，包括：

- 网络结构
- 数据集划分
- 训练顺序
- 学习规则
- baseline 设置
- 评价指标

### 阶段 B：在复现结果上扩展 NGSG

等阶段 A 跑通以后，再逐步加入：

- winner-frequency 统计
- reserve neuron 发现
- novelty score
- neuron partition / reserve growth branch
- silent synapse growth

## 3. 复现记录必须写清楚什么

每个 baseline 或诊断实验至少记录：

- 原论文设定
- 当前配置文件
- 当前实现路径
- 和原论文仍不一致的地方
- 训练命令
- 结果文件位置
- Task1 after Task1
- Task1 after Task2
- Task2 after Task2
- Forgetting
- Avg Acc
- 当前判断和下一步

只要有任意一个地方和原论文不完全一致，就要登记偏差，因为后面写论文 related work、experiment 和 limitation 时会反复用到。

## 4. 推荐复现顺序

1. catastrophic forgetting baseline
2. joint training
3. frozen large weights
4. Langevin dynamics
5. winner-frequency logging, only after the paper-source baseline is trusted
6. NGSG growth logic, only after baseline behavior is stable

catastrophic forgetting baseline 仍然是第一优先级，因为它最容易验证数据流、训练流和测试流是否正确。如果它都不稳定，后面的保护机制和 NGSG 对比没有意义。

## 5. 当前 catastrophic baseline 实现方式

Task 1 先训练 MNIST，Task 2 再继续在同一网络上训练 EMNIST letters，不加入额外保护项，不冻结参数，不引入额外容量。

当前 paper-source 路线的关键点：

- 输入预处理：DoG filter、local normalization、Intensity2Latency。
- S1/S2：SpykeTorch convolution + STDP。
- S3：SpykeTorch convolution + reward / anti-reward STDP。
- 输出映射：200 个 S3 feature map，每类 20 个。
- EMNIST：优先 torchvision，必要时从 raw idx / idx.gz 文件直接读取。
- 预处理缓存：写入 `data/preprocessed/paper_source/<hash>/`，避免服务器长跑时重复做昂贵预处理。

## 6. 代码仓库落地方式

### 配置文件

当前优先维护：

- `configs/baseline/catastrophic_mnist_emnist.yaml`
- `configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml`

后续再补齐：

- `configs/baseline/joint_training.yaml`
- `configs/baseline/frozen_large_weights.yaml`
- `configs/baseline/langevin.yaml`

### 训练脚本

统一入口：

- `scripts/run_baseline.py`

这个脚本负责读取配置、运行 baseline、保存日志和结果。

### 结果汇总

生成结果可以放在本地：

- `results/baseline_summary.csv`
- `experiments/<run_name>/result.json`
- `logs/<run_name>.log`

这些是运行产物，默认不进 git。需要写入论文或长期保留的信息，应转写到 Markdown 文档或最终表格中。

## 7. 实验记录模板

### 实验名称

`paper_medium_source_port_seed0`

### 目标

复现原论文中不加保护时的持续学习结果，验证训练与评估流程正确。

### 对应原论文方法

Catastrophic forgetting baseline

### 当前实现说明

- Task 1 先训练 MNIST。
- Task 2 再继续训练 EMNIST letters。
- 不加参数保护。
- 不加额外容量。
- 使用 paper-source SpykeTorch 移植路线。

### 配置

- 配置文件：待填
- 数据集：待填
- 任务划分：待填
- 网络结构：待填
- epoch：待填
- seed：待填

### 结果

- Task1 after Task1：待填
- Task1 after Task2：待填
- Task2 after Task2：待填
- Forgetting：待填
- Avg Acc：待填

### 和原论文对比

- 原论文：待填
- 当前结果：待填
- 差异：待填
- 可能原因：待填

### 当前判断

- 是否算复现成功：待填
- 是否需要继续排查：待填

### 下一步

- 待填

## 8. 如何判断“复现成功”

不要要求第一版完全一模一样，但至少满足下面三点：

1. 训练和测试流程完整跑通。
2. 结果趋势与原论文一致。
3. 数值误差在可解释范围内。

这里最重要的是趋势一致，比如：

- catastrophic forgetting 明显差。
- joint training 最好或接近最好。
- Langevin 比无保护 baseline 更稳。
- frozen large weights 有一定缓解但不一定最优。

如果趋势都不对，就不要急着做 NGSG。

## 9. 复现阶段最容易犯的错

### 错误 1：一边复现一边改方法

这会导致你最后根本不知道结果来自原方法还是自己的改动。

### 错误 2：没有记录配置

今天能跑，明天不能复现，是最常见的问题。

### 错误 3：只看最终准确率，不看趋势

持续学习最重要的是：

- Task 1 学完时怎样。
- Task 2 学完后旧任务掉了多少。
- 新任务是否学会了。

### 错误 4：原论文和当前实现不一致但没有登记

后面会非常痛苦，因为你会忘记到底哪里改过。

## 10. 下一步优先级

1. 用中等规模配置确认 paper-source 路线稳定学习并产生遗忘趋势。
2. 在服务器上用 `dev` 跑完整规模或更接近论文规模的实验。
3. 把结果从 `experiments/` 和 `logs/` 中提炼到 `CATASTROPHIC_FORGETTING_REPRODUCTION.md`。
4. 再补 joint training / frozen large weights / Langevin 配置。
5. baseline 趋势可信后，再进入 winner-frequency logging 和 NGSG 实现。