# NGSG SpykeTorch 项目手册

最后更新：2026-07-01

这个仓库只保留两个主要 Markdown 入口：

- `README.md`：当前项目状态、配置选择、运行命令、复现计划和 NGSG 实现路线。
- `CATASTROPHIC_FORGETTING_REPRODUCTION.md`：灾难性遗忘 baseline 的历史实验记录和结果日志。

其他旧的计划文档、配置 README 和模块 README 已合并到本文件，避免之后不知道该看哪一个。

## 1. 当前目标

当前目标不是立刻写完整 NGSG，而是按顺序推进：

1. 先复现 paper-source catastrophic forgetting baseline。
2. 确认 MNIST -> EMNIST 连续学习流程、缓存、checkpoint 和评估都稳定。
3. 再加入 winner-frequency logging。
4. 最后在 S3 输出层实现 NGSG 的 novelty gate、reserve neuron 调用和 synapse growth 相关机制。

当前不复现 joint training。旧的 frozen/Langevin 配置也已从 active YAML 中删除，只有论文对比确实需要时再重新建立。

## 2. 分支和代码状态

- `dev`：当前本地和服务器共同使用的主集成分支。
- `baseline/continuous-learning`：baseline 复现分支。
- `ngsg/novelty-gated-growth`：NGSG 创新实现分支。
- 当前 baseline 使用 `src/trainers/baseline_trainer.py` 和 `src/utils/data.py` 中的 paper-source SpykeTorch/Mozafari 路线。
- EMNIST raw idx fallback、paper-source 预处理缓存、S1/S2 feature checkpoint 复用和 C2 feature cache 已在 `dev` 中。

## 3. 当前只保留的 baseline YAML

`configs/baseline/` 现在只保留 3 个 active YAML：

| 配置 | 什么时候跑 | 说明 |
| --- | --- | --- |
| `configs/baseline/catastrophic_mnist_emnist.yaml` | 服务器正式完整 baseline | 完整 paper-source MNIST -> EMNIST catastrophic forgetting 流程。默认加载 `checkpoints/features/` 中已跟踪的小 checkpoint，并在本地生成/复用 C2 cache。 |
| `configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml` | 只有 checkpoint 缺失或要重建时才跑 | feature-only 模式，只训练 S1/S2 并生成 checkpoint/C2 cache，跳过 S3 R-STDP 和评估。 |
| `configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml` | 本地中等规模诊断 | 每类 100 个样本，训练更短，用来检查代码路径和学习曲线，不作为最终论文数字。 |

已删除的旧 YAML：`catastrophic.yaml`、`joint_training.yaml`、`frozen_large_weights.yaml`、`langevin.yaml`、`catastrophic_mnist_emnist_probe.yaml`、`catastrophic_mnist_emnist_medium.yaml`、`catastrophic_mnist_emnist_medium_stabilizer_off.yaml`。

## 4. 推荐命令

服务器正式完整 baseline：

```bash
git fetch origin
git checkout dev
git pull origin dev
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist.yaml --device cuda --run-name paper_ch4_catastrophic_source_seed0
```

只做配置检查，不训练：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist.yaml --device auto --dry-run --run-name paper_source_strict_dryrun
```

本地中等规模诊断：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml --device auto --run-name paper_medium_source_port_seed0
```

重建 S1/S2 feature checkpoint 和 C2 cache：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml --device cuda --run-name paper_feature_checkpoint_full
```

## 5. 数据、缓存和 checkpoint

当前 paper-source 路线有三层复用：

1. 输入预处理缓存：`data/preprocessed/paper_source/<hash>/`。
2. S1/S2 feature checkpoint：`checkpoints/features/`。
3. C2 pooled feature cache：`data/features/c2/<hash>/`。

已进入 git 的小型 S1/S2 checkpoint：

```text
checkpoints/features/paper_task1_s1e2_s2e4_f26edcfb75b5d681.pt
checkpoints/features/paper_task2_s1e2_s2e4_60c0a06b55746fb6.pt
```

这些 checkpoint 使用完整 paper-aligned feature schedule 生成：每个任务 24,000 个训练样本，S1 STDP 2 epoch，S2 STDP 4 epoch。

不进入 git 的内容：

- `data/` 下的数据集。
- `data/preprocessed/` 下的预处理 `.pt` 缓存。
- `data/features/c2/` 下的大型 C2 feature cache。
- `experiments/`、`logs/`、`results/` 下的运行产物。
- `SERVER_LATEST_STATUS.md` 这类服务器临时状态快照。

C2 cache 体积很大，服务器首次运行时本地重建即可。如果要同步，应通过单独文件传输处理，不要提交到 git。

## 6. 当前 baseline 实现要点

Task 1 先训练 MNIST digits，Task 2 再在同一网络上训练 EMNIST letters，不额外加保护、不冻结参数、不引入额外容量。

paper-source 路线关键点：

- 输入预处理：DoG filter、local normalization、Intensity2Latency。
- S1/S2：SpykeTorch convolution + STDP。
- S3：SpykeTorch convolution + reward / anti-reward STDP。
- 输出映射：200 个 S3 feature map，每类 20 个。
- EMNIST：优先 torchvision，必要时从 raw idx / idx.gz 文件直接读取。
- 训练入口：`scripts/run_baseline.py`。
- 主要 trainer：`src/trainers/baseline_trainer.py`。
- 主要模型：`src/models/paper_mozafari.py`。
- 数据入口：`src/utils/data.py`。

## 7. 复现阶段计划

阶段 A：paper-source baseline 复现。

- 确认数据集划分和 label mapping。
- 确认 S1/S2 checkpoint 复用逻辑。
- 确认 C2 cache 能在服务器本地生成并复用。
- 跑完整 `catastrophic_mnist_emnist.yaml`。
- 记录 Task1 after Task1、Task1 after Task2、Task2 after Task2、forgetting 和 avg acc。

阶段 B：可解释统计。

- 在 S3 训练中记录 winner id、winner frequency 和 winner label count。
- 输出 Task 1 后的 `f_i`、`q_i`、`I_i` 分布。
- 生成 stable/shared/reserve neuron partition。
- 确认统计模块不改变 baseline 学习行为。

阶段 C：NGSG。

- 加入 SDPM soft protection。
- 校准 novelty score。
- 实现 class-local reserve activation。
- 对比 baseline、SDPM only、NGSG only、random reserve 和完整 NGSG。

## 8. NGSG 当前设计共识

NGSG 的创新点不在于重写整个 SpykeTorch 网络，而是在已复现的 Antonov/Mozafari 三层 SNN 基础上，主要在 S3 输出层加入持续学习机制。

核心模块：

- winner-frequency tracker：统计 Task 1 中每个 S3 neuron 的获胜频率。
- neuron partition：根据获胜频率和类别选择性划分 stable/shared/reserve neurons。
- synaptic importance：估计旧任务关键连接的重要性。
- SDPM plasticity gate：对重要旧连接缩放 R-STDP 更新幅度。
- novelty detector：判断当前输入是否对旧网络足够新。
- reserve activation：对高 novelty 样本调用低使用率、低旧任务重要性的 reserve neurons。

推荐第一版采用保守实现：先做 novelty-guided activation 和 plasticity gating，不急着声明真实动态新增结构。等 mask 或低权重 silent synapse 版本稳定后，再决定论文中是否使用 structural growth 的强表述。

## 9. S1/S2 是否冻结

这里要区分 baseline 复现和 NGSG 主实验：

- baseline 复现阶段：按论文/作者源码协议执行，Task 2 仍可训练 S1/S2，用于对齐 catastrophic forgetting baseline。
- NGSG 主方法阶段：优先采用 Task 1 后冻结或复用 S1/S2 特征的设置，让创新点集中在 S3。
- 必要消融：保留 `NGSG with S1/S2 retraining`，用于说明性能变化不是单纯来自低层特征重学习。

因此，S1/S2 是否冻结是实验协议变量，不应和 NGSG 核心创新混为一谈。

## 10. 需要记录的实验结果

每次正式 baseline 或 NGSG 实验至少记录：

- 配置文件和 run name。
- 当前 git commit。
- 数据集、任务顺序和样本规模。
- 是否加载 S1/S2 checkpoint。
- 是否使用 C2 cache。
- Task1 after Task1。
- Task1 after Task2。
- Task2 after Task2。
- Forgetting。
- Avg Acc。
- 与论文或上一次结果的差异。
- 下一步判断。

长期实验记录写入 `CATASTROPHIC_FORGETTING_REPRODUCTION.md`，不要散落在新的临时 Markdown 文件里。

## 11. 代码目录说明

- `src/models/`：SNN 网络定义和可复用层模块。
- `src/trainers/`：训练和评估流程。
- `src/utils/`：配置、文件 IO、随机种子和数据处理工具。
- `src/analysis/`：遗忘指标、结果汇总脚本和可视化辅助工具。
- `src/continual/`：winner-frequency、novelty score、neuron partition、reserve branch 等持续学习逻辑。
- `src/plasticity/`：R-STDP、mask、silent-synapse update，以及后续可能重新引入的 Langevin 相关逻辑。

## 12. 下一步优先级

1. 服务器拉取 `dev`，确认已经有两份 `paper_task*_s1e2_s2e4_*.pt` checkpoint。
2. 直接运行 `configs/baseline/catastrophic_mnist_emnist.yaml`。
3. 跑完后把关键结果整理到 `CATASTROPHIC_FORGETTING_REPRODUCTION.md`。
4. baseline 趋势可信后，开始实现 winner-frequency logging。
5. 再进入 NGSG 的 SDPM、novelty gate 和 reserve activation。