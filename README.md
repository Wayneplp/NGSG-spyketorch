# NGSG SpykeTorch 项目手册

最后更新：2026-07-03（同步 no-op 对照与下一步）

这个仓库只保留两个主要 Markdown 入口：

- `README.md`：当前项目状态、配置选择、运行命令、复现计划和 NGSG 实现路线。
- `CATASTROPHIC_FORGETTING_REPRODUCTION.md`：灾难性遗忘 baseline 的历史实验记录和结果日志。

其他旧的计划文档、配置 README 和模块 README 已合并到本文件，避免之后不知道该看哪一个。

## 0. 最新正式 baseline 结果（2026-07-02）

服务器完整 `catastrophic_mnist_emnist.yaml` 已在开启 winner-frequency logging 后跑完，运行版本为 `e86a263`，run name 为 `paper_ch4_catastrophic_optimized_winnerlog_seed0`。

| 指标 | 本次结果 | 论文 catastrophic forgetting 参考 |
| --- | ---: | ---: |
| Initial MNIST / Task1 after Task1 | 92.84% | 90.8 ± 0.9% |
| Subsequent MNIST / Task1 after Task2 | 48.42% | 48.1 ± 4.8% |
| Subsequent EMNIST / Task2 after Task2 | 74.91% | 78.4 ± 1.2% |
| Forgetting | 44.42% | 约 42.7% |
| Avg Acc | 61.67% | - |

结论：catastrophic forgetting 趋势已经稳定复现。Task2 之后 MNIST 保留率 48.42%，和论文 48.1% 基本对齐；EMNIST 仍低约 3.5 个点，后续如果要追表格数字，需要继续核对 EMNIST 数据处理、作者 notebook 中的保存张量格式和随机种子细节。winner-frequency logging 已开启，可作为后续 NGSG 的统计基线。

可提交的精简结果文件：`published_results/baseline/paper_ch4_catastrophic_optimized_winnerlog_seed0.json`。

本机原始服务器产物副本：`experiments/server_paper_ch4_catastrophic_optimized_winnerlog_seed0/`，其中包含完整 `result.json`、`resolved_config.json`、`baseline_summary.csv` 和运行日志。该目录属于运行产物，不进入 git。

## 0.1 当前推进状态（2026-07-03）

当前项目已经不再停留在“只复现 baseline”的阶段，但也还没有进入完整 NGSG full 实验。最新状态如下：

| 模块 | 当前状态 | 判断 |
| --- | --- | --- |
| paper-source catastrophic baseline | 已完成完整服务器复现 | 遗忘趋势和论文基本对齐，可作为主 baseline。 |
| winner-frequency logging | 已接入，并完成 medium no-op 对照 | seed 0 medium 下不扰动 baseline 学习行为，可作为后续统计基线。 |
| winner label count | medium 已验证 | `paper_medium_partition_seed0` 中已导出 `winner_label_counts`，shape 为 200 x 10，总计 50,000 次 Task1 winner 记录。 |
| `neuron_partition.py` | medium 已验证，并完成 no-op 对照 | 已从 Task1 winner counts 和 per-neuron label counts 计算 `f_i/q_i/I_i`，并划分 stable/shared/reserve/dead；seed 0 medium 下拟合 partition 不改变指标，full 结论仍需后续 600 epoch 实验确认。 |
| SDPM gate | 已完成 medium 机制验证 | 确认能从 Task1 统计拟合，并在 Task2 R-STDP 更新中生效。 |
| novelty gate / reserve activation | 尚未实现 | 这是后续补足新任务学习能力的关键。 |
| full NGSG | 尚未开始正式 full 运行 | no-op 对照已完成；下一步是统一 SDPM 统计基础并实现 novelty/reserve。 |

服务器 2 号 medium 对照结果：

| 配置 | Task1 after Task1 | Task1 after Task2 | Task2 after Task2 | Forgetting | Avg Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-SDPM same-code | 79.6% | 65.9% | 60.3% | 13.7 pp | 63.10% |
| SDPM-only | 79.6% | 68.1% | 54.8% | 11.5 pp | 61.45% |
| SDPM - no-SDPM | 0.0 pp | +2.2 pp | -5.5 pp | -2.2 pp | -1.65 pp |

结论：SDPM-only 已证明“旧任务保护”方向有效，但它会牺牲 Task2 学习；这符合模块定位。完整 NGSG 还需要 novelty gate 和 reserve activation 给新任务分配容量，不能只凭 SDPM-only 结果声称整体方法有效。


服务器 medium partition 验证结果（`paper_medium_partition_seed0`，`dev@9d9bded`）：

| 验证项 | 结果 | 判断 |
| --- | --- | --- |
| Metrics | Task1 after Task1 77.2%，Task1 after Task2 69.8%，Task2 after Task2 58.7%，Forgetting 7.4 pp，Avg Acc 64.25% | medium 诊断结果，不作为最终论文数字。 |
| `winner_label_counts` | shape 200 x 10，总数 50,000 | 已确认真实 per-neuron label counts 写入 `result.json`。 |
| partition | stable 61，shared 59，reserve 80，dead 0 | 基本合理；stable 只比建议上界 60 多 1 个，先不急调阈值。 |
| q_i vs decision_map fallback | 真实 q mean 0.504，fallback active q=1.0；86/200 active neurons 的 dominant label 与 decision_map 不一致 | fallback 明显高估选择性，后续 SDPM/reserve 应以真实 `winner_label_counts` 计算的 q_i 为准。 |

本机服务器产物副本：`experiments/server_paper_medium_partition_seed0/`。本机诊断产物：`experiments/diagnostics/paper_medium_partition_seed0/`，包含 `partition_validation.json` 和三张图：`f_i` 直方图、`q_i` vs `f_i` 散点图、dominant neuron per class 分布图。以上目录属于运行产物，不进入 git。

## 0.2 当前实验结论和后续决策（2026-07-02）

当前已经可以比较明确地得到三个实验结论：

1. **baseline 复现已经成立。** 完整 paper-source baseline 在服务器上跑完后，Task1 after Task2 为 48.42%，与论文参考 48.1% 基本对齐；forgetting 为 44.42 pp，也落在预期灾难性遗忘区间。因此后续论文叙事可以把这个结果作为主 baseline，而不是继续把主要精力花在“是否复现出遗忘”上。
2. **统计链路已经有 medium 级证据。** `winner_label_counts`、`f_i/q_i/I_i` 和 stable/shared/reserve/dead partition 已能从真实 Task1 winner 统计中产生。尤其是 q_i 不能再用 decision map fallback 替代：fallback 会把 active neuron 的 q_i 近似推到 1.0，而真实 q_i 均值只有约 0.504，会明显高估类别选择性。
3. **SDPM-only 的定位已经清楚。** medium 对照中，SDPM-only 把 Task1 after Task2 从 65.9% 提到 68.1%，forgetting 从 13.7 pp 降到 11.5 pp；但 Task2 after Task2 从 60.3% 降到 54.8%，Avg Acc 也下降。因此 SDPM 是“保护旧任务”的有效子模块，但不是完整 NGSG；只靠 SDPM 不能声称整体方法已经解决 continual learning。

当前还不能写成论文结论的内容：

- 不能声称完整 NGSG 已经有效，因为 novelty gate 和 reserve activation 还没有实现。
- 不能把 medium 数字当作最终论文表格数字；medium 只用于机制诊断和消融预筛。
- 不能把 SDPM-only 结果解释为整体性能提升；它目前体现的是稳定性-可塑性 trade-off。

下一步优先级应该是：

1. **固定当前 no-op 对照结论。** 当前 seed 0 medium 的 pure baseline、logging-only、logging+partition 三组结果完全一致，可作为统计模块 no-op 证据写入实验记录。
2. **统一 SDPM 与 partition 的统计基础。** SDPM 后续应使用真实 `winner_label_counts` 计算出的 q_i、f_i 和 I_i，而不是 decision map fallback 或另一套临时 importance。
3. **实现 novelty gate 和 reserve activation。** 先在 medium 上完成机制闭环，再做消融：baseline、SDPM-only、reserve/novelty-only、random reserve、full NGSG。
4. **medium 消融稳定后再跑 full。** full 规模优先顺序建议为 baseline 已完成 -> full SDPM-only -> full NGSG；每次都同步 result、resolved_config、summary 和关键日志到本地记录。
5. **维护服务器空间策略。** `/root/autodl-tmp` 已扩到 100G，当前足够继续 medium；full 实验前仍建议保留 `data/features/c2/`、`data/preprocessed/`，必要时迁移到 `/autodl-pub/data` 并软链接。

## 0.3 Logging / partition no-op 对照（2026-07-03）

为了确认统计模块本身不改变 baseline 学习行为，服务器上补跑了三组 seed 0 medium 对照。三组除统计开关外保持相同数据、epoch、checkpoint、C2 cache 和随机种子；SDPM 均未启用。

| 组别 | run name | winner logging | partition | Task1 after Task1 | Task1 after Task2 | Task2 after Task2 | Forgetting | Avg Acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| pure baseline | `noop_medium_baseline_seed0` | off | off | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |
| logging-only | `noop_medium_logging_seed0` | on | off | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |
| logging + partition | `paper_medium_partition_seed0` | on | on | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |

结论：在 seed 0 medium 配置下，winner logging 和 neuron partition 拟合均没有可观察到的学习行为扰动。这个结论只支持“当前统计路径是 no-op 诊断模块”，不等价于 full 规模或多 seed 的最终证明。

本机服务器产物副本：`experiments/server_noop_medium_baseline_seed0/`、`experiments/server_noop_medium_logging_seed0/`、`experiments/server_paper_medium_partition_seed0/`。本机汇总产物：`experiments/diagnostics/noop_medium_controls/noop_medium_controls_summary.json` 和 `experiments/diagnostics/noop_medium_controls/noop_medium_controls_summary.csv`。

## 1. 当前目标

当前目标已经从“确认统计模块是否 no-op”推进到“把统计基础接入真正的 NGSG 机制”：

1. 以已完成的 paper-source catastrophic baseline 作为主对照。
2. 使用已验证的 Task1 S3 统计基础：winner counts、winner label counts、`f_i/q_i/I_i` 和 neuron partition。
3. 保留 no-op 对照作为诊断证据：seed 0 medium 下 logging 和 partition 不改变 baseline 指标。
4. 将 SDPM 接到统一的 neuron partition / importance 统计基础上。
5. 实现 novelty gate、reserve activation 和完整 NGSG medium 消融。
6. medium 结果稳定后，再启动 full 规模 SDPM-only 和 full NGSG。

当前不复现 joint training。旧的 frozen/Langevin 配置也已从 active YAML 中删除，只有论文对比确实需要时再重新建立。

## 2. 分支和代码状态

- `dev`：当前本地和服务器共同使用的主集成分支。
- `baseline/continuous-learning`：baseline 复现分支。
- `ngsg/novelty-gated-growth`：NGSG 创新实现分支。
- 当前 baseline 使用 `src/trainers/baseline_trainer.py` 和 `src/utils/data.py` 中的 paper-source SpykeTorch/Mozafari 路线。
- EMNIST raw idx fallback、paper-source 预处理缓存、S1/S2 feature checkpoint 复用和 C2 feature cache 已在 `dev` 中。

## 3. 当前常用 YAML

`configs/baseline/` 现在保留 5 个常用 YAML：

| 配置 | 什么时候跑 | 说明 |
| --- | --- | --- |
| `configs/baseline/catastrophic_mnist_emnist.yaml` | 服务器正式完整 baseline | 完整 paper-source MNIST -> EMNIST catastrophic forgetting 流程。默认加载 `checkpoints/features/` 中已跟踪的小 checkpoint，并在本地生成/复用 C2 cache。 |
| `configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml` | 只有 checkpoint 缺失或要重建时才跑 | feature-only 模式，只训练 S1/S2 并生成 checkpoint/C2 cache，跳过 S3 R-STDP 和评估。 |
| `configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml` | 本地中等规模诊断 | 每类 100 个样本，训练更短，用来检查代码路径和学习曲线，不作为最终论文数字。 |
| `configs/baseline/catastrophic_mnist_emnist_paper_medium_sdpm.yaml` | SDPM / partition medium 诊断 | medium 规模，开启 SDPM gate 和 neuron partition，用来验证机制是否生效。 |
| `configs/baseline/catastrophic_mnist_emnist_sdpm.yaml` | 服务器完整 SDPM-only 候选 | full 规模 SDPM-only 配置；当前不建议直接运行，先完成 medium 统计和 partition 验证。 |

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

SDPM / partition 中等规模诊断：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium_sdpm.yaml --device auto --run-name paper_medium_sdpm_only_seed0
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

当前 checkpoint 加载支持 fallback 匹配：当服务器本地 cache 路径导致精确 fingerprint 不同时，只要 stage、S1/S2 epoch、seed 和模型结构一致，就可以加载已有 S1/S2 checkpoint，避免重复训练特征层。C2/S3-input cache 使用独立小 batch（完整配置为 1024），避免一次加载 24,000 个 C2 tensor 导致 GPU OOM。

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

阶段 A：paper-source baseline 复现。当前状态：已完成。

- 确认数据集划分和 label mapping。
- 确认 S1/S2 checkpoint 复用逻辑。
- 确认 C2 cache 能在服务器本地生成并复用。
- 跑完整 `catastrophic_mnist_emnist.yaml`。
- 记录 Task1 after Task1、Task1 after Task2、Task2 after Task2、forgetting 和 avg acc。

阶段 B：可解释统计。当前状态：medium no-op 对照已完成，统计路径可作为诊断模块使用；后续只需随新实验继续整理产物。

- 在 S3 训练中记录 winner id、winner frequency 和 winner label count。
- 输出 Task 1 后的 `f_i`、`q_i`、`I_i` 分布；当前由 `src/continual/neuron_partition.py` 计算。
- 生成 stable/shared/reserve/dead neuron partition。
- 确认统计模块不改变 baseline 学习行为。
- 已完成 medium no-op 对照：pure baseline、logging-only、logging+partition 三组指标完全一致。

阶段 C：NGSG。当前状态：SDPM-only 已完成 medium 机制验证；novelty/reserve 尚未实现。

- 加入 SDPM soft protection。
- 校准 novelty score。
- 实现 class-local reserve activation。
- 对比 baseline、SDPM only、NGSG only、random reserve 和完整 NGSG。

## 8. NGSG 当前设计共识

NGSG 的创新点不在于重写整个 SpykeTorch 网络，而是在已复现的 Antonov/Mozafari 三层 SNN 基础上，主要在 S3 输出层加入持续学习机制。

核心模块：

- winner-frequency tracker：统计 Task 1 中每个 S3 neuron 的获胜频率；已在 baseline 和 SDPM 路径中启用。
- winner label count：统计每个 S3 neuron 对各类别的获胜次数；已接入 S3 训练统计，仍需结果产物验证。
- neuron partition：根据获胜频率和类别选择性划分 stable/shared/reserve/dead neurons；`src/continual/neuron_partition.py` 已有第一版。
- synaptic importance：估计旧任务关键连接的重要性；当前 SDPM 使用 winner frequency 和权重强度。
- SDPM plasticity gate：对重要旧连接缩放 R-STDP 更新幅度；medium 已验证能保护 Task1，但会压制 Task2。
- novelty detector：判断当前输入是否对旧网络足够新；尚未实现。
- reserve activation：对高 novelty 样本调用低使用率、低旧任务重要性的 reserve neurons；尚未实现。

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

1. 将 SDPM 的 importance/gate 逻辑和 `NeuronPartition` 的统一统计基础对齐，避免后续 SDPM、novelty gate、reserve activation 各算一套指标。
2. 实现 novelty gate 和 reserve activation，并先跑 medium 消融：baseline、SDPM-only、reserve/novelty-only、random reserve、full NGSG。
3. medium 消融稳定后，再启动 full 规模实验并同步记录到 `CATASTROPHIC_FORGETTING_REPRODUCTION.md`。
4. full 实验前继续维护服务器空间：不要删除 `checkpoints/`、`data/features/`、`data/preprocessed/`；必要时迁移大缓存到 `/autodl-pub/data` 并创建软链接。
