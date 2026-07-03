# NGSG SpykeTorch 项目手册

最后更新：2026-07-03（统计对齐、paired 对照、novelty/reserve 实现与 medium 消融）

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

## 0.1 当前推进状态（2026-07-03 晚）

当前 `dev` HEAD 为 `8a9e78a`（本地另有未提交的 reserve shape 修复）。项目处于 **medium 消融阶段**：统计对齐与 SDPM paired 对照已完成；reserve-only 与 full NGSG 在 Task2 首样本崩溃，待修复后重跑。

| 模块 | 当前状态 | 判断 |
| --- | --- | --- |
| paper-source catastrophic baseline | 已完成完整服务器复现 | 遗忘趋势和论文基本对齐，可作为主 baseline。 |
| winner-frequency / winner_label_counts | 已接入并完成 no-op 对照 | medium 下不扰动 baseline；`winner_label_counts` 200×10 已验证。 |
| `occupancy_stats.py` + `neuron_partition.py` | 已实现并验证 | 统一计算 `f_i/q_i/I_i`；partition stable 61 / shared 59 / reserve 80。 |
| SDPM gate（统计对齐后） | medium paired 已完成 | 与 partition 共用 occupancy；`q_i_mean≈0.504`，`unified_occupancy=True`。 |
| `novelty_gate.py` | 已实现（`8a9e78a`） | 用 natural winner 的 `combined_score` 作为 novelty。 |
| `reserve_activation.py` | 已实现并接入 Task2（`8a9e78a`）；**4D potentials shape bug 已本地修复** | Task2 首样本崩溃（48000 vs 200）；待 push 后重跑 #4/#6。 |
| full NGSG（600 epoch） | 尚未开始 | 先完成 medium 五组消融再进 full。 |

### 代码 commit 时间线（近期）

| commit | 内容 |
| --- | --- |
| `4f03da0` | SDPM 与 partition 共用 `occupancy_stats.py`（`f_i/q_i/I_i` 对齐） |
| `8a9e78a` | novelty gate + reserve activation + `configs/ngsg/` medium 消融 YAML |

### 历史参考：对齐前 SDPM medium（`dev` 旧版，2026-07-02）

| 配置 | Task1 after Task1 | Task1 after Task2 | Task2 after Task2 | Forgetting | Avg Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-SDPM same-code | 79.6% | 65.9% | 60.3% | 13.7 pp | 63.10% |
| SDPM-only（未对齐 q_i） | 79.6% | 68.1% | 54.8% | 11.5 pp | 61.45% |

这组结果说明旧版 SDPM 能保护 Task1，但明显牺牲 Task2。**不能**与下表新 paired 结果直接混用。


服务器 medium partition 验证结果（`paper_medium_partition_seed0`，`dev@9d9bded`）：

| 验证项 | 结果 | 判断 |
| --- | --- | --- |
| Metrics | Task1 after Task1 77.2%，Task1 after Task2 69.8%，Task2 after Task2 58.7%，Forgetting 7.4 pp，Avg Acc 64.25% | medium 诊断结果，不作为最终论文数字。 |
| `winner_label_counts` | shape 200 x 10，总数 50,000 | 已确认真实 per-neuron label counts 写入 `result.json`。 |
| partition | stable 61，shared 59，reserve 80，dead 0 | 基本合理；stable 只比建议上界 60 多 1 个，先不急调阈值。 |
| q_i vs decision_map fallback | 真实 q mean 0.504，fallback active q=1.0；86/200 active neurons 的 dominant label 与 decision_map 不一致 | fallback 明显高估选择性，后续 SDPM/reserve 应以真实 `winner_label_counts` 计算的 q_i 为准。 |

本机服务器产物副本：`experiments/server_paper_medium_partition_seed0/`。本机诊断产物：`experiments/diagnostics/paper_medium_partition_seed0/`，包含 `partition_validation.json` 和三张图：`f_i` 直方图、`q_i` vs `f_i` 散点图、dominant neuron per class 分布图。以上目录属于运行产物，不进入 git。

## 0.2 medium 消融矩阵（seed 0，`8a9e78a` 同 lineage）

所有下列 medium 实验使用相同数据规模（每类 100 train/test）、Task1 S3 50 epoch、Task2 S3 10 epoch、seed 0、相同 feature checkpoint 与 C2 cache。

| # | 组别 | run name | SDPM | reserve | 状态 | Task1→1 | Task1→2 | Task2→2 | Forgetting | Avg Acc |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline / no-op | `noop_medium_*` / `paper_medium_partition_seed0` | off | off | ✅ | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |
| 2 | no-SDPM paired | `paper_medium_no_sdpm_aligned_seed0` | off | off | ✅ | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |
| 3 | SDPM aligned | `paper_medium_sdpm_aligned_seed0` | on | off | ✅ | 77.2% | **74.2%** | 57.8% | **3.0 pp** | **66.0%** |
| 4 | reserve-only | `paper_medium_reserve_only_seed0` | off | on | ❌ Task2 崩溃 | - | - | - | - | - |
| 5 | random reserve | `paper_medium_random_reserve_seed0` | off | random | ⬜ 待跑 | - | - | - | - | - |
| 6 | full NGSG | `paper_medium_ngsg_seed0` | on | on | ❌ Task2 崩溃 | - | - | - | - | - |

**崩溃原因（#4/#6）：** C2 cache 路径下 `ctx["potentials"]` 为 4D `[1, 200, H, W]`，`aggregate_s3_neuron_potentials` 误 flatten 为 48000 维，与 partition mask（200）不匹配。已在本地 `reserve_activation.py` 修复（按 `s3_neurons=200` 解析 3D/4D shape）。

服务器路径：`/root/autodl-tmp/NGSG-spyketorch-4a958ae`，tmux 会话 `pw`。GPU 已空闲；修复 push 后重跑 #4 → #6。

### 对齐后 SDPM vs no-SDPM（paired，`4f03da0`+）

| 对比项 | no-SDPM | SDPM aligned | 差值 |
| --- | ---: | ---: | ---: |
| Task1 after Task1 | 77.2% | 77.2% | 0.0 |
| Task1 after Task2 | 69.8% | 74.2% | **+4.4 pp** |
| Task2 after Task2 | 58.7% | 57.8% | -0.9 pp |
| Forgetting | 7.4 pp | 3.0 pp | **-4.4 pp** |
| Avg Acc | 64.25% | 66.0% | **+1.75 pp** |

**当前可写结论：**

1. **baseline 复现成立**（full 48.42% Task1 after Task2，见 §0）。
2. **统计模块 no-op**（logging / partition 不改变 medium 指标）。
3. **对齐后 SDPM 在 medium 上同时改善旧任务保持与平均准确率**；Task2 仅小幅下降 0.9 pp，优于对齐前 SDPM 的 trade-off。
4. **partition + reserve 代码已接入**，但 medium 实验 #4/#6 因 potentials shape bug 未完成；修复后需重跑。

**当前还不能写成论文结论的内容：**

- reserve-only / full NGSG 的 medium 数字尚未产出（Task2 首样本崩溃）。
- medium 数字仍不是 final 600/100 epoch 论文主表。
- random reserve 对照尚未跑完。

**下一步：**

1. commit + push reserve shape 修复，服务器 `git pull` 后重跑 `paper_medium_reserve_only_seed0` → `paper_medium_ngsg_seed0`。
2. 补跑 `paper_medium_random_reserve_seed0`。
3. medium 稳定后启动 full 规模：baseline（已有）→ full SDPM aligned → full NGSG。
4. 汇总 #4/#6 结果到 `CATASTROPHIC_FORGETTING_REPRODUCTION.md`。

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

1. 以已完成的 paper-source catastrophic baseline 作为主对照。
2. 使用统一 occupancy 统计（`occupancy_stats.py`）：winner counts、winner label counts、`f_i/q_i/I_i`、partition。
3. SDPM 保护旧任务突触更新；reserve activation 为高 novelty 的 Task2 样本分配 reserve 容量。
4. 完成 medium 五组消融（baseline / SDPM / reserve-only / random reserve / full NGSG）。
5. medium 稳定后启动 full 规模实验。

当前不复现 joint training。旧的 frozen/Langevin 配置也已从 active YAML 中删除，只有论文对比确实需要时再重新建立。

## 2. 分支和代码状态

- `dev`：当前本地和服务器共同使用的主集成分支。
- `baseline/continuous-learning`：baseline 复现分支。
- `ngsg/novelty-gated-growth`：NGSG 创新实现分支。
- 当前 baseline 使用 `src/trainers/baseline_trainer.py` 和 `src/utils/data.py` 中的 paper-source SpykeTorch/Mozafari 路线。
- EMNIST raw idx fallback、paper-source 预处理缓存、S1/S2 feature checkpoint 复用和 C2 feature cache 已在 `dev` 中。

## 3. 当前常用 YAML

### `configs/baseline/`

| 配置 | 什么时候跑 | 说明 |
| --- | --- | --- |
| `catastrophic_mnist_emnist.yaml` | 服务器正式完整 baseline | 完整 paper-source MNIST → EMNIST；600/100 epoch。 |
| `catastrophic_mnist_emnist_feature_checkpoint.yaml` | checkpoint 缺失或重建时 | feature-only：只训 S1/S2，跳过 S3。 |
| `catastrophic_mnist_emnist_paper_medium.yaml` | medium 诊断 | 每类 100 样本；纯 baseline。 |
| `catastrophic_mnist_emnist_paper_medium_sdpm.yaml` | SDPM + partition medium | SDPM 与 reserve 开关见 YAML；当前 SDPM 默认 on。 |
| `catastrophic_mnist_emnist_paper_medium_no_sdpm.yaml` | paired 对照 | 与 medium_sdpm 相同，仅 `sdpm_gate.enabled: false`。 |
| `catastrophic_mnist_emnist_sdpm.yaml` | full SDPM-only | medium 消融稳定后再跑 full。 |

### `configs/ngsg/`（`8a9e78a` 新增）

| 配置 | SDPM | reserve | 用途 |
| --- | --- | --- | --- |
| `catastrophic_mnist_emnist_paper_medium_reserve_only.yaml` | off | on | 只验证 reserve 招募 |
| `catastrophic_mnist_emnist_paper_medium_ngsg.yaml` | on | on | full NGSG medium |
| `catastrophic_mnist_emnist_paper_medium_random_reserve.yaml` | off | random | reserve 随机对照 |

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

SDPM aligned medium：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium_sdpm.yaml --device cuda --run-name paper_medium_sdpm_aligned_seed0
```

no-SDPM paired medium：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium_no_sdpm.yaml --device cuda --run-name paper_medium_no_sdpm_aligned_seed0
```

reserve-only / full NGSG medium：

```bash
python scripts/run_baseline.py --config configs/ngsg/catastrophic_mnist_emnist_paper_medium_reserve_only.yaml --device cuda --run-name paper_medium_reserve_only_seed0
python scripts/run_baseline.py --config configs/ngsg/catastrophic_mnist_emnist_paper_medium_ngsg.yaml --device cuda --run-name paper_medium_ngsg_seed0
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

阶段 C：NGSG。当前状态：SDPM 统计对齐与 paired 对照已完成；novelty/reserve 已接入；medium 消融 #4/#6 运行中。

- ✅ SDPM soft protection（对齐 occupancy 后 medium paired 验证）。
- ✅ novelty score（`novelty_gate.py`，基于 partition `combined_score`）。
- ✅ class-local reserve activation（`reserve_activation.py`，Task2 STDP 路由）。
- ⏳ medium 消融：baseline、SDPM、reserve-only、random reserve、full NGSG。
- ⬜ full 规模 SDPM / full NGSG。

## 8. NGSG 当前设计共识

NGSG 的创新点不在于重写整个 SpykeTorch 网络，而是在已复现的 Antonov/Mozafari 三层 SNN 基础上，主要在 S3 输出层加入持续学习机制。

核心模块（`src/continual/`）：

| 模块 | 文件 | 状态 |
| --- | --- | --- |
| occupancy 统计 | `occupancy_stats.py` | ✅ `f_i/q_i/I_i` 统一来源 |
| neuron partition | `neuron_partition.py` | ✅ stable/shared/reserve 划分 |
| SDPM gate | `sdpm_gate.py` | ✅ 对齐 occupancy；Task2 缩放 R-STDP |
| novelty gate | `novelty_gate.py` | ✅ natural winner 的 occupancy 分数 |
| reserve activation | `reserve_activation.py` | ✅ 高 novelty → reserve STDP 路由 |

Task2 训练顺序（full NGSG）：forward → novelty 判定 → 可选 reserve  reroute → reward/punish → SDPM 缩放更新。

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

1. ⏳ 等 medium #4 reserve-only 与 #6 full NGSG 跑完，更新 §0.2 表格与 `CATASTROPHIC_FORGETTING_REPRODUCTION.md`。
2. ⬜ 补跑 random reserve（`paper_medium_random_reserve_seed0`）。
3. ⬜ 若 full NGSG medium 优于 SDPM-only，启动 full 规模 SDPM aligned 与 full NGSG。
4. ⬜ 多 seed（0/1/2）重复关键 medium 配置。
5. 维护服务器缓存与磁盘；实验产物同步到本机 `experiments/server_*` 副本（不进 git）。
