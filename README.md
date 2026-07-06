# NGSG SpykeTorch 项目手册

最后更新：2026-07-06（**§0.0 服务器 P0 PASS** — 6.1/6.2 + random/shuffle 反事实已同步）

这个仓库只保留三个主要 Markdown 入口：

- `README.md`：当前项目状态、配置选择、运行命令、复现计划和 NGSG 实现路线。
- `CATASTROPHIC_FORGETTING_REPRODUCTION.md`：灾难性遗忘 baseline 的历史实验记录和结果日志。
- `实验列表.md`：WTA 分区生死实验优先级与判据。

其他旧的计划文档、配置 README 和模块 README 已合并到本文件，避免之后不知道该看哪一个。

## 0.0 WTA 分区组诊断 + P0 反事实（medium seed0，服务器 2026-07-06）— **PASS**

**Run：** partition_group_diagnosis_medium_seed0 · **服务器 commit：** 3f97631  
**协议：** medium Task1 结束后（未训 Task2）；100/class，S3 50 epoch · **分区：** stable **61** / shared **59** / reserve **80** / dead **0**  
**产物：** 服务器 diagnostics/partition_counterfactuals.json；git published_results/diagnostics/partition_counterfactuals_medium_seed0.json  
**本机镜像（不进 git）：** experiments/server_partition_group_diagnosis_medium_seed0/

### 6.1 Group-only（只用某一组神经元推理 Task1）

| 条件 | 用哪些神经元 | Task1 准确率 | vs 200 全用 |
| --- | ---: | ---: | ---: |
| **all-200（baseline）** | 200 | **77.50%** | — |
| **stable-only** | 61 | **77.50%** | **0.00 pp** |
| shared-only | 59 | 20.30% | -57.20 pp |
| reserve-only | 80 | 7.30% | -70.20 pp |

测试时 natural WTA 的 winner：**98.4%** 来自 stable，shared **0.8%**，reserve **0.8%**。

### 6.2 Group-masked（屏蔽某一组再推理 Task1）

| 条件 | 屏蔽谁 | Task1 准确率 | vs 200 全用 |
| --- | ---: | ---: | ---: |
| **mask-reserve** | reserve（80） | **77.40%** | **-0.10 pp** |
| mask-shared | shared（59） | 77.40% | -0.10 pp |
| **mask-stable** | stable（61） | **19.30%** | **-58.20 pp** |

embedded trainer.evaluate Task1 acc：**77.2%**。

### P0 反事实对照（10 seeds，k=61）

| 对照 | Task1 准确率 | vs WTA | 判据 |
| --- | ---: | ---: | --- |
| random stable-only | **52.06% ± 6.09 pp** | -25.4 pp vs WTA stable-only | **PASS** |
| random mask-stable | **70.19% ± 2.94 pp** | vs mask WTA stable 19.30% | **PASS** |
| shuffled winner-history stable-only | 64.30% / 49.30% / 55.90% | 不再 ≈ all | **PASS** |
| frequency-only stable-only | 77.50%（= WTA） | — | **NARROW** |

**Caveat：** dead=0，matched-active random 与普通 random 相同。

### 结论

1. **Task1 几乎全靠 stable** — 61 个 stable = 200 个全用（77.5%）。
2. **reserve / shared 对 Task1 自然推理几乎无贡献** — mask-reserve / mask-shared 各 -0.1 pp；reserve-only ≈ 7.3%。
3. **stable 是 Task1 推理必要条件** — mask-stable → 19.3%。
4. **非随机、非任意 mask** — random 61 仅 52%；random mask 70%；只有 mask **WTA stable** 才崩。
5. **工程含义** — shared + reserve（**139 个**）可作 Task2 容量池；Task1 由 stable **61** 承担。

**仍缺：** P0-3 seed 1/2、**P0-4 full Task1**（主文硬门槛）。详见 实验列表.md。

## 0. 最新正式 SDPM-only full 结果（2026-07-06）

服务器完整 `catastrophic_mnist_emnist_sdpm.yaml` 已跑完，运行版本为 `3f97631`，run name 为 `paper_full_sdpm_only_seed0`（600/100 epoch，seed 0，aligned occupancy + SDPM on Task2）。

| 指标 | SDPM-only full | no-SDPM baseline full（§0.1） | 论文参考 |
| --- | ---: | ---: | ---: |
| Initial MNIST / Task1 after Task1 | 93.14% | 92.84% | 90.8 ± 0.9% |
| Subsequent MNIST / Task1 after Task2 | **57.45%** | 48.42% | 48.1 ± 4.8% |
| Subsequent EMNIST / Task2 after Task2 | 75.01% | 74.91% | 78.4 ± 1.2% |
| Forgetting | **35.69 pp** | 44.42 pp | 约 42.7% |
| Avg Acc | **66.23%** | 61.67% | - |

相对同一 lineage 的 no-SDPM full baseline（`paper_ch4_catastrophic_optimized_winnerlog_seed0`）：

| 对比项 | no-SDPM full | SDPM-only full | 差值 |
| --- | ---: | ---: | ---: |
| Task1 after Task2 | 48.42% | 57.45% | **+9.03 pp** |
| Task2 after Task2 | 74.91% | 75.01% | +0.10 pp |
| Forgetting | 44.42 pp | 35.69 pp | **-8.73 pp** |
| Avg Acc | 61.67% | 66.23% | **+4.56 pp** |

SDPM 摘要（Task2，`result.json` extra）：`gate_mean≈0.973`，`protected_fraction≈0.30`，`occupancy_q_i_mean≈0.760`，`unified_occupancy=True`；partition stable 60 / shared 60 / reserve 80。

结论：full 规模下 **SDPM-only 显著缓解灾难性遗忘**（Task1 after Task2 +9.03 pp，Forgetting -8.73 pp），且 **Task2 与 Avg Acc 不降反升**。这是当前最强、可写进论文主表的正式结果；reserve / full NGSG 的 full 实验尚未启动。

可提交的精简结果文件：`published_results/baseline/paper_full_sdpm_only_seed0.json`。

本机服务器产物副本：`experiments/server_paper_full_sdpm_only_seed0/`（含 `result.json`、`resolved_config.json`）。该目录属于运行产物，不进入 git。

## 0.1 最新正式 baseline 结果（2026-07-02）

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

## 0.2 当前推进状态（2026-07-06）

当前 `dev` 待推送：partition 组诊断 + task-aware 读出脚本与 `published_results/diagnostics/`。服务器已跑完 full SDPM-only（`3f97631`）；medium 消融 **#1–#4、#6 已完成**；**#5 random reserve 待跑**。

| 模块 | 当前状态 | 判断 |
| --- | --- | --- |
| paper-source catastrophic baseline | 已完成完整服务器复现 | 遗忘趋势和论文基本对齐，可作为主 baseline（§0.1）。 |
| **SDPM-only full（600/100 epoch）** | **已完成**（`paper_full_sdpm_only_seed0`） | **Task1 after Task2 +9.03 pp vs baseline full**；当前主方法数值结果。 |
| **partition 组推理 + P0 反事实** | **medium seed0 PASS** | **§0.0：random/shuffle 证伪失败；mask WTA stable 特异** |
| **task-aware readout 消融** | **medium Task2 已完成** | MNIST +4.1 pp / EMNIST +3.3 pp vs 标准 WTA（§0.6）。 |
| winner-frequency / winner_label_counts | 已接入并完成 no-op 对照 | medium 下不扰动 baseline。 |
| `occupancy_stats.py` + `neuron_partition.py` | 已实现并验证 | full：stable 60 / shared 60 / reserve 80；medium：61/63/76。 |
| `readout.py` + 诊断脚本 | **新增** | `eval_partition_group_diagnosis.py`、`eval_readout_ablation.py`。 |
| SDPM gate（统计对齐后） | **medium + full 均已完成** | full：Forgetting 44.42→35.69 pp。 |
| `reserve_activation.py` | medium 已跑通 | Task2 掉点；train–test mismatch（§0.3.1）。 |
| full NGSG（600 epoch） | 尚未开始 | 待 reserve 机制修正 + full partition/readout 诊断。 |

### 代码 commit 时间线（近期）

| commit | 内容 |
| --- | --- |
| `4f03da0` | SDPM 与 partition 共用 `occupancy_stats.py`（`f_i/q_i/I_i` 对齐） |
| `8a9e78a` | novelty gate + reserve activation + `configs/ngsg/` medium 消融 YAML |
| `cf2ae88` | 修复 C2 cache 下 4D S3 potentials 解析 |
| `0aa21a3` | 修复 `_select_neuron` 语法错误 |
| `9b55533` | test-time winner 诊断 + `stable_mismatch` reserve reroute |
| `3f97631` | STDP feedback 与 rerouted winner 对齐；**full SDPM-only 在此 commit 跑完** |

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

## 0.3 medium 消融矩阵（seed 0，`0aa21a3` 同 lineage）

所有下列 medium 实验使用相同数据规模（每类 100 train/test）、Task1 S3 50 epoch、Task2 S3 10 epoch、seed 0、相同 feature checkpoint 与 C2 cache。

| # | 组别 | run name | SDPM | reserve | 状态 | Task1→1 | Task1→2 | Task2→2 | Forgetting | Avg Acc |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline / no-op | `noop_medium_*` / `paper_medium_partition_seed0` | off | off | ✅ | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |
| 2 | no-SDPM paired | `paper_medium_no_sdpm_aligned_seed0` | off | off | ✅ | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |
| 3 | SDPM aligned | `paper_medium_sdpm_aligned_seed0` | on | off | ✅ | 77.2% | **74.2%** | 57.8% | **3.0 pp** | **66.0%** |
| 4 | reserve-only | `paper_medium_reserve_only_seed0` | off | on | ✅ | 77.2% | 72.7% | **33.6%** | 4.5 pp | 53.15% |
| 5 | random reserve | `paper_medium_random_reserve_seed0` | off | random | ⬜ 待跑 | - | - | - | - | - |
| 6 | full NGSG | `paper_medium_ngsg_seed0` | on | on | ✅ | 77.2% | 71.3% | **36.3%** | 5.9 pp | 53.80% |

服务器路径：`/root/autodl-tmp/NGSG-spyketorch-4a958ae`。`#4/#6` 于 `0aa21a3` 重跑完成（`dev@cf2ae88+`）。

### 对齐后 SDPM vs no-SDPM（paired，`4f03da0`+）

| 对比项 | no-SDPM | SDPM aligned | 差值 |
| --- | ---: | ---: | ---: |
| Task1 after Task1 | 77.2% | 77.2% | 0.0 |
| Task1 after Task2 | 69.8% | 74.2% | **+4.4 pp** |
| Task2 after Task2 | 58.7% | 57.8% | -0.9 pp |
| Forgetting | 7.4 pp | 3.0 pp | **-4.4 pp** |
| Avg Acc | 64.25% | 66.0% | **+1.75 pp** |

### 0.3.1 reserve / NGSG Task2 掉点诊断（2026-07-03）

**现象：** reserve-only 与 full NGSG 的 Task1 after Task2 略优于 no-SDPM（72.7 / 71.3 vs 69.8%），但 **Task2 从 58.7% 跌至 33–36%**，Avg Acc 反而最低。

**Task2 训练统计（10 epoch × 1000 samples）：**

| 指标 | reserve-only | full NGSG |
| --- | ---: | ---: |
| `novel_fraction` | 38.4% | 38.3% |
| `recruited_updates` | 3842 | 3825 |
| `recruitment_rate` | 0.384 | 0.383 |
| `skipped_low_novelty` | 6158 | 6175 |
| `mean_score`（natural winner occupancy） | 0.136 | 0.136 |
| Task2 末 epoch train acc proxy | ~41% | ~41% |

**最可能根因：训练 STDP 目标与推理 winner 不一致（train–test mismatch）**

当前 Task2 循环（`baseline_trainer.py`）为：

```text
forward → natural WTA winner → decision（用于 acc 统计）
       → maybe_reroute：改 ctx["winners"] 为 reserve 神经元
       → reward/punish：对 rerouted winner 做 STDP
```

推理 / 测试时 **不做 reroute**，预测仍由 natural WTA + `decision_map` 决定。约 **38%** 的 STDP 更新被写入 reserve 神经元，但这些神经元在测试竞争里往往 **仍输给 Task1 已占用的 stable/shared 神经元**，导致 EMNIST 学不上去。

**次要因素：**

1. **`is_novel` 语义：** 代码里 `occupancy >= 0.15` 才 recruit，实际是「旧任务高占用 winner 触发避让」，不是「低占用才算 novel」；命名易误解，但逻辑本身是保护旧神经元。
2. **recruit 策略：** `_select_neuron` 在 class-local reserve 里取 **potential 最大**者，不等于测试时会赢的 neuron；reserve 在 Task1 几乎未训练，potential 排序噪声大。
3. **SDPM + reserve 叠加：** full NGSG 的 Task2 比 reserve-only 还低 2.7 pp，SDPM 进一步压低 shared 神经元可塑性，可能加剧「能赢的 neuron 学不动」。

**建议下一步（按优先级）：**

1. **诊断实验：** Task2 结束后统计 reserve 神经元 test-time win rate vs stable/shared；确认 train–test mismatch。
2. **机制修正（择一或组合）：** 仅对「natural winner 为 stable 且 decision≠target」reroute；或 recruit 后同步更新 `decision_map`/boost reserve 在 WTA 中的竞争；或降低 `novelty_threshold` 减少 reroute 比例做 sensitivity。
3. **对照：** 跑 `#5 random reserve`；试 threshold ∈ {0.25, 0.35, 0.50}。
4. **论文叙事：** 当前 SDPM-only 已足够支撑「importance-aware plasticity allocation」；reserve 需 fix 后再 claim「novelty-conditioned capacity recruitment」。

**当前可写结论：**

1. **baseline 复现成立**（full 48.42% Task1 after Task2，见 §0.1）。
2. **统计模块 no-op**（logging / partition 不改变 medium 指标）。
3. **对齐后 SDPM 在 medium 与 full 上均改善旧任务保持与 Avg Acc**（medium §0.3 表；full §0）。
4. **reserve 代码跑通但 medium 上 Task2 失效**，根因高度指向 STDP reroute 与推理 WTA 脱节；**不能**写「完整 NGSG 优于 SDPM-only」。

**当前还不能写成论文结论的内容：**

- reserve / full NGSG 作为有效创新点的 efficacy claim（full NGSG 未跑；reserve medium 失效）。
- full SDPM-only 目前仅 seed 0；多 seed 待补。

**下一步：**

1. full 上复跑 §0.5 partition 组诊断（`save_task1_model: true` + full config）。
2. 补跑 `paper_medium_random_reserve_seed0` 与 random-partition 对照。
3. 修 reserve train–test mismatch；测 Task2 reserve test-time win rate。
4. full SDPM-only 多 seed；reserve 稳定后再考虑 full NGSG。
5. 汇总到 `CATASTROPHIC_FORGETTING_REPRODUCTION.md`。

## 0.4 Logging / partition no-op 对照（2026-07-03）

为了确认统计模块本身不改变 baseline 学习行为，服务器上补跑了三组 seed 0 medium 对照。三组除统计开关外保持相同数据、epoch、checkpoint、C2 cache 和随机种子；SDPM 均未启用。

| 组别 | run name | winner logging | partition | Task1 after Task1 | Task1 after Task2 | Task2 after Task2 | Forgetting | Avg Acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| pure baseline | `noop_medium_baseline_seed0` | off | off | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |
| logging-only | `noop_medium_logging_seed0` | on | off | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |
| logging + partition | `paper_medium_partition_seed0` | on | on | 77.2% | 69.8% | 58.7% | 7.4 pp | 64.25% |

结论：在 seed 0 medium 配置下，winner logging 和 neuron partition 拟合均没有可观察到的学习行为扰动。这个结论只支持“当前统计路径是 no-op 诊断模块”，不等价于 full 规模或多 seed 的最终证明。

本机服务器产物副本：`experiments/server_noop_medium_baseline_seed0/`、`experiments/server_noop_medium_logging_seed0/`、`experiments/server_paper_medium_partition_seed0/`。本机汇总产物：`experiments/diagnostics/noop_medium_controls/noop_medium_controls_summary.json` 和 `experiments/diagnostics/noop_medium_controls/noop_medium_controls_summary.csv`。

## 0.5 Partition 组推理诊断 — 协议与复现（结论见 §0.0）

§0.0 已记录 **6.1 / 6.2 完整数据与结论**。本节补充实验目的、协议细节与命令。

**目的：** 验证 stable/shared/reserve 分区是否具有 **功能意义**（不是贴标签），并判断 Task2 可动用哪些神经元池。

**协议：** Task1 训练完成后立即在 MNIST 测试集上评估；**未训练 Task2**。每类 100 train/test，S3 50 epoch；partition 为 stable 61 / shared 63 / reserve 76。推理使用标准 `decision_map` + global WTA，仅在指定神经元子集内竞争（group-only / group-masked）。

**结果摘要（与 §0.0 一致）：**

| 类型 | 关键条件 | Task1 acc | vs all-200 |
| --- | --- | ---: | ---: |
| 6.1 | stable-only (61) | 76.40% | -0.10 pp |
| 6.1 | reserve-only (76) | 10.10% | -66.40 pp |
| 6.2 | mask-reserve | 76.60% | +0.10 pp |
| 6.2 | mask-stable | 25.40% | -51.10 pp |

natural WTA winner：stable 97.8% / shared 1.6% / reserve 0.6%。

**注意：** medium 上 embedded Task1 acc ≈ 77.9%（非 full ~93%）；相对关系（stable 主导）待 full Task1 checkpoint 复验（§0.8 P0-4）。

运行：

```bash
python scripts/eval_partition_group_diagnosis.py --config configs/ngsg/partition_group_diagnosis_medium.yaml --train-task1 --device cuda
python scripts/eval_partition_group_diagnosis.py --run-dir experiments/partition_group_diagnosis_medium_seed0 --device cuda
```

精简结果：`published_results/diagnostics/partition_group_diagnosis_medium_seed0.json`。完整产物在 `experiments/partition_group_diagnosis_medium_seed0/`（不进 git）。

## 0.6 Task-aware 读出消融（Task2 后，medium，2026-07-06）

目的：比较标准 WTA + `decision_map` 与基于 `winner_label_counts` 的 **task-aware class-max 读出**（不改训练，只改推理）。

协议：medium SDPM+reserve（同 full NGSG 结构），Task2 结束后评估。checkpoint：`model_after_task2.pt`。

| 测试集 | 标准 WTA + decision_map | task-aware class-max | 差值 |
| --- | ---: | ---: | ---: |
| MNIST after Task2 | 55.6% | **59.7%**（Task1 dominant label） | **+4.1 pp** |
| EMNIST after Task2 | 54.3% | **57.6%**（Task2 dominant label） | **+3.3 pp** |

注意：必须用 **分任务 label 表**（MNIST 用 Task1 统计，EMNIST 用 Task2 统计）；混用 Task1 label 测 EMNIST 会跌至 16.5%。

运行：

```bash
python scripts/eval_readout_ablation.py --run-dir experiments/readout_ablation_medium_seed0 --device cuda
```

精简结果：`published_results/diagnostics/readout_ablation_medium_seed0.json`。

## 0.7 论文叙事收敛（2026-07-06）

**不能 claim**：提出 WTA；静态 stable/shared/reserve 贴标签；完整 NGSG 已优于 continue 表格方法。

**可以 claim（需实验支撑）**：

> 基于 WTA 竞争历史的输出神经元功能分区 + task-aware group 竞争读出。

证据链：

| 论点 | 支撑 |
| --- | --- |
| 分区不是随机分组 | **P0 PASS**：random 61 → 52% vs WTA stable 77.5%（§0.0） |
| 分区不是贴标签 no-op | logging+partition 不改变训练指标（§0.4）；功能意义在读出/招募 |
| stable 承载 Task1 | §0.5 group 诊断 |
| task-aware 读出有效 | §0.6 readout 消融 |
| SDPM 缓解遗忘 | §0 full SDPM-only（+9.03 pp Task1 after Task2 vs baseline full） |
| reserve 学 Task2 且不抢 Task1 | **未完成**；当前 reserve 训练有 train–test mismatch（§0.3.1） |

当前主结果仍是 **SDPM-only full**；下一阶段重点是 **reserve 定向学习 + 推理时 group 竞争不与 stable 抢 Task1**。

## 0.8 WTA 分区生死实验路线图（2026-07-06）

排序原则：**跑完能证明或证伪 WTA 分区 claim**；若无效则立刻舍弃或重做算法。

### 已有正证据 + P0 反事实（seed 0 medium，2026-07-06）

- stable-only ≈ all-200（**77.50%**）
- mask WTA stable 崩（**19.30%**）
- **random stable-only 52.06%** — 随机不行 ✅
- **random mask-stable 70.19%** vs mask WTA stable 19.30% — 只有 WTA stable 特异 ✅
- **shuffled history** stable-only 64/49/56% — 打乱不行 ✅
- frequency-only = WTA — claim 可收窄为 winner-frequency ⚠️

**仍缺：** P0-3 seed 1/2、**P0-4 full**

### P0：生死实验（先跑）

| # | 实验 | 证明/证伪 | 失败含义 |
| ---: | --- | --- | --- |
| 1 | **random same-size partition 对照** | 随机选同样数量神经元，是否也能 stable-only≈all | 随机 61 个也接近 all → WTA stable 不特殊，claim 倒 |
| 2 | **winner-history shuffled partition** | 打乱 winner history 后重分区是否失效 | 打乱仍有效 → 不是 WTA 历史在起作用 |
| 3 | **multi-seed medium diagnosis** | seed 0/1/2 是否都 stable-only≈all、mask-stable 崩 | 仅 seed0 → 不稳定，不能作主 claim |
| 4 | **full-scale partition diagnosis** | full Task1 后是否 stable 主导 | full 不成立 → 只能写 medium 诊断现象 |
| 5 | **random mask-stable 对照** | 随机 mask 同数量神经元是否也崩 | 随机 mask 也崩 → 冗余太低，非 stable 特异 |

### P1：强证据（决定 claim 能写多强）

| # | 实验 | 证明/证伪 | 失败含义 |
| ---: | --- | --- | --- |
| 6 | per-class stable coverage | 每类是否都有 stable 承载 | 只覆盖少数类 → 改写成部分类别主导 |
| 7 | stable selectivity / purity | stable 是否高频且对 Task1 有选择性 | 只高响应不 selective → 解释变弱 |
| 8 | frequency / selectivity / combo 消融 | 分区规则是否需组合指标 | frequency-only 就够 → 算法应简化 |
| 9 | threshold sensitivity | 改 percentile / q_min 结论是否稳 | 稍变就失效 → 调参产物 |
| 10 | training-time emergence | stable 是否随 Task1 训练逐渐形成 | 后验偶然切出 → 机制说法变弱 |

### P2：应用验证（非 WTA 生死线）

| # | 实验 | 作用 |
| ---: | --- | --- |
| 11 | task-aware readout full 复验 | 分区/label counts 能否改进读出（需 task-aware 设定） |
| 12 | SDPM random protection vs WTA protection | occupancy 软保护是否优于随机少更新 |
| 13 | hard-freeze stable vs SDPM | 是否等价于冻结旧神经元 |
| 14 | SDPM multi-seed | 加强应用结果 |
| 15 | reserve 修正 / random reserve / Task2 win rate | 暂后置；当前 reserve 干扰主线 |

### 最短必跑组合（5 项）

1. random same-size stable-only / mask 对照
2. winner-history shuffled partition
3. medium seed 1/2 partition diagnosis
4. full partition group diagnosis
5. frequency-only vs current partition 消融

**跑完后的决策树：**

- random/shuffled 也有效 → 当前分区 claim **不成立**，重做算法
- medium 支持、full 不支持 → 只能写 medium 诊断，**不能作主贡献**
- 仅 frequency-only 有效 → 收窄为「winner-frequency identifies task-bearing neurons」
- 五项全支持 → WTA 分区是真实、非随机、可复现的机制点

### 命令（P0 #1/#2/#5 可在已有 checkpoint 上后验跑）

```bash
# 反事实对照（#1/#2/#5）：无需重训，10 个 random seed
python scripts/eval_partition_group_diagnosis.py \
  --run-dir experiments/partition_group_diagnosis_medium_seed0 \
  --counterfactuals --device cuda \
  --write-json experiments/partition_group_diagnosis_medium_seed0/diagnostics/partition_counterfactuals.json

# multi-seed medium（#3）
python scripts/eval_partition_group_diagnosis.py --config configs/ngsg/partition_group_diagnosis_medium_seed1.yaml --train-task1 --device cuda
python scripts/eval_partition_group_diagnosis.py --config configs/ngsg/partition_group_diagnosis_medium_seed2.yaml --train-task1 --device cuda

# full Task1 诊断（#4，600 epoch，耗时长）
python scripts/eval_partition_group_diagnosis.py --config configs/ngsg/partition_group_diagnosis_full.yaml --train-task1 --device cuda
```

## 1. 当前目标

1. 以已完成的 paper-source catastrophic baseline 作为主对照。
2. 使用统一 occupancy 统计（`occupancy_stats.py`）：winner counts、winner label counts、`f_i/q_i/I_i`、partition。
3. SDPM 保护旧任务突触更新；reserve activation 为高 novelty 的 Task2 样本分配 reserve 容量。
4. 完成 medium 五组消融（baseline / SDPM / reserve-only / random reserve / full NGSG）。
5. medium 稳定后启动 full 规模实验；**full SDPM-only 已完成**（§0），full NGSG 待 reserve 机制修正。

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
| `partition_group_diagnosis_medium.yaml` | off | off | Task1 后 partition 组诊断；`save_task1_model: true` |
| `readout_ablation_medium_seed0.yaml` | on | on | task-aware 读出消融；`save_task2_model: true` |

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

SDPM-only full（服务器正式）：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_sdpm.yaml --device cuda --run-name paper_full_sdpm_only_seed0
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

**可提交 git 的实验摘要**：`published_results/baseline/*.json`（正式 run 指标）、`published_results/diagnostics/*.json`（组诊断与读出消融）。完整 `experiments/`、`result.json`、模型 checkpoint 不进 git。

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

阶段 C：NGSG。当前状态：SDPM medium + **full 均已完成**；novelty/reserve 已接入；medium 消融 #4/#6 完成；full NGSG 未开始。

- ✅ SDPM soft protection（medium + full 验证，见 §0 / §0.3）。
- ✅ novelty score（`novelty_gate.py`，基于 partition `combined_score`）。
- ✅ class-local reserve activation（`reserve_activation.py`，Task2 STDP 路由；medium Task2 掉点待修）。
- ⏳ medium 消融：baseline、SDPM、reserve-only、random reserve（待跑）、full NGSG。
- ⬜ full 规模 NGSG（SDPM-only full 已完成：`paper_full_sdpm_only_seed0`）。

## 8. NGSG 当前设计共识

NGSG 的创新点不在于重写 SpykeTorch 网络，而是在已复现的 Antonov/Mozafari SNN 上，**把 WTA 竞争历史变成神经元功能分区依据，并据此设计 task-aware group 竞争读出与容量分配**。

一句话：

> WTA 本身不能 claim；能 claim 的是：**竞争历史 → stable/shared/reserve 功能分区 → 按任务选择神经元池做 group-WTA 读出**。

核心模块（`src/continual/`）：

| 模块 | 文件 | 状态 |
| --- | --- | --- |
| occupancy 统计 | `occupancy_stats.py` | ✅ `f_i/q_i/I_i` 统一来源 |
| neuron partition | `neuron_partition.py` | ✅ 竞争历史驱动的功能分区 |
| task-aware readout | `readout.py` | ✅ group-only / class-max 读出策略 |
| SDPM gate | `sdpm_gate.py` | ✅ Task2 突触级 soft protection |
| novelty gate | `novelty_gate.py` | ✅ occupancy 分数 |
| reserve activation | `reserve_activation.py` | ⚠️ Task2 训练路由；推理 mismatch 待修 |

诊断脚本：

| 脚本 | 用途 |
| --- | --- |
| `scripts/eval_partition_group_diagnosis.py` | §6.1/6.2 stable/shared/reserve 组推理 |
| `scripts/eval_readout_ablation.py` | task-aware vs 标准 WTA 读出对比 |
| `scripts/diagnose_test_winner_roles.py` | Task2 后 test-time winner 角色分布 |

Task2 训练顺序（full NGSG）：forward → novelty 判定 → 可选 reserve reroute → reward/punish → SDPM 缩放更新。

**当前论文可写边界**：SDPM-only full + partition 功能诊断 + task-aware readout 消融；**暂不能** claim 完整 NGSG 优于 SDPM-only 或 continue 表格全线方法。

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

**WTA 分区生死线（见 §0.8 P0）— 最短必跑 5 项优先于 reserve/SDPM 扩展。**

1. ⬜ P0 #1/#2/#5：`--counterfactuals` 于 seed0 checkpoint（脚本已支持）。
2. ⬜ P0 #3：medium seed 1/2 partition diagnosis。
3. ⬜ P0 #4：full partition group diagnosis（600 epoch）。
4. ⬜ P1：per-class coverage、threshold sweep、frequency-only 正式对比。
5. ⬜ P2：task-aware readout full、SDPM multi-seed；reserve 修正暂后置。
6. 维护 `published_results/` 精简 JSON；大产物保留在 `experiments/`（不进 git）。
