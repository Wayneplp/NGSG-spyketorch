# NGSG SpykeTorch 项目手册

最后更新：2026-07-09（**主线：方案 B + HTM；R0 full + P1/R3 medium/full 已记录**）

这个仓库只保留三个主要 Markdown 入口：

- `README.md`：当前项目状态、方案 B / HTM 方法路线、配置选择、运行命令与历史实验数字。
- `CATASTROPHIC_FORGETTING_REPRODUCTION.md`：灾难性遗忘 baseline 的历史实验记录和结果日志。
- `实验列表.md`：实验优先级与 Claim 对照（待与 HTM 叙事同步更新）。

其他旧的计划文档、配置 README 和模块 README 已合并到本文件，避免之后不知道该看哪一个。

---

## 当前主线：方案 B + HTM（Role-Aware CL + Hippocampus-inspired Task Memory）

### 我们在做什么

在已复现的 **Mozafari / Antonov paper-source SNN**（MNIST → EMNIST，无回放）上，解决：

> **训练阶段按 stable / shared / reserve 做了分工，但推理阶段仍是 200 神经元裸 WTA → train–test mismatch，Task2 学在 reserve、测时 stable 抢答。**

完整方法 = **训练阶段神经元分组** + **推理阶段任务记忆检索** + **按组路由的竞争式分类**：

```text
Step 1  Role assignment      Task1 后按 winner frequency f_i → stable / shared / reserve
Step 2  Role-aware training  Task2：Δw' = λ_{g(i)} · Δw（stable 保护，reserve 正常学）
Step 3  HTM task memory      每个 task 形成内部记忆痕迹 m_k；推理时 pattern completion 推断 t̂
Step 4  Group routing + WTA  t̂=1 → stable∪shared；t̂=2 → reserve∪shared → 组内 WTA → 类别
```

### HTM 是什么（为什么不写成外置 `prototype_bank.pt`）

**不推荐**叙事：

```text
x → r(x) → 加载外部 .pt 原型表 → 查表得 task
```

**推荐**叙事：**HTM（Hippocampus-inspired Task Memory，海马体启发任务记忆模块）** — 任务原型存在模型内部的 **task memory neurons（任务记忆神经元）** `m_k` 中，随 checkpoint 一起保存，是网络的一部分而非「小抄文件」。

```text
x → r(x) → HTM(m_1, m_2, …) → t̂ → G_{t̂} 路由 → WTA(G_{t̂}) → y
```

论文表述（英）：

> We introduce a hippocampus-inspired task memory module to store task-level spiking response patterns and perform task inference through similarity-based memory retrieval.

### HTM 机制（第一版，K=2 固定任务）

**响应向量**（来自 S3 分类层，不参与 HTM 训练 STDP）：

```text
r(x) ∈ R^200
r_i(x) = max_{t,h,w} S3_potential_i(t,h,w)
```

**记忆神经元**（每个已学 task 一条 engram）：

```text
m_1  Task1（MNIST）记忆原型
m_2  Task2（EMNIST）记忆原型
```

**Task1 训完后写入 m_1**（running average，非外置文件）：

```text
m_1 ← (1-η) m_1 + η r(x)    对 x ∈ D_1 训练集遍历
```

亦可写为 Hebbian 形式：`Δm_k = η · a_k · r(x)`，其中 `a_k = sim(r(x), m_k)`。

**推理：记忆检索 + WTA 选 task**

```text
a_k = cos(r(x), m_k) = (r(x)·m_k) / (||r(x)|| ||m_k||)
t̂   = argmax_k a_k
```

**组路由（role-aware competition）**

```text
t̂=1  →  G_1 = stable ∪ shared      （reserve mask）
t̂=2  →  G_2 = shared ∪ reserve     （stable mask）
y     =  WTA(G_{t̂})  →  decision_map[winner]
```

**后续扩展（K>2 / 真 continual）**：若 `max_k a_k < θ_novel`，分配新记忆神经元 `m_{K+1}`（快速任务编码，类似海马对新场景的编码）。当前 MNIST→EMNIST 协议可先固定 K=2。

### 生物启发对照（论文 Discussion / Figure）

| 生物概念 | 模型对应 |
| --- | --- |
| hippocampus（海马体） | HTM 模块 |
| memory engram（记忆痕迹） | task memory neuron `m_k` |
| pattern completion（模式补全） | `r(x)` 与 `m_k` 相似度匹配 |
| pattern separation（模式分离） | 不同 task 的 `m_k` 可分 |
| cortex-like classifier（皮层样分类器） | S3 200 神经元 + decision_map |
| top-down gating（自上而下门控） | stable / shared / reserve 组路由 |

### 实验设定：Task-IL，不是 Class-IL

| 项目 | 本仓库 / continue 论文 |
| --- | --- |
| 任务顺序 | Task1 MNIST 0–9 → Task2 EMNIST 十字母 → 映射到 0–9 |
| 输出头 | 固定 **10 类、200 S3 神经元**（每类 20 个） |
| 是否「类数递增」 | **否** |
| 测试方式 | **分开测** MNIST / EMNIST 测试集（Task-IL 协议） |
| continue 推理 | **无** task memory；200 路同一套 WTA |

HTM 使模型在 **不 oracle 给 task id** 的情况下做 task inference，再触发组路由。与 continue 比：**多了内部记忆检索模块**（但记忆权重在 checkpoint 内，不是运行时外挂查表）。

### 训练（Role-train，λ）与 HTM 的关系

```text
stable:  λ ≈ 0
shared:  λ ≈ 0.3
reserve: λ = 1.0
```

SDPM 可作为突触级 λ 的一种实现（§0，**辅助对照**）。HTM 只管 **推理阶段 task 推断**，不参与 S3 STDP。

### 指标分层（读数前必读）

本仓库有 **两套准确率**，不要混用：

| 路径 | 脚本 / API | 用途 | 典型 Task1 after T2 | 典型 Task2 after T2 |
| --- | --- | --- | ---: | ---: |
| **主表** | `trainer.evaluate` | 与论文 Table 1 对齐（**R0/R1 主表**） | **47.54%**（full） | **75.43%**（full） |
| **组诊断** | `eval_partition_group_diagnosis.py` / `eval_oracle_group_routing.py` | P1/R3 机制诊断（`forward_potentials` + mask WTA） | **34.88%** natural → **55.66%** oracle | **70.44%** natural → **66.57%** oracle |

**R3 Oracle** = 训练与 catastrophic **完全相同**；测试时 **已知** task id，按组 mask WTA（Task1→mask reserve；Task2→mask stable）。**不是**论文 joint/frozen 方法，**不能**用诊断 oracle 61% 直接比论文主表 75%。


| 对比行 | 训练 | 推理 | 说明 |
| --- | --- | --- | --- |
| **R0 catastrophic** | 无保护 | natural WTA | continue 复现 |
| **R1 Role-train** | λ 按角色 | natural WTA | **公平主表**（推理与 continue 相同） |
| **R2 HTM + Role-routing** | λ 按角色 | HTM→t̂→G_{t̂} WTA | **方案 B 完整方法** |
| **R3 Oracle routing** | 同 R0（当前已跑） | 已知 q→G_q WTA | **推理路由上界**（非 Role-train） |
| **R4 Static mean prototype** | λ | 外置 mean(r) 无 HTM 模块 | 消融：证明 HTM 叙事 vs 简单均值向量 |
| SDPM-only | SDPM | natural WTA | 历史辅助（§0） |

**主表公平对比：R0 vs R1。** 机制表：**R2 vs R1**（HTM 是否带来分类增益）。

### 论文必须证明的两个指标

| 指标 | 含义 | 怎么报 |
| --- | --- | --- |
| **Acc_task** | task inference accuracy | MNIST 测试集上 t̂=1 比例；EMNIST 上 t̂=2 比例 |
| **Acc_class** | 最终分类准确率 | Task1 after T2、Task2 after T2；**须优于 R1** 才有机制价值 |

若 Acc_task 高但 Acc_class 不涨 → 记忆能分清 task 但路由无增益。  
若 Acc_class 涨但 Acc_task 低 → 可能是侥幸，需看混淆矩阵。

### 实验优先级（当前）

| 优先级 | 实验 | 状态 | 目的 |
| ---: | --- | --- | --- |
| **P1** | Task2 checkpoint，**EMNIST + mask-stable** | ✅ medium | 验证「屏蔽 stable 能否救 Task2」— **+2.50 pp**（§0.0.1） |
| **P2** | **Acc_task**：`m_1,m_2` 分离度（可先 static mean，再接 HTM） | ⬜ | 任务记忆可辨识度 |
| **P3** | **Role-train** vs R0（full，≥3 seed） | ⬜ | 公平主表 R1 |
| **P4** | **HTM + Role-routing** 端到端（R2） | ⬜ | Acc_task + Acc_class |
| **P5** | Oracle routing（R3） | ✅ medium + **full** | medium avg +7.80 pp；full avg +8.46 pp（§0.0.2–3） |
| **P6** | R4 static prototype vs R2 HTM | ⬜ | 证明「内部记忆模块」非多余包装 |
| A | Tier A 现象（§0.0） | ✅ | Step 1 动机 |

### 实现状态（代码）

| 模块 | 路径 | 状态 |
| --- | --- | --- |
| occupancy + partition | `occupancy_stats.py`, `neuron_partition.py` | ✅ Step 1 |
| role-aware inference（α / mask） | `role_aware_inference.py` | 🟡 未接入 evaluate |
| **HTM task memory** | `src/continual/` **待建**（如 `task_memory.py`） | ⬜ Step 3–4 |
| group / readout 诊断 | `readout.py`, 诊断脚本 | ✅ |
| Role-train（λ） | freeze / SDPM 等碎片能力 | 🟡 需统一配置 |

**禁止**单独维护 `prototype_bank.pt` 作为主方案交付物；`m_k` 应作为 `nn.Module` 状态写入 `model_after_task2.pt`。

### 一句话 Thesis

> WTA winner frequency 驱动输出层角色分工与 role-aware 可塑性分配；**海马体启发的 HTM 模块**将各 task 的 S3 响应模式编码为内部记忆痕迹，推理时经 **similarity-based retrieval（相似度检索）** 完成 task inference 并 **top-down 路由** stable/shared/reserve，使训练分工在测试阶段生效。

### 不能 claim（截至当前证据）

- HTM **自动** task inference + routing **已验证**端到端提升 Acc_class（P4 未跑）
- Oracle **已跑**（medium + full）：见 §0.0.2–3；**不等于 Task1 已保护**（主表仍 ~47%）
- SDPM alone 为核心创新或全面 SOTA
- reserve / Phased / full NGSG 有效
- Class-IL 混合测试集结果
- 「海马体」生物同源性（仅 **inspired by**，非神经科学 claim）

---
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
5. **工程含义（方案 B + HTM）** — stable 承载 Task1；reserve 承载 Task2；推理须 **按 task 路由神经元组**。**P1 已验证**：EMNIST 上 mask-stable（= shared∪reserve）**+2.50 pp**（§0.0.1）。

**仍缺：** P0-3 seed 1/2、**P0-4 full Task1**（主文硬门槛）。详见 实验列表.md。

## 0.0.1 Tier A′ — Task2/EMNIST 组诊断（P1，2026-07-08）✅

**协议：** `readout_ablation_medium_seed0` 的 `model_after_task2.pt`；partition 来自 Task1 后 `f_i`（stable 61 / shared 63 / reserve 76）；**EMNIST 测试集**（1000 样本）；只改推理 group-only / group-masked，不改权重。

**结果文件：** `experiments/readout_ablation_medium_seed0/diagnostics/p1_task2_emnist_group_diagnosis.json`

### 6.1 Group-only（Task2/EMNIST）

| 条件 | 神经元数 | EMNIST Acc | vs all-200 |
| --- | ---: | ---: | ---: |
| **all-200（baseline）** | 200 | **54.30%** | — |
| stable-only | 61 | 12.00% | -42.30 pp |
| shared-only | 63 | 28.10% | -26.20 pp |
| **reserve-only** | 76 | **53.40%** | **-0.90 pp** |

测试时 natural WTA 的 winner：**reserve 64.8%**，shared **23.2%**，stable **12.0%**。

### 6.2 Group-masked（屏蔽某一组再推理 Task2）

| 条件 | 屏蔽谁 | EMNIST Acc | vs all-200 |
| --- | ---: | ---: | ---: |
| **mask-stable** | stable（61） | **56.80%** | **+2.50 pp** |
| mask-shared | shared（63） | 49.40% | -4.90 pp |
| **mask-reserve** | reserve（76） | **28.60%** | **-25.70 pp** |

embedded Task1 acc（同 checkpoint）：**77.9%**。

### 与 §0.0（Task1/MNIST）的对称关系

| 维度 | Task1 / MNIST（§0.0） | Task2 / EMNIST（P1） |
| --- | --- | --- |
| 承载组 | **stable-only ≈ all-200**（77.5%） | **reserve-only ≈ all-200**（53.4%） |
| 自然 WTA winner 主导组 | stable **98.4%** | reserve **64.8%** |
| 屏蔽承载组 | mask-stable → **-58.2 pp** | mask-reserve → **-25.7 pp** |
| 屏蔽「干扰组」 | mask-reserve → -0.1 pp | **mask-stable → +2.5 pp** |
| oracle 路由含义 | Task1 → stable∪shared | Task2 → shared∪reserve |

### P1 结论

1. **Task2 知识主要在 reserve** — reserve-only 仅比 all-200 低 0.9 pp；与 MNIST 上 stable-only ≈ all-200 **对称**。
2. **stable 在 EMNIST 上是干扰源** — 虽只占 12% winner，mask-stable 仍 **+2.5 pp**；stable-only 仅 12%。
3. **train–test mismatch 的推理侧证据成立** — Task1 应用 stable∪shared，Task2 应用 shared∪reserve；用错组（mask-reserve on EMNIST）跌至 28.6%。
4. **HTM + group routing 有诊断级上界** — oracle mask-stable（56.8%）接近 §0.6 task-aware class-max（57.6%），机制不同、增益同量级。
5. **尚未 claim** — 这是 **oracle 组 mask**，不是 HTM 自动推断 t̂；Acc_task（P2）与 Role-train 公平主表（P3）仍待跑。

**Caveat：** medium checkpoint（EMNIST 54.3% baseline）；full scale 与 multi-seed 待复验。NGSG reserve **训练 reroute** 实验 Task2 仅 33–36%（§0.3.1）与本诊断 **不同 checkpoint/训练协议** — 本结果说明「reserve 神经元已学到 EMNIST」，瓶颈在 **自然 WTA 未按 task 选组**。

## 0.0.2 R3 — Oracle group routing，medium（2026-07-08）✅

**训练：** 与 catastrophic 相同（`readout_ablation_medium_seed0`）  
**脚本：** `scripts/eval_oracle_group_routing.py`  
**精简结果：** `published_results/diagnostics/r3_oracle_group_routing_medium_seed0.json`

| metric | natural all-200 | oracle routing | delta |
| --- | ---: | ---: | ---: |
| Task1 / MNIST | 55.60% | **68.70%** (mask_reserve) | **+13.10 pp** |
| Task2 / EMNIST | 54.30% | **56.80%** (mask_stable) | **+2.50 pp** |
| **Avg** | **54.95%** | **62.75%** | **+7.80 pp** |

Task2 行与 P1（§0.0.1）一致；Task1 侧 reserve winner 占 25.6%。

## 0.0.3 R0 full + R3 full — `paper_full_partition_seed0`（2026-07-08 服务器）✅

**配置：** `configs/ngsg/paper_full_partition_seed0.yaml`（600/100 epoch，2400/class，partition on，pure catastrophic，无 SDPM/reserve，`save_task2_model: true`）  
**Run：** `paper_full_partition_seed0` · **服务器 commit：** `761d6f3`  
**精简主表：** `published_results/baseline/paper_full_partition_seed0.json`  
**本机镜像（不进 git）：** `experiments/server_paper_full_partition_seed0/`

### 主表（trainer.evaluate，与论文对齐 = **R0**）

| 指标 | paper_full_partition_seed0 | 论文参考 | §0.1 历史 full |
| --- | ---: | ---: | ---: |
| Task1 after Task1 | **93.14%** | 90.8 ± 0.9% | 92.84% |
| Task1 after Task2 | **47.54%** | 48.1 ± 4.8% | 48.42% |
| Task2 after Task2 | **75.43%** | 78.4 ± 1.2% | 74.91% |
| Forgetting | **45.6 pp** | ~42.7% | 44.4 pp |
| Avg Acc | **61.48%** | — | 61.67% |

**结论：** full catastrophic **复现成功**；Task1 **未被保护**（93%→47%），这是要解决的遗忘问题本身。

**Partition（Task1 后）：** stable **60** / shared **60** / reserve **80** / dead **0**

### P1 — Task2/EMNIST 组诊断（full）

**精简结果：** `published_results/diagnostics/p1_task2_emnist_group_diagnosis_full_seed0.json`

| 条件 | EMNIST Acc | vs all-200 |
| --- | ---: | ---: |
| all-200 natural WTA | **70.44%** | — |
| reserve-only | 57.14% | -13.30 pp |
| **mask-stable**（shared∪reserve） | **66.57%** | **-3.86 pp** |
| mask-reserve | 30.56% | -39.88 pp |

natural WTA winner：reserve **72.6%**，stable **8.9%**。

**与 medium P1 对比：** medium 上 mask-stable **+2.50 pp**；**full 上 -3.86 pp** — scale 不同，路由规则不能照搬。

### R3 — Oracle group routing（full，诊断路径）

**精简结果：** `published_results/diagnostics/r3_oracle_group_routing_full_seed0.json`

| metric | natural all-200 | oracle routing | delta |
| --- | ---: | ---: | ---: |
| Task1 / MNIST | 34.88% | **55.66%** (mask_reserve) | **+20.78 pp** |
| Task2 / EMNIST | 70.44% | **66.57%** (mask_stable) | **-3.86 pp** |
| **Avg** | **52.66%** | **61.12%** | **+8.46 pp** |

**如何读 R3（避免误判）：**

1. **+20.78 pp（Task1）** = 在**已遗忘权重**上，屏蔽 reserve 抢答还能**相对**多捞分；**不是** Task1 保护成功（主表仍 47.54%，oracle 绝对值 55.66% 仍很低）。
2. **诊断 stable-only（Task1）= 63.75%** > mask_reserve 55.66% → 当前 Oracle 规则（stable∪shared）**不是 Task1 最优**；shared 在拖后腿。
3. **Task2 oracle 略亏** → full 上「mask stable」不能作为无脑默认；需配合 **R1 Role-train** 或调路由规则。
4. **R3 平均 +8.46 pp** 是「路由机制 headroom」，**不是**可部署方法；**HTM（R2）** 目标是闭卷逼近此上界。

**诊断命令（full checkpoint）：**

```bash
python scripts/eval_partition_group_diagnosis.py \
  --run-dir experiments/paper_full_partition_seed0 \
  --checkpoint-stage task2 --test-task task2 --device cuda \
  --write-json experiments/paper_full_partition_seed0/diagnostics/p1_task2_emnist_group_diagnosis.json

python scripts/eval_oracle_group_routing.py \
  --run-dir experiments/paper_full_partition_seed0 \
  --checkpoint-stage task2 --device auto \
  --write-json experiments/paper_full_partition_seed0/diagnostics/r3_oracle_group_routing.json
```

## 0. 历史：SDPM-only full 结果（2026-07-06）

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

结论：full 规模下 SDPM-only 缓解遗忘（Task1 after Task2 +9.03 pp），可作为 **Role-train 中 λ 实现的一种对照**；**不足以单独作为方案 B 主结论**。

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

## 0.2 当前推进状态（2026-07-09）

**主线：方案 B + HTM（Role-Aware CL + Hippocampus-inspired Task Memory）**，见文首「当前主线」一节。

| 模块 | 状态 | 在方案 B 中的位置 |
| --- | --- | --- |
| **Tier A 现象诊断** | ✅ medium PASS（§0.0） | Step 1 动机：stable 承载 Task1 |
| **R0 catastrophic full** | ✅ `paper_full_partition_seed0`（§0.0.3） | 主表对齐论文 |
| **P1 Task2 EMNIST mask-stable** | ✅ medium +2.50 pp；full **-3.86 pp**（§0.0.1/3） | 路由规则 scale 敏感 |
| **R3 Oracle routing** | ✅ medium +7.80 pp avg；full +8.46 pp avg（§0.0.2/3） | 推理 headroom；非保护成功 |
| **train–test mismatch 诊断** | ✅ reserve 38% STDP→reserve（§0.3.1） | 训练侧问题定义 |
| **SDPM-only full** | ✅ +9.03 pp Task1 retention（§0） | 辅助对照 |
| **Role-train / HTM / R2** | ⬜ 未闭环 | **方案 B 主实现** |
| **Phased Task2** | ❌ medium FAIL（§0.9） | 历史 |
| **reserve / full NGSG** | ❌ Task2 掉点（§0.3.1） | 历史 |

历史进度：partition 组诊断 + P0 反事实 PASS；full SDPM-only；medium 消融 #1–#7；Phased 首跑 FAIL。

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
| 7 | **Phased Task2** | `phased_task2_medium_seed0` | off | off（phased） | ✅ **FAIL** | 77.9% | **11.0%** | **40.9%** | 66.9 pp | 25.95% |

第 7 行详见 §0.9；与 reserve 无关，启用 `continual.phased_task2`。

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
4. **论文叙事（已更新为方案 B + HTM）：** 主方法为 **role-aware training + HTM task memory retrieval + group routing**；SDPM / reserve / Phased 为历史或辅助路线（见文首「当前主线」）。

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

**协议：** Task1 训练完成后立即在 **MNIST 测试集**上评估；**未训练 Task2**。每类 100 train/test，S3 50 epoch。推理使用 `decision_map` + global WTA，仅在指定神经元子集内竞争（group-only / group-masked）。

**Task2/EMNIST 对称诊断（P1）** 见 §0.0.1（`model_after_task2.pt`，mask-stable **+2.50 pp**）。

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

目的：比较标准 WTA + `decision_map` 与基于 `winner_label_counts` 的 **task-aware class-max 读出**（不改训练，只改推理；**不是** role-mask WTA，也 **不是** HTM 记忆路由）。

协议：medium SDPM+reserve（同 full NGSG 结构），Task2 结束后评估。checkpoint：`model_after_task2.pt`。

| 测试集 | 标准 WTA + decision_map | task-aware class-max | 差值 |
| --- | ---: | ---: | ---: |
| MNIST after Task2 | 55.6% | **59.7%**（Task1 dominant label） | **+4.1 pp** |
| EMNIST after Task2 | 54.3% | **57.6%**（Task2 dominant label） | **+3.3 pp** |

注意：必须用 **分任务 label 表**（MNIST 用 Task1 统计，EMNIST 用 Task2 统计）；混用 Task1 label 测 EMNIST 会跌至 16.5%。

**与方案 B 的关系：** 证明「换推理规则能涨分」，但是 **换 label 映射 / class-max**，不是 **按角色 mask WTA**。P1（§0.0.1）已验证 **mask-stable on EMNIST +2.50 pp**；与 class-max +3.3 pp 同量级、机制不同。

运行：

```bash
python scripts/eval_readout_ablation.py --run-dir experiments/readout_ablation_medium_seed0 --device cuda
```

精简结果：`published_results/diagnostics/readout_ablation_medium_seed0.json`。

## 0.7 论文叙事（方案 B + HTM，2026-07-08）

完整 **Claim vs 实验对照表** 见 [`实验列表.md`](实验列表.md)（**待与 HTM 叙事同步**）。

### 一句话 Thesis

> Task1 的 WTA winner frequency 估计输出层占用；Task2 在角色约束下分配可塑性；**海马体启发的 HTM 模块**将各 task 的 S3 响应模式编码为内部记忆痕迹，推理时经 **similarity-based retrieval** 完成 task inference 并 **top-down 路由** stable/shared/reserve，使训练分工在测试阶段生效。

### 证据链（方案 B + HTM）

```text
现象层  →  stable 几乎单独承载 Task1 推理（§0.0）
问题层  →  train–test mismatch：训练写 reserve，测试 stable 赢（§0.3.1）
方法层  →  Role-train（λ）+ HTM memory retrieval + group routing
对照层  →  R0 catastrophic vs R1 Role-train（公平）；R2 HTM vs R1；R3 oracle 上界
```

### 可以 claim（需实验支撑）

| 层级 | 论点 | 支撑 | 状态 |
| --- | --- | --- | --- |
| **现象** | 高频 winner 子集承载 Task1 | §0.0 P0 | ✅ medium |
| **问题** | 训练分工与推理 WTA 不一致损害 Task2 | §0.3.1 reserve 统计 | ✅ 训练侧 |
| **方法** | Role-train 缓解遗忘且推理协议与 continue 一致 | R1 vs R0 full | ⬜ |
| **机制** | HTM 可推断 task（Acc_task）；group routing 提升分类（Acc_class） | P2–P6；P1 oracle mask **+2.5 pp** ✅ | 🟡 P1 ✅；P2–P4 ⬜ |
| **辅助** | SDPM soft protection 缓解遗忘 | §0 full +9 pp | ✅ 但非主叙事 |

### 不能 claim

- SDPM 作为唯一/核心创新已全面优于市面方法
- reserve / full NGSG / Phased Task2 已验证有效
- 三区划分算法本身优于 f_i-only（frequency-only = WTA）
- Class-IL 混合测试无 task 信息下的结果
- HTM 自动 task inference **已验证**端到端（P4 未跑）；P1 仅验证 **oracle mask-stable** 诊断增益
- 「海马体」生物同源性（仅 **hippocampus-inspired**，非神经科学 claim）

### 历史叙事（SDPM 线，保留数字、降级为支撑）

SDPM-only full：Task1 after Task2 **+9.03 pp** vs no-SDPM（§0）。可作为 Role-train 中 λ 的一种实现对照，**不足以单独支撑主论文**（`gate_mean≈0.97`，绝对 retention 仍仅 57%）。

## 0.8 实验路线图（方案 B + HTM，2026-07-08）

排序原则：**P1 诊断 → Acc_task（HTM 分离度）→ Role-train 公平主表 → HTM + routing 闭环 → Tier A 收尾 → 历史 SDPM/Phased 归档**。

### 方案 B 主实验（当前优先级）

| ID | 实验 | 规模 | 状态 | 说明 |
| ---: | --- | --- | --- | --- |
| **P1** | Task2 checkpoint，**EMNIST 测试 + mask-stable** | medium | ✅ | mask-stable **+2.50 pp**（§0.0.1）；reserve-only ≈ all-200 |
| **P2** | HTM 记忆分离度（`m_1,m_2`；可先 static mean） | medium | ⬜ | 报 **Acc_task** / 混淆矩阵 |
| **P3** | **Role-train** vs R0 catastrophic | full, ≥3 seed | ⬜ | **与 continue 公平主表** |
| **P4** | **HTM + group routing** 端到端（R2） | full | ⬜ | **Acc_task + Acc_class** |
| **P5** | Oracle routing（R3） | medium + full | ✅ | medium +7.80 pp；full +8.46 pp（§0.0.2–3） |
| **P6** | R4 static mean prototype vs R2 HTM | medium | ⬜ | 消融：内部记忆模块 vs 外置均值 |

**诊断命令（P1，在已有 Task2 checkpoint 上）：**

```bash
# 需扩展脚本：--checkpoint-stage task2 --test-task emnist --group-mask stable
# 当前 eval_partition_group_diagnosis.py 默认只测 MNIST（task_bundles[0]）
python scripts/eval_partition_group_diagnosis.py \
  --run-dir experiments/readout_ablation_medium_seed0 \
  --checkpoint-stage task2 --device cuda
```

### Tier A — 现象层（已完成 medium，支撑 Step 1）

- stable-only ≈ all-200（**77.50%**）✅
- mask-stable on **Task1/MNIST** → **19.30%** ✅（§0.0）
- mask-stable on **Task2/EMNIST** → **56.80%**（+2.50 pp）✅（§0.0.1，P1）
- random / shuffle 反事实 PASS ✅
- **仍缺：** P0-4 full Task1、P0-3 multi-seed

### 历史路线 — SDPM / Phased / reserve（归档，非主表）

| 路线 | 代表结果 | 状态 |
| --- | --- | --- |
| SDPM-only full | Task1 after T2 +9.03 pp | ✅ 辅助 |
| Tier B hard-freeze vs SDPM | — | ⬜ SDPM 线对照 |
| Phased Task2 medium | Task1 after T2 **11%** | ❌ §0.9 |
| reserve / full NGSG medium | Task2 **33–36%** | ❌ §0.3.1 |

### Tier A 后验命令（不变）

```bash
python scripts/eval_partition_group_diagnosis.py \
  --run-dir experiments/partition_group_diagnosis_medium_seed0 \
  --counterfactuals --device cuda \
  --write-json experiments/partition_group_diagnosis_medium_seed0/diagnostics/partition_counterfactuals.json

python scripts/eval_partition_group_diagnosis.py --config configs/ngsg/partition_group_diagnosis_medium_seed1.yaml --train-task1 --counterfactuals --device cuda
python scripts/eval_partition_group_diagnosis.py --config configs/ngsg/partition_group_diagnosis_medium_seed2.yaml --train-task1 --counterfactuals --device cuda
python scripts/eval_partition_group_diagnosis.py --config configs/ngsg/partition_group_diagnosis_full.yaml --train-task1 --counterfactuals --device cuda
```

## 0.9 Phased Task2 三阶段方案（2026-07-07）— **历史尝试，medium FAIL**

> **非当前主线。** 方案 B 改为「Role-train + HTM task memory + group routing」；本节保留失败经验与数字。

**目标（当时）：** Task1 记忆留在 stable ~61；Task2 在 **reset 池（shared+reserve，~139）** 上学习 EMNIST；stable 在 Task2 训练时不「抢答」、权重不被覆盖；联合阶段允许 200 路 WTA 但 stable 赢则 reroute STDP。

**架构前提（与 continue 一致）：**

- **S1/S2 共用**（一套 conv1/conv2），所有 200 个 S3 neuron 看到同一份 C2 特征；不是 per-neuron 独立通路。
- **S3**：200 neuron，各 conv3 权重独立；**k=1 global WTA**；仅 winner 做 R-STDP。
- Task2 训练可走 **C2 cache**（`forward_from_s3_input`）；推理为 **S1→S2→S3 全链路 200 路 WTA**（与 continue 相同，无 partition mask）。

### 三阶段协议

| 阶段 | 时机 | 行为 |
| --- | --- | --- |
| **Task1 结束** | partition 拟合 | 用 winner frequency + q_i 标 stable ~61 / shared ~59 / reserve ~80 |
| **Task2 Phase1（isolated）** | 前 `phase1_epochs` | WTA **mask stable**；仅 reset 池（~139）参与竞争与 STDP |
| **Task2 Phase2（joint）** | 后 `phase2_epochs` | 200 路 WTA；**freeze stable conv3**；stable 赢 → STDP **reroute** 到 reset 池 argmax-potential neuron |
| **Reset 时机** | `task1_after_task1` 评估**之后**、Task2 训练**之前** | 将 non-stable conv3 重置为 Task1 初值（`reset_non_stable_conv3: true`） |

**与 reserve_activation 互斥**；启用 `phased_task2` 时必须 `neuron_partition.enabled: true`。

### 代码与配置

| 路径 | 说明 |
| --- | --- |
| `src/continual/phased_task2.py` | `PhasedTask2Controller`：reset、WTA mask、stable-win reroute、joint 阶段 freeze |
| `src/models/paper_mozafari.py` | `s3_wta_allow_mask`（仅训练时 mask potential） |
| `src/trainers/baseline_trainer.py` | 集成三阶段调度；保存 init conv3；reset 在 Task1 eval 后执行 |
| `configs/baseline/catastrophic_mnist_emnist_paper_medium_phased_task2.yaml` | medium：phase1=8，phase2=2，Task2 S3=10 |
| `configs/baseline/catastrophic_mnist_emnist_phased_task2.yaml` | full：phase1=80，phase2=20，Task2 S3=100 |
| `tests/test_phased_task2.py` | schedule / reset 单测 |

运行（medium）：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium_phased_task2.yaml --device cuda --run-name phased_task2_medium_seed0
```

本机产物（不进 git）：`experiments/phased_task2_medium_seed0/result.json`。

### medium seed0 结果（`phased_task2_medium_seed0`，修复 reset 时序后）

| 指标 | Phased Task2 | no-op baseline medium |
| --- | ---: | ---: |
| Task1 after Task1 | **77.9%** | 77.2% |
| Task1 after Task2（200 路 WTA） | **11.0%** | 69.8% |
| Task2 after Task2（200 路 WTA） | **40.9%** | 58.7% |

**Task2 训练诊断：**

| 指标 | Phase1 isolated（8 ep） | Phase2 joint（2 ep） |
| --- | --- | --- |
| stable_win_frac | **0%**（mask 生效） | 0.4% / 0.1% |
| `winner_active` | **1/200**（全程） | — |
| `max_winner_fraction` | **1.0** | — |
| train acc proxy | **~10%**（单 neuron 垄断 → 单类 decision_map） | — |

测试时 winner 角色：**shared+reset 池 100% 赢**；stable **0%**（权重虽冻住，200 路推理几乎用不上 stable）。

**根因：** 139 个 non-stable conv3 重置到相近初值 + S3 k=1 WTA + winner-only STDP → **rich-get-richer 独赢**；共用 S1/S2 特征不能解决 S3 层 credit assignment。Phase2 reroute 样本过少，无法挽回。

### 当前共识（2026-07-07 讨论）

1. **freeze stable 肯定有用（定义上）** — 防止 stable conv3 被 Task2 STDP 覆盖；**不必单独做「freeze 有没有用」实验**。
2. **freeze 不充分** — stable-only 测 Task1 仍有 ~77%，但 200 路 WTA 测 Task1 可崩至 ~11%；问题在 **推理时谁赢 WTA**，不在权重是否被改。
3. **只 reset reserve（不 reset shared）也不会自动好** — S1/S2 共用下，Task1 已训过的 **shared potential 通常高于 cold-start reserve**；默认 WTA 下 Task2 更新会偏向 shared，reserve 仍饿肚子（与 §0.3.1 reserve reroute 同类问题，失败模式不同）。
4. **139 全 reset vs 只 reset reserve** — 前者易 **init neuron 垄断**（proxy ~10%）；后者易 **shared 垄断**（Task2 可能略能学，reserve 仍废）。二者都是 **赢者通吃**，需额外机制：**mask shared、reroute、homeostatic boost、per-class WTA** 等。
5. **full 80+20** — Phase1 已见独赢；不加竞争多样性机制时，加长 epoch **可能更糟**；先修 Phase1 `winner_active` 再跑 full。

### 建议下一步（Phased 线 — 已搁置，见方案 B P1–P5）

| 优先级 | 方向 | 说明 |
| ---: | --- | --- |
| — | 仅训练侧 mask 不够 | 须配合推理 group routing 或 HTM（方案 B） |
| — | Task2 测 EMNIST + mask-stable | 已提升为文首 **P1** |

## 1. 当前目标（方案 B + HTM）

1. ~~**P1**~~：✅ Task2 checkpoint + **EMNIST + mask-stable**（§0.0.1，+2.50 pp）。
2. **实现 HTM 模块**：`m_k` 记忆神经元、Task 训完后 EMA 写入、`cos(r(x), m_k)` 推断 t̂；`src/continual/task_memory.py`（待建）。
3. **Role-train 配置**：stable freeze + shared/reserve 分工；与 **R0 catastrophic** full 对比（公平主表）。
4. **接入 group routing**：HTM→t̂→G_{t̂} mask WTA + `role_aware_inference.py` + trainer `evaluate`。
5. **HTM + routing 端到端（R2）**；对照 **oracle routing（R3）** 上界与 **R4 static mean** 消融。
6. 收尾 **Tier A**：P0-4 full + P0-3 multi-seed。
7. **SDPM / Phased / reserve** 保留为历史实验与辅助对照，见 §0 / §0.3 / §0.9。

当前不复现 joint training。Langevin / frozen 对照需要时可单独重建（continue Table 1 其他行）。

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
| `catastrophic_mnist_emnist_paper_medium_phased_task2.yaml` | **Phased Task2 medium** | phase1=8，phase2=2；reset non-stable conv3；与 reserve 互斥。 |
| `catastrophic_mnist_emnist_phased_task2.yaml` | **Phased Task2 full** | phase1=80，phase2=20；修 Phase1 竞争后再跑。 |

### `configs/ngsg/`（`8a9e78a` 新增）

| 配置 | SDPM | reserve | 用途 |
| --- | --- | --- | --- |
| `catastrophic_mnist_emnist_paper_medium_reserve_only.yaml` | off | on | 只验证 reserve 招募 |
| `catastrophic_mnist_emnist_paper_medium_ngsg.yaml` | on | on | full NGSG medium |
| `catastrophic_mnist_emnist_paper_medium_random_reserve.yaml` | off | random | reserve 随机对照 |
| `partition_group_diagnosis_medium.yaml` | off | off | Task1 后 partition 组诊断；`save_task1_model: true` |
| `paper_full_partition_seed0.yaml` | off | off | **R0 full**（600/100）；partition；`save_task2_model: true`；P1/R3 |
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

Phased Task2 medium（§0.9）：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium_phased_task2.yaml --device cuda --run-name phased_task2_medium_seed0
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

### 6.1 S3「电压 / 电位」如何产生（与 WTA 的关系）

SpykeTorch `snn.Convolution` 对每个时间步独立做卷积（**不是**生物式跨步膜电位泄漏积分）：

```text
输入：脉冲序列 x[t]，t = 1..T（T=15，Intensity2Latency 编码）
每层：pot[t] = conv2d(x[t], W)     # 感受野内：u = Σ w·spike
S3：  raw_pot[t,i,h,w] 为第 i 个神经元在时刻 t、位置 (h,w) 的电位
```

**分类决策：**

```text
wta_pot = raw_pot（经可选 mask/boost）
winners = get_k_winners(wta_pot, k=1)   # 全图 200 神经元竞争
        → 优先最早 spike 时间，再比电位大小
class   = decision_map[winner_neuron_index]
```

**诊断用标量分数**（`readout.aggregate_neuron_scores`）：对每个神经元取空间 max，用于 group-only / HTM 响应向量 `r(x)`。

**方案 B + HTM：** `r_i(x) = max_{t,h,w} raw_pot_i` 作为 HTM 输入；group routing 在 WTA 前对 `raw_pot` 按角色 mask 或乘 α。

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

阶段 C：方案 B + HTM 与历史 NGSG 路线。当前状态：**HTM 主实现进行中**；SDPM / Phased / reserve 已有结果见 §0 / §0.3 / §0.9。

- ✅ Step 1：partition + Tier A 现象（§0.0）
- ✅ train–test mismatch 问题刻画（§0.3.1）
- ⬜ Step 2：Role-train（λ）统一配置与 full 主表
- ⬜ Step 3–4：HTM task memory + group routing
- ✅ 历史：SDPM（§0）、Phased（§0.9 FAIL）、reserve（§0.3.1 FAIL）

## 8. NGSG / 方案 B + HTM 设计共识

创新点不在重写 SpykeTorch，而是在已复现的 Mozafari SNN 上：**把 WTA 竞争历史变成神经元角色，训练与推理成对地按角色分配可塑性 / 竞争权限；task 由内部 HTM 记忆检索推断，而非 oracle 或外置查表。**

一句话（方案 B + HTM）：

> Task1 winner frequency → stable/shared/reserve 角色 → Task2 role-aware training（λ）→ HTM 记忆检索推断 task → group routing WTA → 缓解 train–test mismatch。

核心模块（`src/continual/`）：

| 模块 | 文件 | 方案 B 角色 | 状态 |
| --- | --- | --- | --- |
| occupancy 统计 | `occupancy_stats.py` | Step 1 输入 | ✅ |
| neuron partition | `neuron_partition.py` | Step 1 角色划分 | ✅ |
| **HTM task memory** | `task_memory.py`（待建） | Step 3 记忆形成与检索 | ⬜ |
| **group routing** | `role_aware_inference.py` | Step 4 按 t̂ mask WTA | 🟡 未接入 evaluate |
| task-aware readout | `readout.py` | 诊断 / 与 HTM 不同轴 | ✅ |
| SDPM gate | `sdpm_gate.py` | 可选 λ 实现 | ✅ 历史 |
| reserve activation | `reserve_activation.py` | 历史；train–test mismatch 例 | ⚠️ |
| Phased Task2 | `phased_task2.py` | 历史 FAIL | ⚠️ |

诊断脚本：

| 脚本 | 用途 |
| --- | --- |
| `scripts/eval_partition_group_diagnosis.py` | P1 group-only / group-masked（`--test-task task1|task2`） |
| `scripts/eval_oracle_group_routing.py` | R3 oracle：Task1 mask_reserve / Task2 mask_stable |
| `scripts/eval_readout_ablation.py` | class-max readout 消融 |
| `scripts/diagnose_test_winner_roles.py` | test-time winner 角色分布 |

**当前论文可写边界：** Tier A 现象 ✅；train–test mismatch 机制叙事 ✅；**HTM 端到端数字 ⬜**；SDPM +9 pp 作辅助；Phased/reserve/full NGSG 不作主 claim。

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

## 12. 下一步优先级（方案 B + HTM）

详见文首「实验优先级」与 §0.8。

**方案 B + HTM 主线**

1. ✅ **P1**：Task2 checkpoint + **EMNIST + mask-stable**（§0.0.1，+2.50 pp）。
2. 🔴 **P2**：实现 HTM 记忆神经元 + **Acc_task**（可先 static mean 验证分离度）。
3. 🔴 **P3**：Role-train full vs catastrophic（R0/R1 公平主表，≥3 seed）。
4. ⬜ **P4**：HTM + group routing 端到端（R2，Acc_task + Acc_class）。
5. ✅ **P5**：Oracle routing（R3，medium + full，§0.0.2–3）。
6. ⬜ **P6**：R4 static mean vs R2 HTM 消融。

**收尾**

7. 🟡 Tier A：P0-4 full、P0-3 multi-seed。
8. 维护 `published_results/`；大产物在 `experiments/`（不进 git）。

**历史路线（非阻塞）**

9. SDPM Tier B hard-freeze / random protection（SDPM 线论文对照）。
10. Phased / reserve 归档，不投入 full 除非方案 B 失败。
