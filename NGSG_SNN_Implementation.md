# NGSG-SNN 实现设计文档

> Novelty-Guided Structural Growth Spiking Neural Network  
> 基于 SpykeTorch 的持续学习脉冲神经网络  
> 共同认知版，最后更新：2026-06-29

---

## 0. 当前共同认知

本方法的创新点不是重写整个 SpykeTorch 网络，而是在已复现的 Antonov / Mozafari 风格三层 SNN 基础上，只在 S3 输出层加入持续学习机制：

1. **旧知识识别**：Task 1 后统计 S3 神经元的 winner frequency 和 class selectivity，得到神经元级与突触级重要性。
2. **旧知识软保护**：用 SDPM 对重要突触的 R-STDP 更新幅度做门控，尽量减少 Task 2+ 对旧任务关键连接的破坏。
3. **新知识容量调用**：用 novelty score 判断当前样本是否对旧网络“新颖”，优先调用低使用率、低旧任务重要性、对当前输入有响应潜力的 reserve neurons。
4. **局部学习约束**：NGSG 不依赖反向传播，仍沿用 SpykeTorch / R-STDP 的局部学习框架。

第一版实现必须保持保守：先不声称已经实现物理意义上的动态新增参数，而是实现 **预留容量神经元与低可塑性连接的 novelty-guided activation**。如果后续确实实现稀疏 mask 或动态扩容，再在论文中升级表述为 structural growth。

---

## 1. 整体架构

网络主干沿用当前仓库中正在复现的 paper-source SpykeTorch 路线。基础结构为：

```text
Input Image
  -> DoG filtering + Intensity-to-Latency spike encoding
  -> S1: STDP convolutional spiking layer
  -> C1: spike pooling
  -> S2: STDP convolutional spiking layer
  -> C2: spike pooling
  -> S3-NGSG: continual-learning core
       - Winner-Frequency Tracker
       - Stable / Shared / Reserve neuron partition
       - Synaptic importance estimation
       - SDPM plasticity gate
       - Novelty detector
       - Reserve neuron activation
  -> Readout by S3 winner / membrane potential
  -> Prediction
```

### 1.1 关于 S1/S2 是否冻结

这里需要区分 baseline 复现和 NGSG 实验：

- **baseline 复现阶段**：按原论文 / 作者源码协议执行，Task 2 仍可训练 S1/S2，用于对齐 catastrophic forgetting baseline。
- **NGSG 主方法阶段**：优先采用 S1/S2 在 Task 1 后冻结的设置，使创新点集中在 S3，避免把性能变化混入低层特征重学习。
- **必要消融**：需要保留 `NGSG with S1/S2 retraining` 对照，用来说明冻结低层特征不是唯一性能来源。

因此，文中应表述为：NGSG 的持续学习机制作用于 S3；S1/S2 是否冻结是实验协议变量，不应和核心创新混为一谈。

---

## 2. Task 1 训练与统计

Task 1 按当前 paper-source R-STDP 路线正常训练。S3 的所有神经元正常参与 WTA，不预先分组或屏蔽。

训练过程中记录每个 S3 神经元的获胜统计：

$$
f_i = \frac{n_i^{win}}{N_{update}}
$$

- $n_i^{win}$：第 $i$ 个 S3 神经元在 Task 1 S3 训练期间的获胜次数。
- $N_{update}$：Task 1 S3 R-STDP 的总更新样本数；若多 epoch 训练，则为样本数乘以 epoch 数。
- $f_i$：第 $i$ 个神经元的 winner frequency。

同时记录该神经元获胜时的标签分布，用于计算 class selectivity。若某神经元从未获胜，则其 selectivity 记为 0，避免除零。

---

## 3. Task 1 后的神经元分区

Task 1 结束后，根据 S3 winner frequency 将神经元分为三组：

$$
\text{Stable}: f_i > \theta_{stable}
$$

$$
\text{Shared}: \theta_{reserve} < f_i \leq \theta_{stable}
$$

$$
\text{Reserve}: f_i \leq \theta_{reserve}
$$

初始比例建议为：

- Stable：winner frequency 最高 30%
- Shared：中间 40%
- Reserve：winner frequency 最低 30%

对应阈值：

$$
\theta_{stable} = P_{70}(f), \quad \theta_{reserve} = P_{30}(f)
$$

如果后续实验发现 reserve 容量不足，可把比例调整为 25/45/30 或 20/50/30，但每次调整必须记录在实验配置中。

---

## 4. 重要性计算

### 4.1 神经元级重要性

类别选择性定义为：

$$
q_i = 1 - \frac{H_i}{\log C}
$$

$$
H_i = -\sum_{c=1}^{C} p_i(c)\log p_i(c)
$$

- $p_i(c)$：神经元 $i$ 获胜时对应类别为 $c$ 的比例。
- $C$：Task 1 类别数。
- $q_i = 1$：该神经元高度专门响应某一类。
- $q_i = 0$：该神经元对各类响应接近均匀，或从未稳定获胜。

神经元级重要性：

$$
I_i = \text{Norm}(f_i) \cdot q_i
$$

其中 `Norm` 使用 min-max normalization，并加入 $\epsilon$ 防止分母为 0。

### 4.2 突触级重要性

$$
I_{ij} = I_i \cdot \text{Norm}(w_{ij} - w_{min})
$$

- $w_{ij}$：S3 神经元 $i$ 对输入位置 / 通道 $j$ 的权重。
- $w_{min}$：权重下界。

直觉：旧任务中经常获胜、类别选择性强、且权重较大的连接更重要，应在后续任务中被更强保护。

---

## 5. SDPM：可塑性门控机制

SDPM（Spike-Distribution-aware Plasticity Modulation）不改变 R-STDP 的奖励 / 惩罚方向，只缩放实际权重更新幅度。

### 5.1 Gate 定义

$$
g_{ij} = \exp(-\lambda I_{ij})
$$

- $\lambda$：保护强度，初始值建议为 3。
- $I_{ij}\to0$：$g_{ij}\to1$，接近正常更新。
- $I_{ij}\to1$：$g_{ij}\to e^{-\lambda}$，更新被强烈压缩。

### 5.2 当前代码中的可靠实现方式

当前 paper-source 路线中，S3 更新由 SpykeTorch 的 `stdp3` / `anti_stdp3` 直接修改权重。因此 SDPM 不能假设代码里天然暴露了 $\Delta w_{ij}^{raw}$。

第一版可靠实现应采用 delta-replay 方式：

```python
w_before = conv3.weight.clone()

# 调用原始 reward 或 punish，得到 SpykeTorch 官方 R-STDP 行为
model.reward()  # or model.punish()

w_after = conv3.weight
delta_raw = w_after - w_before

delta_gated = gate * delta_raw
conv3.weight.copy_(clamp(w_before + delta_gated, w_min, w_max))
```

这样能最大限度保持原始 SpykeTorch R-STDP 行为，只在外部加入软保护。

### 5.3 不同神经元类型的 gate

| 神经元类型 | 初始 gate 策略 |
| --- | --- |
| Stable neurons | $g_{ij}=\exp(-\lambda I_{ij})$ |
| Shared neurons | $g_{ij}=\exp(-\lambda I_{ij})$，通常保护弱于 stable |
| Reserve neurons | 默认 $g_{ij}=1$；被 novelty 激活时可设为 $1+\beta_g N(x)$ |

注意：reserve neurons 的增益不能无限放大，实际实现中应设置上界，例如 `g_max = 1.5`。

---

## 6. NGSG：新颖性引导的 reserve 激活

### 6.1 Novelty Score

使用 S3 winner margin 定义 novelty：

$$
N(x) = 1 - \frac{V_{winner} - V_{second}}{|V_{winner}| + \epsilon}
$$

实现时将 $N(x)$ clamp 到 $[0, 1]$。若无有效 winner 或 $V_{winner}$ 接近 0，则直接记为高 novelty。

直觉：

- winner 明显强于 second winner：网络确定，novelty 低。
- winner 与 second winner 接近：网络不确定，novelty 高。
- 无明显响应：网络无法解释输入，novelty 高。

### 6.2 Novelty 阈值修正

原方案中写“取 Task 1 novelty 的第 5 百分位数”是错误的。因为 novelty 越高越新颖，如果用第 5 百分位数作为阈值，Task 1 自己大多数样本都会被误判为新颖。

修正为：

$$
\theta_N = P_{95}\left(\{N(x_i)\}_{i=1}^{N_{Task1}}\right)
$$

含义：只有高于 Task 1 绝大多数样本 novelty 的输入，才被判为对当前网络新颖。

可选稳健版本：

$$
\theta_N = \mu_N + k\sigma_N
$$

其中 $k$ 初始取 1 或 2。第一版建议先用 $P_{95}$，因为更容易解释和复现。

### 6.3 Reserve Neuron 选择

当 $N(x)>\theta_N$ 时，从 reserve 集合中选择要激活的神经元：

$$
i^* = \arg\min_{i \in \mathcal{R}} \left(\alpha f_i + \beta I_i - \gamma A_i(x)\right)
$$

- $f_i$：Task 1 winner frequency，越低越适合作为 reserve。
- $I_i$：旧任务重要性，越低越适合被新任务使用。
- $A_i(x)$：当前输入对神经元 $i$ 的激活潜力，可用 S3 原始膜电位表示。
- $\alpha,\beta,\gamma$：初始值建议为 1, 1, 0.5。

### 6.4 Readout / 类别映射约束

当前 paper-source S3 readout 是每类固定 20 个神经元。因此第一版 NGSG 必须避免全局随机调用 reserve neuron 导致类别映射混乱。

推荐第一版采用 **class-local reserve**：

- 若训练阶段有目标标签，则只在目标类别对应的 S3 neuron block 内选择 reserve neuron。
- 若未来做无标签 novelty 检测，则需要额外设计 readout remapping 或任务头，目前不作为第一版目标。

这条约束很重要：它保证 NGSG 不破坏当前 `decision_map` 的基本语义。

---

## 7. Silent Synapse 的保守定义

当前 S3 是 dense convolution 权重，不能直接声称“凭空新增连接”。第一版将 silent synapse 定义为：

> reserve neurons 中一部分低可塑性或低有效权重的连接，在 novelty 触发后开放更高可塑性。

实现方式可选：

1. **mask 版本**：给 S3 权重增加 `plasticity_mask`，silent synapse 初始 mask 为 0 或低值，激活后设为 1。
2. **低权重版本**：reserve neurons 的部分连接初始化或重置为 $w_{min}+\delta$，激活后允许快速更新。
3. **第一版建议**：先不改变初始化，只实现 reserve neuron 的 novelty-gated learning gain；等 SDPM 和 novelty 跑稳后，再加入 mask 版本。

论文写法也应保守：第一版称为 “silent-synapse-inspired plasticity gating”，不要过早宣称真实动态结构扩张。

---

## 8. Task 2+ 训练流程

每个训练样本 $x$ 的处理步骤：

```text
1. 前向传播到 S3，得到 potentials、spikes、winner 和 second winner。

2. 计算 novelty score N(x)。

3. 若 N(x) > theta_N：
     在目标类别对应的 reserve neuron block 内选择 i*
     对 i* 及其连接提高 plasticity gate
   否则：
     使用常规 SDPM gate

4. 调用原始 SpykeTorch reward / punish R-STDP。

5. 用 delta-replay 得到 delta_raw，并应用：
     delta = gate * delta_raw

6. 写回权重并 clamp 到 [w_min, w_max]。

7. 记录本次是否触发 novelty、被选中的 reserve neuron、gate 分布和预测结果。
```

---

## 9. 超参数汇总

| 超参数 | 符号 | 推荐初始值 | 含义 |
| --- | --- | --- | --- |
| 神经元分区比例 | - | 30 / 40 / 30 | stable / shared / reserve |
| 保护强度 | $\lambda$ | 3 | SDPM gate 衰减系数 |
| novelty 阈值 | $\theta_N$ | Task 1 novelty 的第 95 百分位数 | 触发 reserve 激活 |
| reserve 选择权重 | $\alpha,\beta,\gamma$ | 1, 1, 0.5 | 使用率、旧重要性、当前激活潜力 |
| reserve 学习增益 | $\beta_g$ | 0.5 | novelty 高时的额外学习增益 |
| reserve gate 上界 | $g_{max}$ | 1.5 | 防止更新过大 |
| silent synapse 偏移 | $\delta$ | $10^{-3}$ | 仅用于后续 mask / 低权重版本 |
| 防零常数 | $\epsilon$ | $10^{-6}$ | 数值稳定 |

---

## 10. 需要新增的记录逻辑

```python
# Task 1 S3 training
win_count[i]
win_label_count[i][c]
total_s3_updates

# Task 1 post-hoc statistics
f[i] = win_count[i] / total_s3_updates
p[i][c] = win_label_count[i][c] / max(win_count[i], 1)
H[i] = -sum(p[i][c] * log(p[i][c]) for c if p[i][c] > 0)
q[i] = 0 if win_count[i] == 0 else 1 - H[i] / log(C)
I_neuron[i] = norm(f[i]) * q[i]
I_synapse[i][j] = I_neuron[i] * norm(w[i][j] - w_min)

# Partition
theta_stable = percentile(f, 70)
theta_reserve = percentile(f, 30)
stable_set = {i for i if f[i] > theta_stable}
shared_set = {i for i if theta_reserve < f[i] <= theta_stable}
reserve_set = {i for i if f[i] <= theta_reserve}

# Novelty calibration
theta_N = percentile([N(x) for x in task1_train_or_calibration_set], 95)

# Task 2+ logs
novelty_score_per_sample
novelty_trigger_count
selected_reserve_neuron_count
gate_mean_by_group
task1_retention_snapshot
task2_learning_snapshot
```

---

## 11. Ablation 设计

| 实验 | 目的 |
| --- | --- |
| Baseline catastrophic forgetting | 证明无保护时存在遗忘 |
| + winner-frequency logging only | 验证统计不改变学习行为 |
| + SDPM only | 验证软保护是否提升 Task 1 retention |
| + novelty detector only | 验证 novelty 是否能区分 Task 1 / Task 2 分布 |
| + reserve activation only | 验证 reserve 调用是否提升新任务学习 |
| + SDPM + NGSG | 完整方法 |
| NGSG w/ random reserve | 证明 novelty-guided 优于随机 reserve |
| NGSG w/ P5 threshold | 反例消融，证明低分位 novelty 阈值会过度触发 |
| NGSG with S1/S2 retraining | 分离 S3 机制和低层特征重学习的影响 |

---

## 12. 分阶段 TODO List

### Phase 0：固定 baseline 认知

- [ ] 用 paper-source 配置继续确认 MNIST -> EMNIST catastrophic forgetting baseline 能稳定学习并产生遗忘趋势。
- [ ] 明确主实验使用的任务：MNIST digits -> EMNIST capitals ABDEGHNQRS。
- [ ] 保存 baseline 配置、运行名、结果 JSON 和复现日志。

完成标准：baseline 不再是随机水平，并能观察到 Task 1 after Task 2 下降。

### Phase 1：无侵入统计

- [ ] 在 S3 训练中记录 winner id、winner frequency、winner label count。
- [ ] 输出 Task 1 后的 `f_i`、`q_i`、`I_i` 分布。
- [ ] 生成 stable/shared/reserve 分区，并保存到实验结果。
- [ ] 确认加入统计后准确率与 baseline 基本一致。

完成标准：统计模块不改变训练行为，只增加可解释性。

### Phase 2：SDPM soft protection

- [ ] 实现 delta-replay 版 gated R-STDP。
- [ ] 对 stable/shared/reserve 分别应用 gate。
- [ ] 记录每组 gate 均值和权重更新幅度。
- [ ] 跑 `+ SDPM only` 消融。

完成标准：Task 1 retention 有提升，Task 2 学习不能被完全压制。

### Phase 3：Novelty calibration

- [ ] 在 Task 1 训练后计算 novelty 分布。
- [ ] 使用第 95 百分位数设定 $\theta_N$。
- [ ] 检查 Task 1 样本触发率应接近或低于 5%。
- [ ] 检查 Task 2 样本触发率是否明显高于 Task 1。

完成标准：novelty score 能区分旧任务熟悉输入和新任务输入。

### Phase 4：Class-local reserve activation

- [ ] 在每个类别 block 内划分 reserve neurons。
- [ ] 当 $N(x)>\theta_N$ 时，只在目标类别 block 内选择 reserve neuron。
- [ ] 对被选中的 reserve neuron 提高 gate，上限为 `g_max`。
- [ ] 跑 `+ NGSG only` 和 `NGSG w/ random reserve` 消融。

完成标准：reserve 激活能帮助新任务学习，并优于随机 reserve。

### Phase 5：完整 NGSG

- [ ] 合并 SDPM 与 novelty-guided reserve activation。
- [ ] 跑完整 `+ SDPM + NGSG`。
- [ ] 对比 baseline、SDPM only、NGSG only、random reserve。
- [ ] 输出 BWT、Task 1 retention、Task 2 accuracy、Avg Acc。

完成标准：完整方法在旧任务保持和新任务学习之间优于单独模块。

### Phase 6：Silent synapse 版本

- [ ] 设计 S3 plasticity mask。
- [ ] 实现 silent synapse-inspired low-plasticity connections。
- [ ] 和无 mask 的 reserve activation 对比。
- [ ] 决定论文中是否使用 “structural growth” 强表述。

完成标准：mask / silent synapse 版本带来明确收益，否则论文中只保留为机制解释或未来工作。

---

## 13. 写论文时的保守表述

推荐表述：

> NGSG-SNN introduces a novelty-guided plasticity allocation mechanism at the output spiking layer. It identifies stable, shared, and reserve neurons from winner-frequency statistics after the initial task, softly protects important synapses through distribution-aware plasticity modulation, and allocates reserve neurons to novel inputs under local R-STDP learning.

避免过早表述：

- “动态新增神经元”，除非代码真的扩容。
- “完全解决灾难性遗忘”，除非结果充分。
- “SOTA”，除非有系统对比。
- “无额外参数”，如果使用 mask、gate、统计表，应说清楚额外状态量。

---

## 14. 当前最关键的修正点

1. Novelty 阈值必须从第 5 百分位数改为第 95 百分位数。
2. reserve activation 第一版必须遵守 class-local readout，避免破坏当前 20 neurons/class 的 decision map。
3. SDPM 要用 delta-replay 接入 SpykeTorch 官方 STDP，而不是假设已有 raw delta。
4. S1/S2 冻结是 NGSG 主方法的实验设定，不是 baseline 复现设定。
5. silent synapse 第一版应保守解释为可塑性 mask / 预留连接激活，不直接声称真实新增参数。
