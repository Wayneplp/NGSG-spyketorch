# NGSG 创新总结笔记

## 一句话概括

你的工作不是简单复现 SpykeTorch 持续学习，而是在**局部学习规则驱动的脉冲神经网络持续学习框架**上，提出一套面向**新颖样本检测、空闲神经元调度和结构性增量生长**的 NGSG 扩展机制。

---

## 1. 你的核心创新主线

### 1.1 问题切入点有明确特色

你关注的不是标准 ANN 持续学习，而是：

- **SNN 场景**
- **局部学习规则场景**
- **持续学习场景**

这个切入点本身就比较有辨识度，因为它同时受三种约束：

- 不能依赖常规反向传播式全局更新；
- 旧知识容易在新任务中被覆盖；
- 现有容量固定时，新类别/新分布难以稳定吸收。

所以你的工作重点不是只做“抗遗忘”，而是进一步回答：

> 当局部学习的 SNN 遇到新任务时，如何判断“这是新东西”，如何少破坏旧知识地吸收它，以及如何有节制地扩展可塑性容量。

---

## 2. 方法层面的创新点

### 2.1 从“参数保护”走向“神经元资源管理”

已有 baseline 主要是：

- catastrophic forgetting
- joint training
- frozen large weights
- Langevin dynamics

这些方法更多是在已有参数上做保护、对冲或平滑更新。

而你的 NGSG 思路往前走了一步：  
不只问“哪些权重该少改”，而是问：

- 哪些神经元经常被使用；
- 哪些神经元其实长期空闲；
- 哪些输入说明当前表征不够；
- 是否应该把“保留资源”分配给真正的新信息。

这意味着你的创新重点从**权重级保护**提升到了**神经元级资源分配与结构性适应**。

### 2.2 引入 winner frequency 作为内部状态统计

你计划先加入 **winner frequency** 统计，这很关键。

它的作用不只是做分析，而是为后续机制提供依据：

- 衡量哪些神经元被频繁竞争获胜；
- 识别哪些神经元使用过度；
- 找出长期沉默、可能尚未承载稳定语义的单元。

这一步的意义在于，你不是盲目扩展网络，而是先建立一个**可解释的内部使用率视角**。

### 2.3 用 novelty score 触发“是否需要新资源”的判断

你设计中的 **novelty score** 是另一个很强的点。

它对应的不是普通分类分数，而是：

- 当前输入是否能被现有神经元充分解释；
- 现有竞争格局是否出现明显失配；
- 该样本是否应被视为“需要新表征资源”的候选。

这使 NGSG 不只是“有备用神经元”，而是具备一个**何时扩展**的判据。  
也就是说，你的方法是“检测新颖性后再调度资源”，而不是固定地增加容量。

### 2.4 reserve neuron / reserve branch 体现受控可塑性

你提出的 **reserve neuron** 和 **reserve branch**，体现的是一种受控增量学习思想：

- 常规神经元负责维持已有稳定表征；
- 保留神经元或保留分支负责吸收新模式；
- 新旧知识在功能上被部分隔离，减少直接冲突。

这比单纯冻结大权重更细致，因为它不是只限制更新，而是主动为新知识准备“落点”。

### 2.5 silent synapse growth 体现结构生长而非单纯重训练

**silent synapse growth** 是你整套思路里最像“结构创新”的部分。

它说明你的目标不是只在固定连接上反复调参，而是：

- 在需要的时候激活潜在连接；
- 让新知识通过新增或唤醒的通路接入网络；
- 以较小代价获得新的表征能力。

这个点很重要，因为它把你的方法从“持续学习技巧”推进到了**持续学习中的结构可塑性机制**。

---

## 3. 这套创新为什么有逻辑闭环

你的设计不是几个零散技巧拼在一起，而是有比较完整的链路：

1. 先复现局部学习 SNN 的持续学习 baseline；
2. 用 winner frequency 建立神经元使用统计；
3. 用 novelty score 判断当前样本是否超出已有表征；
4. 用 reserve neuron / reserve branch 提供低干扰的新学习空间；
5. 用 silent synapse growth 完成必要的结构性接入；
6. 最终目标是在减少遗忘的同时提升对新任务的吸收能力。

这个链路的优点是：

- 每一步都有前一步作为依据；
- 每个模块都能单独做消融；
- 很适合写成“从分析到机制”的论文叙事。

---

## 4. 你的创新和普通持续学习工作的区别

如果要简洁地区分，你的工作和常见方法的差别可以概括为：

### 常见持续学习方法更像

- 正则化旧参数
- 冻结部分权重
- 回放旧样本
- 在固定容量里降低冲突

### 你的 NGSG 更像

- 监控神经元竞争行为
- 识别新颖输入
- 调度保留神经资源
- 在必要时进行结构性生长

所以你的亮点不是“再做一个抗遗忘 trick”，而是：

> 在局部学习 SNN 中，把新颖性检测、资源调度和结构可塑性整合成一个持续学习框架。

---

## 5. 目前最适合对外表达的创新表述

你现在可以把自己的创新先表述成下面这版：

> We build a continual-learning framework for locally trained spiking neural networks, where neuron usage statistics are first monitored to expose under-utilized capacity, novelty signals are then used to detect inputs that are insufficiently represented by existing neurons, and reserved neural resources with silent-synapse growth are recruited to absorb new knowledge while reducing interference with previously learned tasks.

对应中文可以写成：

> 我们面向局部学习规则训练的脉冲神经网络，构建了一套持续学习框架。该框架首先通过神经元获胜频率统计刻画内部资源使用状态，再利用新颖性信号识别现有表征难以覆盖的输入，并通过保留神经元/保留分支及静默突触生长机制，为新知识提供低干扰的吸收通道，从而在减轻遗忘的同时提升网络对新任务的适应能力。

---

## 6. 现阶段最稳妥的结论

基于当前仓库内容，你的创新可以分成两层：

### 已经明确成型的创新方向

- 在局部学习 SNN 持续学习中引入 NGSG 机制；
- 用 winner frequency 和 novelty score 做内部状态感知；
- 用 reserve neuron / reserve branch 做资源隔离；
- 用 silent synapse growth 做结构增量适应。

### 还需要实验进一步坐实的部分

- novelty score 的具体定义与阈值机制；
- reserve neuron 的分配策略；
- reserve branch 的接入时机；
- silent synapse growth 的触发条件与稳定性；
- 这些模块分别对 forgetting 和 avg acc 的贡献。

所以目前最准确的说法是：

> 你的创新框架已经非常清楚，接下来最关键的是把这些模块逐步实现，并通过消融实验把“资源统计、 novelty 检测、保留容量、生长机制”四个部分的贡献拆开证明。

---

## 7. 后面写论文时可以直接复用的标题思路

- NGSG: Novelty-Guided Structural Growth for Continual Learning in Locally Trained Spiking Neural Networks
- Reserve Neuron Allocation and Silent Synapse Growth for Continual Learning in Spiking Networks
- Novelty-Aware Resource Recruitment for Continual Learning with Local Plasticity Rules

---

## 8. 备注

这份总结是基于当前仓库里的项目说明、baseline 结构和 NGSG 预留模块整理的。  
它更适合作为你现在的“创新框架笔记”，后面等具体公式、实验结果和模块实现更完整后，可以再升级成论文摘要版和 introduction 版。
