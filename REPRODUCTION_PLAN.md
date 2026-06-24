# Reproduction Plan for "Continuous Learning of Spiking Networks Trained with Local Rules"

## 0. 这份文档是干什么的

这份文档用于指导我们先复现原论文，再在复现成功的基础上加入 NGSG。

核心原则只有一句话：

> 先严格复现原论文 baseline，不要一开始就改模型；先证明原方法能跑通，再开始加自己的东西。

---

## 1. 复现目标

当前复现目标不是“立刻做出 NGSG”，而是分成两个阶段：

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

- winner frequency 统计
- reserve neuron 发现
- novelty score
- reserve branch
- silent synapse growth

---

## 2. 你现在到底要“写”什么

如果你现在是准备动手复现，那你真正要写的不是论文正文，而是下面四类东西：

### 2.1 复现实验说明

要写清楚：

- 原论文用了什么任务设置；
- 每个 baseline 的核心机制是什么；
- 我们当前代码里准备怎么对应实现；
- 哪些地方已经完全一致；
- 哪些地方暂时只能近似实现。

### 2.2 复现代码入口

要写清楚：

- 哪个脚本负责训练 baseline；
- 哪个脚本负责测试；
- 哪个配置文件对应哪个方法；
- 训练输出存到哪里。

### 2.3 复现记录表

要写清楚：

- 哪次实验用了什么配置；
- 是否成功跑通；
- 准确率是多少；
- 和论文结果差多少；
- 可能误差来源是什么。

### 2.4 偏差说明

只要有任意一个地方和原论文不完全一致，就要记录：

- 原论文设定
- 当前实现设定
- 差异
- 可能影响

这个非常重要，因为以后写论文 related work 或 experiment 时，你会反复用到。

---

## 3. 最推荐的复现顺序

不要同时复现所有东西。建议按下面顺序做。

### Step 1：确认原论文实验最小闭环

先从论文里抽出最小闭环：

1. 用了什么数据集
2. 任务是怎么分的
3. 网络结构是什么
4. 学习规则是什么
5. baseline 有哪些
6. 指标是什么

你先不要纠结所有细节，先把这 6 个问题单独写出来。

### Step 2：先跑最简单 baseline

第一个应该先做：

- catastrophic forgetting baseline

原因：

- 它最简单；
- 最容易验证数据流、训练流和测试流是否正确；
- 如果这个都跑不通，后面的 Langevin 也没法做。

### Step 3：再做 joint training

然后做：

- joint training

原因：

- 它相当于理想上界；
- 可以帮助判断数据、标签、评估流程是不是有问题。

### Step 4：再做 frozen large weights

这是保护式方法里相对更直接的一种。

### Step 5：最后做 Langevin dynamics

Langevin 是你后面要重点对比的强 baseline，所以要最后认真做、认真对齐。

---

## 4. 复现时每个 baseline 应该怎么写

后面无论是写代码说明、实验日志还是论文实验部分，都建议按同一个模板写。

模板如下。

### 4.1 方法名

例如：

`Catastrophic Forgetting Baseline`

### 4.2 它在原论文里的作用

例如：

用于展示在没有保护机制时，旧任务性能会因新任务训练而明显下降。

### 4.3 当前实现方式

例如：

Task 1 正常训练，Task 2 继续在同一网络上训练，不加入额外保护项，不冻结参数，不引入额外容量。

### 4.4 关键配置

例如：

- 数据集
- S1/S2/S3 结构
- 学习率或 R-STDP 参数
- 训练 epoch
- 任务顺序

### 4.5 输出指标

- Task1 after Task1
- Task1 after Task2
- Task2 after Task2
- Forgetting
- Avg Acc

### 4.6 和原论文结果对比

- 原论文结果
- 当前结果
- 差异
- 可能原因

---

## 5. 你在代码仓库里应该怎么对应落地

推荐这样对应：

### 5.1 配置文件

放在：

- `configs/baseline/catastrophic.yaml`
- `configs/baseline/joint_training.yaml`
- `configs/baseline/frozen_large_weights.yaml`
- `configs/baseline/langevin.yaml`

每个配置文件至少写清楚：

- 数据集
- 任务设置
- 网络参数
- 训练参数
- 评估参数
- 输出目录

### 5.2 训练脚本

放在：

- `scripts/run_baseline.py`

建议这个脚本先只做一件事：

> 读取配置，跑指定 baseline，然后保存日志和结果。

### 5.3 结果汇总

放在：

- `results/baseline_summary.csv`

建议字段包括：

- method
- seed
- task1_after_task1
- task1_after_task2
- task2_after_task2
- forgetting
- avg_acc
- notes

### 5.4 每次实验的单独记录

放在：

- `experiments/exp001_baseline_reproduce/`

里面可以放：

- `plan.md`
- `run_log.md`
- 导出的图
- 对应结果表

---

## 6. 第一阶段你应该先写哪些文件

如果现在马上开始，建议第一批只写下面这些：

### 在代码仓库里

- `configs/baseline/catastrophic.yaml`
- `configs/baseline/joint_training.yaml`
- `configs/baseline/frozen_large_weights.yaml`
- `configs/baseline/langevin.yaml`
- `scripts/run_baseline.py`
- `experiments/exp001_baseline_reproduce/run_log.md`

### 在论文仓库里

- 更新 `NGSG_framework_and_log.md`
- 把 baseline 跑通情况填进实验总表

---

## 7. 复现记录应该怎么写

下面这个模板你可以直接照着填。

## 实验记录模板

### 实验名称

例如：

`exp001_catastrophic_baseline_seed0`

### 目标

复现原论文中不加保护时的持续学习结果，验证训练与评估流程正确。

### 对应原论文方法

Catastrophic forgetting baseline

### 当前实现说明

- Task 1 先训练
- Task 2 再继续训练
- 不加参数保护
- 不加额外容量

### 配置

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

### 当前判断

- 是否算复现成功：待填
- 是否需要继续排查：待填

### 下一步

- 待填

---

## 8. 如何判断“复现成功”

不要要求第一版完全一模一样，但至少满足下面三点：

1. 训练和测试流程完整跑通；
2. 结果趋势与原论文一致；
3. 数值误差在可解释范围内。

这里最重要的是“趋势一致”，比如：

- catastrophic forgetting 明显差；
- joint training 最好或接近最好；
- Langevin 比无保护 baseline 更稳；
- frozen large weights 有一定缓解但不一定最优。

如果趋势都不对，就不要急着做 NGSG。

---

## 9. 复现阶段最容易犯的错

### 错误 1：一边复现一边改方法

这会导致你最后根本不知道结果来自原方法还是你自己的改动。

### 错误 2：没有记录配置

今天能跑，明天不能复现，是最常见的问题。

### 错误 3：只看最终准确率，不看趋势

持续学习最重要的是：

- Task 1 学完时怎样
- Task 2 学完后旧任务掉了多少
- 新任务是否学会了

### 错误 4：原论文和当前实现不一致但没有登记

后面会非常痛苦，因为你会忘记到底哪里改过。

---

## 10. 对你当前任务的最直接建议

你现在不要先写 NGSG 代码。

你现在应该先写的是：

1. baseline 配置文件
2. baseline 统一训练入口
3. 第一份复现实验日志

也就是说，当前最合理的实际动作是：

> 先把原论文的四个 baseline 在你的代码仓库里组织出来，形成一个干净、可重复运行的 baseline 框架。

等这个框架稳定之后，再开始加 winner frequency 统计。

---

## 11. 你下一步可以直接做什么

最推荐的顺序：

1. 先把原论文 baseline 名单定死
2. 给每个 baseline 建一个 config 文件
3. 写一个统一的 `run_baseline.py`
4. 先跑 catastrophic baseline
5. 再跑 joint training
6. 再补 frozen 和 Langevin
7. 跑通后再做 winner frequency logging

如果你愿意，下一步我可以直接继续帮你把这些“第一批该写的空文件”也生成出来。
