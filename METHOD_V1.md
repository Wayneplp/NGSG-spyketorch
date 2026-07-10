# METHOD v1：Role-Train + Group Confidence Routing

**版本：** v1 保守可落地版 · **最后更新：** 2026-07-09  
**不含 SDPM** · **证据链仅 full scale**

---

## 0. 一句话

Task1 后按 winner frequency 将 S3 200 神经元分为 stable / shared / reserve；Task2 用 **确定性 WTA mask + S3 STDP lr 缩放** 塑造分工；推理时以 **stable vs reserve 的 group confidence** 判任务，不确定时 fallback 到 prototype similarity；最后按任务做 **route mask** 组内 WTA 分类。

---

## 1. 核心原则

```text
训练阶段：gate 控制谁能赢 WTA、谁能更新 STDP
推理阶段：gate 控制谁参与最终分类（route mask ≠ 训练 schedule）
任务判定：不看 shared，主要看 stable / reserve 的可信度
```

---

## 2. 问题与动机（full 诊断）

| 现象 | 数据 | 对策 |
|------|------|------|
| Task2 后 reserve 抢 MNIST winner | reserve 44.7%，all-200 34.88% | Task1 route 关闭 reserve |
| stable 权重未全丢 | stable-only 63.75% | Task2 训练保护 stable |
| Oracle 路由有增益 | avg +8.46 pp | 需 Role-train + 推理路由 |
| reserve 高响应乱答 | vote 胶着 confidence≈0 | group confidence 判任务 |

---

## 3. 流程

```text
Step 1  Task1           200 神经元全开 → winner_count 分区
Step 2  Task2 训练      schedule WTA mask + S3 STDP lr（前 80% / 后 20%）
Step 3  推理判任务       conf_stable vs conf_reserve；不确定 → prototype sim
Step 4  Route mask      Task1 stable-only；Task2 reserve+shared → WTA → 类别
```

---

## 4. Step 1：Task1 与分区

- Task1：200 神经元全开，正常 WTA + STDP
- 分区：`stable` = 高频 winner，`shared` = 中间，`reserve` = 低频
- 实现：`neuron_partition.py`（`f_stable_percentile=0.70`, `f_shared_percentile=0.40`）

---

## 5. Step 2：Task2 Role-Train

### 5.1 WTA gate（确定性 mask）

```python
score_masked = score.clone()
score_masked[closed_group] = -inf
winner = argmax(score_masked)
```

- **不用**随机屏蔽、概率 gate、`score * gate`
- 训练用 **schedule gate**；推理用 **route mask**（两套独立）

### 5.2 STDP lr

- **仅 S3（conv3）** 按组缩放；S1/S2 保持原样或统一降 lr，不按组缩放

### 5.3 两阶段日程（固定 epoch）

前 **80%** / 后 **20%**（例：100 epoch → 1–80 / 81–100）

| 阶段 | stable WTA | stable lr | shared WTA | shared lr | reserve WTA | reserve lr |
|------|------------|-----------|------------|-----------|-------------|------------|
| 前 80% | closed | 0 | open | 0.1–0.3 | open | 1.0 |
| 后 20% | closed | 0 | open | 0.1 | open | 0.5 |

v1 **不做** Task2 后期 stable `gate=0.1`（即使 lr=0 也可能抢 WTA）。

---

## 6. Step 3：推理任务判定

### 6.1 Group confidence

对 `stable` / `reserve`（**不含 shared**）：

```text
votes_g[c] = 组内神经元对类别 c 的 response 累加
conf_g = (top1(votes_g) - top2(votes_g)) / (sum(votes_g) + eps)
```

### 6.2 映射规则

```python
if conf_stable - conf_reserve > tau:   task_hat = Task1
elif conf_reserve - conf_stable > tau:  task_hat = Task2
else:                                 prototype fallback
```

- `tau` 默认 0.05–0.1，验证集扫描
- prototype：`cos(r(x), m_k)`，`m_k` 存 checkpoint 内（`task_memory.py`）

### 6.3 冲突与 uncertain

confidence 与 prototype 严重冲突 → uncertain route（保守压低 reserve）。

---

## 7. Step 4：Route mask

| 路由 | stable | shared | reserve |
|------|--------|--------|---------|
| **Task1（v1 首跑）** | 1.0 | 0.0 | 0.0 |
| **Task2（v1 首跑）** | 0.0 | 0.6 | 1.0 |
| **uncertain** | 0.7 | 0.5 | 0.3 |

gate ≥ 0.5 → 参与 WTA；否则 closed（`-inf` mask）。

---

## 8. 实验对照

| 行 | 训练 | 推理 | 配置 |
|----|------|------|------|
| R0 | 无保护 | natural WTA | `paper_full_partition_seed0.yaml` |
| **R1** | role-train | natural WTA | `method_v1_role_train_r1_full_seed0.yaml` |
| **R2** | **同 R1（不重训）** | confidence routing on R1 ckpt | R2 YAML 仅提供 `eval.method_v1` 参数 |
| R3 | 同 R0 | Oracle mask | 诊断脚本 |

**R1 vs R2：** 训练配置完全相同；R2 = 在 R1 checkpoint 上做 routing eval，**不需要再跑 600+100 epoch**。

**指标：**
- **Acc_task**：MNIST 上 t̂=Task1 比例；EMNIST 上 t̂=Task2 比例
- **Acc_class**：Task1 after T2、Task2 after T2

---

## 9. 运行命令

```bash
# R1：Role-train 主表（推理仍为 natural WTA）
python scripts/run_baseline.py --config configs/ngsg/method_v1_role_train_r1_full_seed0.yaml

# R2：在 R1 checkpoint 上做 routing eval（不重训）
python scripts/eval_method_v1_routing.py \
  --run-dir experiments/method_v1_role_train_r1_full_seed0 \
  --config configs/ngsg/method_v1_role_routing_r2_full_seed0.yaml \
  --checkpoint-stage task2 \
  --write-markdown

# 或 R1 结束后自动执行：
bash scripts/run_method_v1_chain.sh
```

---

## 10. 代码模块

| 模块 | 路径 | 职责 |
|------|------|------|
| Role-train | `src/continual/role_train.py` | schedule WTA mask + S3 STDP 缩放 |
| Group confidence | `src/continual/group_confidence.py` | 组内 vote margin |
| Task memory | `src/continual/task_memory.py` | `m_k` 原型 + cosine sim |
| Role-aware inference | `src/continual/role_aware_inference.py` | 判任务 + route mask + 预测 |
| WTA mask（模型） | `src/models/paper_mozafari.py` | `set_s3_wta_allow_mask` |
| 训练集成 | `src/trainers/baseline_trainer.py` | Task2 role-train + routing eval |

---

## 11. v2 延后项

- stable 弱开 WTA 扫描（0.05–0.2）
- 自适应 epoch 切换（val winner-ratio）
- 期望向量匹配 task 判定
- confidence + prototype 加权融合
- S1/S2 统一冻结

---

## 12. 实现状态（2026-07-09）

| 模块 | 状态 |
|------|------|
| `role_train.py` | ✅ |
| `group_confidence.py` | ✅ |
| `task_memory.py` | ✅ |
| `role_aware_inference.py` | ✅ |
| `paper_mozafari.py` WTA mask API | ✅ |
| `baseline_trainer.py` 接线 | ✅ |
| R1 YAML | `configs/ngsg/method_v1_role_train_r1_full_seed0.yaml` |
| R2 YAML | `configs/ngsg/method_v1_role_routing_r2_full_seed0.yaml` |
| 路由诊断脚本 | `scripts/eval_method_v1_routing.py` |

**审查要点：**
1. Task2 训练时 stable 全程 WTA closed + STDP lr=0
2. R1 主表用 natural WTA；R2 用 confidence routing
3. Task1 after Task1 评估不走 routing（`routing_active=False`）
4. `result.json` extra 含完整 `task_memory` 原型（`to_dict()`）
