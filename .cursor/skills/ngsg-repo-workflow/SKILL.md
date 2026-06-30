---
name: ngsg-repo-workflow
description: >-
  NGSG-spyketorch 论文仓库的分支策略、合并流程与租服务器部署规范。
  在 dev/baseline/ngsg 分支切换、合并 codex/server-preprocess-cache、
  推送远程、服务器拉代码或跑实验时使用。
---

# NGSG-spyketorch 仓库工作流

## 仓库定位

本仓库同时承担三件事，按优先级排序：

1. **baseline 复现** — 对齐 SpykeTorch / Mozafari 持续学习论文
2. **NGSG 创新实现** — 在 S3 层加入 novelty-gated growth
3. **实验基础设施** — 数据加载、预处理缓存、训练日志（本地与租服务器共用）

代码与配置进 git；数据集、日志、checkpoint、预处理 `.pt` 缓存不进 git（见 `.gitignore`）。

## 分支拓扑

```
main                          稳定正式版，只从 dev 合并已验证内容
  └── dev                     日常集成，所有功能先到这里
        ├── baseline/continuous-learning   仅 baseline 复现相关改动
        └── ngsg/novelty-gated-growth      仅 NGSG 创新相关改动
```

| 分支 | 用途 | 合并方向 |
|------|------|----------|
| `main` | 可对外引用的稳定快照 | 仅接受 `dev` 的 PR/merge |
| `dev` | 日常开发集成 | 接收 baseline/ngsg 的功能 PR；接收 infra 改进 |
| `baseline/continuous-learning` | baseline 实验与对齐 | 从 `dev` 定期 rebase/merge；完成后 merge 回 `dev` |
| `ngsg/novelty-gated-growth` | NGSG 模块 | 从 `dev`（或 baseline 稳定点）分出；完成后 merge 回 `dev` |

**禁止**：在 `main` 上直接开发；在 feature 分支上长期堆 infra 而不回 `dev`。

## 当前已知状态（2026-06-30）

- 本地当前分支：`dev`，与 `origin/dev` 同步
- `baseline/continuous-learning`、`ngsg/novelty-gated-growth`：**仅本地**，尚未 push
- `origin/codex/server-preprocess-cache`：**遗留服务器分支**，在 `4a958ae` 后与 `dev` 分叉

### codex 分支相对 dev 多出的功能提交（应并入 dev）

| 提交 | 内容 | 是否并入 dev |
|------|------|-------------|
| `42ababb` | paper-source 离线预处理缓存 | **是** — 全分支共享 infra |
| `ce88943` | 训练 epoch 进度日志 | **是** |
| `9fcd9f7` | EMNIST letters 从 raw idx 读取（绕过 torchvision 崩溃） | **是** |
| `e86a263` | `SERVER_LATEST_STATUS.md` | **否** — 服务器运行时快照，不应长期留在 git |

### dev 相对 codex 多出的提交

| 提交 | 内容 |
|------|------|
| `7e3dd16` | 删除 `.DS_Store` |

**结论**：codex 上的 infra 代码应 merge 进 `dev`，之后服务器跟 `dev` 跑，不再维护独立 codex 分支。

## 立即待办（一次性整理）

按顺序执行，Agent 可在用户确认后直接操作：

```bash
# 1. 在 dev 上合并 codex 的功能提交（推荐 merge，保留历史）
git checkout dev
git fetch origin
git merge origin/codex/server-preprocess-cache -m "merge: bring server infra (preprocess cache, EMNIST fix, logging) into dev"

# 2. 解决冲突（若有）：dev 的 .DS_Store 删除应保留；不要重新引入 .DS_Store
# 3. 若合并进了 SERVER_LATEST_STATUS.md，从 dev 删除并加入 .gitignore（可选）
git rm --cached SERVER_LATEST_STATUS.md 2>/dev/null || true

# 4. 推送 dev
git push origin dev

# 5. 更新 feature 分支到最新 dev
git checkout baseline/continuous-learning
git merge dev
git push -u origin baseline/continuous-learning

git checkout ngsg/novelty-gated-growth
git merge dev
git push -u origin ngsg/novelty-gated-growth

# 6. 确认服务器已切到 dev 且缓存任务正常后，删除远程 codex 分支
git push origin --delete codex/server-preprocess-cache
```

若用户希望历史更干净，可对 `42ababb`~`9fcd9f7` 做 cherry-pick 而非全量 merge；**不要** cherry-pick `e86a263`。

## 日常开发流程

### 选分支

```
改 baseline 配置/训练器/对齐论文？  → baseline/continuous-learning
改 NGSG 模块（S3 分区、SDPM、novelty）？ → ngsg/novelty-gated-growth
改 data.py、缓存、日志、requirements？   → dev（或从 dev 拉 infra 到 feature 分支）
```

### Feature 分支生命周期

1. 从最新 `dev` 创建或更新 feature 分支
2. 小步 commit，消息说明是 baseline 还是 ngsg
3. 功能验证通过后 merge 回 `dev`
4. `dev` 稳定后 merge 到 `main` 并打 tag（如 `v0.2.0-baseline-repro`）

### 目录约定

| 路径 | 归属 |
|------|------|
| `configs/baseline/` | baseline 分支主战场 |
| `configs/ngsg/` | ngsg 分支主战场（dev 上可留 `.gitkeep`） |
| `src/trainers/baseline_trainer.py` | baseline + 共享 infra |
| `src/continual/` | ngsg 专用逻辑 |
| `src/utils/data.py` | 共享 infra，改时两边实验都要能跑 |
| `approx/legacy_approx/` | 历史近似实现，非主路径 |
| `experiments/`, `logs/`, `checkpoints/`, `results/`, `data/` | 本地/服务器产物，不进 git |

## 租服务器部署

### 原则

- **服务器跟 `dev`**（或 `dev` 上的 release tag），不再维护 `codex/*` 长期分支
- 服务器路径示例：`/root/autodl-tmp/NGSG-spyketorch-<short-sha>`
- 运行时状态（tmux 会话名、缓存进度、日志路径）写在服务器本地，**不要** commit `SERVER_LATEST_STATUS.md`

### 服务器首次 / 更新代码

```bash
cd /root/autodl-tmp/NGSG-spyketorch-*
git fetch origin
git checkout dev
git pull origin dev
pip install -r requirements.txt
```

### 预处理缓存

- 缓存目录：`data/preprocessed/paper_source/<hash>/`
- 由 `src/utils/data.py` + `baseline_trainer.py` 写入，已在 codex 分支实现
- 全量缓存用 tmux 长跑；检查进度：

```bash
tmux attach -t ngsg
find data/preprocessed/paper_source -name '*.pt' | wc -l
tail -n 50 logs/preprocess_*.log
```

### EMNIST 注意

服务器上若 `torchvision.datasets.EMNIST(split="letters")` 崩溃，代码会 fallback 到 `data/emnist/EMNIST/raw/gzip/` 的 idx 文件。确保 raw 文件在服务器上存在。

## 合并冲突处理优先级

1. **infra 改进**（data.py、缓存、日志）→ 以功能更完整的版本为准，通常来自 dev/codex 合并结果
2. **baseline 行为** → 以 `REPRODUCTION_PLAN.md` 和论文对齐为准
3. **NGSG 行为** → 以 `NGSG_SNN_Implementation.md` 为准
4. **不要** 为了消冲突而删除 winner-frequency / cache 逻辑

## Agent 检查清单

开始改代码前：

- [ ] `git branch --show-current` 确认在正确分支
- [ ] feature 分支是否基于最新 `dev`
- [ ] 改动属于 baseline、ngsg 还是 infra

提交 / PR 前：

- [ ] 未提交 `data/`、`logs/`、`checkpoints/`、`.pt` 缓存
- [ ] 未提交 `SERVER_LATEST_STATUS.md` 或 `.DS_Store`
- [ ] baseline 改动未混入 NGSG 模块（除非是有意的 dev 集成）
- [ ] 若改 `data.py` 或 trainer，本地或服务器至少 smoke test 一次

用户问「仓库怎么处理」时：

1. 说明各分支现状（见上文「当前已知状态」）
2. 建议 merge codex → dev，push feature 分支，废弃 codex
3. 给出下一步具体 git 命令，等用户确认再执行 destructive 操作（删远程分支）

## 参考文档

- 复现计划：[REPRODUCTION_PLAN.md](../../REPRODUCTION_PLAN.md)
- NGSG 设计：[NGSG_SNN_Implementation.md](../../NGSG_SNN_Implementation.md)
- 仓库说明：[README.md](../../README.md)
