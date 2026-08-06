---
name: ngsg-repo-workflow
description: >-
  Wayneplp/NGSG-spyketorch 的 GitHub 账户、分支策略、commit/PR/push 规范与
  租服务器代码同步。在 dev/baseline/ngsg 分支切换、推送远程、创建 PR、合并、
  废弃 codex 分支、或用户问「仓库怎么处理 / GitHub 怎么推」时使用。
  服务器 SSH 跑实验见 ngsg-server-experiments。
---

# NGSG-spyketorch GitHub 工作流

## GitHub 身份（固定，勿重复询问用户）

| 项 | 值 |
| --- | --- |
| GitHub 用户 | `Wayneplp` |
| 仓库 | `https://github.com/Wayneplp/NGSG-spyketorch.git` |
| remote 名 | `origin` |
| 本地路径 | `F:/paper_code/NGSG-spyketorch`（Windows） |
| 服务器路径 | `/root/autodl-tmp/NGSG-spyketorch-4a958ae` |

**默认集成分支**：`dev`（本地开发、服务器实验都跟这条分支）。

## 分支拓扑

```
main                          稳定正式版，只从 dev 合并已验证内容
  └── dev                     日常集成（默认工作分支）
        ├── baseline/continuous-learning   baseline 复现
        └── ngsg/novelty-gated-growth      NGSG 创新
```

| 分支 | 用途 | 合并方向 |
| --- | --- | --- |
| `main` | 对外稳定快照 | 仅接受 `dev` |
| `dev` | 日常集成 | 接收 feature 分支与 infra |
| `baseline/continuous-learning` | baseline 实验与论文对齐 | 从 `dev` 更新；完成后 merge 回 `dev` |
| `ngsg/novelty-gated-growth` | NGSG 模块 | 同上 |

**禁止**：在 `main` 直接开发；长期维护 `codex/*` 服务器分支。

## 当前状态（2026-07-01）

| 分支 | HEAD | 说明 |
| --- | --- | --- |
| `dev` / `origin/dev` | `f90f6b7` | 已同步；含 catastrophic baseline 正式结果文档 |
| `main` / `origin/main` | `7e3dd16` | **落后 dev**，待 dev 稳定后 merge |
| `baseline/continuous-learning` | 已 push | 与 dev 同 lineage |
| `ngsg/novelty-gated-growth` | 已 push | 与 dev 同 lineage |
| `origin/codex/server-preprocess-cache` | 遗留 | **已 merge 进 dev**（`e32298e`），可删远程 |

codex 的 infra（预处理缓存、EMNIST idx fallback、训练日志）已在 dev；之后服务器只跟 `dev`。

## 选分支（改代码前）

```
baseline 配置/训练器/论文对齐？     → baseline/continuous-learning
NGSG（S3 分区、SDPM、novelty）？    → ngsg/novelty-gated-growth
data.py、缓存、日志、requirements？ → dev（或 feature 从 dev 拉 infra）
```

## 日常 Git 流程

### 1. 开始工作

```bash
git fetch origin
git checkout dev          # 或对应 feature 分支
git pull origin dev       # feature 分支先 merge/rebase dev
git branch --show-current
```

### 2. 提交（仅用户明确要求时 commit）

- **不要**主动 commit，除非用户说「提交 / commit」
- **不要**改 git config；**不要** `--no-verify` / force push main
- commit 前并行检查：`git status`、`git diff`、`git log -5`
- 消息 1–2 句，说明 **why**；不提交 `data/`、`logs/`、大 `.pt`、`SERVER_LATEST_STATUS.md`、`.DS_Store`

PowerShell 提交示例：

```powershell
git add <paths>
git commit -m "feat: short reason"
git status
```

### 3. 推送

```bash
git push origin dev
# feature 分支首次：
git push -u origin baseline/continuous-learning
```

**不要** push 除非用户明确要求。

### 4. dev → main（里程碑）

dev 上 baseline 或文档已验证后：

```bash
git checkout main
git pull origin main
git merge dev -m "release: baseline catastrophic repro verified"
git push origin main
# 可选 tag：git tag v0.2.0-baseline-repro && git push origin v0.2.0-baseline-repro
```

### 5. Feature 分支生命周期

1. 从最新 `dev` 创建/更新 feature 分支
2. 小步 commit（baseline / ngsg 标明方向）
3. 验证通过后 merge 回 `dev`
4. `dev` 稳定后 merge 到 `main`

### 6. 创建 PR

本机 **未安装 `gh` CLI**。可选：

- 用户安装 [GitHub CLI](https://cli.github.com/) 后用 `gh pr create`
- 或 push 分支后在浏览器打开：`https://github.com/Wayneplp/NGSG-spyketorch/compare`

PR 前检查：`git status`、与 base 的 `git diff main...HEAD`（或 `dev...HEAD`）、全部相关 commit。

## 目录归属

| 路径 | 归属 |
| --- | --- |
| `configs/baseline/` | baseline |
| `configs/ngsg/` | ngsg |
| `src/trainers/baseline_trainer.py` | baseline + 共享 infra |
| `src/continual/` | ngsg |
| `src/utils/data.py` | 共享 infra |
| `experiments/`, `logs/`, `checkpoints/`, `results/`, `data/` | 不进 git |

## 服务器代码同步

服务器跟 `dev`，不在 git 里维护运行时快照。SSH 跑实验完整流程见 [ngsg-server-experiments](../ngsg-server-experiments/SKILL.md)。

快速更新：

```bash
cd /root/autodl-tmp/NGSG-spyketorch-4a958ae
git fetch origin && git checkout dev && git pull origin dev
```

## 合并冲突优先级

1. infra（data.py、缓存、日志）→ 功能更完整者
2. baseline → `README.md` / `CATASTROPHIC_FORGETTING_REPRODUCTION.md`
3. NGSG → `NGSG_SNN_Implementation.md`（若仍存在）或 README §8
4. **不要**为消冲突删掉 winner-frequency / cache 逻辑

## Agent 检查清单

**改代码前**

- [ ] 当前分支正确
- [ ] feature 基于最新 `dev`
- [ ] 改动属 baseline / ngsg / infra

**提交/推送/PR 前**

- [ ] 用户已明确要求 commit/push/PR
- [ ] 无数据集、日志、大 checkpoint、服务器状态文件
- [ ] baseline 未混入 NGSG（除非 dev 有意集成）

**用户问「仓库怎么处理」**

1. 报上表「当前状态」
2. 说明 codex 已废弃、服务器跟 dev
3. 若 `main` 落后，建议 dev 稳定后 merge
4. destructive 操作（删远程分支）等用户确认

## 待办（可选，非阻塞）

- [ ] 确认服务器已切到 `dev` 后：`git push origin --delete codex/server-preprocess-cache`
- [ ] baseline 文档稳定后：`dev` → `main` merge + tag

## 参考

- 项目手册：[README.md](../../README.md)
- 实验记录：[CATASTROPHIC_FORGETTING_REPRODUCTION.md](../../CATASTROPHIC_FORGETTING_REPRODUCTION.md)
- 服务器实验：[ngsg-server-experiments/SKILL.md](../ngsg-server-experiments/SKILL.md)
- Git 命令速查：[references/git-commands.md](references/git-commands.md)
