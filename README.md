# NGSG SpykeTorch 工作区

这个仓库目前服务于三件事：

- 先复现论文 *Continuous Learning of Spiking Networks Trained with Local Rules* 中的 SpykeTorch 持续学习 baseline；
- 在 baseline 稳定之后加入 NGSG 创新模块；
- 维护共享实验基础设施，包括数据加载、预处理缓存、特征 checkpoint、训练日志和结果记录。

## 当前主线状态（2026-07-01）

- `dev` 是本地和服务器共同使用的集成分支。
- `baseline/continuous-learning` 用于 baseline 复现工作。
- `ngsg/novelty-gated-growth` 用于 NGSG 相关改动。
- 当前 paper-source catastrophic baseline 使用 `src/trainers/baseline_trainer.py` 和 `src/utils/data.py` 中的 SpykeTorch/Mozafari 风格路径。
- 服务器预处理缓存、EMNIST raw idx fallback、S1/S2 feature checkpoint 复用和 C2 feature cache 已合并到 `dev`。
- `SERVER_LATEST_STATUS.md` 属于服务器运行时笔记，不应提交。

## 最新情况

- `dev` 已包含 feature-cache 实现和可复用的 paper-source S1/S2 checkpoint。
- 已跟踪的小 checkpoint：
  - `checkpoints/features/paper_task1_s1e2_s2e4_f26edcfb75b5d681.pt`
  - `checkpoints/features/paper_task2_s1e2_s2e4_60c0a06b55746fb6.pt`
- 这些 checkpoint 使用完整 paper-aligned feature schedule 生成：每个任务 24,000 个训练样本，S1 STDP 2 epoch，S2 STDP 4 epoch。
- `data/preprocessed/` 和 `data/features/c2/` 仍然是本地/服务器运行时缓存，不进 git。
- `configs/baseline/` 现在只保留 3 个 active YAML。具体该跑哪个，见 `configs/baseline/README.md`。

## 推荐工作流

1. 共享基础设施放在 `dev`。
2. baseline 复现放在 `baseline/continuous-learning`，稳定后合回 `dev`。
3. NGSG 实现放在 `ngsg/novelty-gated-growth`，等 baseline 行为清楚后再合回 `dev`。
4. 只有在得到可复用、可对外说明的快照后，才把 `dev` 合到 `main`。

## 顶层目录

- `configs/`：实验配置。
- `scripts/`：可直接运行的入口脚本。
- `src/`：模型、数据、可塑性规则和训练代码。
- `approx/legacy_approx/`：早期近似实现，仅作为参考保留。
- `experiments/`：本地生成的单次运行输出。
- `logs/`：本地生成的原始日志。
- `checkpoints/`：checkpoint 目录；其中正式 S1/S2 feature checkpoint 已选择性进入 git。
- `results/`：本地生成的汇总表和图。
- `data/`：下载的数据集、预处理缓存和 C2 feature cache。

## 在另一台机器上复现

这个仓库跟踪代码、配置和说明文档；不跟踪下载数据、大型缓存、普通运行日志和普通运行输出。

进入 git 的内容：

- `src/` 下的源代码；
- `scripts/` 下的运行脚本；
- `configs/` 下的 active 配置；
- Markdown 文档和流程说明；
- `checkpoints/features/` 下正式共享的小 S1/S2 feature checkpoint。

不进入 git 的内容：

- `data/` 下的数据集；
- `data/preprocessed/` 下的预处理 `.pt` 缓存；
- `data/features/c2/` 下的大型 C2 feature cache；
- `experiments/` 下的每次运行输出；
- `logs/` 下的原始日志；
- `results/` 下的生成结果；
- `SERVER_LATEST_STATUS.md` 这类服务器运行时快照。

## 环境与基础检查

1. 克隆仓库。
2. 创建并激活 Python 环境。
3. 执行 `pip install -r requirements.txt`。
4. 长时间训练前先做 dry-run。

配置检查示例：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist.yaml --device auto --dry-run --run-name paper_source_strict_dryrun
```

中等规模诊断示例：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml --device auto --run-name paper_medium_source_port_seed0
```

## 数据与缓存

- MNIST 在需要时通过 torchvision 下载。
- EMNIST letters 优先通过 torchvision 读取；如果服务器上的 `torchvision.datasets.EMNIST(split="letters")` 不稳定，会从 raw idx / idx.gz 文件回退读取。
- paper-source 预处理缓存写入 `data/preprocessed/paper_source/<hash>/`。
- C2 pooled feature cache 写入 `data/features/c2/<hash>/`。
- 这些缓存都是运行产物，默认不进 git。

服务器建议跟随 `dev` 或从 `dev` 切出的 release tag：

```bash
git fetch origin
git checkout dev
git pull origin dev
pip install -r requirements.txt
```

服务器缓存检查示例：

```bash
find data/preprocessed/paper_source -name '*.pt' | wc -l
tail -n 50 logs/preprocess_*.log
```

## 训练复用策略

paper-source 路线现在有三层复用：

- 输入预处理缓存：`data/preprocessed/paper_source/<hash>/`。
- S1/S2 feature checkpoint：`checkpoints/features/`。
- C2 pooled feature cache：`data/features/c2/<hash>/`，后续反复调 S3、NGSG、winner frequency、reserve neuron 和 synapse growth 时优先从这里进入。

长期共享的 feature checkpoint 应使用论文对齐设定：S1 STDP 2 epoch，S2 STDP 4 epoch。smoke test 可以用更小数据，但不能替代正式 baseline 和 NGSG 对比使用的共享特征层。

如果需要重建这些可复用特征产物，运行：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml --device cuda --run-name paper_feature_checkpoint_full
```

正式 `S1=2/S2=4` 的 `checkpoints/features/paper_task*_s1e2_s2e4_*.pt` 文件体积小，已经跟随 git，方便服务器直接复用。更大的 `data/preprocessed/` 和 `data/features/c2/` 缓存仍然留在本机或服务器本地；服务器首次运行会重建，也可以后续用单独文件传输方式同步。

## 当前服务器命令

服务器拉取 `dev` 后，应该已经能拿到小型 S1/S2 checkpoint。正式 catastrophic baseline 运行命令：

```bash
git fetch origin
git checkout dev
git pull origin dev
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist.yaml --device cuda --run-name paper_ch4_catastrophic_source_seed0
```

第一次服务器运行仍可能生成 `data/preprocessed/paper_source/` 和 `data/features/c2/`；后续运行可以在服务器本地复用。

## 文档地图

- active baseline 配置指南：`configs/baseline/README.md`
- baseline 复现计划：`REPRODUCTION_PLAN.md`
- catastrophic forgetting 状态和实验记录：`CATASTROPHIC_FORGETTING_REPRODUCTION.md`
- NGSG 设计笔记：`NGSG_SNN_Implementation.md`
- agent 工作流 skill：`.cursor/skills/ngsg-repo-workflow/SKILL.md`
