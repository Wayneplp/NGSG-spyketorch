# Baseline 配置说明

这个目录现在只保留 active baseline 入口。旧的 toy、probe、stabilizer、frozen-weights、Langevin 和 joint-training YAML 已从 active config tree 删除，避免以后不知道该跑哪个配置。

## 现在应该跑哪个 YAML？

| 配置 | 用途 | 说明 |
| --- | --- | --- |
| `catastrophic_mnist_emnist.yaml` | 服务器正式完整 baseline | 完整 paper-source MNIST -> EMNIST catastrophic forgetting 流程。存在已跟踪 S1/S2 checkpoint 时会从 `checkpoints/features/` 加载，并在本地生成/复用 `data/features/c2/`。 |
| `catastrophic_mnist_emnist_feature_checkpoint.yaml` | 重建 S1/S2 checkpoint 和 C2 cache | feature-only 模式。按论文对齐设定训练 S1 2 epoch、S2 4 epoch，然后跳过 S3 R-STDP 和评估。只有 checkpoint 缺失或需要重建时才跑。 |
| `catastrophic_mnist_emnist_paper_medium.yaml` | 本地中等规模诊断 | 每类 100 个样本，S3 训练轮次更短。用于检查代码路径和学习曲线，不作为最终论文数字。 |

## 推荐命令

服务器正式完整 baseline：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist.yaml --device cuda --run-name paper_ch4_catastrophic_source_seed0
```

仅在需要时重建 feature checkpoint 和 C2 cache：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml --device cuda --run-name paper_feature_checkpoint_full
```

本地中等规模诊断：

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml --device cuda --run-name paper_medium_source_port_seed0
```

## 哪些内容进入 git？

- 本目录中的 active YAML 配置。
- 小型可复用 S1/S2 checkpoint：
  - `checkpoints/features/paper_task1_s1e2_s2e4_f26edcfb75b5d681.pt`
  - `checkpoints/features/paper_task2_s1e2_s2e4_60c0a06b55746fb6.pt`

## 哪些内容不进入 git？

- `data/preprocessed/`：paper-source 输入预处理缓存。
- `data/features/c2/`：大型 C2 pooled feature cache。
- `experiments/`、`logs/`、`results/`：运行生成产物。

C2 cache 体积很大，服务器首次运行时本地重建即可；如果确实要同步，应通过单独文件传输方式处理，不要提交到 git。

## 已删除的旧配置

以下 YAML 已从 active 配置中删除：

- `catastrophic.yaml`
- `joint_training.yaml`
- `frozen_large_weights.yaml`
- `langevin.yaml`
- `catastrophic_mnist_emnist_probe.yaml`
- `catastrophic_mnist_emnist_medium.yaml`
- `catastrophic_mnist_emnist_medium_stabilizer_off.yaml`

当前范围是先复现 catastrophic forgetting，再加入 winner-frequency logging 和 NGSG。现在不复现 joint training。
