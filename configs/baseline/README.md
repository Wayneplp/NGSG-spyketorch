# Baseline Configs

This directory now keeps only the active baseline entry points. Older toy,
probe, stabilizer, frozen-weights, Langevin, and joint-training YAMLs were
removed from the active config tree so the current reproduction path is not
ambiguous.

## Which YAML Should I Run?

| Config | Use it for | Notes |
| --- | --- | --- |
| `catastrophic_mnist_emnist.yaml` | Main server run and final catastrophic baseline reproduction | Full paper-source MNIST -> EMNIST protocol. Loads the tracked S1/S2 checkpoints from `checkpoints/features/` when present and builds/reuses local C2 cache under `data/features/c2/`. |
| `catastrophic_mnist_emnist_feature_checkpoint.yaml` | Rebuilding reusable S1/S2 checkpoints and C2 cache | Feature-only mode. It trains S1/S2 with the paper-aligned `S1=2` and `S2=4` epoch schedule, then skips S3 R-STDP and evaluation. Run this only when checkpoints are missing or need to be regenerated. |
| `catastrophic_mnist_emnist_paper_medium.yaml` | Local medium diagnostic | Uses 100 samples per label and shorter S3 training. Good for checking code paths and learning curves, but not for final reported numbers. |

## Recommended Commands

Main full baseline on server:

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist.yaml --device cuda --run-name paper_ch4_catastrophic_source_seed0
```

Rebuild feature checkpoints and C2 cache only when needed:

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml --device cuda --run-name paper_feature_checkpoint_full
```

Medium local diagnostic:

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml --device cuda --run-name paper_medium_source_port_seed0
```

## What Is In Git?

- Active YAML configs in this directory.
- Small reusable S1/S2 checkpoints:
  - `checkpoints/features/paper_task1_s1e2_s2e4_f26edcfb75b5d681.pt`
  - `checkpoints/features/paper_task2_s1e2_s2e4_60c0a06b55746fb6.pt`

## What Is Not In Git?

- `data/preprocessed/`: paper-source input preprocessing cache.
- `data/features/c2/`: large C2 pooled feature cache.
- `experiments/`, `logs/`, and `results/`: generated run outputs.

The C2 cache is large, so the server should rebuild it locally on first run or
receive it through a separate file-transfer path. It should not be committed.

## Removed Legacy Configs

The following YAMLs were removed from active use:

- `catastrophic.yaml`
- `joint_training.yaml`
- `frozen_large_weights.yaml`
- `langevin.yaml`
- `catastrophic_mnist_emnist_probe.yaml`
- `catastrophic_mnist_emnist_medium.yaml`
- `catastrophic_mnist_emnist_medium_stabilizer_off.yaml`

Current scope: reproduce catastrophic forgetting first, then add
winner-frequency logging and NGSG. Joint training is not part of the current
reproduction target.