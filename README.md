# NGSG SpykeTorch Workspace

This repository is for three connected jobs:

- reproducing SpykeTorch continual-learning baselines from *Continuous Learning of Spiking Networks Trained with Local Rules*,
- implementing NGSG components after the baseline is stable,
- maintaining shared experiment infrastructure such as data loading, preprocessing cache, and training logs.

Current mainline status, 2026-06-30:

- `dev` is the integration branch used for local and server runs.
- `baseline/continuous-learning` is for baseline reproduction work.
- `ngsg/novelty-gated-growth` is for NGSG-specific changes.
- The paper-source catastrophic baseline uses the SpykeTorch/Mozafari-style path in `src/trainers/baseline_trainer.py` and `src/utils/data.py`.
- The server-side preprocessing cache and EMNIST raw idx fallback have been merged into `dev`.
- Server runtime notes such as `SERVER_LATEST_STATUS.md` are local-only and should not be committed.

Current latest status:

- `dev` contains the feature-cache implementation and the reusable paper-source S1/S2 checkpoints.
- The tracked checkpoints are:
  - `checkpoints/features/paper_task1_s1e2_s2e4_f26edcfb75b5d681.pt`
  - `checkpoints/features/paper_task2_s1e2_s2e4_60c0a06b55746fb6.pt`
- These checkpoint files were built with the full paper-aligned feature schedule: S1 STDP 2 epochs and S2 STDP 4 epochs on 24,000 samples per task.
- `data/preprocessed/` and `data/features/c2/` are still local/server runtime caches and are not tracked in git.
- The next server run should usually use `configs/baseline/catastrophic_mnist_emnist.yaml`; use `configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml` only if the tracked checkpoints are missing or need to be regenerated. See `configs/baseline/README.md` for the active config guide.
Recommended workflow:

1. Keep shared infrastructure on `dev`.
2. Do baseline reproduction on `baseline/continuous-learning`, then merge back to `dev`.
3. Do NGSG implementation on `ngsg/novelty-gated-growth`, then merge back to `dev` after the baseline is understood.
4. Merge `dev` to `main` only after a verified, externally usable snapshot.

Top-level folders:

- `configs/`: experiment configs
- `scripts/`: runnable entry scripts
- `src/`: model, data, plasticity, and training code
- `approx/legacy_approx/`: older approximate implementations kept for reference
- `experiments/`: per-run outputs generated locally
- `logs/`: raw logs generated locally
- `checkpoints/`: saved models generated locally
- `results/`: aggregated tables and figures generated locally
- `data/`: downloaded datasets and preprocessing cache generated locally

## Reproducing On Another Computer

This repository tracks code, configs, and instructions. It does not track downloaded datasets, generated caches, logs, checkpoints, or run outputs.

What is kept in git:

- source code under `src/`
- runnable scripts under `scripts/`
- experiment configs under `configs/`
- documentation and workflow notes

What is intentionally not kept in git:

- downloaded datasets under `data/`
- preprocessed `.pt` cache files under `data/preprocessed/`
- per-run outputs under `experiments/`
- raw logs under `logs/`
- generated checkpoints under `checkpoints/`, except the small reusable S1/S2 feature checkpoints in `checkpoints/features/`
- aggregated generated results under `results/`
- server runtime snapshots such as `SERVER_LATEST_STATUS.md`

### Setup Steps

1. Clone the repository.
2. Create and activate a Python environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run a dry run before long experiments.

Example dry run:

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist.yaml --device auto --dry-run --run-name paper_source_strict_dryrun
```

Example medium diagnostic run:

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist_paper_medium.yaml --device auto --run-name paper_medium_source_port_seed0
```

## Data And Cache Notes

- MNIST is downloaded through torchvision when needed.
- EMNIST letters are read from torchvision when possible, with a raw idx / idx.gz fallback for server environments where `torchvision.datasets.EMNIST(split="letters")` is unstable.
- Paper-source preprocessing cache is written under `data/preprocessed/paper_source/<hash>/`.
- Cache files are generated artifacts and must stay out of git.

On a server, keep the checkout on `dev` or a tag cut from `dev`:

```bash
git fetch origin
git checkout dev
git pull origin dev
pip install -r requirements.txt
```

Useful cache checks on the server:

```bash
find data/preprocessed/paper_source -name '*.pt' | wc -l
tail -n 50 logs/preprocess_*.log
```


## Training Reuse Strategy

Paper-source runs now have three reuse layers:

- Input preprocessing cache under `data/preprocessed/paper_source/<hash>/`.
- S1/S2 feature checkpoints under `checkpoints/features/`.
- C2 pooled feature cache under `data/features/c2/<hash>/`, which is the preferred fast entry point for repeated S3/NGSG experiments.

The long-lived feature checkpoint should use the paper-aligned feature schedule: S1 STDP for 2 epochs and S2 STDP for 4 epochs. Smoke tests may use smaller datasets, but should not replace the shared S1/S2 checkpoint used for real baseline and NGSG comparisons.

Use `configs/baseline/catastrophic_mnist_emnist_feature_checkpoint.yaml` to build these reusable artifacts without running S3 R-STDP or evaluation.

The full `S1=2/S2=4` `checkpoints/features/paper_task*_s1e2_s2e4_*.pt` files are small and tracked in git for server reuse. The larger `data/preprocessed/` and `data/features/c2/` caches stay out of git; rebuild them on the server or transfer them separately.


### Current Server Command

After pulling `dev`, the server should already have the small S1/S2 checkpoints from git. Run the full catastrophic baseline like this:

```bash
git fetch origin
git checkout dev
git pull origin dev
python scripts/run_baseline.py --config configs/baseline/catastrophic_mnist_emnist.yaml --device cuda --run-name paper_ch4_catastrophic_source_seed0
```

The first server run may still materialize `data/preprocessed/paper_source/` and `data/features/c2/`; later runs can reuse them locally.

## Documentation Map

- Active baseline config guide: `configs/baseline/README.md`
- Baseline reproduction plan: `REPRODUCTION_PLAN.md`
- Catastrophic forgetting status and experiment log: `CATASTROPHIC_FORGETTING_REPRODUCTION.md`
- NGSG design notes: `NGSG_SNN_Implementation.md`
- Agent workflow skill: `.cursor/skills/ngsg-repo-workflow/SKILL.md`