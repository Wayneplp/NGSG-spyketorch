# NGSG SpykeTorch Workspace

This repository is for:

- reproducing SpykeTorch continual-learning baselines,
- implementing NGSG components,
- storing experiment outputs, logs, and checkpoints.

Recommended workflow:

1. Reproduce the original baselines first.
2. Add winner-frequency analysis without changing learning behavior.
3. Implement minimal NGSG modules incrementally.

Top-level folders:

- `configs/`: experiment configs
- `data/`: datasets or dataset links/instructions
- `scripts/`: runnable entry scripts
- `src/`: model and training code
- `experiments/`: per-run notes and outputs
- `logs/`: raw logs
- `checkpoints/`: saved models
- `results/`: aggregated tables and figures

## Reproducing on another computer

This repository is meant to track the code, configs, and instructions needed to
reproduce experiments, rather than every generated artifact from each run.

What is kept in git:

- source code under `src/`
- runnable scripts under `scripts/`
- experiment configs under `configs/`
- setup instructions in this README

What is intentionally not kept in git:

- downloaded datasets under `data/`
- per-run outputs under `experiments/`
- raw logs under `logs/`
- saved checkpoints under `checkpoints/`
- aggregated result files under `results/`
- local cache files such as `__pycache__/` and `.pyc`

### Setup steps

1. Clone the repository on the new machine.
2. Create and activate a fresh Python virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run a baseline script with one of the configs in `configs/baseline/`.

Example:

```bash
python scripts/run_baseline.py --config configs/baseline/catastrophic.yaml
```

### Notes about data

- The current runnable baseline uses `torchvision.datasets.MNIST` with
  `download=True`, so MNIST will be downloaded automatically to `data/mnist`
  the first time you run it.
- Because datasets and experiment outputs are not stored in git, cloning on a
  new machine will give you the code and configs first, then regenerate data
  downloads and run outputs locally.

### If you want exact continuity across machines

If you need to continue from an existing run instead of re-running it, manually
copy the relevant files from the old machine:

- `checkpoints/` for saved models
- `results/` for summary tables and figures
- specific run folders under `experiments/`

That way, git stays clean while your experiment state can still move between
machines when needed.
