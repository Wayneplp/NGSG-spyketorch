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
