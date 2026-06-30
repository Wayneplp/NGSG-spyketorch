# utils

Shared helpers for configuration, file IO, seeds, and data handling live here.

Current data responsibilities:

- `data.py` owns dataset construction for the baseline runs.
- Paper-source preprocessing cache is written under `data/preprocessed/paper_source/<hash>/`.
- EMNIST letters can fall back to raw idx / idx.gz files under `data/emnist/EMNIST/raw/gzip/` when torchvision's processed split path is unstable.
- Generated datasets and `.pt` cache files are runtime artifacts and must stay out of git.

When changing data loading or cache keys, run at least a dry run and document any compatibility change in the top-level reproduction notes.
Feature-cache notes:

- Cached datasets carry a `feature_kind` marker so trainer code can distinguish raw/preprocessed inputs from cached C2/S3-input tensors.
- Cache fingerprints include source cache directories to avoid collisions between tasks with the same length.
