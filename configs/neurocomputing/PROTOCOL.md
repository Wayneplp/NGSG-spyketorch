# Neurocomputing Formal Experiment Protocol

Frozen for seeds **1–5** (seed 0 is development-only and must not enter main-table statistics).

## Four main-text components

1. **Role-aware training** — winner-frequency roles after Task 1; Task-2 WTA/STDP schedule on stable/shared/reserve.
2. **K=8 multi-prototype routing** — k-means prototypes from train responses; cosine task scores; no Task ID at test.
3. **Unified Route Gate** — single `τ_route` form; **selected per seed on that seed’s validation split only**.
4. **Bidirectional Gain Selector** — two MLPs (`T1→T2`, `T2→T1`); gain labels `{-1,0,+1}`; weights `(1.0, 3.0, 0.25)`; `τ_gain = 0` (not scanned).

## Supervision sources (must be stated in the paper)

| Component | Supervision |
| --- | --- |
| Role partition | Task-1 winner frequency |
| Prototypes | Train-sample S3 responses |
| Selector | Dual-path correctness change labels (validation fit split) |
| `τ_route` | Validation macro only |
| Test inference | No Task ID; official test used once |

## Matrix

- Orders: `mnist_to_emnist` (`task_order: [0,1]`), `emnist_to_mnist` (`task_order: [1,0]`).
- Seeds: `1,2,3,4,5` with `seed == subset_seed`.
- Methods per cell: `unprotected`, `frozen_large_weights`, `langevin`, `role_train` (+ post-hoc `k8`, `gain_selector`, `oracle`).

## Fairness / comparability

- **Frozen Large Weights**: same network, data, epochs, seeds as unprotected; freeze top 10% Task-1 conv3 weights (`high_percentile: 90`).
- **Langevin**: same protocol plus Gaussian noise on conv3 after each R-STDP update (`noise_std: 0.001` default). If the original notebook uses a different `noise_std`, report as same-protocol reconstruction and list unmatched hyperparameters under “incomparable items”; **never** place literature Table-1 numbers in the fair main table.

## Artifacts per run

```
experiments/<run_name>/
  resolved_config.json
  subset_indices.json
  result.json
  artifacts/model_after_task{1,2}.pt
  diagnostics/   # K8 / Gain / Oracle post-hoc
```

## Stop conditions

- Do not retune `w_h` / directional route gates from seed-0 test reads.
- Oracle is a diagnostic ceiling only.
- Claim scope: mitigate WTA inference competition mismatch in locally trained SNNs — not general catastrophic forgetting for arbitrary task sequences.
