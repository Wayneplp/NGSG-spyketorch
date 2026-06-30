# trainers

Training and evaluation routines live here.

Current baseline route:

- `baseline_trainer.py` is the main entry used by `scripts/run_baseline.py`.
- `model.architecture: paper_spyketorch` selects the paper-source SpykeTorch/Mozafari-style path.
- `learning_rule: paper_source_rstdp` uses the reward / anti-reward STDP behavior aligned with the source-paper route.
- Long runs should write generated logs and results under ignored runtime folders, not into git-tracked status files.

Before changing trainer behavior, check whether the change is baseline reproduction, shared infra, or NGSG-specific logic. Shared infra belongs on `dev`; NGSG-only behavior should stay out of the baseline route unless intentionally integrated.