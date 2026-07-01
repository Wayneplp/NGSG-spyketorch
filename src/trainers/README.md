# trainers

这里放训练和评估流程。

当前 baseline 路线：

- `baseline_trainer.py` 是 `scripts/run_baseline.py` 使用的主入口。
- `model.architecture: paper_spyketorch` 会选择 paper-source SpykeTorch/Mozafari 风格路径。
- `learning_rule: paper_source_rstdp` 使用和论文源代码路线对齐的 reward / anti-reward STDP 行为。
- 长时间运行产生的日志和结果应写到被忽略的运行目录，不要写进 git 跟踪的状态文件。

修改 trainer 行为前，先判断它属于 baseline 复现、共享基础设施，还是 NGSG 专属逻辑。共享基础设施放在 `dev`；NGSG 专属行为不要混进 baseline 路线，除非已经明确要集成。

复用入口：

- `train.feature_checkpoint` 可以加载/保存 S1/S2 paper feature checkpoint。
- `train.c2_feature_cache` 可以生成 C2 pooled spikes，并让 S3 直接从缓存特征训练。
- 这些入口是本地运行优化；生成文件放在被忽略的 `checkpoints/` 和 `data/` 路径下。
