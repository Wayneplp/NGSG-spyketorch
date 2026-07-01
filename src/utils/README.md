# utils

这里放配置、文件 IO、随机种子和数据处理等共享工具。

当前数据相关职责：

- `data.py` 负责 baseline run 的数据集构建。
- paper-source 预处理缓存写入 `data/preprocessed/paper_source/<hash>/`。
- 当 torchvision 的 EMNIST processed split 路径不稳定时，EMNIST letters 可以从 `data/emnist/EMNIST/raw/gzip/` 下的 raw idx / idx.gz 文件回退读取。
- 下载数据集和 `.pt` 缓存都是运行产物，不应进入 git。

修改数据加载或 cache key 时，至少跑一次 dry-run，并把兼容性变化记录到顶层复现文档。

feature-cache 注意点：

- cached dataset 带有 `feature_kind` 标记，方便 trainer 区分原始/预处理输入和缓存的 C2/S3-input tensor。
- cache fingerprint 会包含源 cache 目录，避免不同任务长度相同导致冲突。
