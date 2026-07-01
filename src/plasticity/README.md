# plasticity

这里放可塑性规则相关逻辑，例如 R-STDP、mask、silent-synapse update，以及后续可能重新引入的 Langevin 相关实验代码。

当前重点仍是 paper-source baseline 的 reward / anti-reward STDP 对齐。旧的 frozen/Langevin 配置已经从 active YAML 中删除；只有在论文对比确实需要时，再重新建立清晰、可复现的配置和实现路径。
