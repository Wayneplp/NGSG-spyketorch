# continual

这里放持续学习和 NGSG 相关逻辑，例如 winner-frequency 统计、novelty score、neuron partition、reserve branch 和 reserve-neuron 调用策略。

当前顺序：

1. 先保证 catastrophic baseline 跑通并可解释。
2. 再加入 winner-frequency logging。
3. 最后接入 NGSG 的 novelty gate、reserve neuron 和 synapse growth。

这里的代码应尽量不破坏 paper-source baseline 路线。
