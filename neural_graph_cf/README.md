# Neural Graph Collaborative Filtering

Pytorch implementation of the algorithm described in [Xiang Wang, et. all
2019, Neural Graph Collaborative
Filtering](https://arxiv.org/pdf/1905.08108.pdf).

I believe the nature of this algorithm, where a huge graph is executed in
every forward pass, makes it more suitable for Deep Learning frames that use
static graphs (e.g. TF). Therefore, [the original
implementation](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
by the authors is faster than my Pytorch implementation. The results (as
tested with the Gowalla dataset. See their paper) are entirely consistent.

Nonetheless, a detail description of the algorithm components and the
training/validation process can be found in the notebooks.

Examples of how to run it can be found in `run_experiments.sh`. I have tried a
number of different combinations (not many) including [Cyclic learning
rates](https://arxiv.org/pdf/1506.01186.pdf) and
[RAdam](https://arxiv.org/abs/1908.03265v1) optimizer.

