# Neural Graph Collaborative Filtering

Pytorch implementation of the algorithm described in Dawen Liang, et. all
2019, [Variational Autoencoders for Collaborative
Filtering](https://arxiv.org/pdf/1802.05814.pdf). The companion post can be
found
[here](https://jrzaurin.github.io/infinitoml/jupyter/2020/05/15/mult-vae.html).

The reason to focus in this algorithm is because, as shown by Maurizio Ferrari
Dacrema an co-authors in their fantastic paper [Are We Really Making Much
Progress? A Worrying Analysis of Recent Neural Recommendation
Approaches](https://arxiv.org/abs/1907.06902), this is, to that date, the only
Deep-Learning algorithm that obtained competitive results when compared with
non-Deep Learning techniques.

A detail description of the algorithm components and the training/validation
process can be found in the notebooks and the post.

Examples of how to run it can be found in `run_experiments.sh` and the results
of those experiments are in the file `all_results.csv` .

