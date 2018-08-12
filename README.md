# RecoTour

This repo intends to be a tour through some recommendation algorithms in
python using various dataset.

At the moment there is only one dataset, the
[Ponpare](https://www.kaggle.com/c/coupon-purchase-prediction) coupon dataset,
which corresponds to a coupon purchase prediction competition at Kaggle (i.e.
recommending coupons to customers).

**The core of the repo are the notebooks** in the  `Ponpare` directory. They
intend to be self-contained and in consequence, there is some of code
repetition. The code is, of course, "notebook-oriented". In the future I will
include a more modular, nicer version of the code in the directory
`py_scrips`. If you look at it now it might burn your eyes (you are warned, I
would not go there). The notebooks have plenty of explanations and references
to relevant papers or packages. My intention was to focus on the code, but you
will also find some math.

Overall, this is what you will find in the notebooks:

1. Data processing, with a deep dive into feature engineering
2. Most Popular recommendations (the baseline)
3. Item-User similarity based recommendations
4. kNN Collaborative Filtering recommendations
5. GBM based recommendations using `lightGBM` with a tutorial on how to optimize gbms
6. Non-Negative Matrix Factorization recommendations
7. Factorization Machines recommendations using `xlearn`
8. Field Aware Factorization Machines recommendations using `xlearn`
9. Deep Learning based recommendations (Wide and Deep) using `pytorch`

In the last notebook I have also included what it could be a good solution for
this problem in "the real word". So, where do we go from here? These are some
of the things I intend to include when I have the time:

### Datasets

1. The [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/) dataset (or well, a fraction of it)
2. Other datasets that are better suited for Deep Learning based algorithms,
containing text, images if possible and user behavior.

### Algorithms

1. Graph based recommendation algorithms
2. [Neural Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)
3. [Sequence based recommendation algorithms](https://florianwilhelm.info/2018/08/multiplicative_LSTM_for_sequence_based_recos/)
4. Others...

### Miscellaneous

1. Illustration of how to use other evaluation metrics apart from the one
shown in the notebooks ( the mean average precision or MAP) such as the
Normalized Discounted Cumulative Gain
([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)).

Anyway, I hope the code here is useful to someone. If you have any idea on how to improve the content of the repo, or you want to contribute, let me know.