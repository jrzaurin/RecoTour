import numpy as np

# let's build a fake example to illustrate how one could use the normalised
# discounted cumulative gain (ndcg) for the excercise here.

# To use this metric one needs an estimate of the real relevance of the items
# we recommend. In our case, this would be the interest.

# As it happened before, there is a caveat worth mentioning here. We are
# recommending a few hundred coupons while the users only interacted with a
# few. Therefore, it is not possible to know/compute the real relevance for
# all recommended items. there are 2 ways to go about it:

# 1-We can rank based on interest, keep the top 10 and compute the metric with
# the intersect between those 10 and the ones that the user interacted with, or

# 2-We can extract all the coupons that the user interacted with, with their
# rank. If there are more than 10, we cut at 10, if there are less, we
# proceed.

# We will implement both methods

# let's assume we have 10 users
user_ids = ['user_'+str(i) for i in np.arange(10)]

# and we recommend 50 coupons
recommended_coupon_ids = ['coupon_'+str(i) for i in np.arange(50)]
n_rec = len(recommended_coupon_ids)


# each user interacted with a number of coupons (between 1-50) during testing
np.random.seed(1)
real_interactions = {}
for user in user_ids:
	n_coupons = np.random.randint(1,50)
	coupon_ids = random.sample(recommended_coupon_ids, n_coupons)
	interest = [random.random() for _ in range(len(coupon_ids))]
	real_interactions[user] = dict(zip(coupon_ids,interest))

np.random.seed(2)
recommendations = {}
for user in user_ids:
	interest = [np.random.random() for _ in range(len(recommended_coupon_ids))]
	recommendations[user] = dict(zip(recommended_coupon_ids,interest))

# In reality what we need to evaluate is a list of dictionaries
real_interactions = [v for v in real_interactions.values()]
recommendations = [v for v in recommendations.values()]

def dcgk(actual, predicted, method=1, k=10):
    """
    Computes the discounted cumulative gain at k.
    This function computes the discounted cumulative gain at k between two
    dictionaries of items: scores

    Parameters
    ----------
    actual : dict
             A dict of elements: {'item': score} for the actual interactions
    predicted : dict
                A dict of predicted elements {'item': score}
    k : int, optional
        The maximum number of predicted elements
	method: int
        1: sort the recommendations and cut at k. Then intersect with actual
        2: sort the recommendations and intersect with actual. Then cut at min(len(actual), k)

    Returns
    -------
    score : double
            discounted cumulative gain at k
    """
	if method==1:
		# First rank the recommendations based on predicted interest and keep the top 10
		ranked_rec = sorted([(k,v) for k,v in predicted.items()],
			key=lambda x: x[1], reverse=True)[:k]
		# Among these top 10 keep those that the user did interact with and their rank
		ranked_rec = [ (i,k) for i,(k,v) in enumerate(ranked_rec) if k in actual.keys()]
		# then extract the rank and the relevance (real interest)
		ranked_scores = [(k,actual[v]) for k,v in dict(ranked_rec).items()]
	elif method==2:
		# First rank the recommendations based on predicted interest
		ranked_rec = sorted([(k,v) for k,v in predicted.items()],
			key=lambda x: x[1], reverse=True)
		# Select those that the user did interact with and their rank
		ranked_rec = [ (i,k) for i,(k,v) in enumerate(ranked_rec) if k in actual.keys()]
		# then extract the rank and the relevance (real interest) and keep min(len(actual), k)
		ranked_scores = [(k,actual[v]) for k,v in dict(ranked_rec).items()][:min(len(actual), k)]

	if not ranked_scores:
		return 0.
	else:
		rank = np.asarray([s[0] for s in ranked_scores])
		score = np.asarray([s[1] for s in ranked_scores])
		return np.sum((2**score - 1) / np.log2(rank+1))

def ndcgk(actual, predicted, method=1, k=10):
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Parameters
    ----------
    actual : dict
             A dict of elements: {'item': score} for the actual interactions
    predicted : dict
                A dict of predicted elements {'item': score}
    k : int, optional
        The maximum number of predicted elements
	method: int
        1: sort the recommendations and cut at k. Then intersect with actual
        2: sort the recommendations and intersect with actual. Then cut at min(len(actual), k)

    Returns
    -------
    score : double
            normalised discounted cumulative gain at k
    """
	return dcgk(actual,predicted) / dcgk(actual,actual)


def mndcgk(actual, predicted, method=1, k=10):
	"""
	Mean normalized discounted cumulative gain (NDCG) at rank K over all recommendations
    Parameters
    ----------
    actual : dict
             A dict of elements: {'item': score} for the actual interactions
    predicted : dict
                A dict of predicted elements {'item': score}
    k : int, optional
        The maximum number of predicted elements
	method: int
        1: sort the recommendations and cut at k. Then intersect with actual
        2: sort the recommendations and intersect with actual. Then cut at min(len(actual), k)

    Returns
    -------
    score : double
            mean normalised discounted cumulative gain at k
    """
	return np.mean([ndcgk(a,p,method,k) for a,p in zip(actual,predicted)])


# An alternative way of computing it based on
# [https://www.kaggle.com/davidgasquez/ndcg-scorer]

def dcg_score(y_true, y_score, k=10):
    """
    Computes the discounted cumulative gain at k.
    This function computes the discounted cumulative gain at k

    Parameters
    ----------
    y_true : array
             binary array with 1 in locations corresponding to actual interactions
    y_score : array
             array with the scores for all recommendations
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    np.sum(gain / discounts) : double
            discounted cumulative gain at k
    """

    rec = np.argsort(y_score)[::-1][:k]
    y_true = np.take(y_true, rec)
    gain = y_true
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(actual, predicted, k=10):
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Parameters
    ----------
    actual : dict
             A dict of elements: {'item': score} for the actual interactions
    predicted : dict
                A dict of predicted elements {'item': score}
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    np.mean(scores) : double
            mean normalised discounted cumulative gain at k
    """

	scores = []
	for i, (t,r) in enumerate(zip(actual,predicted)):
		idx = [int(c.split("_")[1]) for c in t.keys()]
		y_true = np.zeros(n_rec, dtype='int')
		y_true[idx] = 1
		y_score = np.array(list(r.values()))
		dcg = dcg_score(y_true, y_score, k)
		idcg = dcg_score(y_true, y_true, k)
        score = dcg / idcg
        scores.append(score)

    return np.mean(scores)
