import numpy as np
import argparse
import csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate toy/small dataset.")
    parser.add_argument('--n_users', type=int, default=100,
        help="number of users")
    parser.add_argument('--n_items', type=int, default=200,
        help="number of n_items")
    parser.add_argument('--min_interactions', type=int, default=11,
        help="min number of interactions per user")
    parser.add_argument('--max_interactions', type=int, default=51,
        help="man number of interactions per user")
    args = parser.parse_args()

    np.random.seed(1)

    n_users = args.n_users
    n_items = args.n_items
    min_interactions = args.min_interactions
    max_interactions = args.max_interactions

    interactions = []
    for u in range(n_users):
        # number of interactions between min-max
    	n_interactions = np.random.randint(min_interactions, max_interactions)
        # random choice n_interactions
    	items = np.random.choice(range(n_items), n_interactions, replace=False)
    	interactions.append((u, items))

    def train_test_split(u, i_l, p=0.8):
        s = np.floor(len(i_l)*p).astype('int')
        train = [u] + list(np.random.choice(i_l, s, replace=False))
        test  = [u] + list(np.setdiff1d(i_l, train))
        return (train, test)

    train_test = [train_test_split(i[0],i[1]) for i in interactions]

    train_fname = 'Data/toy_data/train.txt'
    test_fname = 'Data/toy_data/test.txt'

    with open(train_fname, 'w') as trf, open(test_fname, 'w') as tef:
        trwrt = csv.writer(trf, delimiter=' ')
        tewrt = csv.writer(tef, delimiter=' ')
        trwrt.writerows([train_test[i][0] for i in range(len(train_test))])
        tewrt.writerows([train_test[i][1] for i in range(len(train_test))])