import numpy as np
import random as rd
import os
import scipy.sparse as sp

from time import time


class Data(object):
    def __init__(self, path, batch_size, val=False):
        self.path = path
        self.batch_size = batch_size

        train_file = os.path.join(path,'train.txt')
        # are we running validation or "final" test
        test_file = os.path.join(path,'valid.txt') if val else  os.path.join(path,'test.txt')

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.Rtr = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.Rte = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.Rtr[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]

                    for i in train_items:
                        self.Rte[uid, i] = 1.

                    self.test_set[uid] = test_items

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def normalized_adj_single(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.Rtr.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        norm_adj_mat = self.normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = self.normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def sample_pos_items_for_u(self, u, num):
        pos_items = self.train_items[u]
        n_pos_items = len(pos_items)
        pos_batch = []
        while True:
            if len(pos_batch) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch

    def sample_neg_items_for_u(self, u, num):
        neg_items = []
        while True:
            if len(neg_items) == num: break
            neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
            if neg_id not in self.train_items[u] and neg_id not in neg_items:
                neg_items.append(neg_id)
        return neg_items

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []
        for u in users:
            pos_items += self.sample_pos_items_for_u(u, 1)
            neg_items += self.sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


# This should go to a tests folder
def test_data_building_process(data_path, org_dataset, user_map_fname, item_map_fname,
    generator, N=10):
    """
    Simple funcion to test we have built datasets correctly
    """

    # Read the original dataset (no mapping)
    df = pd.read_csv(data_path/org_dataset)
    # Read the mappings
    user_map = pd.read_csv(data_path/user_map_fname, sep=" ").set_index('orig_id').to_dict()['remap_id']
    item_map = pd.read_csv(data_path/item_map_fname, sep=" ").set_index('orig_id').to_dict()['remap_id']
    # pick N random users
    random_users = np.random.choice(df.user.unique(), N, replace=False)
    temp = df[df.user.isin(random_users)]

    # for each user, make sure that all items in data_generator train and test
    # sets are in  the original dataset. Note that since the Data class only
    # loads two datasets (train/valid) or (train/test) there will be some items
    # in the original dataset that will not be present in "all_items"
    users_check = []
    for u in random_users:
        mapped_user_id = user_map[u]
        # get training and test/valid items
        train_items = data_generator.train_items[mapped_user_id]
        test_items = data_generator.test_set[mapped_user_id]
        all_items = train_items+test_items
        # map them back to their original IDs
        org_items_id1 = [k for k,v in item_map.items() if v in all_items]
        # get their original IDs
        org_items_id2 = temp[temp.user == u].item.tolist()
        # check that every item in all_items is in org_items_id2
        users_check.append(len(np.setdiff1d(org_items_id1, org_items_id2)) == 0)

    return np.all(users_check)