'''
Test that the loss function I implemented lead to identical results to that of
the original code release. The tf code here is a direct copy and paste from
here:
https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/NGCF.py
'''
import numpy as np

import torch
import tensorflow as tf

tf.enable_eager_execution()

n_users = 100
n_items = 1000
n_emb = 12
reg = 1e-5

pos_items = np.random.choice(n_items, n_users, replace=False)
temp = np.setdiff1d(np.arange(n_items), pos_items)
neg_items = np.random.choice(temp, n_users, replace=False)

emb_users = torch.from_numpy(np.random.rand(n_users, n_emb))
emb_items = torch.from_numpy(np.random.rand(n_items, n_emb))
pos_emb = emb_items[pos_items]
neg_emb = emb_items[neg_items]

def torch_bpr_loss(u,i,j):

    y_ui = torch.mul(u, i).sum(dim=1)
    y_uj = torch.mul(u, j).sum(dim=1)
    log_prob = (torch.log(torch.sigmoid(y_ui-y_uj))).mean()

    l2norm = (torch.sum(u**2)/2. + torch.sum(i**2)/2. + torch.sum(j**2)/2.).mean()
    l2reg  = reg*l2norm

    return -log_prob + l2reg

torch_loss = torch_bpr_loss(emb_users,pos_emb,neg_emb).numpy()

emb_users = tf.convert_to_tensor(emb_users.numpy())
pos_emb = tf.convert_to_tensor(pos_emb.numpy())
neg_emb = tf.convert_to_tensor(neg_emb.numpy())

# Original tf code
    # def create_bpr_loss(self, users, pos_items, neg_items):
    #     pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
    #     neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

    #     regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
    #     regularizer = regularizer/self.batch_size

    #     ## In the first version, we implement the bpr loss via the following codes:
    #     # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
    #     # mf_loss = tf.negative(tf.reduce_mean(maxi))

    #     # In the second version, we implement the bpr loss via the following codes to aviod 'NAN' loss during training:
    #     mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))


    #     emb_loss = self.decay * regularizer

    #     reg_loss = tf.constant(0.0, tf.float32, [1])

    #     return mf_loss, emb_loss, reg_loss

# Slightly adapted (using their first version)
def tf_bpr_loss(u, i, j):
    pos_scores = tf.reduce_sum(tf.multiply(u, i), axis=1)
    neg_scores = tf.reduce_sum(tf.multiply(u, j), axis=1)

    regularizer = tf.reduce_mean(tf.nn.l2_loss(u) + tf.nn.l2_loss(i) + tf.nn.l2_loss(j))
    # regularizer = regularizer/self.batch_size

    maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
    mf_loss = tf.negative(tf.reduce_mean(maxi))

    # emb_loss = self.decay * regularizer
    emb_loss = reg * regularizer


    return mf_loss+emb_loss

tf_loss = np.array(tf_bpr_loss(emb_users,pos_emb,neg_emb))

print(torch_loss, tf_loss)