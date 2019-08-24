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


