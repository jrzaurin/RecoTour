import torch

from torch import nn

# -> HERE

def torch_forward():

    ego_embeddings = torch.cat([embeddings_user, embeddings_item], 0)
    pred_embeddings = [ego_embeddings]

    for k in range(n_layers):


        temp_embed = []
        for f in range(n_fold):
            temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))

        weighted_sum_emb = torch.mm(A, ego_embeddings)
        affinity_emb = ego_embeddings.mul(weighted_sum_emb)

        t1 = W1[k](weighted_sum_emb)
        t2 = W2[k](affinity_emb)

        ego_embeddings = F.leaky_relu(t1 + t2)
        norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

        pred_embeddings += [norm_embeddings]

    pred_embeddings = torch.cat(pred_embeddings, 1)
    g_embeddings_user, g_embeddings_item = pred_embeddings.split([n_users, n_items], 0)

    return g_embeddings_user, g_embeddings_item


def tf_forward():

    ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
    all_embeddings = [ego_embeddings]

    for k in range(0, n_layers):

        # sum messages of neighbors.
        side_embeddings = tf.matmul(A, ego_embeddings)
        # transformed sum messages of neighbors.
        sum_embeddings = tf.nn.leaky_relu(
            tf.matmul(side_embeddings, weights['W_gc_%d' % k]) + weights['b_gc_%d' % k])

        # bi messages of neighbors.
        bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
        # transformed bi messages of neighbors.
        bi_embeddings = tf.nn.leaky_relu(
            tf.matmul(bi_embeddings, weights['W_bi_%d' % k]) + weights['b_bi_%d' % k])

        # non-linear activation.
        ego_embeddings = sum_embeddings + bi_embeddings

        # normalize the distribution of embeddings.
        norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

        all_embeddings += [norm_embeddings]

    all_embeddings = tf.concat(all_embeddings, 1)
    u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)
    return u_g_embeddings, i_g_embeddings
