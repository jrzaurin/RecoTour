'''
Pytorch Implementation of Neural Graph Collaborative Filtering (NGCF) by:
Wang Xiang et al. Neural Graph Collaborative Filtering
'''
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import pdb

from torch import nn


class NGCF_BPR(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, adjacency_matrix, layers,
        node_dropout, mess_dropout, regularization, n_fold, batch_size):
        super(NGCF_BPR, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.A = adjacency_matrix
        self.emb_dim = emb_dim
        self.layers = layers
        self.n_layers = len(layers)
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout*self.n_layers
        self.reg = regularization
        self.n_fold = n_fold
        self.batch_size = batch_size

        self.embeddings_user = nn.Embedding(n_users, emb_dim)
        self.embeddings_item = nn.Embedding(n_items, emb_dim)
        self.g_embeddings_user = nn.Embedding(n_users, np.sum(layers) + emb_dim)
        self.g_embeddings_item = nn.Embedding(n_items, np.sum(layers) + emb_dim)
        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()

        features = [emb_dim] + layers
        for i in range(1,len(features)):
                self.W1.append(nn.Linear(features[i-1],features[i]))
                self.W2.append(nn.Linear(features[i-1],features[i]))

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    @staticmethod
    def convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _edge_dropout_sparse(self, X, keep_prob):

        random_tensor = keep_prob
        random_tensor += torch.FloatTensor(X._nnz()).uniform_()
        dropout_mask = random_tensor.floor()
        dropout_tensor = torch.sparse.FloatTensor(X.coalesce().indices(), dropout_mask, X.size())
        X_w_dropout = X.mul(dropout_tensor)

        return  X_w_dropout.mul(1./keep_prob)

    def _node_dropout_sparse(self, X, keep_prob):

        random_array = keep_prob
        random_array += np.random.rand(X.size()[0])
        dropout_mask = np.floor(random_array)
        dropout_mask = np.tile(dropout_mask.reshape(-1,1), X.size()[1])
        dropout_tensor = self.convert_sp_mat_to_sp_tensor(sp.csr_matrix(dropout_mask))
        X_w_dropout = X.mul(dropout_tensor)

        return  X_w_dropout.mul(1./keep_prob)

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self.convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self.convert_sp_mat_to_sp_tensor(X[start:end])
            A_fold_hat.append(self._edge_dropout_sparse(temp, 1 - self.node_dropout))

        return A_fold_hat

    def _create_ngcf_embed(self):

        if self.node_dropout > 0.:
            A_fold_hat = self._split_A_hat_node_dropout(self.A)
        else:
            A_fold_hat = self._split_A_hat(self.A)

        ego_embeddings = torch.cat([self.embeddings_user.weight, self.embeddings_item.weight], 0)
        pred_embeddings = [ego_embeddings]

        for k in range(self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))

            weighted_sum_emb = torch.cat(temp_embed, 0)
            affinity_emb = ego_embeddings.mul(weighted_sum_emb)

            t1 = self.W1[k](weighted_sum_emb)
            t2 = self.W2[k](affinity_emb)

            ego_embeddings = nn.Dropout(self.mess_dropout[k])(F.leaky_relu(t1 + t2))
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            pred_embeddings += [norm_embeddings]

        pred_embeddings = torch.cat(pred_embeddings, 1)
        u_g_embeddings, i_g_embeddings = pred_embeddings.split([self.n_users, self.n_items], 0)
        self.g_embeddings_user.weight = torch.nn.Parameter(u_g_embeddings)
        self.g_embeddings_item.weight = torch.nn.Parameter(i_g_embeddings)

    def forward(self,u,i,j):

        self._create_ngcf_embed()
        u_emb = self.g_embeddings_user(u)
        p_emb = self.g_embeddings_item(i)
        n_emb = self.g_embeddings_item(j)

        return u_emb, p_emb, n_emb

    def bpr_loss(self, u, i, j):
        y_ui = torch.mul(u, i).sum(dim=1)
        y_uj = torch.mul(u, j).sum(dim=1)
        log_prob = (torch.log(torch.sigmoid(y_ui-y_uj))).mean()

        l2norm = (torch.sum(u**2)/2. + torch.sum(i**2)/2. + torch.sum(j**2)/2.).mean()
        l2reg  = self.reg*l2norm

        return -log_prob + l2reg


class NGCF_BCE(NGCF_BPR):
    def __init__(self, n_users, n_items, emb_dim, adjacency_matrix, layers,
        node_dropout, mess_dropout, regularization, n_fold, mlp_layers, mlp_dropouts):
        super().__init__(n_users, n_items, emb_dim, adjacency_matrix, layers,
        node_dropout, mess_dropout, regularization, n_fold)

        self.mlp = nn.Sequential()
        features = [emb_dim + np.sum(layers)] + mlp_layers
        for i in range(1,len(features)):
            self.mlp.add_module("linear%d" %i, nn.Linear(features[i-1],features[i]))
            self.mlp.add_module("relu%d" %i, torch.nn.ReLU())
            self.mlp.add_module("dropout%d" %i , torch.nn.Dropout(p=mlp_dropouts[i-1]))
        self.out = nn.Linear(in_features=features[-1], out_features=1)

    def forward(self,u,i):

        self._create_ngcf_embed()

        u_emb = self.g_embeddings_user(u)
        i_emb = self.g_embeddings_item(i)

        emb_vector = torch.cat([u_emb,i_emb], dim=1)
        emb_vector = self.mlp(emb_vector)
        preds = torch.sigmoid(self.out(emb_vector))

        return preds
