"""
Pytorch Implementation of Neural Graph Collaborative Filtering (NGCF) by:
Wang Xiang et al. Neural Graph Collaborative Filtering
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import pdb

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BPR(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, reg):
        super().__init__()

        self.u_g_embeddings = nn.Parameter(torch.rand(n_users, emb_dim))
        self.i_g_embeddings = nn.Parameter(torch.rand(n_items, emb_dim))
        self.reg = reg

        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            nn.init.xavier_uniform_(p)

    def forward(self, u, i, j):

        u_emb = self.u_g_embeddings[u]
        p_emb = self.i_g_embeddings[i]
        n_emb = self.i_g_embeddings[j]

        y_ui = torch.mul(u_emb, p_emb).sum(dim=1)
        y_uj = torch.mul(u_emb, n_emb).sum(dim=1)
        log_prob = (torch.log(torch.sigmoid(y_ui - y_uj))).mean()

        bpr_loss = -log_prob
        if self.reg > 0.0:
            l2norm = (
                torch.sum(u_emb ** 2) / 2.0
                + torch.sum(p_emb ** 2) / 2.0
                + torch.sum(n_emb ** 2) / 2.0
            ) / u_emb.shape[0]
            l2reg = self.reg * l2norm
            bpr_loss = -log_prob + l2reg

        return bpr_loss


class NGCF(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        emb_dim,
        layers,
        reg,
        node_dropout,
        mess_dropout,
        adj_mtx,
        n_fold,
        dropout_mode="edge",
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.adj_mtx = adj_mtx
        self.reg = reg
        self.layers = layers
        self.n_layers = len(self.layers)
        self.n_fold = n_fold
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        self.dropout_mode = dropout_mode

        self.u_embeddings = nn.Parameter(torch.rand(n_users, emb_dim))
        self.i_embeddings = nn.Parameter(torch.rand(n_items, emb_dim))

        # Let's define them here so we can save/load them with the state_dict
        self.u_g_embeddings = nn.Parameter(
            torch.zeros(n_users, emb_dim + np.sum(self.layers))
        )
        self.i_g_embeddings = nn.Parameter(
            torch.zeros(n_items, emb_dim + np.sum(self.layers))
        )

        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()
        features = [emb_dim] + layers
        for i in range(1, len(features)):
            self.W1.append(nn.Linear(features[i - 1], features[i]))
            self.W2.append(nn.Linear(features[i - 1], features[i]))

        self._init_weights()
        self.A_fold_hat = self._split_A_hat(self.adj_mtx)

    def _init_weights(self):
        for n, p in self.named_parameters():
            if "bias" not in n:
                nn.init.xavier_uniform_(p)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape)
        return res.to(device)

    def _edge_dropout_sparse(self, X, keep_prob):
        """
        Drop individual locations (edges) in X
        """
        random_tensor = keep_prob
        random_tensor += torch.FloatTensor(X._nnz()).uniform_()
        dropout_mask = random_tensor.floor().to(device)
        idx = X.coalesce().indices().to(device)
        dropout_tensor = torch.sparse.FloatTensor(idx, dropout_mask, X.size())
        X_w_dropout = X.mul(dropout_tensor)

        return X_w_dropout.mul(1.0 / keep_prob)

    def _node_dropout_sparse(self, X, keep_prob):
        """
        Drop entire rows (nodes) in X
        """
        random_array = keep_prob
        random_array += np.random.rand(X.size()[0])
        dropout_mask = np.floor(random_array)
        dropout_mask = np.tile(dropout_mask.reshape(-1, 1), X.size()[1])
        dropout_tensor = self._convert_sp_mat_to_sp_tensor(sp.csr_matrix(dropout_mask))
        X_w_dropout = X.mul(dropout_tensor)

        return X_w_dropout.mul(1.0 / keep_prob)

    def _split_A_hat(self, X):
        """
        Split the Adjacency matrix into self.n_folds
        """
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        """
        This is really time consuming and has to run in every forward pass
        so be really careful
        """
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            if self.dropout_mode == "edge":
                A_fold_hat.append(
                    self._edge_dropout_sparse(temp, 1 - self.node_dropout)
                )
            elif self.dropout_mode == "node":
                A_fold_hat.append(
                    self._node_dropout_sparse(temp, 1 - self.node_dropout)
                )

        return A_fold_hat

    def forward(self, u, i, j):
        """
        Implementation of Figure 2 in Wang Xiang et al. Neural Graph
        Collaborative Filtering. Returns the BPR loss directly.
        """
        if self.node_dropout > 0.0:
            self.A_fold_hat = self._split_A_hat_node_dropout(self.adj_mtx)

        ego_embeddings = torch.cat([self.u_embeddings, self.i_embeddings], 0)
        pred_embeddings = [ego_embeddings]

        for k in range(self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(self.A_fold_hat[f], ego_embeddings))

            weighted_sum_emb = torch.cat(temp_embed, 0)
            affinity_emb = ego_embeddings.mul(weighted_sum_emb)

            t1 = self.W1[k](weighted_sum_emb)
            t2 = self.W2[k](affinity_emb)

            ego_embeddings = nn.Dropout(self.mess_dropout[k])(F.leaky_relu(t1 + t2))
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            pred_embeddings.append(norm_embeddings)

        pred_embeddings = torch.cat(pred_embeddings, 1)
        u_g_embeddings, i_g_embeddings = pred_embeddings.split(
            [self.n_users, self.n_items], 0
        )

        self.u_g_embeddings = nn.Parameter(u_g_embeddings)
        self.i_g_embeddings = nn.Parameter(i_g_embeddings)

        u_emb = u_g_embeddings[u]
        p_emb = i_g_embeddings[i]
        n_emb = i_g_embeddings[j]

        y_ui = torch.mul(u_emb, p_emb).sum(dim=1)
        y_uj = torch.mul(u_emb, n_emb).sum(dim=1)
        log_prob = (torch.log(torch.sigmoid(y_ui - y_uj))).mean()

        bpr_loss = -log_prob
        if self.reg > 0.0:
            l2norm = (
                torch.sum(u_emb ** 2) / 2.0
                + torch.sum(p_emb ** 2) / 2.0
                + torch.sum(n_emb ** 2) / 2.0
            ) / u_emb.shape[0]
            l2reg = self.reg * l2norm
            bpr_loss = -log_prob + l2reg

        return bpr_loss
