import numpy as np
import torch
import os
import re
import scipy.sparse as sp
import multiprocessing
import torch.nn.functional as F

from time import time
from functools import partial
from utils.dataset import Data
from utils.metrics import ranklist_by_heapq, get_performance
from utils.parser import parse_args
from ngcf import NGCF_BPR
from multiprocessing import Pool

cores = multiprocessing.cpu_count()
use_cuda = torch.cuda.is_available()


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', patience=10):

    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= patience:
        print("Early stopping is trigger at step: {} log:{}".format(patience, log_value))
        should_stop = True
    else:
        should_stop = False

    return best_value, stopping_step, should_stop


def train(model, data_generator, optimizer):
    model.train()
    n_batch = data_generator.n_train // data_generator.batch_size + 1
    running_loss=0
    for _ in range(n_batch):
        u, i, j = data_generator.sample()
        u, i, j = torch.LongTensor(u), torch.LongTensor(i), torch.LongTensor(j)
        if use_cuda:
            u, i, j = u.cuda(), i.cuda(), j.cuda()
        optimizer.zero_grad()
        u_emb, p_emb, n_emb =  model(u, i, j)
        loss = model.bpr_loss(u_emb,p_emb,n_emb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def test_one_user(x):
    u = x[0]
    rating = x[1]

    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []

    user_pos_test = data_generator.test_set[u]
    all_items = set(range(data_generator.n_items))
    test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test_cpu(model, data_generator):
    result = {
        'precision': np.zeros(len(Ks)),
        'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks)),
        'hit_ratio': np.zeros(len(Ks)),
        'auc': 0.
        }

    u_batch_size = data_generator.batch_size * 2

    test_users = list(data_generator.test_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    n_test_items = data_generator.n_items

    count = 0
    p = Pool(cores)
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        item_batch = np.arange(n_test_items)

        rate_batch  = torch.mm(model.g_embeddings_user.weight, model.g_embeddings_item.weight.t())
        rate_batch_np = rate_batch.detach().cpu().numpy()
        batch_result = p.map(test_one_user, zip(user_batch,rate_batch_np))

        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
    assert count == n_test_users
    p.close()
    return result


def split_mtx(X, n_folds=10):
    """
    Split a matrix/tensor in n_folds folds

    There is some redundancy with the split methods within the
    NGCF_BPR class...I am ok with that, or almost.
    """
    X_folds = []
    fold_len = X.shape[0]//n_folds
    for i in range(n_folds):
        start = i * fold_len
        if i == n_folds -1:
            end = X.shape[0]
        else:
            end = (i + 1) * fold_len
        X_folds.append(X[start:end])
    return X_folds


def test_gpu(user_emb, item_emb, R_tr, R_te, Ks):

    tr_folds = split_mtx(R_tr)
    te_folds = split_mtx(R_te)
    ue_folds = split_mtx(user_emb.weight)

    fold_prec, fold_rec = {}, {}
    for ue_fold, tr_fold, te_fold in zip(ue_folds, tr_folds, te_folds):

        result = torch.sigmoid(torch.mm(ue_fold, item_emb.weight.t()))
        test_pred_mask = torch.from_numpy(1 - tr_fold.todense())
        test_true_mask = torch.from_numpy(te_fold.todense())
        if use_cuda:
            test_pred_mask, test_true_mask = test_pred_mask.cuda(), test_true_mask.cuda()
        test_pred = test_pred_mask * result
        test_true = test_true_mask * result

        _, test_indices = torch.topk(test_pred, dim=1, k=max(Ks))
        for k in Ks:
            topk_mask = torch.zeros_like(test_pred)
            topk_mask.scatter_(dim=1, index=test_indices[:, :k], src=torch.tensor(1.0).cuda())
            test_pred_topk = topk_mask * test_pred
            acc_result = (test_pred_topk != 0) & (test_pred_topk == test_true)
            pr_k = acc_result.sum().float() / (user_emb.weight.shape[0] * k)
            rec_k = (acc_result.float().sum(dim=1) / test_true_mask.float().sum(dim=1))
            try:
                fold_prec[k].append(pr_k)
                fold_rec[k].append(rec_k)
            except KeyError:
                fold_prec[k] = [pr_k]
                fold_rec[k] = [rec_k]

    precision, recall = {}, {}
    for k in Ks:
        precision[k] = np.sum(fold_prec[k])
        recall[k] = torch.cat(fold_rec[k]).mean()
    return precision, recall


if __name__ == '__main__':

    args = parse_args()
    Ks = eval(args.Ks)

    # data_path = args.data_path + args.dataset
    # batch_size = args.batch_size
    data_path = "Data/toy_data/"
    batch_size = 1024
    data_generator = Data(data_path, batch_size, val=False)
    _, _, mean_adj = data_generator.get_adj_mat()
    adjacency_matrix = mean_adj + sp.eye(mean_adj.shape[0])
    n_users = data_generator.n_users
    n_items = data_generator.n_items

    # emb_size = args.emb_size
    emb_size = 12
    # layers = eval(args.layers)
    layers = [12, 6]
    # node_dropout = args.node_dropout
    node_dropout = 0.1
    # mess_dropout = [args.mess_dropout]*len(layers)
    mess_dropout = [0.1]*len(layers)
    # regularization = args.reg
    regularization = 1e-5
    lr = args.lr
    # n_fold = 100
    n_fold = 10

    modelfname =  "NeuGCF" + \
        "_nemb_" + str(emb_size) + \
        "_layers_" + re.sub(" ", "", str(layers)) + \
        "_nodedr_" + str(node_dropout) + \
        "_messdr_" + re.sub(" ", "", str(mess_dropout)) + \
        "_reg_" + str(regularization) + \
        ".pt"
    modelpath = os.path.join(args.model_path, modelfname)
    res_path = os.path.join(args.model_path, 'results_df.p')

    model = NGCF_BPR(n_users, n_items, emb_size, adjacency_matrix, layers,
        node_dropout, mess_dropout, regularization, n_fold, batch_size)
    if use_cuda:
        model = model.cuda()
    print(model)

    if args.pretrain == 1:
        assert os.path.isfile(modelpath)
        model.load_state_dict(torch.load(modelpath))
        res = test_cpu(model, data_generator)
        cur_best_pre = res['recall'][0]
        print("Pretrained model", "\n"
            "Recall@{}: {:.4f}, Recall@{}: {:.4f}".format(Ks[0], res['recall'][0],  Ks[-1], res['recall'][-1]), "\n"
            "Precision@{}: {:.4f}, Precision@{}: {:.4f}".format(Ks[0], res['precision'][0],  Ks[-1], res['precision'][-1]), "\n"
            "Hit_ratio@{}: {:.4f}, Hit_ratio@{}: {:.4f}".format(Ks[0], res['hit_ratio'][0],  Ks[-1], res['hit_ratio'][-1]), "\n"
            "NDCG@{}: {:.4f}, NDCG@{}: {:.4f}".format(Ks[0], res['ndcg'][0],  Ks[-1], res['ndcg'][-1])
            )
    elif args.pretrain == -1:
        assert os.path.isfile(modelpath)
        state_dict = torch.load(modelpath)
        model.embeddings_user.weight = torch.nn.Parameter(state_dict['embeddings_user.weight'])
        model.embeddings_item.weight = torch.nn.Parameter(state_dict['embeddings_item.weight'])
        cur_best_pre = 0
    else:
        cur_best_pre = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopping_step, should_stop = 0, False
    for epoch in range(args.epochs):
        t1 = time()
        loss = train(model, data_generator, optimizer)
        if epoch % args.print_every  == (args.print_every - 1):
            print("Epoch:{} {:.2f}s, Loss = {:.4f}".
                format(epoch, time()-t1, loss))
        if epoch % args.eval_every  == (args.eval_every - 1):
            t2 = time()
            res = test_cpu(model, data_generator)
            print("VALIDATION.","\n"
                "Epoch: {}, {:.2f}s".format(epoch, time()-t2), "\n",
                "Recall@{}: {:.4f}, Recall@{}: {:.4f}".format(Ks[0], res['recall'][0],  Ks[-1], res['recall'][-1]), "\n"
                "Precision@{}: {:.4f}, Precision@{}: {:.4f}".format(Ks[0], res['precision'][0],  Ks[-1], res['precision'][-1]), "\n"
                "Hit_ratio@{}: {:.4f}, Hit_ratio@{}: {:.4f}".format(Ks[0], res['hit_ratio'][0],  Ks[-1], res['hit_ratio'][-1]), "\n"
                "NDCG@{}: {:.4f}, NDCG@{}: {:.4f}".format(Ks[0], res['ndcg'][0],  Ks[-1], res['ndcg'][-1])
                )
            cur_best_pre, stopping_step, should_stop = \
            early_stopping(res['recall'][0], cur_best_pre, stopping_step)
        if epoch % args.save_every == (args.save_every - 1):
            torch.save(model.state_dict(), modelpath)
