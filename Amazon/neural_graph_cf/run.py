import numpy as np
import pandas as pd
import torch
import os
import re
import scipy.sparse as sp
import multiprocessing

from time import time
from collections import defaultdict
from multiprocessing import Pool
from utils.load_data import Data
from utils.metrics import ranklist_by_heapq, get_performance
from utils.parser import parse_args
from utils.radam import RAdam, AdamW
from ngcf import NGCF, BPR

import pdb

cores = multiprocessing.cpu_count()
use_cuda = torch.cuda.is_available()

def early_stopping(log_value, best_value, stopping_step, patience, expected_order='asc'):
    """
    if log_value >= best_value (asc) or log_value <= best_value (des) more
    than our patience, stop
    """
    assert expected_order in ['asc', 'des']
    if (expected_order == 'asc' and log_value >= best_value) or (expected_order == 'des' and log_value <= best_value):
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
        optimizer.zero_grad()
        loss = model(u,i,j)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def test_one_user(x):
    rating = x[0]
    u = x[1]

    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []

    user_pos_test = data_generator.test_set[u]
    all_items = set(range(data_generator.n_items))
    test_items = list(all_items - set(training_items))
    r = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, Ks)


def test_CPU(model, users_to_test):
    model.eval()
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = batch_size * 2
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):

        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        item_batch = range(data_generator.n_items)

        user_emb = model.u_g_embeddings[user_batch].detach()

        rate_batch  = torch.mm(user_emb, model.i_g_embeddings.t().detach()).cpu().numpy()

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users

    assert count == n_test_users
    pool.close()
    return result


def split_mtx(X, n_folds=10):
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


def ndcg_at_k_gpu(test_items, scores, test_indices, k):
    preds = torch.zeros_like(scores).float()
    preds.scatter_(dim=1,index=test_indices,src=torch.tensor(1.0).cuda())
    r = (test_items * preds).gather(1, test_indices)
    f = torch.from_numpy(np.log2(np.arange(2, k+2))).float().cuda()
    dcg = (r[:, :k]/f).sum(1)
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k]/f).sum(1)
    ndcg = dcg/dcg_max
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg


def test_GPU(u_emb, i_emb, Rtr, Rte, Ks):

    ue_folds = split_mtx(u_emb)
    tr_folds = split_mtx(Rtr)
    te_folds = split_mtx(Rte)

    fold_prec, fold_rec, fold_ndcg, fold_hr = \
        defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for ue_f, tr_f, te_f in zip(ue_folds, tr_folds, te_folds):

        scores = torch.mm(ue_f, i_emb.t())
        test_items = torch.from_numpy(te_f.todense()).float().cuda()
        non_train_items = torch.from_numpy(1-(tr_f.todense())).float().cuda()
        scores = scores * non_train_items
        _, test_indices = torch.topk(scores, dim=1, k=max(Ks))

        for k in Ks:
            topk_preds = torch.zeros_like(scores).float()
            topk_preds.scatter_(dim=1,index=test_indices[:, :k],src=torch.tensor(1.0))

            TP = (test_items * topk_preds).sum(1)
            prec = TP/k
            rec = TP/test_items.sum(1)
            hit_r = (TP > 0).float()
            ndcg = ndcg_at_k_gpu(test_items, scores, test_indices, k)

            fold_prec[k].append(prec)
            fold_rec[k].append(rec)
            fold_ndcg[k].append(ndcg)
            fold_hr[k].append(hit_r)

    result = {'precision': [], 'recall': [], 'ndcg': [], 'hit_ratio': []}
    for k in Ks:
        result['precision'].append(torch.cat(fold_prec[k]).mean())
        result['recall'].append(torch.cat(fold_rec[k]).mean())
        result['ndcg'].append(torch.cat(fold_ndcg[k]).mean())
        result['hit_ratio'].append(torch.cat(fold_hr[k]).mean())
    return result


if __name__ == '__main__':

    args = parse_args()
    data_dir = args.data_dir
    dataset = args.dataset
    batch_size = args.batch_size

    layers = eval(args.layers)
    emb_dim = args.emb_dim
    reg = args.reg
    mess_dropout = [args.mess_dropout]*len(layers)
    node_dropout = args.node_dropout
    n_fold = args.n_fold
    adj_type = args.adj_type

    Ks = eval(args.Ks)

    data_generator = Data(path=data_dir + dataset, batch_size=batch_size)
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    if adj_type == 'plain':
        adj_mtx = plain_adj
    elif adj_type == 'norm':
        adj_mtx = norm_adj
    elif adj_type == 'gcmc':
        adj_mtx = mean_adj
    else:
        adj_mtx = mean_adj + sp.eye(mean_adj.shape[0])

    modelfname =  "NeuGCF" + \
        "_nemb_" + str(emb_dim) + \
        "_layers_" + re.sub(" ", "", str(layers)) + \
        "_nodedr_" + str(node_dropout) + \
        "_messdr_" + re.sub(" ", "", str(mess_dropout)) + \
        "_reg_" + str(reg) + \
        ".pt"
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    model_weights = os.path.join(args.results_dir, modelfname)
    results_tab = os.path.join(args.results_dir, 'results_df.csv')

    if args.model == 'ngcf':
        model = NGCF(data_generator.n_users, data_generator.n_items, emb_dim, layers, reg,
            node_dropout, mess_dropout, adj_mtx, n_fold)
    elif args.model == 'bpr':
        model = BPR(data_generator.n_users, data_generator.n_items, emb_dim, reg)

    if use_cuda:
        model = model.cuda()

    if args.pretrain == 1:
        assert os.path.isfile(model_weights)
        model.load_state_dict(torch.load(model_weights))
        print("Computing metrics with pretrained weights")
        if args.test_with == 'cpu':
            users_to_test = list(data_generator.test_set.keys())
            res = test_CPU(model, users_to_test)
        elif args.test_with == 'gpu':
            res = test_GPU(
                model.u_g_embeddings.detach(),
                model.i_g_embeddings.detach(),
                data_generator.Rtr,
                data_generator.Rte,
                Ks)
        print(
            "Pretrained model", "\n",
            "Recall@{}: {:.4f}, Recall@{}: {:.4f}".format(Ks[0], res['recall'][0],  Ks[-1], res['recall'][-1]), "\n",
            "Precision@{}: {:.4f}, Precision@{}: {:.4f}".format(Ks[0], res['precision'][0],  Ks[-1], res['precision'][-1]), "\n",
            "Hit_ratio@{}: {:.4f}, Hit_ratio@{}: {:.4f}".format(Ks[0], res['hit_ratio'][0],  Ks[-1], res['hit_ratio'][-1]), "\n",
            "NDCG@{}: {:.4f}, NDCG@{}: {:.4f}".format(Ks[0], res['ndcg'][0],  Ks[-1], res['ndcg'][-1])
            )
        # cur_best_metric = res['recall'][0]
    elif args.pretrain == -1:
        assert os.path.isfile(model_weights)
        state_dict = torch.load(model_weights)
        model.u_embeddings = torch.nn.Parameter(state_dict['u_embeddings'])
        model.i_embeddings = torch.nn.Parameter(state_dict['i_embeddings'])
    #     cur_best_metric = 0.
    # else:
    #     cur_best_metric = 0.

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.lr)

    cur_best_loss, stopping_step, should_stop = 1e3, 0, False
    for epoch in range(args.n_epochs):

        t1 = time()
        loss = train(model, data_generator, optimizer)

        if epoch % args.print_every  == (args.print_every - 1):
            print("Epoch:{} {:.2f}s, Loss = {:.4f}".
                format(epoch, time()-t1, loss))

        if epoch % args.eval_every  == (args.eval_every - 1):
            with torch.no_grad():
                if args.test_with == 'cpu':
                    t2 = time()
                    users_to_test = list(data_generator.test_set.keys())
                    res = test_CPU(model, users_to_test)
                if args.test_with == 'gpu':
                    t2 = time()
                    res = test_GPU(
                        model.u_g_embeddings.detach(),
                        model.i_g_embeddings.detach(),
                        data_generator.Rtr,
                        data_generator.Rte,
                        Ks)

            print(
                "VALIDATION:","\n",
                "Epoch: {}, {:.2f}s".format(epoch, time()-t2),"\n",
                "Recall@{}: {:.4f}, Recall@{}: {:.4f}".format(Ks[0], res['recall'][0],  Ks[-1], res['recall'][-1]), "\n",
                "Precision@{}: {:.4f}, Precision@{}: {:.4f}".format(Ks[0], res['precision'][0],  Ks[-1], res['precision'][-1]), "\n",
                "Hit_ratio@{}: {:.4f}, Hit_ratio@{}: {:.4f}".format(Ks[0], res['hit_ratio'][0],  Ks[-1], res['hit_ratio'][-1]), "\n",
                "NDCG@{}: {:.4f}, NDCG@{}: {:.4f}".format(Ks[0], res['ndcg'][0],  Ks[-1], res['ndcg'][-1])
                )

            # log_value = res['recall'][0]
            # cur_best_metric, stopping_step, should_stop = \
            # early_stopping(log_value, cur_best_metric, stopping_step, args.patience)

        log_value = loss
        cur_best_loss, stopping_step, should_stop = \
        early_stopping(log_value, cur_best_loss, stopping_step, args.patience, expected_order='des')

        if should_stop == True: break

        if (stopping_step == 0) & (args.save_results):
            torch.save(model.state_dict(), model_weights)
            try:
                final_loss, final_res = loss, res
            except NameError:
                pass

    if args.save_results:
        cols, vals = [], []
        for m in final_res.keys():
            for i,k in enumerate(Ks):
                cols.append(m+'@'+str(k))
                vals.append(final_res[m][i].cpu().numpy())
        cols = ['modelname', 'loss'] + cols
        vals = [modelfname, final_loss] + vals
        if not os.path.isfile(results_tab):
            results_df = pd.DataFrame(columns=cols)
            experiment_df = pd.DataFrame(data=[vals], columns=cols)
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_csv(results_tab, index=False)
        else:
            results_df = pd.read_csv(results_tab)
            experiment_df = pd.DataFrame(data=[vals], columns=cols)
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_csv(results_tab, index=False)
