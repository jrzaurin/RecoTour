"""
@author: Javier Rodriguez (jrzaurin@gmail.com)
"""

import numpy as np
import pandas as pd
import os
import torch
import argparse
import heapq
import pdb

from tqdm import tqdm,trange
from time import time
from scipy.sparse import load_npz
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import get_train_instances, get_hitratio, get_ndcg


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, default="../datasets/Amazon",
        help="data directory.")
    parser.add_argument("--modeldir", type=str, default="../datasets/Amazon/models",
        help="models directory")
    parser.add_argument("--dataname", type=str, default="neuralcf_split.npz",
        help="chose a dataset.")
    parser.add_argument("--train_matrix", type=str, default="neuralcf_train_sparse.npz",
        help="chose a dataset.")
    parser.add_argument("--epochs", type=int, default=20,
        help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="batch size.")
    parser.add_argument("--n_emb", type=int, default=8,
        help="embedding size.")
    parser.add_argument("--lr", type=float, default=0.01,
        help="learning rate.")
    parser.add_argument("--learner", type=str, default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--validate_every", type=int, default=1,
        help="validate every n epochs")
    parser.add_argument("--save_model", action="store_false")
    parser.add_argument("--n_neg", type=int, default=4,
        help="number of negative instances to consider per positive instance.")
    parser.add_argument("--topK", type=int, default=10,
        help="number of items to retrieve for recommendation.")

    return parser.parse_args()


class GMF(nn.Module):
    def __init__(self, n_user, n_item, n_emb=8):
        super(GMF, self).__init__()

        self.n_emb = n_emb
        self.n_user = n_user
        self.n_item = n_item

        self.embeddings_user = nn.Embedding(n_user, n_emb)
        self.embeddings_item = nn.Embedding(n_item, n_emb)
        self.out = nn.Linear(in_features=n_emb, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight)

    def forward(self, users, items):

        user_emb = self.embeddings_user(users)
        item_emb = self.embeddings_item(items)
        prod = user_emb*item_emb
        preds = torch.sigmoid(self.out(prod))

        return preds


def train(model, criterion, optimizer, epoch, batch_size, use_cuda,
    train_ratings, negatives, n_items, n_neg):
    model.train()
    train_dataset = get_train_instances(train_ratings,
        negatives,
        n_items,
        n_neg)
    train_loader = DataLoader(dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    train_steps = (len(train_loader.dataset) // train_loader.batch_size) + 1
    running_loss=0
    for data in train_loader:
        users = data[:,0]
        items = data[:,1]
        labels = data[:,2].float()
        if use_cuda:
            users, items, labels = users.cuda(), items.cuda(), labels.cuda()
        optimizer.zero_grad()
        preds =  model(users, items)
        loss = criterion(preds.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss/train_steps


def evaluate(model, test_loader, use_cuda, topK):
    model.eval()
    hits, ndcgs = [],[]
    test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1
    with torch.no_grad():
        for data in test_loader:
            users = data[:,0]
            items = data[:,1]
            labels = data[:,2].float()
            if use_cuda:
                users, items, labels = users.cuda(), items.cuda(), labels.cuda()
            preds = model(users, items)
            ---->

            gtItem = items[0].item()
            map_item_score = dict( zip(items.cpu().numpy(), preds.squeeze(1).detach().cpu().numpy()) )
            ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
            hr = get_hitratio(ranklist, gtItem)
            ndcg = get_ndcg(ranklist, gtItem)
            hits.append(hr)
            ndcgs.append(ndcg)

    return (np.array(hits).mean(),np.array(ndcgs).mean())


def checkpoint(model, modelpath):
    torch.save(model.state_dict(), modelpath)


if __name__ == '__main__':
    args = parse_args()

    datadir = args.datadir
    dataname = args.dataname
    train_matrix = args.train_matrix
    modeldir = args.modeldir
    n_emb = args.n_emb
    batch_size = args.batch_size
    epochs = args.epochs
    learner = args.learner
    lr = args.lr
    validate_every = args.validate_every
    save_model = args.save_model
    topK = args.topK
    n_neg = args.n_neg

    modelfname = "pytorch_GMF" + \
        "_".join(["_bs", str(batch_size)]) + \
        "_".join(["_lr", str(lr).replace(".", "")]) + \
        "_".join(["_n_emb", str(n_emb)]) + ".pt"
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_df.p')

    dataset = np.load(os.path.join(datadir, dataname))
    train_ratings = load_npz(os.path.join(datadir, train_matrix)).todok()
    test_ratings, negatives = dataset['test_negative'], dataset['negatives']
    n_users, n_items = dataset['n_users'].item(), dataset['n_items'].item()

    test_loader = DataLoader(dataset=test_ratings,
        batch_size=100,
        shuffle=False
        )

    model = GMF(n_users, n_items, n_emb=n_emb)
    if learner.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif learner.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif learner.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    best_hr, best_ndcgm, best_iter=0,0,0
    for epoch in range(1,epochs+1):
        t1 = time()
        train(model, criterion, optimizer, epoch, batch_size, use_cuda,
            train_ratings, negatives, n_items, n_neg)
        t2 = time()
        if epoch % validate_every == 0:
            (hr, ndcg) = evaluate(model, test_loader, use_cuda, topK)
            print("Epoch: {} {:.2f}s, HR = {:.4f}, NDCG = {:.4f}, validated in {:.2f}s".
                format(epoch, t2-t1, hr, ndcg, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter, train_time = hr, ndcg, epoch, t2-t1
                if save_model:
                    checkpoint(model, modelpath)

    print("End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}. ".format(best_iter, best_hr, best_ndcg))
    if save_model:
        print("The best GMF model is saved to {}".format(modelpath))

    if not os.path.isfile(resultsdfpath):
        results_df = pd.DataFrame(columns = ["modelname", "best_hr", "best_ndcg", "best_iter",
            "train_time"])
        experiment_df = pd.DataFrame([[modelfname, best_hr, best_ndcg, best_iter, train_time]],
            columns = ["modelname", "best_hr", "best_ndcg", "best_iter","train_time"])
        results_df = results_df.append(experiment_df, ignore_index=True)
        results_df.to_pickle(resultsdfpath)
    else:
        results_df = pd.read_pickle(resultsdfpath)
        experiment_df = pd.DataFrame([[modelfname, best_hr, best_ndcg, best_iter, train_time]],
            columns = ["modelname", "best_hr", "best_ndcg", "best_iter","train_time"])
        results_df = results_df.append(experiment_df, ignore_index=True)
        results_df.to_pickle(resultsdfpath)
