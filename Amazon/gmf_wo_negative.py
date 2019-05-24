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
from sklearn.model_selection import train_test_split
from utils import get_train_instances, get_scores


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, default="../datasets/Amazon",
        help="data directory.")
    parser.add_argument("--modeldir", type=str, default="../datasets/Amazon/models",
        help="models directory")
    parser.add_argument("--dataname", type=str, default="neuralcf_split.npz",
        help="npz file with dataset")
    parser.add_argument("--train_matrix", type=str, default="neuralcf_train_sparse.npz",
        help="train matrix for faster iteration")
    parser.add_argument("--epochs", type=int, default=5,
        help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="batch size.")
    parser.add_argument("--n_emb", type=int, default=8,
        help="embedding size.")
    parser.add_argument("--lr", type=float, default=0.01,
        help="if lr_scheduler this will be max_lr")
    parser.add_argument("--learner", type=str, default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--validate_every", type=int, default=1,
        help="validate every n epochs")
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument("--n_neg", type=int, default=4,
        help="number of negative instances to consider per positive instance")
    parser.add_argument("--topk", type=int, default=10,
        help="number of items to retrieve for recommendation")

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


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    train_steps = (len(train_loader.dataset) // train_loader.batch_size) + 1
    running_loss=0
    with trange(train_steps) as t:
        for i, data in zip(t, train_loader):
            t.set_description('epoch %i' % epoch)
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
            avg_loss = running_loss/(i+1)
            t.set_postfix(loss=np.sqrt(avg_loss))
    return running_loss/train_steps


def valid(model, valid_loader, criterion):
    model.eval()
    valid_steps = (len(valid_loader.dataset) // valid_loader.batch_size) + 1
    running_loss=0
    with torch.no_grad():
        with trange(valid_steps) as t:
            for i, data in zip(t, valid_loader):
                t.set_description('valid')
                users = data[:,0]
                items = data[:,1]
                labels = data[:,2].float()
                if use_cuda:
                    users, items, labels = users.cuda(), items.cuda(), labels.cuda()
                preds =  model(users, items)
                loss = criterion(preds.squeeze(1), labels)
                running_loss += loss.item()
                avg_loss = running_loss/(i+1)
                t.set_postfix(loss=np.sqrt(avg_loss))
    return running_loss/valid_steps


def test(model, test_loader, topk):
    model.eval()
    test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1
    scores=[]
    with torch.no_grad():
        with trange(test_steps) as t:
            for i, data in zip(t, test_loader):
                t.set_description('test')
                users = data[:,0]
                items = data[:,1]
                labels = data[:,2].float()
                if use_cuda:
                    users, items, labels = users.cuda(), items.cuda(), labels.cuda()
                preds = model(users, items)
                items_cpu = items.cpu().numpy()
                preds_cpu = preds.squeeze(1).detach().cpu().numpy()
                litems=np.split(items_cpu, test_loader.batch_size//100)
                lpreds=np.split(preds_cpu, test_loader.batch_size//100)
                scores += [get_scores(it,pr,topk) for it,pr in zip(litems,lpreds)]
    hits = [s[0] for s in scores]
    ndcgs = [s[1] for s in scores]
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
    topk = args.topk
    n_neg = args.n_neg

    modelfname = "GMF_wo_neg" + \
        "_".join(["_bs", str(batch_size)]) + \
        "_".join(["_lr", str(lr).replace(".", "")]) + \
        "_".join(["_n_emb", str(n_emb)]) + \
        "_".join(["_lrnr", learner]) + \
        ".pt"
    if not os.path.exists(modeldir): os.makedirs(modeldir)
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_wo_negatove_df.p')

    dataset = np.load(os.path.join(datadir, dataname))
    df_train = pd.DataFrame(dataset['train'], columns=['user', 'item', 'rating'])
    df_train['rating'] = df_train.rating.apply(lambda x: 3 if x==5 else 2 if (x==4 or x==3) else 1)
    train_dataset, valid_dataset = train_test_split(df_train.values, test_size=0.2, stratify=df_train['rating'])
    test_ratings, negatives = dataset['test_negative'], dataset['negatives']
    n_users, n_items = dataset['n_users'].item(), dataset['n_items'].item()

    train_loader = DataLoader(dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    test_loader = DataLoader(dataset=test_ratings,
        batch_size=1000,
        shuffle=False
        )

    model = GMF(n_users, n_items, n_emb=n_emb)

    if learner.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif learner.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    elif learner.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    criterion = nn.MSELoss()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    best_hr, best_ndcgm, best_iter=0,0,0
    for epoch in range(1,epochs+1):
        tr_loss  = train(model, train_loader, criterion, optimizer, epoch)
        val_loss = valid(model, valid_loader, criterion)
        if epoch % validate_every == 0:
            (hr, ndcg) = test(model, test_loader, topk)
            print("epoch: {} HR = {:.4f}, NDCG = {:.4f}".format(epoch, hr, ndcg))
            if hr > best_hr:
                iter_tr_loss, iter_val_loss, best_hr, best_ndcg, best_iter = \
                    tr_loss, val_loss, hr, ndcg, epoch
                if save_model:
                    checkpoint(model, modelpath)

    print("End. Best Iteration {}: HR = {:.4f}, NDCG = {:.4f}. ".format(best_iter, best_hr, best_ndcg))
    if save_model:
        print("The best GMF model wihtout negative feedback is saved to {}".format(modelpath))

    if save_model:
        cols = ["modelname", "iter_tr_loss", "iter_val_loss", "best_hr",
            "best_ndcg", "best_iter"]
        vals = [modelfname, iter_tr_loss, iter_val_loss, best_hr, best_ndcg,
            best_iter]
        if not os.path.isfile(resultsdfpath):
            results_df = pd.DataFrame(columns=cols)
            experiment_df = pd.DataFrame(data=[vals], columns=cols)
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(resultsdfpath)
        else:
            results_df = pd.read_pickle(resultsdfpath)
            experiment_df = pd.DataFrame(data=[vals], columns=cols)
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(resultsdfpath)
