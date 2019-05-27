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
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset
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
    parser.add_argument("--epochs", type=int, default=20,
        help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="batch size.")
    parser.add_argument("--n_emb", type=int, default=8,
        help="embedding size.")
    parser.add_argument("--lr", type=float, default=0.01,
        help="if lr_scheduler this will be max_lr")
    parser.add_argument("--learner", type=str, default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--lr_scheduler", action="store_true",
        help="boolean to set the use of CyclicLR during training")
    parser.add_argument("--loss_criterion", type=str, default="mse",
        help="Specify the criterion: mse or bce")
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


def train(model, criterion, optimizer, scheduler, epoch, batch_size,
    use_cuda, train_ratings, negatives, n_items, n_neg):
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
        if scheduler:
            scheduler.step()
        loss = criterion(preds.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss/train_steps


def evaluate(model, test_loader, use_cuda, topk):
    model.eval()
    scores=[]
    with torch.no_grad():
        for data in test_loader:
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
    lr_scheduler = args.lr_scheduler
    lrs = "wlrs" if lr_scheduler else "wolrs"
    loss_criterion = args.loss_criterion
    validate_every = args.validate_every
    save_model = args.save_model
    topk = args.topk
    n_neg = args.n_neg

    modelfname = "GMF" + \
        "_".join(["_bs", str(batch_size)]) + \
        "_".join(["_lr", str(lr).replace(".", "")]) + \
        "_".join(["_n_emb", str(n_emb)]) + \
        "_".join(["_lrnr", learner]) + \
        "_".join(["_lrs", lrs]) + \
        ".pt"
    if not os.path.exists(modeldir): os.makedirs(modeldir)
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_df.p')

    dataset = np.load(os.path.join(datadir, dataname))
    train_ratings = load_npz(os.path.join(datadir, train_matrix)).todok()
    test_ratings, negatives = dataset['test_negative'], dataset['negatives']
    n_users, n_items = dataset['n_users'].item(), dataset['n_items'].item()

    test_loader = DataLoader(dataset=test_ratings,
        batch_size=10000,
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

    if loss_criterion.lower() == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    training_steps = ((len(train_ratings)+len(train_ratings)*n_neg)//batch_size)+1
    step_size = training_steps*3 # one cycle every 6 epochs
    cycle_momentum=True
    if learner.lower() == "adagrad" or learner.lower()=="adam":
        cycle_momentum=False
    if lr_scheduler:
        scheduler = CyclicLR(optimizer, step_size_up=step_size, base_lr=lr/10., max_lr=lr,
            cycle_momentum=cycle_momentum)
    else:
        scheduler = None

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    best_hr, best_ndcgm, best_iter=0,0,0
    for epoch in range(1,epochs+1):
        t1 = time()
        loss = train(model, criterion, optimizer, scheduler, epoch, batch_size,
            use_cuda, train_ratings, negatives, n_items, n_neg)
        t2 = time()
        if epoch % validate_every == 0:
            (hr, ndcg) = evaluate(model, test_loader, use_cuda, topk)
            print("Epoch: {} {:.2f}s, LOSS = {:.4f}, HR = {:.4f}, NDCG = {:.4f}, validated in {:.2f}s".
                format(epoch, t2-t1, loss, hr, ndcg, time()-t2))
            if hr > best_hr:
                iter_loss, best_hr, best_ndcg, best_iter, train_time = \
                    loss, hr, ndcg, epoch, t2-t1
                if save_model:
                    checkpoint(model, modelpath)

    print("End. Best Iteration {}: HR = {:.4f}, NDCG = {:.4f}. ".format(best_iter, best_hr, best_ndcg))
    if save_model:
        print("The best GMF model is saved to {}".format(modelpath))

    if save_model:
        cols = ["modelname", "iter_loss","best_hr", "best_ndcg", "best_iter","train_time"]
        vals = [modelfname, iter_loss, best_hr, best_ndcg, best_iter, train_time]
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
