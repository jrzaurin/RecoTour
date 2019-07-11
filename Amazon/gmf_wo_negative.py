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
    parser.add_argument("--dataname", type=str, default="standard_split.npz",
        help="npz file with dataset")
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
        preds = self.out(prod)

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


def valid(model, data_loader, criterion, mode='valid'):
    model.eval()
    steps = (len(data_loader.dataset) // data_loader.batch_size) + 1
    running_loss=0
    with torch.no_grad():
        with trange(steps) as t:
            for i, data in zip(t, data_loader):
                t.set_description(mode)
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
    return running_loss/steps


def checkpoint(model, modelpath):
    torch.save(model.state_dict(), modelpath)


if __name__ == '__main__':

    args = parse_args()

    datadir = args.datadir
    dataname = args.dataname
    modeldir = args.modeldir
    n_emb = args.n_emb
    batch_size = args.batch_size
    epochs = args.epochs
    learner = args.learner
    lr = args.lr

    # I am going to perform a simple train/valid/test exercise, predicting
    # directly the ratings. I will leave it to you to adapt the datasets so
    # that we could get some ranking metrics
    dataset = np.load(os.path.join(datadir, dataname))
    train_dataset, test_dataset = dataset['train'][:, [0,1,3]], dataset['test'][:, [0,1,3]]
    train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.2, stratify=train_dataset[:,2])
    n_users, n_items = dataset['n_users'], dataset['n_items']

    train_loader = DataLoader(dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
        batch_size=batch_size,
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

    # There are better ways of structuring the code, but since I already had
    # it from the other experiments I will run it like this:
    for epoch in range(1,epochs+1):
        tr_loss  = train(model, train_loader, criterion, optimizer, epoch)
        val_loss = valid(model, valid_loader, criterion, mode='valid')
    test_loss = valid(model, test_loader, criterion, mode='test')
    print("test loss: {}".format(np.sqrt(test_loss)))
