import numpy as np
import pandas as pd
import os
import torch
import argparse
import heapq
import pdb

from time import time
from scipy.sparse import load_npz
from torch import nn
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset
from utils import get_train_instances, get_scores
from gmf import GMF, train, evaluate, checkpoint
from mlp import MLP


def parse_args():
    parser = argparse.ArgumentParser()

    # dirnames
    parser.add_argument("--datadir", type=str, default="../datasets/Amazon",
        help="data directory.")
    parser.add_argument("--modeldir", type=str, default="../datasets/Amazon/models",
        help="models directory")
    parser.add_argument("--dataname", type=str, default="neuralcf_split.npz",
        help="npz file with dataset")
    parser.add_argument("--train_matrix", type=str, default="neuralcf_train_sparse.npz",
        help="train matrix for faster iteration")

    # general parameter
    parser.add_argument("--epochs", type=int, default=20,
        help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="batch size.")
    parser.add_argument("--lr", type=float, default=0.001,
        help="learning rate.")
    parser.add_argument("--learner", type=str, default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--lr_scheduler", action="store_true",
        help="boolean to set the use of CyclicLR during training")

    # GMF set up
    parser.add_argument("--n_emb", type=int, default=8,
        help="embedding size for the GMF part.")

    # MLP set up
    parser.add_argument("--layers", type=str, default="[64,32,16,8]",
        help="layer architecture. The first elements is used for the embedding \
        layers for the MLP part and equals n_emb*2")
    parser.add_argument("--dropouts", type=str, default="[0.,0.,0.]",
        help="dropout per dense layer. len(dropouts) = len(layers)-1")

    # regularization
    parser.add_argument("--l2reg", type=float, default=0.,
        help="l2 regularization.")

    # Pretrained model names
    parser.add_argument("--freeze", type=int, default=0,
        help="freeze all but the last output layer where \
        weights are combined")
    parser.add_argument("--mf_pretrain", type=str, default="",
        help="Specify the pretrain model filename for GMF part. \
        If empty, no pretrain will be used")
    parser.add_argument("--mlp_pretrain", type=str, default="",
        help="Specify the pretrain model filename for MLP part. \
        If empty, no pretrain will be used")

    # Experiment set up
    parser.add_argument("--validate_every", type=int, default=1,
        help="validate every n epochs")
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument("--n_neg", type=int, default=4,
        help="number of negative instances to consider per positive instance.")
    parser.add_argument("--topk", type=int, default=10,
        help="number of items to retrieve for recommendation.")

    return parser.parse_args()


class NeuMF(nn.Module):
    def __init__(self, n_user, n_item, n_emb, layers, dropouts):
        super(NeuMF, self).__init__()

        self.layers = layers
        self.n_layers = len(layers)
        self.dropouts = dropouts
        self.n_user = n_user
        self.n_item = n_item

        self.mf_embeddings_user = nn.Embedding(n_user, n_emb)
        self.mf_embeddings_item = nn.Embedding(n_item, n_emb)

        self.mlp_embeddings_user = nn.Embedding(n_user, layers[0]//2)
        self.mlp_embeddings_item = nn.Embedding(n_item, layers[0]//2)
        self.mlp = nn.Sequential()
        for i in range(1,self.n_layers):
            self.mlp.add_module("linear%d" %i, nn.Linear(layers[i-1],layers[i]))
            self.mlp.add_module("relu%d" %i, torch.nn.ReLU())
            self.mlp.add_module("dropout%d" %i , torch.nn.Dropout(p=dropouts[i-1]))

        self.out = nn.Linear(in_features=n_emb+layers[-1], out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)

    def forward(self, users, items):

        mf_user_emb = self.mf_embeddings_user(users)
        mf_item_emb = self.mf_embeddings_item(items)

        mlp_user_emb = self.mlp_embeddings_user(users)
        mlp_item_emb = self.mlp_embeddings_item(items)

        mf_emb_vector = mf_user_emb*mf_item_emb
        mlp_emb_vector = torch.cat([mlp_user_emb,mlp_item_emb], dim=1)
        mlp_emb_vector = self.mlp(mlp_emb_vector)

        emb_vector = torch.cat([mf_emb_vector,mlp_emb_vector], dim=1)
        preds = torch.sigmoid(self.out(emb_vector))

        return preds


def load_pretrain_model(model, gmf_model, mlp_model):

    # MF embeddings
    model.mf_embeddings_item.weight = gmf_model.embeddings_item.weight
    model.mf_embeddings_user.weight = gmf_model.embeddings_user.weight

    # MLP embeddings
    model.mlp_embeddings_item.weight = mlp_model.embeddings_item.weight
    model.mlp_embeddings_user.weight = mlp_model.embeddings_user.weight

    # MLP layers
    model_dict = model.state_dict()
    mlp_layers_dict = mlp_model.state_dict()
    mlp_layers_dict = {k: v for k, v in mlp_layers_dict.items() if 'linear' in k}
    model_dict.update(mlp_layers_dict)
    model.load_state_dict(model_dict)

    # Prediction weights
    mf_prediction_weight, mf_prediction_bias = gmf_model.out.weight, gmf_model.out.bias
    mlp_prediction_weight, mlp_prediction_bias = mlp_model.out.weight, mlp_model.out.bias

    new_weight = torch.cat([mf_prediction_weight, mlp_prediction_weight], dim=1)
    new_bias = mf_prediction_bias + mlp_prediction_bias
    model.out.weight = torch.nn.Parameter(0.5*new_weight)
    model.out.bias = torch.nn.Parameter(0.5*new_bias)

    return model


if __name__ == '__main__':
    args = parse_args()

    datadir = args.datadir
    dataname = args.dataname
    train_matrix = args.train_matrix
    modeldir = args.modeldir

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    learner = args.learner
    lr_scheduler = args.lr_scheduler
    lrs = "wlrs" if lr_scheduler else "wolrs"

    n_emb = args.n_emb

    layers = eval(args.layers)
    dropouts = eval(args.dropouts)

    freeze = bool(args.freeze)
    mf_pretrain = os.path.join(modeldir, args.mf_pretrain)
    mlp_pretrain = os.path.join(modeldir, args.mlp_pretrain)
    with_pretrained = "wpret" if os.path.isfile(mf_pretrain) else "wopret"
    is_frozen = "frozen" if freeze else "trainable"

    l2reg = args.l2reg

    validate_every = args.validate_every
    save_model = bool(args.save_model)
    n_neg = args.n_neg
    topk = args.topk

    modelfname = "NeuMF" + \
        "_" + with_pretrained + \
        "_" + is_frozen + \
        "_" + learner + \
        "_".join(["_lrs", lrs]) + \
        ".pt"
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_df.p')

    dataset = np.load(os.path.join(datadir, dataname))
    train_ratings = load_npz(os.path.join(datadir, train_matrix)).todok()
    test_ratings, negatives = dataset['test_negative'], dataset['negatives']
    n_users, n_items = dataset['n_users'].item(), dataset['n_items'].item()

    test_loader = DataLoader(dataset=test_ratings,
        batch_size=1000,
        shuffle=False
        )

    model = NeuMF(n_users, n_items, n_emb, layers, dropouts)
    if os.path.isfile(mf_pretrain) and os.path.isfile(mlp_pretrain):
        gmf_model = GMF(n_users, n_items, n_emb)
        gmf_model.load_state_dict(torch.load(mf_pretrain))
        mlp_model = MLP(n_users, n_items, layers, dropouts)
        mlp_model.load_state_dict(torch.load(mlp_pretrain))
        model = load_pretrain_model(model, gmf_model, mlp_model)
        print("Load pretrained GMF {} and MLP {} models done. ".format(mf_pretrain, mlp_pretrain))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    if freeze:
        for name, layer in model.named_parameters():
            if not ("out" in name):
                layer.requires_grad = False

    # or this and pass train_parametes to the optimizer
    # train_parametes = model.out.parameters() if freeze else model.parameters()

    if learner.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2reg)
    elif learner.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2reg,
            momentum=0.9)
    elif learner.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2reg,
            momentum=0.9, nesterov=True)

    criterion = nn.BCELoss()

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    # print(trainable_params)

    training_steps = ((len(train_ratings)+len(train_ratings)*n_neg)//batch_size)+1
    step_size = training_steps*10
    cycle_momentum=True
    if learner.lower() == "adagrad" or learner.lower()=="adam":
        cycle_momentum=False
    if lr_scheduler:
        scheduler = CyclicLR(optimizer, step_size_up=step_size, base_lr=lr/10., max_lr=lr,
            cycle_momentum=cycle_momentum)
    else:
        scheduler = None

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
        print("The best NeuMF model is saved to {}".format(modelpath))

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
