# -*- coding: utf-8 -*-
import numpy as np
import pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm,trange

use_cuda = torch.cuda.is_available()

class WideDeepLoader(Dataset):
    """Helper to facilitate loading the data to the pytorch models.
    Parameters:
    --------
    data: namedtuple with 3 elements - (wide_input_data, deep_inp_data, target)
    """
    def __init__(self, data, mode='train'):

        self.X_wide = data['wide']
        self.X_deep = data['deep']
        self.mode = mode
        if self.mode is 'train':
            self.Y = data['target']
        elif self.mode is 'test':
            self.Y = None

    def __getitem__(self, idx):

        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        if self.mode is 'train':
            y  = self.Y[idx]
            return xw, xd, y
        elif self.mode is 'test':
            return xw, xd

    def __len__(self):
        return len(self.X_deep)


class WideDeep(nn.Module):

    def __init__(self,
                 wide_dim,
                 embeddings_input,
                 continuous_cols,
                 deep_column_idx,
                 hidden_layers,
                 dropout,
                 encoding_dict):

        super(WideDeep, self).__init__()
        self.wide_dim = wide_dim
        self.deep_column_idx = deep_column_idx
        self.embeddings_input = embeddings_input
        self.continuous_cols = continuous_cols
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.encoding_dict = encoding_dict

        # Build the embedding layers to be passed through the deep-side
        for col,val,dim in self.embeddings_input:
            setattr(self, 'emb_layer_'+col, nn.Embedding(val, dim))

        # Build the deep-side hidden layers with dropout if specified
        input_emb_dim = np.sum([emb[2] for emb in self.embeddings_input])
        self.linear_1 = nn.Linear(input_emb_dim+len(continuous_cols), self.hidden_layers[0])
        if self.dropout:
            self.linear_1_drop = nn.Dropout(self.dropout[0])
        for i,h in enumerate(self.hidden_layers[1:],1):
            setattr(self, 'linear_'+str(i+1), nn.Linear( self.hidden_layers[i-1], self.hidden_layers[i] ))
            if self.dropout:
                setattr(self, 'linear_'+str(i+1)+'_drop', nn.Dropout(self.dropout[i]))

        # Connect the wide- and dee-side of the model to the output neuron
        self.output = nn.Linear(self.hidden_layers[-1]+self.wide_dim, 1)


    def forward(self, X_w, X_d):

        # Deep Side
        emb = [getattr(self, 'emb_layer_'+col)(X_d[:,self.deep_column_idx[col]].long())
               for col,_,_ in self.embeddings_input]
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X_d[:, cont_idx].float()]
            deep_inp = torch.cat(emb+cont, 1)
        else:
            deep_inp = torch.cat(emb, 1)

        x_deep = F.relu(self.linear_1(deep_inp))
        if self.dropout:
            x_deep = self.linear_1_drop(x_deep)
        for i in range(1,len(self.hidden_layers)):
            x_deep = F.relu( getattr(self, 'linear_'+str(i+1))(x_deep) )
            if self.dropout:
                x_deep = getattr(self, 'linear_'+str(i+1)+'_drop')(x_deep)

        # Deep + Wide sides
        wide_deep_input = torch.cat([x_deep, X_w.float()], 1)

        out = self.output(wide_deep_input)

        return out


    def fit(self, train_loader, criterion, optimizer, n_epochs, eval_loader=None, lr_scheduler=None):

        train_steps =  (len(train_loader.dataset) // train_loader.batch_size) + 1
        if eval_loader:
            eval_steps =  (len(eval_loader.dataset) // eval_loader.batch_size) + 1

        for epoch in range(n_epochs):
            if lr_scheduler: lr_scheduler.step()
            net = self.train()
            with trange(train_steps) as t:
                for i, (X_wide, X_deep, target) in zip(t, train_loader):
                    t.set_description('epoch %i' % (epoch+1))

                    X_w = Variable(X_wide)
                    X_d = Variable(X_deep)
                    y = Variable(target).float()
                    if use_cuda:
                        X_w, X_d, y = X_w.cuda(), X_d.cuda(), y.cuda()

                    optimizer.zero_grad()

                    y_pred =  net(X_w, X_d)

                    loss = criterion(y_pred.squeeze(1), y)
                    t.set_postfix(loss=loss.item())

                    loss.backward()
                    optimizer.step()

            if eval_loader:
                eval_loss=0
                net = self.eval()
                with trange(eval_steps) as v:
                    for i, (X_wide, X_deep, target) in zip(v, eval_loader):
                        v.set_description('valid')

                        X_w = Variable(X_wide)
                        X_d = Variable(X_deep)
                        y = Variable(target).float()
                        if use_cuda:
                            X_w, X_d, y = X_w.cuda(), X_d.cuda(), y.cuda()

                        y_pred = net(X_w,X_d)

                        loss = criterion(y_pred.squeeze(1), y)
                        v.set_postfix(loss=loss.item())
                        eval_loss+=loss.item()

                eval_loss /= eval_steps
                print("Evaluation loss: {:.4f}".format(eval_loss))


    def predict(self, dataloader):

        test_steps =  (len(dataloader.dataset) // dataloader.batch_size) + 1

        net = self.eval()
        preds_l = []
        with trange(test_steps) as t:
            for i, (X_wide, X_deep) in zip(t, dataloader):
                t.set_description('predict')

                X_w = Variable(X_wide)
                X_d = Variable(X_deep)
                if use_cuda:
                    X_w, X_d = X_w.cuda(), X_d.cuda()

                preds_l.append(net(X_w,X_d))

        preds = torch.cat(preds_l, 0).cpu().data.numpy()

        return preds


    def get_embeddings(self, col_name):

        params = list(self.named_parameters())
        emb_layers = [p for p in params if 'emb_layer' in p[0]]
        emb_layer  = [layer for layer in emb_layers if col_name in layer[0]][0]
        embeddings = emb_layer[1].cpu().data.numpy()
        col_label_encoding = self.encoding_dict[col_name]
        inv_dict = {v:k for k,v in col_label_encoding.items()}
        embeddings_dict = {}
        for idx,value in inv_dict.items():
            embeddings_dict[value] = embeddings[idx]

        return embeddings_dict
