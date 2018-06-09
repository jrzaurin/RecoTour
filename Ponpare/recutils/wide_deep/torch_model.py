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
import pdb

use_cuda = torch.cuda.is_available()


wd_dataset = pickle.load(open("wd_dataset.p", "rb"))


class WideDeepLoader(Dataset):
    """Helper to facilitate loading the data to the pytorch models.
    Parameters:
    --------
    data: namedtuple with 3 elements - (wide_input_data, deep_inp_data, target)
    """
    def __init__(self, data):

        self.X_wide = data['wide']
        self.X_deep = data['deep']
        self.Y = data['target']

    def __getitem__(self, idx):

        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        y  = self.Y[idx]

        return xw, xd, y

    def __len__(self):
        return len(self.Y)


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
            cont = [X_d[:, :42].float()]
            deep_inp = torch.cat(cont+emb, 1)
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

# Network set up
wide_dim = wd_dataset['train_dataset']['wide'].shape[1]
deep_column_idx = wd_dataset['deep_column_idx']
continuous_cols = wd_dataset['continuous_cols']
embeddings_input= wd_dataset['embeddings_input']
encoding_dict   = wd_dataset['encoding_dict']
hidden_layers = [100,50]
dropout = [0.5,0.2]

model = WideDeep(wide_dim,embeddings_input,continuous_cols,deep_column_idx,hidden_layers,dropout,encoding_dict)

train_dataset = wd_dataset['train_dataset']
widedeep_dataset = WideDeepLoader(train_dataset)
train_loader = DataLoader(dataset=widedeep_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4)

model.cuda()
criterion = F.mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(3):
for i, (X_wide, X_deep, target) in enumerate(train_loader):
    X_w = Variable(X_wide)
    X_d = Variable(X_deep)
    y = Variable(target).float()

    if use_cuda:
        X_w, X_d, y = X_w.cuda(), X_d.cuda(), y.cuda()

    optimizer.zero_grad()
    y_pred =  model(X_w, X_d)
    loss = criterion(y_pred.squeeze(1), y)
    loss.backward()
    optimizer.step()

    # print ('Epoch {} Loss: {}'.format(epoch+1,
    #     round(loss.item(),3)))
