import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn  as nn
import torch.nn.functional as F

from recutils.average_precision import mapk
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from recutils.wide_deep import WideDeepLoader, WideDeep

wd_dir = "../datasets/Ponpare/data_processed/wide_deep"
wd_dataset_fname = "wd_dataset.p"
wd_interactions_fname = "interactions_dict.p"
wd_dataset = pickle.load(open(os.path.join(wd_dir,wd_dataset_fname), "rb"))
wd_interactions = pickle.load(open(os.path.join(wd_dir,wd_interactions_fname), "rb"))

# model inputs
wide_dim = wd_dataset['train_dataset']['wide'].shape[1]
deep_column_idx = wd_dataset['deep_column_idx']
continuous_cols = wd_dataset['continuous_cols']
embeddings_input= wd_dataset['embeddings_input']
encoding_dict   = wd_dataset['encoding_dict']

# Interactions during "testing period"
df_all_interactions = wd_interactions['all_valid_interactions']

# datasets
train_dataset = wd_dataset['train_dataset']
# train_dataset['wide'] = np.array(train_dataset['wide'].todense())
widedeep_dataset_tr = WideDeepLoader(train_dataset)

valid_dataset = wd_dataset['valid_dataset']
# valid_dataset['wide'] = np.array(valid_dataset['wide'].todense())
widedeep_dataset_val = WideDeepLoader(valid_dataset)

test_dataset = wd_dataset['test_dataset']
# test_dataset['wide'] = np.array(test_dataset['wide'].todense())
widedeep_dataset_te = WideDeepLoader(test_dataset, mode='test')

# Let's manually define some network set_ups for the experiment
set_ups = {}
set_ups['set_up_1'] = {}
set_ups['set_up_1']['batch_size'] = 4096
set_ups['set_up_1']['lr'] = 0.01
set_ups['set_up_1']['hidden_layers'] = [50, 25]
set_ups['set_up_1']['dropout'] = [0.5, 0.2]
set_ups['set_up_1']['n_epochs'] = 3

set_ups['set_up_2'] = {}
set_ups['set_up_2']['batch_size'] = 4096
set_ups['set_up_2']['lr'] = 0.01
set_ups['set_up_2']['hidden_layers'] = [100, 50]
set_ups['set_up_2']['dropout'] = [0.5, 0.5]
set_ups['set_up_2']['n_epochs'] = 6

set_ups['set_up_3'] = {}
set_ups['set_up_3']['batch_size'] = 8192
set_ups['set_up_3']['lr'] = 0.05
set_ups['set_up_3']['hidden_layers'] = [100, 100, 100]
set_ups['set_up_3']['dropout'] = [0.5, 0.5, 0.5]
set_ups['set_up_3']['n_epochs'] = 10

set_ups['set_up_4'] = {}
set_ups['set_up_4']['batch_size'] = 8192
set_ups['set_up_4']['lr'] = 0.05
set_ups['set_up_4']['hidden_layers'] = [100, 50, 25]
set_ups['set_up_4']['dropout'] = [0.5, 0.2, 0]
set_ups['set_up_4']['n_epochs'] = 10

set_ups['set_up_5'] = {}
set_ups['set_up_5']['batch_size'] = 9216
set_ups['set_up_5']['lr'] = 0.05
set_ups['set_up_5']['hidden_layers'] = [100, 50]
set_ups['set_up_5']['dropout'] = [0.5, 0.2]
set_ups['set_up_5']['n_epochs'] = 5

results = {}
for set_up_name, params in set_ups.items():
    print("INFO: {}".format(set_up_name))

    batch_size = params['batch_size']
    hidden_layers = params['hidden_layers']
    dropout = params['dropout']
    n_epochs = params['n_epochs']
    lr = params['lr']

    train_loader = DataLoader(dataset=widedeep_dataset_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    eval_loader = DataLoader(dataset=widedeep_dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    test_loader = DataLoader(dataset=widedeep_dataset_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    model = WideDeep(
        wide_dim,
        embeddings_input,
        continuous_cols,
        deep_column_idx,
        hidden_layers,
        dropout,
        encoding_dict
        )
    model.cuda()

    criterion = F.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # schedulers need to be define after the optimizer
    if set_up_name is 'set_up_1':
        lr_scheduler = None
    elif set_up_name is 'set_up_2':
        lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    elif set_up_name is 'set_up_3':
        lr_scheduler = MultiStepLR(optimizer, milestones=[3,8], gamma=0.1)
    elif set_up_name is 'set_up_4':
        lr_scheduler = MultiStepLR(optimizer, milestones=[3,8], gamma=0.1)
    elif set_up_name is 'set_up_5':
        lr_scheduler = MultiStepLR(optimizer, milestones=[2,4], gamma=0.1)

    model.fit(
        train_loader,
        criterion,
        optimizer,
        n_epochs=n_epochs,
        eval_loader=eval_loader,
        lr_scheduler=lr_scheduler
        )
    preds = model.predict(test_loader)

    df_all_interactions['interest'] = preds
    df_ranked = df_all_interactions.sort_values(['user_id_hash', 'interest'], ascending=[False, False])
    df_ranked = (df_ranked
    	.groupby('user_id_hash')['coupon_id_hash']
    	.apply(list)
    	.reset_index())
    recomendations_dict = pd.Series(df_ranked.coupon_id_hash.values,
    	index=df_ranked.user_id_hash).to_dict()
    true_valid_interactions = wd_interactions['true_valid_interactions']

    actual = []
    pred = []
    for k,_ in recomendations_dict.items():
    	actual.append(list(true_valid_interactions[k]))
    	pred.append(list(recomendations_dict[k]))
    print("Mean Average Precission: {}".format(mapk(actual,pred)))
    results[set_up_name] = mapk(actual,pred)
    del(model, optimizer, criterion)
