import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn  as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from recutils.wide_deep import WideDeepLoader, WideDeep

wd_dir = "../datasets/Ponpare/data_processed/wide_deep"
wd_dataset_fname = "wd_dataset.p"
wd_interactions_fname = "interactions_dict.p"
wd_dataset = pickle.load(open(os.path.join(wd_dir,wd_dataset_fname), "rb"))
wd_interactions = pickle.load(open(os.path.join(wd_dir,wd_interactions_fname), "rb"))

# Network set up
wide_dim = wd_dataset['train_dataset']['wide'].shape[1]
deep_column_idx = wd_dataset['deep_column_idx']
continuous_cols = wd_dataset['continuous_cols']
embeddings_input= wd_dataset['embeddings_input']
encoding_dict   = wd_dataset['encoding_dict']
hidden_layers = [100,50]
dropout = [0.5,0.5]

train_dataset = wd_dataset['train_dataset']
widedeep_dataset_tr = WideDeepLoader(train_dataset)
train_loader = DataLoader(dataset=widedeep_dataset_tr,
    batch_size=5096,
    shuffle=True,
    num_workers=4)

valid_dataset = wd_dataset['valid_dataset']
widedeep_dataset_val = WideDeepLoader(valid_dataset)
eval_loader = DataLoader(dataset=widedeep_dataset_val,
    batch_size=5096,
    shuffle=True,
    num_workers=4)

test_dataset = wd_dataset['test_dataset']
widedeep_dataset_te = WideDeepLoader(test_dataset, mode='test')
test_loader = DataLoader(dataset=widedeep_dataset_te,
    batch_size=5096,
    shuffle=False,
    num_workers=4)

model = WideDeep(wide_dim,embeddings_input,continuous_cols,deep_column_idx,hidden_layers,dropout,encoding_dict)
model.cuda()
criterion = F.mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)

model.fit(train_loader, criterion, optimizer, n_epochs=5, eval_loader=eval_loader, lr_scheduler=lr_scheduler)
preds = model.predict(test_loader)

df_all_interactions = wd_interactions['all_valid_interactions']
df_all_interactions['interest'] = preds
df_ranked = df_all_interactions.sort_values(['user_id_hash', 'interest'], ascending=[False, False])
df_ranked = (df_ranked
	.groupby('user_id_hash')['valid_coupon_id_hash']
	.apply(list)
	.reset_index())
recomendations_dict = pd.Series(df_ranked.valid_coupon_id_hash.values,
	index=df_ranked.user_id_hash).to_dict()
true_valid_interactions = wd_interactions['true_valid_interactions']

actual = []
pred = []
for k,_ in recomendations_dict.items():
	actual.append(list(true_valid_interactions[k]))
	pred.append(list(recomendations_dict[k]))

from recutils.average_precision import mapk
print(mapk(actual,pred))
