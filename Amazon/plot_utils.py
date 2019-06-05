import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: lightgreen' if is_max.any() else '' for v in is_max]


def build_model_df(df):
    df_cp = df.copy()
    df_cp = df_cp[df_cp.modelname.str.contains("MLP|GMF") & ~df_cp.modelname.str.contains("MSE")]
    df_cp['model'] = df_cp.modelname.apply(lambda x: 'GMF' if 'GMF' in x else 'MLP')
    df_cp = (df_cp
    	.sort_values('best_hr', ascending=False)
    	.reset_index(drop=True)
    	)
    n_emb = [int(mn.split("n_emb_")[1].split("_")[0]) for mn in df_cp.modelname.tolist()]
    df_cp['n_emb'] = n_emb
    df_cp = (df_cp
    	.sort_values(by=['model', 'n_emb', 'best_hr'], ascending=[True,True,False])
    	.reset_index(drop=True)
    	)
    df_cp = df_cp.groupby(['model', 'n_emb']).first().reset_index()
    return df_cp


def plot_emb(df):
	sns.set(color_codes=True)
	sns.set_context("notebook", font_scale=1.)
	plt.figure(figsize=(15, 10))
	plt.subplot(2,2,1)
	plt.subplots_adjust(hspace=0.4)
	fig = sns.lineplot(x='n_emb', y='best_hr', hue='model', style='model',
		markers=True, markersize=10, linewidth=2, data=df)
	fig.set(ylabel="HR@10")
	fig.set(xlabel="Number of Embeddings")
	plt.xticks(df.n_emb.unique())
	plt.subplot(2,2,2)
	fig = sns.lineplot(x='n_emb', y='best_ndcg', hue='model', style='model',
		markers=True, markersize=10, linewidth=2, data=df)
	fig.set(ylabel="NDCG@10")
	fig.set(xlabel="Number of Embeddings")
	plt.xticks(df.n_emb.unique())
	plt.subplot(2,2,3)
	fig = sns.lineplot(x='n_emb', y='iter_loss', hue='model', style='model',
		markers=True, markersize=10, linewidth=2, data=df)
	fig.set(ylabel="BCELoss")
	fig.set(xlabel="Number of Embeddings")
	plt.xticks(df.n_emb.unique())


def plot_loss(df):
	sns.set(color_codes=True)
	sns.set_context("notebook", font_scale=1.)
	plt.figure(figsize=(15, 10))
	plt.subplot(2,2,1)
	plt.subplots_adjust(hspace=0.4)
	fig = sns.lineplot(x='iter_loss', y='best_hr', hue='model', style='model',
		markers=True, markersize=10, linewidth=2, data=df.round(4))
	fig.set(ylabel="HR@10")
	fig.set(xlabel="BCE Loss")
	plt.subplot(2,2,2)
	fig = sns.lineplot(x='iter_loss', y='best_ndcg', hue='model', style='model',
		markers=True, markersize=10, linewidth=2, data=df.round(4))
	fig.set(ylabel="NDCG@10")
	fig.set(xlabel="BCE Loss")
