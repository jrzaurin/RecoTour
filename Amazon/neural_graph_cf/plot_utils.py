import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: lightgreen' if is_max.any() else '' for v in is_max]

def plot_loss(df):
	sns.set(color_codes=True)
	sns.set_context("notebook", font_scale=1.)
	plt.figure(figsize=(15, 10))
	plt.subplot(2,2,1)
	plt.subplots_adjust(hspace=0.4)
	fig = sns.lineplot(x='loss', y='precision', hue='k', style='k',
		markers=True, markersize=10, linewidth=2, data=df.round(4), legend="full")
	fig.set(ylabel="PRECISION")
	fig.set(xlabel="BPR Loss")
	plt.subplot(2,2,2)
	fig = sns.lineplot(x='loss', y='recall', hue='k', style='k',
		markers=True, markersize=10, linewidth=2, data=df.round(4), legend="full")
	fig.set(ylabel="RECALL")
	fig.set(xlabel="BPR Loss")
	plt.subplot(2,2,3)
	fig = sns.lineplot(x='loss', y='hit_ratio', hue='k', style='k',
		markers=True, markersize=10, linewidth=2, data=df.round(4), legend="full")
	fig.set(ylabel="HIT_RATIO")
	fig.set(xlabel="BPR Loss")
	plt.subplot(2,2,4)
	fig = sns.lineplot(x='loss', y='ndcg', hue='k', style='k',
		markers=True, markersize=10, linewidth=2, data=df.round(4), legend="full")
	fig.set(ylabel="NDCG")
	fig.set(xlabel="BPR Loss")
