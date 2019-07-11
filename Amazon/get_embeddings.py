import numpy as np
import pandas as pd
import pickle
import torch
import gzip

from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from mlp import MLP
from gmf import GMF


def parse(path):
	g = gzip.open(path, 'rb')
	for l in g:
    	yield eval(l)

def getDF(path):
	i = 0
	df = {}
	for d in parse(path):
		df[i] = d
	    i += 1
	return pd.DataFrame.from_dict(df, orient='index')

DATA_PATH = Path("../datasets/Amazon/")
MODEL_DIR = "models"

asin2id_map = pickle.load(open(DATA_PATH/'item_mappings.p', 'rb'))
id2asin_map = {k:v for v,k in asin2id_map.items()}

df_movies_meta_data = getDF(DATA_PATH/'meta_Movies_and_TV.json.gz')
keep_cols = ['asin', 'title']
df_movies_meta_data = df_movies_meta_data[keep_cols]
df_movies_meta_data = df_movies_meta_data[~df_movies_meta_data.title.isna()]
asin2title_map = dict(df_movies_meta_data.values)

print("number of items with missing title in the core dataset: {}".format(
	np.setdiff1d(list(id2asin_map.values()), list(asin2title_map.keys())).shape[0]))
print("number of items with non missing titles in the core dataset: {}".format(
	len(id2asin_map) \
	- np.setdiff1d(list(id2asin_map.values()), list(asin2title_map.keys())).shape[0]))

id2title_map = {}
for k,v in id2asin_map.items():
	try:
		id2title_map[k] = asin2title_map[v]
	except:
		continue

df_results = pd.read_pickle(DATA_PATH/MODEL_DIR/'results_df.p')
best_gmf = (df_results[df_results.modelname.str.contains('GMF')]
	.sort_values('best_hr', ascending=False)
	.reset_index(drop=True)
	).modelname[0]
n_emb_i = int(np.where([s == 'emb' for s in best_gmf.split("_")])[0])+1
n_emb = int(best_gmf.split("_")[n_emb_i])

dataset = np.load(DATA_PATH/'neuralcf_split.npz')
n_users, n_items = dataset['n_users'].item(), dataset['n_items'].item()

gmf_model = GMF(n_users, n_items, n_emb)
gmf_model.load_state_dict(torch.load(DATA_PATH/MODEL_DIR/best_gmf))
item_embeddings = gmf_model.embeddings_item.weight.data.numpy()

knn_model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
knn_model.fit(item_embeddings)

def get_movie_titles(input_id, n=20):
	"""first movie will be the "query" movie and the remaining n-1 the similar
	movies. Similar defined under the functioning of the algorithm, i.e.
	leading to the same prediction"""
	dist, nnidx = knn_model.kneighbors(
		item_embeddings[input_id].reshape(1, -1),
		n_neighbors = n)
	titles = []
	for idx in nnidx[0]:
		try:
			titles.append(id2title_map[idx])
		except:
			continue
	return titles

similar_movies = get_movie_titles(1234)
# In [13]: similar_movies
# Out[13]:
# ['Ace Ventura: Pet Detective',
#  'Gone in 60 Seconds',
#  'Better Off Dead [VHS]',
#  'Dinosaur [VHS]',
#  'Coming to America [VHS]',
#  'Rush Hour [VHS]',
#  'Tommy Boy [VHS]',
#  'Rush Hour 2 [VHS]',
#  "Bill and Ted's Excellent Adventure [VHS]",
#  'Evolution',
#  'Con Air [VHS]',
#  'Liar Liar [VHS]',
#  'The Karate Kid',
#  'Weird Science [VHS]',
#  'The Fast and the Furious',
#  'Jumanji [VHS]',
#  'Open Season [UMD for PSP]',
#  'Ace Ventura: When Nature Calls [VHS]',
#  'Scary Movie',
#  'Die Hard Trilogy']