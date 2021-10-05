import numpy as np
import pandas as pd
import pickle
import torch
import gzip

from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from ngcf import NGCF


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


DATA_PATH = Path("/home/ubuntu/projects/RecoTour/datasets/Amazon/")
MODEL_DIR = Path("results")

asin2id_df = pd.read_csv(DATA_PATH / "item_list.txt", sep=" ")
id2asin_map = dict(zip(asin2id_df.remap_id, asin2id_df.orig_id))

df_movies_meta_data = getDF(DATA_PATH / "meta_Movies_and_TV.json.gz")
keep_cols = ["asin", "title"]
df_movies_meta_data = df_movies_meta_data[keep_cols]
df_movies_meta_data = df_movies_meta_data[~df_movies_meta_data.title.isna()]
asin2title_map = dict(df_movies_meta_data.values)

print(
    "number of items with missing title in the core dataset: {}".format(
        np.setdiff1d(list(id2asin_map.values()), list(asin2title_map.keys())).shape[0]
    )
)
print(
    "number of items with non missing titles in the core dataset: {}".format(
        len(id2asin_map)
        - np.setdiff1d(list(id2asin_map.values()), list(asin2title_map.keys())).shape[0]
    )
)

id2title_map = {}
for k, v in id2asin_map.items():
    try:
        id2title_map[k] = asin2title_map[v]
    except:
        continue

df_results = pd.read_csv(MODEL_DIR / "results_df.csv")
best_model = df_results.sort_values(by="recall@20", ascending=False).modelname.tolist()[
    0
]

state_dict = torch.load(MODEL_DIR / best_model)

item_embeddings = state_dict["i_embeddings"].cpu().numpy()
item_g_embeddings = state_dict["i_g_embeddings"].cpu().numpy()

knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(item_embeddings)
knn_model_g = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model_g.fit(item_g_embeddings)


def get_movie_titles(input_id, nn_model, embeddings, n=20):
    """first movie will be the "query" movie and the remaining n-1 the similar
    movies. Similar defined under the functioning of the algorithm, i.e.
    leading to the same prediction"""
    dist, nnidx = nn_model.kneighbors(
        embeddings[input_id].reshape(1, -1), n_neighbors=n
    )
    titles = []
    for idx in nnidx[0]:
        try:
            titles.append(id2title_map[idx])
        except:
            continue
    return titles


# similar_movies = get_movie_titles(1234, knn_model_g, item_g_embeddings)
similar_movies = get_movie_titles(1234, knn_model, item_embeddings)
# ['Ace Ventura: Pet Detective',
#  'Rush Hour [VHS]',
#  'The Karate Kid',
#  'Dumb and Dumber',
#  'The Mask',
#  "Brewster's Millions [VHS]",
#  'Teenage Mutant Ninja Turtles - The Original Movie [VHS]',
#  "Wayne's World [VHS]",
#  'Ace Ventura: When Nature Calls [VHS]',
#  "Lara Croft: Tomb Raider - The Cradle of Life (Full Screen Special Collector's Edition)",
#  'Teenage Mutant Ninja Turtles II - The Secret of the Ooze [VHS]',
#  'Jumanji [VHS]',
#  'Teen Wolf [VHS]',
#  'Shanghai Knights',
#  'Superstar',
#  'The Nutty Professor',
#  'Bedazzled',
#  'Tommy Boy [VHS]',
#  'Austin Powers - International Man of Mystery [VHS]',
#  'Eight Legged Freaks (Widescreen Edition) (Snap Case)']
