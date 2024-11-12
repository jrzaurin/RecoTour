from typing import Any, Dict, List, Union, Literal, Sequence

import numpy as np
import cohere
import pandas as pd
import umap.umap_ as umap
from chromadb import Client, Collection
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rec_zoo.tokens_and_api_keys import COHERE_API_KEY
from rec_zoo.feat_engineering.utils import save_objects


class OverviewEmbedder:
    """Embeds movie overviews using either SentenceTransformers or Cohere"""

    def __init__(
        self,
        method: Literal["sentence_transformer", "cohere"] = "sentence_transformer",
        model_name: str = "all-mpnet-base-v2",
        save_dir: str = "feature_store",
    ):
        """
        Args:
            method: str
                Either 'sentence_transformer' or 'cohere'
            model_name:
                Model name for SentenceTransformer
        """
        self.method = method
        if method == "sentence_transformer":
            self.embedder = SentenceTransformer(model_name)
            self.db_name = f"overview_embeddings_st_{model_name}"
        elif method == "cohere":
            self.embedder = cohere.Client(COHERE_API_KEY)  # type: ignore[assignment]
            self.db_name = "overview_embeddings_cohere"

        self.chroma_client = Client(
            Settings(persist_directory=f"{save_dir}/{self.db_name}")
        )
        self.collection: Collection | None = None

    def embed_overviews(
        self, df: pd.DataFrame, overview_col: str = "overview", id_col: str = "item_id"
    ) -> None:
        df = df[df[overview_col].notna()].copy()

        if self.method == "sentence_transformer":
            embeddings = self.embedder.encode(
                df[overview_col].tolist(), show_progress_bar=True, convert_to_numpy=True
            )
        else:  # cohere
            response = self.embedder.embed(
                texts=df[overview_col].tolist(),
                model="embed-english-v3.0",
                input_type="search_document",
            )
            embeddings = np.array(response.embeddings)

        self.collection = self.chroma_client.create_collection(name=self.db_name)

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=df[overview_col].tolist(),
            ids=[str(id_) for id_ in df[id_col]],
        )

    def get_embedding(
        self, movie_id: str | int
    ) -> Sequence[float] | Any | None:  # Any to avoid mypy type error
        """Retrieve embedding for a specific movie ID"""
        if not self.collection:
            raise ValueError("No embeddings found. Run embed_overviews first.")

        result = self.collection.get(ids=[str(movie_id)], include=["embeddings"])  # type: ignore[list-item]

        if result and result["embeddings"]:
            return result["embeddings"][0]
        return None

    def get_similar_movies(
        self, movie_id: Union[str, int], n_similar: int = 5
    ) -> List[Dict]:
        """Find similar movies based on overview embeddings"""
        if not self.collection:
            raise ValueError("No embeddings found. Run embed_overviews first.")

        query_embedding = self.get_embedding(movie_id)
        if not query_embedding:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_similar + 1,  # +1 because it will include the query movie
        )

        similar_movies = []
        for i, (id_, distance) in enumerate(
            zip(results["ids"][0], results["distances"][0])
        ):
            if id_ != str(movie_id):  # Exclude the query movie
                similar_movies.append(
                    {
                        "movie_id": id_,
                        "overview": results["documents"][0][i],
                        "similarity_score": 1
                        - distance,  # Convert distance to similarity
                    }
                )

        return similar_movies[:n_similar]


class EmbeddingDimensionalityReducer:
    """Reduces dimensionality of embeddings using UMAP"""

    def __init__(
        self,
        n_components: int = 5,
        random_state: int = 42,
        umap_config: Dict[str, Any] | None = None,
        save_dir: str | None = "feature_store",
    ):
        """
        Args:
            n_components: int
                Number of dimensions to reduce to
            random_state: int
                Random seed for reproducibility
            umap_config: Dict[str, Any] | None
                Additional UMAP configuration parameters. Defaults to:
                {
                    'n_neighbors': 15,
                    'min_dist': 0.0,
                    'metric': 'cosine',
                    'low_memory': False
                }
        """
        self.n_components = n_components
        self.random_state = random_state
        self.save_dir = save_dir

        default_config = {
            "n_neighbors": 15,
            "min_dist": 0.0,
            "metric": "cosine",
            "low_memory": False,
        }

        if umap_config:
            default_config.update(umap_config)

        self.reducer = umap.UMAP(
            n_components=n_components, random_state=random_state, **default_config
        )

    def reduce(
        self,
        embeddings: np.ndarray | None = None,
        collection: Collection | None = None,
    ) -> pd.DataFrame:
        """
        Reduce dimensionality of embeddings using UMAP.

        Args:
            embeddings: numpy array of embeddings, or
            collection: ChromaDB collection to retrieve embeddings from

        Returns:
            DataFrame with movie IDs and low-dimensional embeddings
        """
        if embeddings is None and collection is None:
            raise ValueError("Either embeddings or collection must be provided")

        if embeddings is not None and collection is not None:
            raise ValueError("Only one of embeddings or collection should be provided")

        if collection is not None:
            result = collection.get(include=["embeddings", "ids"])  # type: ignore[list-item]
            embeddings = np.array(result["embeddings"])
            ids = result["ids"]
        else:
            ids = [str(i) for i in range(len(embeddings))]

        low_dim_embeddings = self.reducer.fit_transform(embeddings)

        embedding_cols = [f"embedding_{i}" for i in range(self.n_components)]
        results_df = pd.DataFrame(low_dim_embeddings, columns=embedding_cols)
        results_df["movie_id"] = ids

        if self.save_dir:
            save_objects([self.reducer], ["umap_reducer"], self.save_dir)

        return results_df[["movie_id"] + embedding_cols]
