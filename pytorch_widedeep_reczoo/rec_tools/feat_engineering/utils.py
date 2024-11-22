import pickle
from typing import Any, List, Tuple, Literal
from pathlib import Path

import pandas as pd

from rec_tools.constants import DATA_DIR


def load_movielens_train_val(
    split_path: str, return_split: Literal["train", "val", "both"]
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    root_dir = Path(f"{DATA_DIR}/train_val_test_splits")
    full_path = root_dir / split_path

    if return_split == "train":
        train_path = full_path / "train.csv"
        return pd.read_csv(train_path), None
    elif return_split == "val":
        val_path = full_path / "val.csv"
        return None, pd.read_csv(val_path)
    else:  # return both
        train_path = full_path / "train.csv"
        val_path = full_path / "val.csv"
        return pd.read_csv(train_path), pd.read_csv(val_path)


def load_movielens(full_path: str | None = None) -> pd.DataFrame:
    ml_path = (
        Path(full_path)
        if full_path
        else Path(f"{DATA_DIR}/raw_data/ml-1m/movielens_ratings_with_info.csv")
    )

    ml_df = pd.read_csv(ml_path)
    ml_df = (
        ml_df[["item_id", "title", "genres"]].drop_duplicates().reset_index(drop=True)
    )

    return ml_df


def load_movie_metadata(full_path: str | None = None) -> pd.DataFrame:
    ml_path = (
        Path(full_path)
        if full_path
        else Path(f"{DATA_DIR}/raw_data/ml-1m/movies_metadata.csv.zip")
    )
    return pd.read_csv(ml_path)


def save_objects(input_objs: List[Any], save_fnames: List[str], save_dir: str) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for obj, fname in zip(input_objs, save_fnames):
        file_path = save_path / fname
        suffix = file_path.suffix

        if suffix == ".pkl":
            with file_path.open("wb") as f:
                pickle.dump(obj, f)
        elif suffix == ".csv":
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(file_path, index=False)
            else:
                pd.DataFrame(obj).to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")
