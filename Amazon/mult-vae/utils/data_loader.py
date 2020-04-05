import os
import pathlib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy import sparse

from typing import Tuple
from scipy.sparse.csr import csr_matrix


class DataLoader(object):
    def __init__(self, path: pathlib.PosixPath):
        super(DataLoader, self).__init__()

        self.path = path
        self.n_items = self._n_items()

    def _n_items(self):
        tr_items = np.load(self.path / "tr_items.npy", allow_pickle=True)
        return len(tr_items)

    def load_data(self, datatype: str = "train"):
        if datatype == "train":
            return self._load_train_data()
        else:
            return self._load_valid_test_data(datatype)

    def _load_train_data(self) -> csr_matrix:
        tp = pd.read_csv(self.path / "train.csv")
        n_users = tp["user"].max() + 1
        rows, cols = tp["user"], tp["item"]
        data = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            dtype="float32",
            shape=(n_users, self.n_items),
        )
        return data

    def _load_valid_test_data(self, datatype: str) -> Tuple[csr_matrix, csr_matrix]:
        tp_tr = pd.read_csv(self.path / "{}_tr.csv".format(datatype))
        tp_te = pd.read_csv(self.path / "{}_te.csv".format(datatype))

        start_idx = min(tp_tr["user"].min(), tp_te["user"].min())
        end_idx = max(tp_tr["user"].max(), tp_te["user"].max())

        rows_tr, cols_tr = tp_tr["user"] - start_idx, tp_tr["item"]
        rows_te, cols_te = tp_te["user"] - start_idx, tp_te["item"]

        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)),
            dtype="float32",
            shape=(end_idx - start_idx + 1, self.n_items),
        )
        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)),
            dtype="float32",
            shape=(end_idx - start_idx + 1, self.n_items),
        )
        return data_tr, data_te
