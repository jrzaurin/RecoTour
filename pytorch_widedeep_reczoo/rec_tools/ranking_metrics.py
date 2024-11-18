from typing import Optional

import numpy as np


def reshape_to_2d(array: np.ndarray, n_columns: int) -> np.ndarray:
    """Reshape 1D or 2D array with one column to 2D array with n_columns"""
    if array.ndim == 1:
        if array.shape[0] % n_columns != 0:
            raise ValueError(
                f"Array length ({array.shape[0]}) must be divisible by n_columns ({n_columns})"
            )
        n_rows = array.shape[0] // n_columns
        return array.reshape(n_rows, n_columns)
    elif array.ndim == 2 and array.shape[1] == 1:
        if array.shape[0] % n_columns != 0:
            raise ValueError(
                f"Array length ({array.shape[0]}) must be divisible by n_columns ({n_columns})"
            )
        n_rows = array.shape[0] // n_columns
        return array.reshape(n_rows, n_columns)
    else:
        raise ValueError(
            "Input array must be 1-dimensional or 2-dimensional with one column"
        )


def ndcg_at_k(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_items: int = 10,
    k: Optional[int] = None,
    eps: float = 1e-8,
) -> float:
    """NumPy implementation of NDCG@k for non-binary relevance scores"""
    if k is None:
        k = n_items

    # Reshape inputs
    y_pred_2d = reshape_to_2d(y_pred, n_items)
    y_true_2d = reshape_to_2d(y_true, n_items)

    # Get top k indices
    top_k_indices = np.argsort(-y_pred_2d, axis=1)[:, :k]

    # Get relevance scores for top k items
    top_k_relevance = np.take_along_axis(y_true_2d, top_k_indices, axis=1)

    # Calculate discounts
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    # Calculate DCG
    dcg = ((2**top_k_relevance - 1) * discounts).sum(axis=1)

    # Calculate IDCG
    ideal_relevance = -np.sort(-y_true_2d, axis=1)[:, :k]
    idcg = ((2**ideal_relevance - 1) * discounts).sum(axis=1)

    # Calculate NDCG
    ndcg = dcg / (idcg + eps)

    return ndcg.mean()


def binary_ndcg_at_k(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_items: int = 10,
    k: Optional[int] = None,
    eps: float = 1e-8,
) -> float:
    """NumPy implementation of NDCG@k for binary relevance scores"""
    if k is None:
        k = n_items

    # Reshape inputs
    y_pred_2d = reshape_to_2d(y_pred, n_items)
    y_true_2d = reshape_to_2d(y_true, n_items)

    # Get top k indices
    top_k_indices = np.argsort(-y_pred_2d, axis=1)[:, :k]

    # Get relevance scores for top k items
    top_k_relevance = np.take_along_axis(y_true_2d, top_k_indices, axis=1)

    # Calculate discounts
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    # Calculate DCG (removed unnecessary mask)
    dcg = (top_k_relevance * discounts[np.newaxis, :k]).sum(axis=1)

    # Calculate IDCG
    n_relevant = np.minimum(y_true_2d.sum(axis=1), k)
    idcg = np.array([discounts[: int(n)].sum() for n in n_relevant])

    # Calculate NDCG
    ndcg = dcg / (idcg + eps)

    return ndcg.mean()


def map_at_k(
    y_pred: np.ndarray, y_true: np.ndarray, n_items: int = 10, k: Optional[int] = None
) -> float:
    """NumPy implementation of MAP@k"""
    if k is None:
        k = n_items

    # Reshape inputs
    y_pred_2d = reshape_to_2d(y_pred, n_items)
    y_true_2d = reshape_to_2d(y_true, n_items)

    # Get top k indices
    top_k_indices = np.argsort(-y_pred_2d, axis=1)[:, :k]

    # Get relevance scores for top k items
    batch_relevance = np.take_along_axis(y_true_2d, top_k_indices, axis=1)

    # Calculate cumulative sum and precision at each position
    cumsum_relevance = np.cumsum(batch_relevance, axis=1)
    precision_at_i = cumsum_relevance / np.arange(1, k + 1)

    # Calculate average precision
    avg_precision = (precision_at_i * batch_relevance).sum(axis=1) / np.maximum(
        y_true_2d.sum(axis=1), 1
    )

    return avg_precision.mean()


def hit_ratio_at_k(
    y_pred: np.ndarray, y_true: np.ndarray, n_items: int = 10, k: Optional[int] = None
) -> float:
    """NumPy implementation of HR@k"""
    if k is None:
        k = n_items

    # Reshape inputs
    y_pred_2d = reshape_to_2d(y_pred, n_items)
    y_true_2d = reshape_to_2d(y_true, n_items)

    # Get top k indices
    top_k_indices = np.argsort(-y_pred_2d, axis=1)[:, :k]

    # Get relevance scores for top k items
    batch_relevance = np.take_along_axis(y_true_2d, top_k_indices, axis=1)

    # Calculate hit ratio
    hit = (batch_relevance.sum(axis=1) > 0).astype(float)

    return hit.mean()
