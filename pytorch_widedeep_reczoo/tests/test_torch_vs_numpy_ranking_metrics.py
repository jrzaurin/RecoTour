import numpy as np
import torch
import pytest
from pytorch_widedeep.metrics import MAP_at_k, NDCG_at_k, HitRatio_at_k, BinaryNDCG_at_k

from rec_tools.ranking_metrics import (
    map_at_k,
    ndcg_at_k,
    hit_ratio_at_k,
    binary_ndcg_at_k,
)


@pytest.fixture
def sample_data():
    # Create sample predictions and ground truth
    y_pred_np = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])  # 2 users, 3 items each
    y_true_np = np.array(
        [1.0, 0.3, 0.7, 0.2, 0.8, 0.4]
    )  # relevance scores between 0 and 1

    # Create PyTorch tensors
    y_pred_torch = torch.FloatTensor(y_pred_np)
    y_true_torch = torch.FloatTensor(y_true_np)

    return {
        "numpy": (y_pred_np, y_true_np),
        "torch": (y_pred_torch, y_true_torch),
        "n_items": 3,
    }


@pytest.mark.parametrize("k", [None, 2])
def test_ndcg_at_k(sample_data, k):
    y_pred_np, y_true_np = sample_data["numpy"]
    y_pred_torch, y_true_torch = sample_data["torch"]
    n_items = sample_data["n_items"]
    np_result = ndcg_at_k(y_pred_np, y_true_np, n_items=n_items, k=k)
    torch_ndcg_at_k = NDCG_at_k(n_items=n_items, k=k)
    torch_result = torch_ndcg_at_k(y_pred_torch, y_true_torch)
    assert np.abs(np_result - torch_result.item()) < 1e-6


@pytest.mark.parametrize("k", [None, 2])
def test_binary_ndcg_at_k(sample_data, k):
    y_pred_np, y_true_np = sample_data["numpy"]
    y_true_np = (y_true_np > 0.5).astype(float)
    y_pred_torch, y_true_torch = sample_data["torch"]
    y_true_torch = (y_true_torch > 0.5).int()

    n_items = sample_data["n_items"]

    np_result = binary_ndcg_at_k(y_pred_np, y_true_np, n_items=n_items, k=k)
    torch_result = BinaryNDCG_at_k(n_items=n_items, k=k)(y_pred_torch, y_true_torch)

    assert np.abs(np_result - torch_result.item()) < 1e-6


@pytest.mark.parametrize("k", [None, 2])
def test_map_at_k(sample_data, k):
    y_pred_np, y_true_np = sample_data["numpy"]
    y_pred_torch, y_true_torch = sample_data["torch"]
    n_items = sample_data["n_items"]

    np_result = map_at_k(y_pred_np, y_true_np, n_items=n_items, k=k)
    torch_result = MAP_at_k(n_items=n_items, k=k)(y_pred_torch, y_true_torch)

    assert np.abs(np_result - torch_result.item()) < 1e-6


@pytest.mark.parametrize("k", [None, 2])
def test_hit_ratio_at_k(sample_data, k):
    y_pred_np, y_true_np = sample_data["numpy"]
    y_pred_torch, y_true_torch = sample_data["torch"]
    n_items = sample_data["n_items"]

    np_result = hit_ratio_at_k(y_pred_np, y_true_np, n_items=n_items, k=k)
    torch_result = HitRatio_at_k(n_items=n_items, k=k)(y_pred_torch, y_true_torch)

    assert np.abs(np_result - torch_result.item()) < 1e-6
