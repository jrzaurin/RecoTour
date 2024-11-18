import numpy as np
import pytest

from rec_tools.ranking_metrics import (
    map_at_k,
    ndcg_at_k,
    reshape_to_2d,
    hit_ratio_at_k,
    binary_ndcg_at_k,
)


@pytest.fixture
def sample_data():
    # Create sample predictions and ground truth
    y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])  # 2 users, 3 items each
    y_true = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.0])  # 2 users, 3 items each
    return y_pred, y_true


def test_reshape_to_2d():
    # Test 1D array
    arr_1d = np.array([1, 2, 3, 4])
    result = reshape_to_2d(arr_1d, n_columns=2)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))

    # Test 2D array with one column
    arr_2d = np.array([[1], [2], [3], [4]])
    result = reshape_to_2d(arr_2d, n_columns=2)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))

    # Test invalid input
    with pytest.raises(ValueError):
        reshape_to_2d(np.array([1, 2, 3]), n_columns=2)  # Not divisible
    with pytest.raises(ValueError):
        reshape_to_2d(
            np.array([[1, 2], [3, 4]]), n_columns=2
        )  # Already 2D with 2 columns


def test_ndcg_at_k(sample_data):
    y_pred, y_true = sample_data

    # Test with default k
    score = ndcg_at_k(y_pred, y_true, n_items=3)
    assert 0 <= score <= 1

    # Test with specific k
    score = ndcg_at_k(y_pred, y_true, n_items=3, k=2)
    assert 0 <= score <= 1

    # Test perfect prediction
    perfect_pred = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.0])
    score = ndcg_at_k(perfect_pred, y_true, n_items=3)
    assert score > 0.99  # Allow for floating point imprecision


def test_binary_ndcg_at_k(sample_data):
    y_pred, y_true = sample_data
    binary_true = (y_true > 0.5).astype(float)

    # Test with default k
    score = binary_ndcg_at_k(y_pred, binary_true, n_items=3)
    assert 0 <= score <= 1

    # Test with specific k
    score = binary_ndcg_at_k(y_pred, binary_true, n_items=3, k=2)
    assert 0 <= score <= 1

    # Test perfect prediction
    perfect_pred = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    score = binary_ndcg_at_k(perfect_pred, binary_true, n_items=3)
    assert score > 0.99  # Allow for floating point imprecision


def test_map_at_k(sample_data):
    y_pred, y_true = sample_data
    binary_true = (y_true > 0).astype(float)

    # Test with default k
    score = map_at_k(y_pred, binary_true, n_items=3)
    assert 0 <= score <= 1

    # Test with specific k
    score = map_at_k(y_pred, binary_true, n_items=3, k=2)
    assert 0 <= score <= 1

    # Test perfect prediction
    perfect_pred = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.0])
    score = map_at_k(perfect_pred, binary_true, n_items=3)
    assert score > 0.99  # Allow for floating point imprecision


def test_hit_ratio_at_k(sample_data):
    y_pred, y_true = sample_data
    binary_true = (y_true > 0).astype(float)

    # Test with default k
    score = hit_ratio_at_k(y_pred, binary_true, n_items=3)
    assert 0 <= score <= 1

    # Test with specific k
    score = hit_ratio_at_k(y_pred, binary_true, n_items=3, k=2)
    assert 0 <= score <= 1

    # Test perfect prediction
    perfect_pred = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.0])
    score = hit_ratio_at_k(perfect_pred, binary_true, n_items=3)
    assert score == 1.0


def test_edge_cases():
    # Test single user case
    y_pred = np.array([0.9, 0.8, 0.7])
    y_true = np.array([1.0, 0.0, 0.0])

    assert 0 <= ndcg_at_k(y_pred, y_true, n_items=3) <= 1
    assert 0 <= binary_ndcg_at_k(y_pred, y_true, n_items=3) <= 1
    assert 0 <= map_at_k(y_pred, y_true, n_items=3) <= 1
    assert 0 <= hit_ratio_at_k(y_pred, y_true, n_items=3) <= 1

    # Test all zeros in ground truth
    y_true_zeros = np.zeros_like(y_true)

    assert ndcg_at_k(y_pred, y_true_zeros, n_items=3) == 0
    assert binary_ndcg_at_k(y_pred, y_true_zeros, n_items=3) == 0
    assert map_at_k(y_pred, y_true_zeros, n_items=3) == 0
    assert hit_ratio_at_k(y_pred, y_true_zeros, n_items=3) == 0


def test_shuffle_invariance(sample_data):
    y_pred, y_true = sample_data

    # Shuffled data (maintaining user groups)
    y_pred_shuffled = np.array(
        [0.9, 0.7, 0.8, 0.6, 0.4, 0.5]
    )  # First 3 and last 3 shuffled separately
    y_true_shuffled = np.array(
        [1.0, 0.5, 0.0, 0.0, 0.0, 1.0]
    )  # First 3 and last 3 shuffled separately

    # Test all metrics with both original and shuffled data
    metrics = [ndcg_at_k, binary_ndcg_at_k, map_at_k, hit_ratio_at_k]

    for metric in metrics:
        score_original = metric(y_pred, y_true, n_items=3)
        score_shuffled = metric(y_pred_shuffled, y_true_shuffled, n_items=3)
        np.testing.assert_almost_equal(score_original, score_shuffled)
