"""Unit tests for the ranking evaluation metrics.

These tests use small, hand-computed cases so a regression in any metric
is caught immediately. No data files are required.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoplay.ranking.evaluation import (
    _average_precision,
    _dcg_at_k,
    _ndcg_at_k,
    _precision_at_k,
    _reciprocal_rank,
    evaluate_predictions,
)


def test_dcg_at_k_hand_computed() -> None:
    # relevances [1, 0, 1], discounts log2(2)=1, log2(3), log2(4)=2
    # DCG = 1/1 + 0 + 1/2 = 1.5
    rels = np.array([1.0, 0.0, 1.0])
    assert _dcg_at_k(rels, 3) == pytest.approx(1.5)


def test_dcg_at_k_truncates_to_k() -> None:
    rels = np.array([1.0, 1.0, 1.0, 1.0])
    # Only the first item contributes at k=1.
    assert _dcg_at_k(rels, 1) == pytest.approx(1.0)


def test_dcg_at_k_empty_is_zero() -> None:
    assert _dcg_at_k(np.array([]), 5) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    rels = np.array([1.0, 1.0, 0.0, 0.0])
    assert _ndcg_at_k(rels, 10) == pytest.approx(1.0)


def test_ndcg_hand_computed() -> None:
    # predicted order [1, 0, 1]: DCG = 1 + 0 + 0.5 = 1.5
    # ideal order [1, 1, 0]: IDCG = 1 + 1/log2(3) = 1.63093
    rels = np.array([1.0, 0.0, 1.0])
    expected = 1.5 / (1.0 + 1.0 / np.log2(3))
    assert _ndcg_at_k(rels, 3) == pytest.approx(expected)


def test_ndcg_no_relevant_items_is_zero() -> None:
    assert _ndcg_at_k(np.array([0.0, 0.0, 0.0]), 3) == 0.0


def test_average_precision_hand_computed() -> None:
    # relevances [1, 0, 1, 0]: precisions at hits = [1/1, 2/3], AP = 0.83333
    rels = np.array([1.0, 0.0, 1.0, 0.0])
    assert _average_precision(rels) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_no_relevant_is_zero() -> None:
    assert _average_precision(np.array([0.0, 0.0])) == 0.0


def test_reciprocal_rank_first_relevant_at_position_two() -> None:
    assert _reciprocal_rank(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.5)


def test_reciprocal_rank_no_relevant_is_zero() -> None:
    assert _reciprocal_rank(np.array([0.0, 0.0])) == 0.0


def test_precision_at_k_hand_computed() -> None:
    rels = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    assert _precision_at_k(rels, 5) == pytest.approx(0.6)
    assert _precision_at_k(rels, 3) == pytest.approx(2.0 / 3.0)


def test_evaluate_predictions_perfect_single_query() -> None:
    labels = np.array([1, 1, 0, 0])
    predictions = np.array([4.0, 3.0, 2.0, 1.0])  # positives scored highest
    group_ids = np.array([0, 0, 0, 0])

    metrics = evaluate_predictions(labels, predictions, group_ids)

    assert metrics.ndcg_10 == pytest.approx(1.0)
    assert metrics.mean_reciprocal_rank == pytest.approx(1.0)
    assert metrics.mean_average_precision == pytest.approx(1.0)
    assert metrics.n_queries_evaluated == 1


def test_evaluate_predictions_averages_across_queries() -> None:
    # Query 0: perfect (ndcg 1.0). Query 1: positive ranked last.
    labels = np.array([1, 0, 1, 0])
    predictions = np.array([2.0, 1.0, 1.0, 2.0])
    group_ids = np.array([0, 0, 1, 1])

    metrics = evaluate_predictions(labels, predictions, group_ids)

    # Query 1 sorted by score desc -> relevances [0, 1], ndcg = 1/log2(3).
    expected_ndcg = (1.0 + 1.0 / np.log2(3)) / 2.0
    assert metrics.ndcg_10 == pytest.approx(expected_ndcg)
    assert metrics.n_queries_evaluated == 2
