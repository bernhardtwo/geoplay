"""Unit tests for the LightGBM group-size computation in model.py.

LightGBM's ranking objective needs the size of each query group, not the
per-row group id. A bug here would silently corrupt every ranking, so it
is worth pinning down.
"""

from __future__ import annotations

import numpy as np

from geoplay.ranking.model import _compute_group_sizes


def test_group_sizes_basic() -> None:
    group_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    result = _compute_group_sizes(group_ids)
    assert result.tolist() == [3, 2, 4]


def test_group_sizes_single_group() -> None:
    result = _compute_group_sizes(np.array([7, 7, 7]))
    assert result.tolist() == [3]


def test_group_sizes_singletons() -> None:
    result = _compute_group_sizes(np.array([0, 1, 2]))
    assert result.tolist() == [1, 1, 1]


def test_group_sizes_empty() -> None:
    result = _compute_group_sizes(np.array([], dtype=np.int64))
    assert result.tolist() == []


def test_group_sizes_sum_equals_total_rows() -> None:
    group_ids = np.array([0, 0, 1, 1, 1, 2])
    result = _compute_group_sizes(group_ids)
    assert int(result.sum()) == len(group_ids)
