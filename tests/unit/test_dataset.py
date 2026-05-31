"""Unit tests for the period-of-day bucketing in dataset.py."""

from __future__ import annotations

import pytest

from geoplay.ranking.dataset import (
    PERIOD_AFTERNOON,
    PERIOD_EVENING,
    PERIOD_MORNING,
    PERIOD_NIGHT,
    hour_to_period,
)


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (0, PERIOD_NIGHT),
        (5, PERIOD_NIGHT),
        (6, PERIOD_MORNING),
        (11, PERIOD_MORNING),
        (12, PERIOD_AFTERNOON),
        (17, PERIOD_AFTERNOON),
        (18, PERIOD_EVENING),
        (23, PERIOD_EVENING),
    ],
)
def test_hour_to_period_boundaries(hour: int, expected: str) -> None:
    assert hour_to_period(hour) == expected


@pytest.mark.parametrize("bad_hour", [-1, 24, 100])
def test_hour_to_period_rejects_invalid(bad_hour: int) -> None:
    with pytest.raises(ValueError, match="Invalid hour"):
        hour_to_period(bad_hour)
