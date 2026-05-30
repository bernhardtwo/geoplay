"""Training dataset construction for learning-to-rank.

The dataset is built from raw event partitions and follows the standard
learning-to-rank format:

- A "query" is a (player_id, day_of_week, hour) tuple. The ranker's job
  is to rank H3 hexes for that player in that time window.
- For each query, we extract the hexes the player actually visited
  (positives, label=1) and sample hexes the player did NOT visit
  (negatives, label=0).
- The negative-to-positive ratio is configurable (default 5).
- Queries are only generated where the player has real activity: empty
  windows are skipped entirely. This produces a dense, signal-rich
  dataset without wasted rows.

Train/test split is temporal: events before the split date go to train,
events after go to test. This is the industry standard for recommender
systems to avoid data leakage from future to past.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoplay.features.h3_utils import latlon_arrays_to_h3

# Default H3 resolution: same as used in feature engineering.
DEFAULT_H3_RESOLUTION = 8

# Number of negative samples per positive sample.
DEFAULT_NEG_RATIO = 5

# Maximum number of positive samples kept per query. Queries with more
# positives have them sub-sampled, prioritizing the most-visited hexes.
# This caps the dataset size and prevents very active players from
# dominating training.
DEFAULT_MAX_POSITIVES_PER_QUERY = 10

# Random seed for reproducibility of negative sampling.
DEFAULT_SEED = 42

# Period-of-day buckets. The query granularity is (player, day_of_week,
# period) instead of (player, day_of_week, hour) to keep the dataset
# manageable while preserving weekday-vs-weekend distinction and rough
# time-of-day patterns.
PERIOD_NIGHT = "night"  # 00h-05h
PERIOD_MORNING = "morning"  # 06h-11h
PERIOD_AFTERNOON = "afternoon"  # 12h-17h
PERIOD_EVENING = "evening"  # 18h-23h

PERIOD_BOUNDARIES = (
    (0, 6, PERIOD_NIGHT),
    (6, 12, PERIOD_MORNING),
    (12, 18, PERIOD_AFTERNOON),
    (18, 24, PERIOD_EVENING),
)


def hour_to_period(hour: int) -> str:
    """Map a 0-23 hour to one of the four period-of-day buckets."""
    for start, end, name in PERIOD_BOUNDARIES:
        if start <= hour < end:
            return name
    raise ValueError(f"Invalid hour: {hour} (expected 0-23)")


@dataclass
class RankingDataset:
    """Container for the learning-to-rank training/test dataset.

    Attributes
    ----------
    pairs : pd.DataFrame
        One row per (query, hex) pair. Columns:
        player_id, day_of_week, hour, hex_id, label, group_id
    n_queries : int
        Number of unique queries (groups).
    n_positives : int
        Number of positive samples (label=1).
    n_negatives : int
        Number of negative samples (label=0).
    """

    pairs: pd.DataFrame
    n_queries: int
    n_positives: int
    n_negatives: int

    @property
    def n_rows(self) -> int:
        return len(self.pairs)

    def summary(self) -> dict[str, int | float]:
        return {
            "n_rows": self.n_rows,
            "n_queries": self.n_queries,
            "n_positives": self.n_positives,
            "n_negatives": self.n_negatives,
            "neg_pos_ratio": round(self.n_negatives / max(self.n_positives, 1), 2),
            "avg_pairs_per_query": round(self.n_rows / max(self.n_queries, 1), 2),
        }


def _split_events_temporal(
    events: pd.DataFrame,
    split_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split events by timestamp: before/after split_date.

    Parameters
    ----------
    events : pd.DataFrame
        Events with a 'timestamp' column.
    split_date : pd.Timestamp
        Events strictly before this date go to train; events on or after
        go to test.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_events, test_events)
    """
    train_mask = events["timestamp"] < split_date
    return events[train_mask].copy(), events[~train_mask].copy()


def _build_visited_hexes(
    events: pd.DataFrame,
    h3_resolution: int,
) -> pd.DataFrame:
    """Aggregate events into (player, day, period, hex) visit records.

    Buckets the hour-of-day into four periods (night/morning/afternoon/evening)
    to keep query count manageable. For each event, computes the H3 hex at
    the given resolution, then groups by (player_id, day_of_week, period, hex_id).

    Parameters
    ----------
    events : pd.DataFrame
        Events with lat, lon, timestamp, player_id, hour, day_of_week columns.
    h3_resolution : int
        H3 resolution for hex aggregation.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, day_of_week, period, hex_id, n_visits
    """
    # Compute H3 hex for each event (vectorized array conversion).
    hex_ids = latlon_arrays_to_h3(
        events["lat"].to_numpy(dtype=np.float64),
        events["lon"].to_numpy(dtype=np.float64),
        resolution=h3_resolution,
    )

    # Map each event's hour to one of the four periods.
    periods = events["hour"].map(hour_to_period).to_numpy()

    events = events.assign(hex_id=hex_ids, period=periods)

    visited = (
        events.groupby(["player_id", "day_of_week", "period", "hex_id"], observed=True)
        .size()
        .reset_index(name="n_visits")
    )
    return visited


def _build_player_hex_universe(
    visited: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Build the universe of hexes that each player could plausibly visit.

    A negative for a player is sampled from this universe (hexes the player
    visited at SOME point), excluding the hex already labeled positive in
    the same query. This is more realistic than sampling from random hexes
    worldwide: we sample hard negatives from the player's actual movement area.

    Parameters
    ----------
    visited : pd.DataFrame
        Output of _build_visited_hexes.

    Returns
    -------
    dict[str, np.ndarray]
        Map from player_id to array of all hex_ids the player has ever visited.
    """
    universe: dict[str, np.ndarray] = {}
    for player_id, group in visited.groupby("player_id", observed=True):
        universe[str(player_id)] = group["hex_id"].unique()
    return universe


def _sample_negatives_for_query(
    player_id: str,
    positive_hexes: set[str],
    player_universe: np.ndarray,
    neg_ratio: int,
    rng: np.random.Generator,
) -> list[str]:
    """Sample negative hexes for one query.

    Negatives are drawn from the player's universe (hexes they have visited
    at some point) but exclude the positive hexes for this specific query.
    The number of negatives is neg_ratio * len(positives), capped by what is
    available.

    Parameters
    ----------
    player_id : str
        Player identifier (used for error context only).
    positive_hexes : set[str]
        Hexes the player visited in this query window.
    player_universe : np.ndarray
        All hexes the player has ever visited.
    neg_ratio : int
        Target ratio of negatives to positives.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    list[str]
        Sampled negative hex IDs.
    """
    candidates = np.array([h for h in player_universe if h not in positive_hexes])
    n_desired = neg_ratio * len(positive_hexes)
    n_actual = min(n_desired, len(candidates))
    if n_actual == 0:
        return []
    sampled_idx = rng.choice(len(candidates), size=n_actual, replace=False)
    return candidates[sampled_idx].tolist()


def build_ranking_dataset(
    visited: pd.DataFrame,
    neg_ratio: int = DEFAULT_NEG_RATIO,
    max_positives_per_query: int = DEFAULT_MAX_POSITIVES_PER_QUERY,
    seed: int = DEFAULT_SEED,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RankingDataset:
    """Build a learning-to-rank dataset from per-window hex visits.

    For each query (player_id, day_of_week, period) with at least one visit:
      - Take up to max_positives_per_query positive hexes (label=1),
        prioritizing hexes with the most visits.
      - Sample neg_ratio * n_positives negatives (label=0) from the
        player's hex universe (excluding the positives for this query).

    Queries with no visits are skipped (dense dataset, no empty windows).

    Parameters
    ----------
    visited : pd.DataFrame
        Output of _build_visited_hexes: per-window visited hexes.
    neg_ratio : int
        Negatives per positive (default 5).
    max_positives_per_query : int
        Cap on positives per query. Reduces dataset size and prevents
        hyperactive players from dominating training (default 10).
    seed : int
        Random seed.
    progress_callback : callable | None
        Optional callback(processed_queries, total_queries) for progress.

    Returns
    -------
    RankingDataset
    """
    rng = np.random.default_rng(seed)
    universe = _build_player_hex_universe(visited)

    # Group by (player_id, day_of_week, period) to iterate per query.
    query_groups = visited.groupby(["player_id", "day_of_week", "period"], observed=True)
    total_queries = len(query_groups)

    rows: list[dict[str, str | int]] = []
    query_id = 0

    for processed, ((player_id, day, period), group) in enumerate(query_groups, start=1):
        # Cap positives: keep at most max_positives_per_query, sorted by
        # n_visits descending so the most-visited hexes are prioritized.
        if len(group) > max_positives_per_query:
            group = group.nlargest(max_positives_per_query, "n_visits")
        positive_hexes = set(group["hex_id"].tolist())
        player_universe = universe[str(player_id)]

        negatives = _sample_negatives_for_query(
            player_id=str(player_id),
            positive_hexes=positive_hexes,
            player_universe=player_universe,
            neg_ratio=neg_ratio,
            rng=rng,
        )

        # Skip queries where we cannot generate any negatives (player has
        # only ever visited the positives for this query — rare edge case).
        if not negatives:
            continue

        for hex_id in positive_hexes:
            rows.append(
                {
                    "player_id": str(player_id),
                    "day_of_week": int(day),
                    "period": str(period),
                    "hex_id": str(hex_id),
                    "label": 1,
                    "group_id": query_id,
                }
            )
        for hex_id in negatives:
            rows.append(
                {
                    "player_id": str(player_id),
                    "day_of_week": int(day),
                    "period": str(period),
                    "hex_id": str(hex_id),
                    "label": 0,
                    "group_id": query_id,
                }
            )
        query_id += 1

        if progress_callback is not None and processed % 10000 == 0:
            progress_callback(processed, total_queries)

    pairs = pd.DataFrame(rows)
    n_positives = int((pairs["label"] == 1).sum())
    n_negatives = int((pairs["label"] == 0).sum())

    return RankingDataset(
        pairs=pairs,
        n_queries=query_id,
        n_positives=n_positives,
        n_negatives=n_negatives,
    )


def build_train_test_datasets(
    events_dir: Path,
    split_date: pd.Timestamp,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    neg_ratio: int = DEFAULT_NEG_RATIO,
    max_positives_per_query: int = DEFAULT_MAX_POSITIVES_PER_QUERY,
    player_ids: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[RankingDataset, RankingDataset]:
    """Build train and test ranking datasets from raw event partitions.

    Streams partitions to avoid loading all 146M events into memory at once.
    Each partition is split temporally before aggregation, then visited hexes
    are accumulated.

    Parameters
    ----------
    events_dir : Path
        Directory with event Parquet partitions.
    split_date : pd.Timestamp
        Events strictly before this date go to train; events on or after
        go to test.
    h3_resolution : int
        H3 resolution for hex aggregation (default 8, ~0.74 km² per cell).
    neg_ratio : int
        Negatives per positive (default 5).
    seed : int
        Random seed.
    progress_callback : callable | None
        Optional callback(message) for progress.

    Returns
    -------
    tuple[RankingDataset, RankingDataset]
        (train_dataset, test_dataset)
    """

    def log(msg: str) -> None:
        if progress_callback is not None:
            progress_callback(msg)

    partitions = sorted(events_dir.glob("*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No Parquet partitions found in {events_dir}")

    log(f"Streaming {len(partitions)} event partitions, splitting at {split_date}")

    # Accumulate per-window visits separately for train and test.
    train_visits: list[pd.DataFrame] = []
    test_visits: list[pd.DataFrame] = []

    player_filter = set(player_ids) if player_ids is not None else None

    t0 = time.time()
    for idx, partition_path in enumerate(partitions, start=1):
        events = pd.read_parquet(
            partition_path,
            columns=["player_id", "timestamp", "lat", "lon", "hour", "day_of_week"],
        )
        if player_filter is not None:
            events = events[events["player_id"].isin(player_filter)]
            if len(events) == 0:
                continue
        train_events, test_events = _split_events_temporal(events, split_date)

        if len(train_events) > 0:
            train_visits.append(_build_visited_hexes(train_events, h3_resolution))
        if len(test_events) > 0:
            test_visits.append(_build_visited_hexes(test_events, h3_resolution))

        if idx % 5 == 0 or idx == len(partitions):
            log(f"  [{idx}/{len(partitions)}] partitions processed ({(time.time()-t0)/60:.1f} min)")

    log("Aggregating visits across partitions")
    train_visited_full = (
        pd.concat(train_visits, ignore_index=True)
        .groupby(["player_id", "day_of_week", "period", "hex_id"], observed=True)["n_visits"]
        .sum()
        .reset_index()
    )
    test_visited_full = (
        pd.concat(test_visits, ignore_index=True)
        .groupby(["player_id", "day_of_week", "period", "hex_id"], observed=True)["n_visits"]
        .sum()
        .reset_index()
    )

    log(f"  Train visits: {len(train_visited_full):,} unique (player, day, hour, hex)")
    log(f"  Test visits: {len(test_visited_full):,} unique")

    log("Building train ranking dataset")
    train_dataset = build_ranking_dataset(
        train_visited_full,
        neg_ratio=neg_ratio,
        max_positives_per_query=max_positives_per_query,
        seed=seed,
    )
    log(f"  Train: {train_dataset.summary()}")

    log("Building test ranking dataset")
    test_dataset = build_ranking_dataset(
        test_visited_full,
        neg_ratio=neg_ratio,
        max_positives_per_query=max_positives_per_query,
        seed=seed,
    )
    log(f"  Test: {test_dataset.summary()}")

    return train_dataset, test_dataset
