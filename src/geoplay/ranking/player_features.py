"""Player features for the ranking task.

Computes 11 curated features per player over a configurable cutoff date,
to avoid temporal leakage: when training a ranker, the model must only see
player history up to the cutoff (i.e., before the test period begins).

Feature design:

- Behavior aggregates (5 features): total events log, unique sessions,
  events per session mean, session duration mean, active days ratio.
- Temporal patterns (3 features): weekday ratio, evening ratio,
  hour concentration (entropy-based).
- Spatial patterns (2 features): mean distance from home, unique hexes log.
- Categorical (1 feature): archetype.

All numeric features are derived from raw events filtered by cutoff_date,
guaranteeing no leakage from future periods into the training signal.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoplay.features.h3_utils import latlon_arrays_to_h3

# Default H3 resolution used elsewhere in the project.
DEFAULT_H3_RESOLUTION = 8

# Hours that count as "evening" for evening_ratio.
EVENING_HOURS_START = 18
EVENING_HOURS_END = 24  # exclusive

# Days that count as weekdays (Monday=0 to Friday=4).
WEEKDAY_DAYS = frozenset({0, 1, 2, 3, 4})


DEFAULT_START_DATE = pd.Timestamp("2025-01-01")


@dataclass
class PlayerFeaturesResult:
    """Container for the player features matrix.

    Attributes
    ----------
    features : pd.DataFrame
        One row per player, columns are the 11 features plus player_id.
    cutoff_date : pd.Timestamp
        Date used to filter events (events strictly before are included).
    n_players : int
        Number of players with at least one event before the cutoff.
    n_events_processed : int
        Total events that passed the cutoff filter.
    """

    features: pd.DataFrame
    cutoff_date: pd.Timestamp
    n_players: int
    n_events_processed: int


def _hour_concentration(hour_counts: np.ndarray) -> float:
    """Normalized inverse entropy of the hour distribution.

    Returns a value in [0, 1] where 1 means all events at one hour (very
    concentrated) and 0 means uniform across 24 hours. Defined as
    1 - H/log(24), where H is Shannon entropy of the hour distribution.

    Parameters
    ----------
    hour_counts : np.ndarray
        Array of length 24 with event counts per hour.

    Returns
    -------
    float
        Concentration in [0, 1].
    """
    total = hour_counts.sum()
    if total == 0:
        return 0.0
    probs = hour_counts / total
    nonzero = probs[probs > 0]
    entropy = -(nonzero * np.log(nonzero)).sum()
    max_entropy = np.log(24)
    return float(1.0 - entropy / max_entropy)


def _process_partition_for_player_aggregates(
    events: pd.DataFrame,
    h3_resolution: int,
) -> dict[str, dict[str, float | int | set[str]]]:
    """Reduce a partition into per-player partial aggregates.

    Streams one partition and produces a dict keyed by player_id with
    intermediate counters. These counters are summed across partitions in
    the orchestrator.

    Parameters
    ----------
    events : pd.DataFrame
        Events with player_id, timestamp, lat, lon, hour, day_of_week,
        session_id columns.
    h3_resolution : int
        H3 resolution for the unique hex set.

    Returns
    -------
    dict
        Per-player partial aggregates with the keys:
        n_events, n_evening_events, n_weekday_events,
        sum_distance_from_home, n_with_distance,
        hour_counts (np.ndarray of length 24),
        session_ids (set), hex_ids (set),
        active_dates (set), session_durations (list).
    """
    # Compute H3 hex for each event.
    hex_ids = latlon_arrays_to_h3(
        events["lat"].to_numpy(dtype=np.float64),
        events["lon"].to_numpy(dtype=np.float64),
        resolution=h3_resolution,
    )
    events = events.assign(hex_id=hex_ids)

    aggregates: dict[str, dict[str, float | int | set[str]]] = {}

    for player_id, group in events.groupby("player_id", observed=True):
        agg: dict[str, float | int | set[str]] = aggregates.setdefault(
            str(player_id),
            {
                "n_events": 0,
                "n_evening_events": 0,
                "n_weekday_events": 0,
                "sum_distance_from_home": 0.0,
                "n_with_distance": 0,
                "hour_counts": np.zeros(24, dtype=np.int64),
                "session_ids": set(),
                "hex_ids": set(),
                "active_dates": set(),
            },
        )

        agg["n_events"] += len(group)
        hours = group["hour"].to_numpy()
        is_evening = (hours >= EVENING_HOURS_START) & (hours < EVENING_HOURS_END)
        agg["n_evening_events"] += int(is_evening.sum())

        dows = group["day_of_week"].to_numpy()
        is_weekday = np.isin(dows, list(WEEKDAY_DAYS))
        agg["n_weekday_events"] += int(is_weekday.sum())

        if "distance_from_home_km" in group.columns:
            dists = group["distance_from_home_km"].to_numpy()
            agg["sum_distance_from_home"] += float(dists.sum())
            agg["n_with_distance"] += len(dists)

        # Hour distribution counter (length 24).
        hour_counts = np.bincount(hours, minlength=24)
        agg["hour_counts"] = agg["hour_counts"] + hour_counts

        agg["session_ids"].update(group["session_id"].tolist())
        agg["hex_ids"].update(group["hex_id"].tolist())
        agg["active_dates"].update(group["timestamp"].dt.date.unique())

    return aggregates


def _merge_partials(
    accumulator: dict[str, dict],
    partition_aggregates: dict[str, dict],
) -> None:
    """Merge a partition's partial aggregates into the running accumulator.

    Modifies the accumulator in place.
    """
    for player_id, agg in partition_aggregates.items():
        if player_id not in accumulator:
            accumulator[player_id] = agg
            continue

        target = accumulator[player_id]
        target["n_events"] += agg["n_events"]
        target["n_evening_events"] += agg["n_evening_events"]
        target["n_weekday_events"] += agg["n_weekday_events"]
        target["sum_distance_from_home"] += agg["sum_distance_from_home"]
        target["n_with_distance"] += agg["n_with_distance"]
        target["hour_counts"] = target["hour_counts"] + agg["hour_counts"]
        target["session_ids"].update(agg["session_ids"])
        target["hex_ids"].update(agg["hex_ids"])
        target["active_dates"].update(agg["active_dates"])


def _finalize_features(
    accumulator: dict[str, dict],
    players_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Convert per-player aggregates into the final feature matrix.

    Parameters
    ----------
    accumulator : dict
        Per-player accumulated aggregates from streaming.
    players_df : pd.DataFrame
        Player table for archetype lookup.
    cutoff_date : pd.Timestamp
        Cutoff date for the train period (used to compute active_days_ratio).
    start_date : pd.Timestamp
        Start date of the simulation.

    Returns
    -------
    pd.DataFrame
        One row per player with player_id and 11 feature columns.
    """
    # Total observation window in days (start_date to cutoff_date).
    observation_days = (cutoff_date - start_date).days

    rows: list[dict[str, str | float | int]] = []
    for player_id, agg in accumulator.items():
        n_events = int(agg["n_events"])
        if n_events == 0:
            continue

        n_sessions = len(agg["session_ids"])
        n_active_days = len(agg["active_dates"])
        avg_dist = (
            agg["sum_distance_from_home"] / agg["n_with_distance"]
            if agg["n_with_distance"] > 0
            else 0.0
        )

        rows.append(
            {
                "player_id": player_id,
                "player_total_events_log": float(np.log1p(n_events)),
                "player_unique_sessions": n_sessions,
                "player_events_per_session_mean": (
                    n_events / n_sessions if n_sessions > 0 else 0.0
                ),
                "player_active_days_ratio": (
                    n_active_days / observation_days if observation_days > 0 else 0.0
                ),
                "player_weekday_ratio": agg["n_weekday_events"] / n_events,
                "player_evening_ratio": agg["n_evening_events"] / n_events,
                "player_hour_concentration": _hour_concentration(agg["hour_counts"]),
                "player_distance_from_home_mean_km": avg_dist,
                "player_unique_hexes_count_log": float(np.log1p(len(agg["hex_ids"]))),
            }
        )

    features = pd.DataFrame(rows)

    # Join with archetype from the players table.
    features = features.merge(
        players_df[["player_id", "archetype"]],
        on="player_id",
        how="left",
    ).rename(columns={"archetype": "player_archetype"})

    return features


def compute_player_features(
    events_dir: Path,
    players_path: Path,
    cutoff_date: pd.Timestamp,
    start_date: pd.Timestamp = DEFAULT_START_DATE,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    player_ids: list[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> PlayerFeaturesResult:
    """Compute per-player features over events strictly before cutoff_date.

    Streams the event Parquet partitions to avoid loading all events at once.

    Note: this function intentionally re-computes player features rather than
    reusing data/processed/features.parquet, because that file aggregates over
    the full 180-day window and would leak future information into a ranker
    trained on the first 150 days.

    Parameters
    ----------
    events_dir : Path
        Directory containing event Parquet partitions.
    players_path : Path
        Path to players.parquet (for archetype lookup).
    cutoff_date : pd.Timestamp
        Events strictly before this date are included; events on or after
        are excluded (no leakage into train).
    start_date : pd.Timestamp
        Start date of the simulation (default 2025-01-01).
    h3_resolution : int
        H3 resolution for unique hex counting (default 8).
    player_ids : list[str] | None
        If given, only compute features for these players.
    progress_callback : callable | None
        Optional callback(partition_idx, total, msg) for progress.

    Returns
    -------
    PlayerFeaturesResult
    """
    partitions = sorted(events_dir.glob("*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No Parquet partitions found in {events_dir}")

    total_parts = len(partitions)
    player_filter = set(player_ids) if player_ids is not None else None

    def log(idx: int, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(idx, total_parts, msg)

    accumulator: dict[str, dict] = {}
    total_events_processed = 0

    t0 = time.time()
    for idx, partition_path in enumerate(partitions, start=1):
        events = pd.read_parquet(
            partition_path,
            columns=[
                "player_id",
                "timestamp",
                "session_id",
                "lat",
                "lon",
                "hour",
                "day_of_week",
                "distance_from_home_km",
            ],
        )

        # Temporal filter: strict less-than to match dataset.py convention.
        events = events[events["timestamp"] < cutoff_date]

        if player_filter is not None:
            events = events[events["player_id"].isin(player_filter)]

        if len(events) == 0:
            continue

        total_events_processed += len(events)
        partition_agg = _process_partition_for_player_aggregates(events, h3_resolution)
        _merge_partials(accumulator, partition_agg)

        if idx % 5 == 0 or idx == total_parts:
            log(
                idx,
                f"  [{idx}/{total_parts}] processed "
                f"({total_events_processed:,} events total, "
                f"{(time.time()-t0)/60:.1f} min)",
            )

    players_df = pd.read_parquet(players_path, columns=["player_id", "archetype"])
    features = _finalize_features(accumulator, players_df, cutoff_date, start_date)

    return PlayerFeaturesResult(
        features=features,
        cutoff_date=cutoff_date,
        n_players=len(features),
        n_events_processed=total_events_processed,
    )
