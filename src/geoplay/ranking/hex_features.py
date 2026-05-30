"""Hex features for the ranking task.

Computes 3 features per H3 hex, characterizing global popularity, traffic
volume, and the archetype diversity of players who visit it. These are
item-side features in the learning-to-rank formulation: they describe each
hex independently of any particular query.

Features:
    hex_global_popularity      - number of distinct players who visited this hex
    hex_total_visits           - total event count in this hex
    hex_n_unique_archetypes    - count of distinct archetypes that visited

A "dominant archetype" feature was considered and discarded: in this dataset,
the archetypes with the largest movement radii (hardcore_raider, weekend_explorer)
visit so many hexes that they dominate the dominance metric in nearly every
hex, even after normalization by archetype population. The signal was not
discriminative and the feature was removed.

All features are computed over events strictly before cutoff_date to avoid
temporal leakage into the test period.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoplay.features.h3_utils import latlon_arrays_to_h3

DEFAULT_H3_RESOLUTION = 8


@dataclass
class HexFeaturesResult:
    """Container for the hex features matrix.

    Attributes
    ----------
    features : pd.DataFrame
        One row per hex, columns: hex_id and the 4 hex features.
    cutoff_date : pd.Timestamp
        Cutoff date used to filter events.
    n_hexes : int
        Number of unique hexes with at least one event.
    n_events_processed : int
        Total events that passed the cutoff filter.
    """

    features: pd.DataFrame
    cutoff_date: pd.Timestamp
    n_hexes: int
    n_events_processed: int


def _process_partition_for_hex_aggregates(
    events: pd.DataFrame,
    players_archetype: pd.Series,
    h3_resolution: int,
) -> pd.DataFrame:
    """Reduce one partition into per-hex partial aggregates.

    Each partition contributes per-hex counters that are summed across
    partitions in the orchestrator.

    Parameters
    ----------
    events : pd.DataFrame
        Events with player_id, lat, lon columns.
    players_archetype : pd.Series
        Map from player_id (index) to archetype (value).
    h3_resolution : int
        H3 resolution.

    Returns
    -------
    pd.DataFrame
        Long-format counts: one row per (hex_id, archetype) pair with two
        columns:
          n_events : event count in that hex for that archetype
          n_players : unique-player count in that hex for that archetype
    """
    hex_ids = latlon_arrays_to_h3(
        events["lat"].to_numpy(dtype=np.float64),
        events["lon"].to_numpy(dtype=np.float64),
        resolution=h3_resolution,
    )
    events = events.assign(hex_id=hex_ids)
    events = events.merge(
        players_archetype.rename("archetype"),
        left_on="player_id",
        right_index=True,
        how="left",
    )

    # Long-format: (hex_id, archetype) -> n_events, n_players
    counts = (
        events.groupby(["hex_id", "archetype"], observed=True)
        .agg(
            n_events=("player_id", "size"),
            n_players=("player_id", "nunique"),
        )
        .reset_index()
    )
    return counts


def _merge_hex_partials(
    accumulator: pd.DataFrame | None,
    partition_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate partition counts into the accumulator.

    We keep the long format throughout streaming and only collapse in the
    finalize step. This avoids holding multiple per-hex sets in memory.
    """
    if accumulator is None:
        return partition_counts
    return pd.concat([accumulator, partition_counts], ignore_index=True)


def _finalize_hex_features(long_counts: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format (hex, archetype) counts into per-hex features.

    Parameters
    ----------
    long_counts : pd.DataFrame
        Columns hex_id, archetype, n_events, n_players from all partitions
        concatenated.

    Returns
    -------
    pd.DataFrame
        Per-hex feature rows with columns:
        hex_id, hex_global_popularity, hex_total_visits,
        hex_n_unique_archetypes.
    """
    # First, aggregate by (hex_id, archetype) across partitions.
    by_hex_archetype = (
        long_counts.groupby(["hex_id", "archetype"], observed=True)
        .agg(
            n_events=("n_events", "sum"),
            n_players=("n_players", "sum"),
        )
        .reset_index()
    )

    # Per-hex aggregates (no dominant archetype: see module docstring).
    by_hex = (
        by_hex_archetype.groupby("hex_id", observed=True)
        .agg(
            hex_global_popularity=("n_players", "sum"),
            hex_total_visits=("n_events", "sum"),
            hex_n_unique_archetypes=("archetype", "nunique"),
        )
        .reset_index()
    )

    return by_hex


def compute_hex_features(
    events_dir: Path,
    players_path: Path,
    cutoff_date: pd.Timestamp,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    player_ids: list[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> HexFeaturesResult:
    """Compute per-hex features over events strictly before cutoff_date.

    Streams the event Parquet partitions to keep memory bounded.

    Note: hex_global_popularity is computed as the SUM of n_players across
    partitions. Since partitions are non-overlapping (each partition contains
    a distinct set of players), this is exactly the number of unique players
    who visited the hex without double-counting.

    Parameters
    ----------
    events_dir : Path
        Directory containing event Parquet partitions.
    players_path : Path
        Path to players.parquet (for archetype lookup).
    cutoff_date : pd.Timestamp
        Events strictly before this date are included.
    h3_resolution : int
        H3 resolution.
    player_ids : list[str] | None
        If given, only include events from these players.
    progress_callback : callable | None
        Optional callback(partition_idx, total, msg) for progress.

    Returns
    -------
    HexFeaturesResult
    """
    partitions = sorted(events_dir.glob("*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No Parquet partitions found in {events_dir}")

    players_df = pd.read_parquet(players_path, columns=["player_id", "archetype"])
    players_archetype = players_df.set_index("player_id")["archetype"]

    total_parts = len(partitions)
    player_filter = set(player_ids) if player_ids is not None else None

    def log(idx: int, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(idx, total_parts, msg)

    accumulator: pd.DataFrame | None = None
    total_events_processed = 0

    t0 = time.time()
    for idx, partition_path in enumerate(partitions, start=1):
        events = pd.read_parquet(
            partition_path,
            columns=["player_id", "timestamp", "lat", "lon"],
        )

        events = events[events["timestamp"] < cutoff_date]

        if player_filter is not None:
            events = events[events["player_id"].isin(player_filter)]

        if len(events) == 0:
            continue

        total_events_processed += len(events)
        partition_counts = _process_partition_for_hex_aggregates(
            events, players_archetype, h3_resolution
        )
        accumulator = _merge_hex_partials(accumulator, partition_counts)

        if idx % 5 == 0 or idx == total_parts:
            log(
                idx,
                f"  [{idx}/{total_parts}] processed "
                f"({total_events_processed:,} events total, "
                f"{(time.time()-t0)/60:.1f} min)",
            )

    if accumulator is None:
        raise RuntimeError("No events passed the filters; nothing to aggregate.")

    features = _finalize_hex_features(accumulator)

    return HexFeaturesResult(
        features=features,
        cutoff_date=cutoff_date,
        n_hexes=len(features),
        n_events_processed=total_events_processed,
    )
