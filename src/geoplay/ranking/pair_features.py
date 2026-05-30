"""Pair (player, hex) features for the ranking task.

Two features per (player, hex) pair:

    pair_past_visits           - number of times the player visited this
                                 hex in the train period (any period of day)
    pair_distance_from_home_km - haversine distance from the player's home
                                 to the centroid of the hex

These features capture the affinity between a specific player and a
specific hex, which is the most direct signal for personalized ranking.

A third feature was considered (pair_visited_in_train, a binary flag for
whether the player ever visited the hex in the train period) but was
dropped after empirical analysis: because negatives in this dataset are
sampled from the player's own visited-hex universe (rather than from
random hexes globally), every pair in the training set has
pair_visited_in_train=1. The feature is perfectly constant and contributes
no information to the model.

All features are computed over events strictly before cutoff_date.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from geoplay.data.geography import haversine_km
from geoplay.features.h3_utils import latlon_arrays_to_h3

DEFAULT_H3_RESOLUTION = 8


@dataclass
class PairFeaturesLookups:
    """Container for the lookup tables used to compute pair features.

    Attributes
    ----------
    player_hex_visits : dict[tuple[str, str], int]
        Map from (player_id, hex_id) to visit count in the train period.
    player_home : dict[str, tuple[float, float]]
        Map from player_id to (home_lat, home_lon).
    hex_centroid : dict[str, tuple[float, float]]
        Map from hex_id to (centroid_lat, centroid_lon).
    cutoff_date : pd.Timestamp
        Cutoff date used.
    n_unique_pairs : int
        Number of distinct (player, hex) pairs in the lookup.
    n_events_processed : int
        Total events processed.
    """

    player_hex_visits: dict[tuple[str, str], int]
    player_home: dict[str, tuple[float, float]]
    hex_centroid: dict[str, tuple[float, float]]
    cutoff_date: pd.Timestamp
    n_unique_pairs: int
    n_events_processed: int


def _process_partition_for_pair_lookups(
    events: pd.DataFrame,
    h3_resolution: int,
) -> pd.DataFrame:
    """Reduce one partition into per-(player, hex) visit counts.

    Parameters
    ----------
    events : pd.DataFrame
        Events with player_id, lat, lon columns.
    h3_resolution : int
        H3 resolution.

    Returns
    -------
    pd.DataFrame
        Columns player_id, hex_id, n_visits.
    """
    hex_ids = latlon_arrays_to_h3(
        events["lat"].to_numpy(dtype=np.float64),
        events["lon"].to_numpy(dtype=np.float64),
        resolution=h3_resolution,
    )
    events = events.assign(hex_id=hex_ids)
    counts = (
        events.groupby(["player_id", "hex_id"], observed=True).size().reset_index(name="n_visits")
    )
    return counts


def _compute_hex_centroids(hex_ids: set[str]) -> dict[str, tuple[float, float]]:
    """Compute (lat, lon) centroid for each H3 hex in the set.

    Uses h3.cell_to_latlng (v4 API).
    """
    centroids: dict[str, tuple[float, float]] = {}
    for hex_id in hex_ids:
        lat, lon = h3.cell_to_latlng(hex_id)
        centroids[hex_id] = (float(lat), float(lon))
    return centroids


def build_pair_features_lookups(
    events_dir: Path,
    players_path: Path,
    cutoff_date: pd.Timestamp,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    player_ids: list[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> PairFeaturesLookups:
    """Build lookup tables for computing pair (player, hex) features.

    Streams event partitions to compute per-(player, hex) visit counts.
    Also extracts player home locations and computes H3 hex centroids
    for distance calculations.

    Parameters
    ----------
    events_dir : Path
        Directory with event Parquet partitions.
    players_path : Path
        Path to players.parquet.
    cutoff_date : pd.Timestamp
        Events strictly before this date are counted.
    h3_resolution : int
        H3 resolution.
    player_ids : list[str] | None
        If given, only include events from these players.
    progress_callback : callable | None
        Optional progress callback.

    Returns
    -------
    PairFeaturesLookups
    """
    partitions = sorted(events_dir.glob("*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No Parquet partitions found in {events_dir}")

    total_parts = len(partitions)
    player_filter = set(player_ids) if player_ids is not None else None

    def log(idx: int, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(idx, total_parts, msg)

    # Accumulate per-(player, hex) visit counts.
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
        partition_counts = _process_partition_for_pair_lookups(events, h3_resolution)

        if accumulator is None:
            accumulator = partition_counts
        else:
            accumulator = pd.concat([accumulator, partition_counts], ignore_index=True)

        if idx % 5 == 0 or idx == total_parts:
            log(
                idx,
                f"  [{idx}/{total_parts}] processed "
                f"({total_events_processed:,} events total, "
                f"{(time.time()-t0)/60:.1f} min)",
            )

    if accumulator is None:
        raise RuntimeError("No events passed the filters; nothing to aggregate.")

    # Aggregate across partitions: sum visit counts for each (player, hex).
    log(total_parts, "Aggregating (player, hex) visit counts")
    final_counts = (
        accumulator.groupby(["player_id", "hex_id"], observed=True)["n_visits"].sum().reset_index()
    )

    # Build dict lookup.
    player_hex_visits: dict[tuple[str, str], int] = {
        (str(row.player_id), str(row.hex_id)): int(row.n_visits)
        for row in final_counts.itertuples(index=False)
    }

    # Player home locations.
    log(total_parts, "Loading player home locations")
    players_df = pd.read_parquet(players_path, columns=["player_id", "home_lat", "home_lon"])
    if player_filter is not None:
        players_df = players_df[players_df["player_id"].isin(player_filter)]
    player_home: dict[str, tuple[float, float]] = {
        str(row.player_id): (float(row.home_lat), float(row.home_lon))
        for row in players_df.itertuples(index=False)
    }

    # Hex centroids (for all hexes seen in the data).
    log(total_parts, "Computing hex centroids")
    unique_hex_ids = set(final_counts["hex_id"].unique())
    hex_centroid = _compute_hex_centroids(unique_hex_ids)

    return PairFeaturesLookups(
        player_hex_visits=player_hex_visits,
        player_home=player_home,
        hex_centroid=hex_centroid,
        cutoff_date=cutoff_date,
        n_unique_pairs=len(player_hex_visits),
        n_events_processed=total_events_processed,
    )


def add_pair_features(
    pairs: pd.DataFrame,
    lookups: PairFeaturesLookups,
) -> pd.DataFrame:
    """Add the 2 pair features to a dataframe of (player_id, hex_id) pairs.

    Parameters
    ----------
    pairs : pd.DataFrame
        Must contain at least player_id and hex_id columns.
    lookups : PairFeaturesLookups
        Lookup tables built by build_pair_features_lookups.

    Returns
    -------
    pd.DataFrame
        Input dataframe with two new columns added:
        pair_past_visits, pair_distance_from_home_km.
    """
    visits_lookup = lookups.player_hex_visits
    home_lookup = lookups.player_home
    centroid_lookup = lookups.hex_centroid

    n = len(pairs)
    past_visits = np.zeros(n, dtype=np.int64)
    home_lats = np.zeros(n, dtype=np.float64)
    home_lons = np.zeros(n, dtype=np.float64)
    hex_lats = np.zeros(n, dtype=np.float64)
    hex_lons = np.zeros(n, dtype=np.float64)

    player_ids = pairs["player_id"].to_numpy()
    hex_ids = pairs["hex_id"].to_numpy()

    for i in range(n):
        pid = str(player_ids[i])
        hid = str(hex_ids[i])
        past_visits[i] = visits_lookup.get((pid, hid), 0)
        home = home_lookup.get(pid, (0.0, 0.0))
        home_lats[i], home_lons[i] = home
        centroid = centroid_lookup.get(hid, (0.0, 0.0))
        hex_lats[i], hex_lons[i] = centroid

    distances = haversine_km(home_lats, home_lons, hex_lats, hex_lons)

    return pairs.assign(
        pair_past_visits=past_visits,
        pair_distance_from_home_km=distances,
    )
