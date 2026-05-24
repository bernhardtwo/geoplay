"""Feature engineering pipeline orchestration with partition streaming.

Coordinates temporal, spatial, and behavioral feature builders into a single
per-player feature matrix. Processes events partition-by-partition to keep
memory footprint bounded (~2 GB constant regardless of dataset size).

The streaming approach accumulates partial aggregates per player across all
partitions, then finalizes the computations once all events have been seen.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from geoplay.features.h3_utils import (
    DEFAULT_H3_RESOLUTION,
    latlon_arrays_to_h3,
    shannon_entropy,
)

# Columns needed across all feature families (union).
ALL_COLUMNS = [
    "player_id",
    "timestamp",
    "session_id",
    "hour",
    "day_of_week",
    "lat",
    "lon",
    "distance_from_home_km",
]


# Constants used for cyclical encoding and ratio calculations.
HOURS_IN_DAY = 24
DAYS_IN_WEEK = 7
WEEKDAY_DAYS = frozenset({0, 1, 2, 3, 4})  # Monday=0 to Friday=4
EVENING_HOURS_START = 18  # events at hour >= 18 count as evening


def _iter_partitions(events_dir: Path) -> Iterable[Path]:
    """Yield Parquet partition paths in deterministic order."""
    yield from sorted(events_dir.glob("part_*.parquet"))


def build_features_from_partitions(
    events_dir: Path,
    players_path: Path,
    output_path: Path,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    observation_days: int = 180,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Build the full per-player feature matrix via partition streaming.

    Strategy: iterate partitions, accumulate per-player partial aggregates,
    then finalize at the end. Memory footprint stays around 2-3 GB
    regardless of total dataset size.

    Parameters
    ----------
    events_dir : Path
        Directory containing event Parquet partitions.
    players_path : Path
        Path to players.parquet for archetype ground truth.
    output_path : Path
        Where to write the resulting feature matrix.
    h3_resolution : int
        H3 resolution for spatial features.
    observation_days : int
        Total observation window in days (for active_days_ratio).
    progress_callback : callable | None
        Optional callback (partition_idx, total_partitions, msg) -> None.

    Returns
    -------
    pd.DataFrame
        Feature matrix with player_id as index and 54 feature columns plus
        archetype as the last column.
    """
    partitions = list(_iter_partitions(events_dir))
    if not partitions:
        raise FileNotFoundError(f"No Parquet partitions found in {events_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_parts = len(partitions)

    # Accumulators (player_id -> partial aggregates).
    # These dicts grow only with unique player_ids (50k max).
    hour_counts: dict[str, np.ndarray] = {}
    dow_counts: dict[str, np.ndarray] = {}
    distance_lists: dict[str, list[np.ndarray]] = {}
    hex_counts: dict[str, dict[str, int]] = {}
    weekday_event_counts: dict[str, int] = {}
    evening_event_counts: dict[str, int] = {}
    total_event_counts: dict[str, int] = {}
    active_dates: dict[str, set] = {}
    session_event_counts: dict[str, dict[str, int]] = {}
    session_min_ts: dict[str, dict[str, pd.Timestamp]] = {}
    session_max_ts: dict[str, dict[str, pd.Timestamp]] = {}

    # Constants used for cyclical encoding and ratio calculations.

    # First pass: stream through partitions and accumulate per-player partials.
    for idx, part_path in enumerate(partitions, start=1):
        if progress_callback is not None:
            progress_callback(idx, total_parts, f"Processing {part_path.name}")

        # Load only one partition. Keep player_id as object (no category conversion).
        part = pd.read_parquet(part_path, columns=ALL_COLUMNS)
        part["distance_from_home_km"] = part["distance_from_home_km"].astype("float32")

        # Compute H3 cells for this partition's events.
        h3_cells = latlon_arrays_to_h3(
            part["lat"].to_numpy(),
            part["lon"].to_numpy(),
            resolution=h3_resolution,
        )
        part["h3_cell"] = h3_cells

        # Group by player_id within this partition.
        for player_id, group in part.groupby("player_id", sort=False):
            # Hour counts.
            h_counts = np.bincount(group["hour"].to_numpy(), minlength=HOURS_IN_DAY).astype(
                np.int64
            )
            if player_id in hour_counts:
                hour_counts[player_id] += h_counts
            else:
                hour_counts[player_id] = h_counts

            # Day-of-week counts.
            d_counts = np.bincount(group["day_of_week"].to_numpy(), minlength=DAYS_IN_WEEK).astype(
                np.int64
            )
            if player_id in dow_counts:
                dow_counts[player_id] += d_counts
            else:
                dow_counts[player_id] = d_counts

            # Distance values (accumulated as list of arrays for percentile calc).
            distances = group["distance_from_home_km"].to_numpy()
            if player_id in distance_lists:
                distance_lists[player_id].append(distances)
            else:
                distance_lists[player_id] = [distances]

            # H3 hex counts per player.
            hex_series = group["h3_cell"].value_counts()
            if player_id not in hex_counts:
                hex_counts[player_id] = {}
            for hex_id, cnt in hex_series.items():
                hex_counts[player_id][hex_id] = hex_counts[player_id].get(hex_id, 0) + int(cnt)

            # Weekday and evening event counts.
            weekday_mask = group["day_of_week"].isin(WEEKDAY_DAYS)
            evening_mask = group["hour"] >= EVENING_HOURS_START
            weekday_event_counts[player_id] = weekday_event_counts.get(player_id, 0) + int(
                weekday_mask.sum()
            )
            evening_event_counts[player_id] = evening_event_counts.get(player_id, 0) + int(
                evening_mask.sum()
            )

            # Total events.
            total_event_counts[player_id] = total_event_counts.get(player_id, 0) + len(group)

            # Active dates (set of unique calendar dates).
            dates = group["timestamp"].dt.date.unique()
            if player_id not in active_dates:
                active_dates[player_id] = set()
            active_dates[player_id].update(dates)

            # Session aggregates: events per session, min/max timestamp per session.
            if player_id not in session_event_counts:
                session_event_counts[player_id] = {}
                session_min_ts[player_id] = {}
                session_max_ts[player_id] = {}
            for session_id, sess_group in group.groupby("session_id", sort=False):
                n = len(sess_group)
                session_event_counts[player_id][session_id] = (
                    session_event_counts[player_id].get(session_id, 0) + n
                )
                t_min = sess_group["timestamp"].min()
                t_max = sess_group["timestamp"].max()
                if session_id in session_min_ts[player_id]:
                    session_min_ts[player_id][session_id] = min(
                        session_min_ts[player_id][session_id], t_min
                    )
                    session_max_ts[player_id][session_id] = max(
                        session_max_ts[player_id][session_id], t_max
                    )
                else:
                    session_min_ts[player_id][session_id] = t_min
                    session_max_ts[player_id][session_id] = t_max

        # Free partition memory before loading next.
        del part, h3_cells

    if progress_callback is not None:
        progress_callback(total_parts, total_parts, "Finalizing features")

    # Second pass: finalize features per player.
    player_ids = list(hour_counts.keys())
    rows = []

    hour_angles = 2.0 * np.pi * np.arange(HOURS_IN_DAY) / HOURS_IN_DAY
    dow_angles = 2.0 * np.pi * np.arange(DAYS_IN_WEEK) / DAYS_IN_WEEK

    for player_id in player_ids:
        row: dict[str, float | int | str] = {"player_id": player_id}

        # Temporal: hour density.
        h_counts = hour_counts[player_id]
        h_total = h_counts.sum()
        h_density = h_counts / h_total if h_total > 0 else np.zeros(HOURS_IN_DAY)
        for h in range(HOURS_IN_DAY):
            row[f"hour_density_{h:02d}"] = float(h_density[h])

        # Temporal: dow density.
        d_counts = dow_counts[player_id]
        d_total = d_counts.sum()
        d_density = d_counts / d_total if d_total > 0 else np.zeros(DAYS_IN_WEEK)
        for d in range(DAYS_IN_WEEK):
            row[f"dow_density_{d}"] = float(d_density[d])

        # Temporal: cyclical features.
        row["hour_mean_sin"] = float(np.sum(h_density * np.sin(hour_angles)))
        row["hour_mean_cos"] = float(np.sum(h_density * np.cos(hour_angles)))
        row["hour_concentration"] = float(
            np.sqrt(row["hour_mean_sin"] ** 2 + row["hour_mean_cos"] ** 2)
        )
        row["dow_mean_sin"] = float(np.sum(d_density * np.sin(dow_angles)))
        row["dow_mean_cos"] = float(np.sum(d_density * np.cos(dow_angles)))
        row["dow_concentration"] = float(
            np.sqrt(row["dow_mean_sin"] ** 2 + row["dow_mean_cos"] ** 2)
        )

        # Temporal: ratios.
        total = total_event_counts[player_id]
        row["weekday_ratio"] = weekday_event_counts.get(player_id, 0) / total if total > 0 else 0.0
        row["evening_ratio"] = evening_event_counts.get(player_id, 0) / total if total > 0 else 0.0

        # Spatial: distance aggregates.
        all_distances = np.concatenate(distance_lists[player_id])
        row["distance_from_home_mean"] = float(all_distances.mean())
        row["distance_from_home_std"] = (
            float(all_distances.std()) if len(all_distances) > 1 else 0.0
        )
        row["distance_from_home_p25"] = float(np.percentile(all_distances, 25))
        row["distance_from_home_p75"] = float(np.percentile(all_distances, 75))
        row["distance_from_home_p95"] = float(np.percentile(all_distances, 95))

        # Spatial: H3 footprint.
        player_hex_counts = hex_counts[player_id]
        row["unique_hexes"] = len(player_hex_counts)
        row["h3_entropy"] = shannon_entropy(np.array(list(player_hex_counts.values())))

        # Behavioral: volume.
        row["total_events"] = total
        row["total_sessions"] = len(session_event_counts[player_id])

        # Behavioral: session structure.
        sess_counts_array = np.array(list(session_event_counts[player_id].values()))
        row["mean_events_per_session"] = float(sess_counts_array.mean())
        row["median_events_per_session"] = float(np.median(sess_counts_array))
        row["std_events_per_session"] = (
            float(sess_counts_array.std()) if len(sess_counts_array) > 1 else 0.0
        )

        # Behavioral: session duration.
        sess_durations = [
            (session_max_ts[player_id][sid] - session_min_ts[player_id][sid]).total_seconds() / 60.0
            for sid in session_event_counts[player_id]
        ]
        row["mean_session_duration_minutes"] = (
            float(np.mean(sess_durations)) if sess_durations else 0.0
        )

        # Behavioral: intensity.
        n_active_days = len(active_dates[player_id])
        row["events_per_active_day"] = total / n_active_days if n_active_days > 0 else 0.0
        row["active_days_ratio"] = n_active_days / observation_days

        rows.append(row)

    features = pd.DataFrame(rows).set_index("player_id")

    # Attach archetype as ground truth.
    players = pd.read_parquet(players_path, columns=["player_id", "archetype"])
    players = players.set_index("player_id")
    features["archetype"] = players["archetype"].reindex(features.index)

    # Persist.
    features.reset_index().to_parquet(output_path, engine="pyarrow", index=False)

    return features
