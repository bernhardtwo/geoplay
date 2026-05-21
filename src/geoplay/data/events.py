"""Event table generation.

Generates per-player event streams that follow archetype-specific temporal
and spatial patterns. Events are grouped into sessions (consecutive events
within a 30-minute gap share a session_id).

For scale (millions of events), use `write_events_to_parquet` which streams
batches of players to disk instead of holding everything in memory.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from geoplay.data.archetypes import ARCHETYPE_PROFILES, Archetype
from geoplay.data.geography import (
    haversine_km,
    sample_points_around_anchor,
    sample_points_in_disk,
)
from geoplay.data.temporal import (
    sample_active_days,
    sample_event_timestamps,
    sample_events_per_day,
)

# Events within this many minutes of each other (same player) share a session.
SESSION_GAP_MINUTES = 30

# Distribution over event types. Tuned to mimic a Pokémon GO-style game:
# spins (PokeStops) are most common, then catches, then raids, then egg walks.
EVENT_TYPE_PROBS: dict[str, float] = {
    "spin_stop": 0.45,
    "catch": 0.40,
    "raid": 0.10,
    "walk_egg": 0.05,
}

# Fraction of events that occur near the work anchor (vs. home anchor) for
# players who have a work location. The rest are around home or freely roaming.
WORK_ANCHOR_FRACTION = 0.35

# Fraction of events that are "free roaming" (sampled uniformly within the
# movement radius of the player, not anchored to home or work).
ROAMING_FRACTION = 0.15


def _sample_event_locations(
    n_events: int,
    home_lat: float,
    home_lon: float,
    work_lat: float | None,
    work_lon: float | None,
    movement_radius_km: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sample lat/lon for n events around a player's home/work anchors.

    The fractions are configurable constants at module level. Players without
    a work location have all their non-roaming events anchored to home.
    """
    has_work = work_lat is not None and work_lon is not None

    # Decide anchor for each event.
    # 0 = home, 1 = work, 2 = roaming
    if has_work:
        anchor_probs = [
            1.0 - WORK_ANCHOR_FRACTION - ROAMING_FRACTION,
            WORK_ANCHOR_FRACTION,
            ROAMING_FRACTION,
        ]
    else:
        anchor_probs = [1.0 - ROAMING_FRACTION, 0.0, ROAMING_FRACTION]

    anchor_choices = rng.choice(3, size=n_events, p=anchor_probs)

    lats = np.empty(n_events, dtype=np.float64)
    lons = np.empty(n_events, dtype=np.float64)

    # Events around home.
    home_mask = anchor_choices == 0
    n_home = int(home_mask.sum())
    if n_home > 0:
        home_lats, home_lons = sample_points_around_anchor(
            home_lat, home_lon, movement_radius_km * 0.4, n_home, rng
        )
        lats[home_mask] = home_lats
        lons[home_mask] = home_lons

    # Events around work.
    if has_work:
        work_mask = anchor_choices == 1
        n_work = int(work_mask.sum())
        if n_work > 0:
            work_lats_arr, work_lons_arr = sample_points_around_anchor(
                work_lat,
                work_lon,
                movement_radius_km * 0.4,
                n_work,
                rng,  # type: ignore[arg-type]
            )
            lats[work_mask] = work_lats_arr
            lons[work_mask] = work_lons_arr

    # Roaming events: sampled uniformly within a disk around home.
    roam_mask = anchor_choices == 2
    n_roam = int(roam_mask.sum())
    if n_roam > 0:
        roam_lats, roam_lons = sample_points_in_disk(
            (home_lat, home_lon), movement_radius_km, n_roam, rng
        )
        lats[roam_mask] = roam_lats
        lons[roam_mask] = roam_lons

    return lats, lons


def _assign_session_ids(
    timestamps: npt.NDArray[np.datetime64],
    rng: np.random.Generator,
) -> list[str]:
    """Group events into sessions by temporal proximity.

    Events sorted by timestamp; a new session starts whenever the gap to the
    previous event exceeds SESSION_GAP_MINUTES.

    Parameters
    ----------
    timestamps : np.ndarray
        Sorted event timestamps for a single player.
    rng : np.random.Generator
        Used to generate session UUIDs.

    Returns
    -------
    list[str]
        Session UUID per event, same length as timestamps.
    """
    n = len(timestamps)
    if n == 0:
        return []

    gap_threshold = np.timedelta64(SESSION_GAP_MINUTES, "m")
    gaps = np.diff(timestamps)
    new_session_starts = np.concatenate([[True], gaps > gap_threshold])

    session_ids: list[str] = []
    current_session = ""
    for is_new in new_session_starts:
        if is_new:
            current_session = str(
                uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8)))
            )
        session_ids.append(current_session)
    return session_ids


def generate_events_for_player(
    player_id: str,
    archetype: Archetype,
    home_lat: float,
    home_lon: float,
    work_lat: float | None,
    work_lon: float | None,
    start_date: datetime,
    n_days: int,
    noise_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate the full event stream for one player over the simulation period.

    The flow:
      1. Sample which days the player is active (archetype weekday/weekend rates).
      2. For each active day, sample number of events (Poisson around archetype mean).
      3. For each event, sample a timestamp within the day (hour density).
      4. For each event, sample a location (anchored to home/work or roaming).
      5. Sample event types from EVENT_TYPE_PROBS.
      6. Group events into sessions and compute denormalized features.

    Parameters
    ----------
    player_id : str
        Player UUID.
    archetype : Archetype
        Latent archetype of the player.
    home_lat, home_lon : float
        Home location.
    work_lat, work_lon : float | None
        Work location, or (None, None) if the player has no work anchor.
    start_date : datetime
        First day of simulation.
    n_days : int
        Number of days to simulate.
    noise_level : float
        Stochastic noise level passed through to events-per-day sampling.
    rng : np.random.Generator
        Player-specific random generator.

    Returns
    -------
    pd.DataFrame
        Event table for this player. May be empty if the player happened to
        be inactive every day.
    """
    profile = ARCHETYPE_PROFILES[archetype]

    # Step 1: which days the player is active.
    active_days = sample_active_days(profile, start_date, n_days, rng)
    n_active = int(active_days.sum())
    if n_active == 0:
        return _empty_events_df()

    # Step 2: events per active day.
    events_per_day = sample_events_per_day(profile, n_active, noise_level, rng)
    total_events = int(events_per_day.sum())
    if total_events == 0:
        return _empty_events_df()

    # Step 3: timestamps for each event.
    timestamp_chunks: list[npt.NDArray[np.datetime64]] = []
    active_day_dates = [start_date + pd.Timedelta(days=int(i)) for i in np.where(active_days)[0]]
    for day_date, n_events in zip(active_day_dates, events_per_day, strict=True):
        if n_events > 0:
            day_timestamps = sample_event_timestamps(profile, day_date, int(n_events), rng)
            timestamp_chunks.append(day_timestamps)
    timestamps = np.concatenate(timestamp_chunks)

    # Sort by timestamp so sessions are coherent.
    order = np.argsort(timestamps)
    timestamps = timestamps[order]

    # Step 4: locations.
    lats, lons = _sample_event_locations(
        n_events=total_events,
        home_lat=home_lat,
        home_lon=home_lon,
        work_lat=work_lat,
        work_lon=work_lon,
        movement_radius_km=profile.movement_radius_km,
        rng=rng,
    )

    # Step 5: event types.
    event_types_list = list(EVENT_TYPE_PROBS.keys())
    event_type_probs = np.array([EVENT_TYPE_PROBS[t] for t in event_types_list], dtype=np.float64)
    event_type_indices = rng.choice(len(event_types_list), size=total_events, p=event_type_probs)
    event_types = np.array(event_types_list)[event_type_indices]

    # Event IDs.
    event_ids = [
        str(uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8))))
        for _ in range(total_events)
    ]

    # Step 6: sessions and denormalized features.
    session_ids = _assign_session_ids(timestamps, rng)

    home_lats_arr = np.full(total_events, home_lat, dtype=np.float64)
    home_lons_arr = np.full(total_events, home_lon, dtype=np.float64)
    distance_from_home = haversine_km(home_lats_arr, home_lons_arr, lats, lons)

    ts_pandas = pd.to_datetime(timestamps)

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "player_id": player_id,
            "timestamp": ts_pandas,
            "lat": lats,
            "lon": lons,
            "event_type": event_types,
            "session_id": session_ids,
            "distance_from_home_km": distance_from_home,
            "hour": ts_pandas.hour.astype(np.int8),
            "day_of_week": ts_pandas.dayofweek.astype(np.int8),
        }
    )


def _empty_events_df() -> pd.DataFrame:
    """Return an empty events DataFrame with the correct schema."""
    return pd.DataFrame(
        {
            "event_id": pd.Series([], dtype="object"),
            "player_id": pd.Series([], dtype="object"),
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "lat": pd.Series([], dtype="float64"),
            "lon": pd.Series([], dtype="float64"),
            "event_type": pd.Series([], dtype="object"),
            "session_id": pd.Series([], dtype="object"),
            "distance_from_home_km": pd.Series([], dtype="float64"),
            "hour": pd.Series([], dtype="int8"),
            "day_of_week": pd.Series([], dtype="int8"),
        }
    )


def write_events_to_parquet(
    players_df: pd.DataFrame,
    start_date: datetime,
    n_days: int,
    noise_level: float,
    output_dir: Path,
    seed: int,
    batch_size: int = 1000,
    progress_callback: callable | None = None,  # type: ignore[type-arg]
) -> dict[str, int]:
    """Generate events for all players and write them to partitioned Parquet.

    Processes players in batches to keep memory usage bounded. Each batch
    writes one Parquet file under `output_dir/events/`. The full event table
    can be read later as a Dataset (multi-file Parquet).

    Parameters
    ----------
    players_df : pd.DataFrame
        Player table (output of generate_players).
    start_date : datetime
        First day of simulation.
    n_days : int
        Number of days to simulate.
    noise_level : float
        Stochastic noise level.
    output_dir : Path
        Base output directory; events go to `output_dir/events/`.
    seed : int
        Master seed; each player gets a derived seed for reproducibility.
    batch_size : int
        Number of players per Parquet partition.
    progress_callback : callable | None
        Optional function called as progress_callback(batch_idx, total_batches,
        n_events_in_batch). Useful for CLI progress bars.

    Returns
    -------
    dict[str, int]
        Summary stats: total_events, total_sessions, n_batches.
    """
    events_dir = output_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    # Master RNG used to derive per-player seeds (so each player is reproducible
    # independent of batch_size).
    master_rng = np.random.default_rng(seed)
    player_seeds = master_rng.integers(0, 2**63 - 1, size=len(players_df), dtype=np.int64)

    total_events = 0
    total_sessions = 0
    n_batches = (len(players_df) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(players_df))
        batch_players = players_df.iloc[batch_start:batch_end]
        batch_seeds = player_seeds[batch_start:batch_end]

        batch_events: list[pd.DataFrame] = []
        for (_, player), player_seed in zip(batch_players.iterrows(), batch_seeds, strict=True):
            player_rng = np.random.default_rng(int(player_seed))
            archetype = Archetype(player["archetype"])
            work_lat = None if pd.isna(player["work_lat"]) else float(player["work_lat"])
            work_lon = None if pd.isna(player["work_lon"]) else float(player["work_lon"])

            player_events = generate_events_for_player(
                player_id=str(player["player_id"]),
                archetype=archetype,
                home_lat=float(player["home_lat"]),
                home_lon=float(player["home_lon"]),
                work_lat=work_lat,
                work_lon=work_lon,
                start_date=start_date,
                n_days=n_days,
                noise_level=noise_level,
                rng=player_rng,
            )
            if len(player_events) > 0:
                batch_events.append(player_events)

        if not batch_events:
            continue

        batch_df = pd.concat(batch_events, ignore_index=True)

        # Normalize timestamp precision for Parquet consistency.
        batch_df["timestamp"] = batch_df["timestamp"].astype("datetime64[ns]")

        # Write batch to a partitioned Parquet file.
        part_path = events_dir / f"part_{batch_idx:05d}.parquet"
        batch_df.to_parquet(
            part_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        batch_event_count = len(batch_df)
        batch_session_count = batch_df["session_id"].nunique()
        total_events += batch_event_count
        total_sessions += batch_session_count

        if progress_callback is not None:
            progress_callback(batch_idx + 1, n_batches, batch_event_count)

    return {
        "total_events": total_events,
        "total_sessions": total_sessions,
        "n_batches": n_batches,
    }
