"""Player table generation.

Produces a DataFrame of players with assigned archetypes and anchor locations
(home, optionally work). Each player is a row; activity is generated separately
in events.py.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from geoplay.data.archetypes import Archetype
from geoplay.data.config import GenerationConfig
from geoplay.data.geography import sample_points_in_disk

# Fraction of players that have a separate work location.
# The rest do all activity around a single home anchor (students, retirees, remote workers).
WORK_LOCATION_FRACTION = 0.70

# Minimum and maximum distance between home and work for players with two anchors.
MIN_HOME_WORK_DISTANCE_KM = 1.0
MAX_HOME_WORK_DISTANCE_KM = 15.0


def assign_archetypes(
    n_players: int,
    distribution: dict[Archetype, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign one archetype to each player following the target distribution.

    Parameters
    ----------
    n_players : int
        Number of players to assign.
    distribution : dict[Archetype, float]
        Proportion of each archetype. Must sum to 1.0.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Array of Archetype values, length n_players.
    """
    archetypes = list(distribution.keys())
    probs = np.array([distribution[a] for a in archetypes], dtype=np.float64)
    probs = probs / probs.sum()
    indices = rng.choice(len(archetypes), size=n_players, p=probs)
    return np.array([archetypes[i] for i in indices])


def generate_players(config: GenerationConfig) -> pd.DataFrame:
    """Generate the full player table.

    Each player gets:
      - a UUID-based identifier;
      - a latent archetype label (ground truth, not exposed to the model);
      - a home location sampled uniformly within the city radius;
      - optionally a work location at 1-15 km from home;
      - a device type (ios/android);
      - a created_at timestamp before the simulation start.

    Parameters
    ----------
    config : GenerationConfig
        Generation parameters.

    Returns
    -------
    pd.DataFrame
        Player table with columns:
        player_id, archetype, created_at, home_lat, home_lon,
        work_lat, work_lon, device_type.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_players

    # Identifiers.
    # numpy's RNG is int64-bounded, so we generate UUIDs from 16 random bytes each.
    # This stays reproducible (driven by config.seed) while producing valid UUID4s.
    player_ids = np.array(
        [
            str(uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8))))
            for _ in range(n)
        ]
    )

    # Archetypes.
    archetypes = assign_archetypes(n, config.archetype_distribution, rng)

    # Home locations: uniform within the city disk.
    home_lats, home_lons = sample_points_in_disk(
        config.geographic_center,
        config.geographic_radius_km,
        n_points=n,
        rng=rng,
    )

    # Work locations: a fraction of players have one.
    has_work = rng.uniform(0.0, 1.0, size=n) < WORK_LOCATION_FRACTION
    work_lats = np.full(n, np.nan, dtype=np.float64)
    work_lons = np.full(n, np.nan, dtype=np.float64)

    n_with_work = int(has_work.sum())
    if n_with_work > 0:
        # Sample work locations uniformly within the city, then we'll filter by distance.
        # Approach: sample around each home with a half-normal scaled to give realistic
        # commute distances. We loop until all are within [MIN, MAX].
        work_indices = np.where(has_work)[0]
        for _batch_attempt in range(5):  # at most 5 resamples for stragglers
            unassigned = work_indices[np.isnan(work_lats[work_indices])]
            if len(unassigned) == 0:
                break
            candidate_lats, candidate_lons = sample_points_in_disk(
                config.geographic_center,
                config.geographic_radius_km,
                n_points=len(unassigned),
                rng=rng,
            )
            # Compute distance from each candidate to that player's home.
            from geoplay.data.geography import haversine_km

            distances = haversine_km(
                home_lats[unassigned],
                home_lons[unassigned],
                candidate_lats,
                candidate_lons,
            )
            ok = (distances >= MIN_HOME_WORK_DISTANCE_KM) & (distances <= MAX_HOME_WORK_DISTANCE_KM)
            valid_indices = unassigned[ok]
            work_lats[valid_indices] = candidate_lats[ok]
            work_lons[valid_indices] = candidate_lons[ok]

        # Any stragglers that didn't converge get their work set to home as fallback.
        still_missing = work_indices[np.isnan(work_lats[work_indices])]
        work_lats[still_missing] = home_lats[still_missing]
        work_lons[still_missing] = home_lons[still_missing]

    # Device type.
    device_types = rng.choice(["ios", "android"], size=n, p=[0.45, 0.55])

    # Created_at: between 365 and 30 days before start_date.
    created_days_before = rng.integers(30, 365, size=n)
    created_at = pd.to_datetime(
        [config.start_date - pd.Timedelta(days=int(d)) for d in created_days_before]
    )

    return pd.DataFrame(
        {
            "player_id": player_ids,
            "archetype": archetypes.astype(str),
            "created_at": created_at,
            "home_lat": home_lats,
            "home_lon": home_lons,
            "work_lat": work_lats,
            "work_lon": work_lons,
            "device_type": device_types,
        }
    )
