"""Temporal pattern generation for synthetic player activity.

Generates event timestamps that follow archetype-specific patterns of
weekday/weekend activity, hour-of-day preferences, and session bursts.
The resulting distributions are recoverable by a downstream model using
cyclical time features (sin/cos of hour and day).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

from geoplay.data.archetypes import ArchetypeProfile

# Standard deviation (in hours) of the Gaussian bumps placed at each peak hour.
HOUR_PEAK_SIGMA = 1.5


def build_hour_density(peak_hours: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """Build a 24-element probability density over hours of the day.

    Places a Gaussian bump at each peak hour and wraps around midnight.
    The result sums to 1.0 and represents P(event happens at hour h).

    Parameters
    ----------
    peak_hours : tuple[int, ...]
        Hours (0-23) where activity should be concentrated.

    Returns
    -------
    np.ndarray
        Shape (24,), probability density over hours.
    """
    hours = np.arange(24, dtype=np.float64)
    density = np.zeros(24, dtype=np.float64)

    for peak in peak_hours:
        # Compute distance in hours with wraparound (so peak=23 is close to hour=0).
        diff = np.abs(hours - peak)
        diff = np.minimum(diff, 24.0 - diff)
        density += np.exp(-0.5 * (diff / HOUR_PEAK_SIGMA) ** 2)

    # Normalize to a probability distribution.
    return density / density.sum()


def is_weekend(date: datetime) -> bool:
    """Return True if the date falls on Saturday or Sunday."""
    # weekday(): Monday=0, Sunday=6.
    return date.weekday() >= 5


def sample_active_days(
    profile: ArchetypeProfile,
    start_date: datetime,
    n_days: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.bool_]:
    """For each day in the range, sample whether the player is active that day.

    Parameters
    ----------
    profile : ArchetypeProfile
        Archetype-specific activity levels.
    start_date : datetime
        First day of the simulation period.
    n_days : int
        Number of days to simulate.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Boolean array of shape (n_days,), True if active on that day.
    """
    activity_probs = np.empty(n_days, dtype=np.float64)
    for i in range(n_days):
        date = start_date + timedelta(days=i)
        activity_probs[i] = (
            profile.weekend_activity if is_weekend(date) else profile.weekday_activity
        )
    return rng.uniform(0.0, 1.0, size=n_days) < activity_probs


def sample_events_per_day(
    profile: ArchetypeProfile,
    n_active_days: int,
    noise_level: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.int32]:
    """Sample number of events for each active day.

    Uses a Poisson distribution centered at the archetype's mean, with
    Gaussian noise on the rate to add realistic variability between days.

    Parameters
    ----------
    profile : ArchetypeProfile
        Archetype-specific activity intensity.
    n_active_days : int
        Number of days the player will be active.
    noise_level : float
        Multiplicative noise on the Poisson rate (0.0 = deterministic mean).
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Integer array of shape (n_active_days,), events per active day.
    """
    base_rate = profile.avg_events_per_active_day
    noise_factors = rng.normal(loc=1.0, scale=noise_level, size=n_active_days)
    noise_factors = np.clip(noise_factors, 0.3, 2.0)  # avoid extreme outliers
    rates = base_rate * noise_factors
    return rng.poisson(lam=rates).astype(np.int32)


def sample_event_timestamps(
    profile: ArchetypeProfile,
    day_date: datetime,
    n_events: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.datetime64]:
    """Sample timestamps for `n_events` events on a given day.

    Hours are sampled from the archetype-specific hour density (weekday or
    weekend). Minutes and seconds are sampled uniformly within the hour.

    Parameters
    ----------
    profile : ArchetypeProfile
        Archetype profile defining hour preferences.
    day_date : datetime
        The date on which events occur (time component is ignored).
    n_events : int
        Number of timestamps to generate.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Array of np.datetime64 timestamps of shape (n_events,).
    """
    peaks = profile.weekend_hour_peaks if is_weekend(day_date) else profile.weekday_hour_peaks
    hour_density = build_hour_density(peaks)

    # Sample hours by inverse CDF over the discrete 24-hour density.
    hours = rng.choice(24, size=n_events, p=hour_density).astype(np.int32)

    # Sample minutes and seconds uniformly.
    minutes = rng.integers(0, 60, size=n_events, dtype=np.int32)
    seconds = rng.integers(0, 60, size=n_events, dtype=np.int32)

    day_start = np.datetime64(day_date.replace(hour=0, minute=0, second=0, microsecond=0))
    offsets = (
        hours.astype("timedelta64[h]")
        + minutes.astype("timedelta64[m]")
        + seconds.astype("timedelta64[s]")
    )
    return day_start + offsets
