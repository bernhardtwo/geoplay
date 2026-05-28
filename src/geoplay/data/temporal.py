"""Temporal pattern generation for synthetic player activity.

Generates event timestamps that follow archetype-specific patterns of
weekday/weekend activity, hour-of-day preferences, and session bursts.
The resulting distributions are recoverable by a downstream model using
cyclical time features (sin/cos of hour and day).

Noise model:
- HOUR_PEAK_SIGMA controls how tightly activity concentrates around each
  archetype's peak hours. Larger = more overlap between archetypes.
- ATYPICAL_DAY_PROBABILITY: chance that any active day is "atypical",
  where the player's behavior departs from their archetype pattern
  (vacations, illness, social events, etc.). On atypical days, hours
  are sampled uniformly across the day instead of from archetype peaks.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

from geoplay.data.archetypes import ArchetypeProfile

# Standard deviation (in hours) of the Gaussian bumps placed at each peak hour.
# Increased from 1.5 to 2.5 to create realistic overlap between archetypes.
HOUR_PEAK_SIGMA = 2.5

# Probability that an active day is "atypical": the player breaks their
# archetype pattern and behaves randomly. Simulates vacations, illness,
# social events, or any disruption to their habitual routine.
ATYPICAL_DAY_PROBABILITY = 0.08


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


def build_uniform_hour_density() -> npt.NDArray[np.float64]:
    """Build a uniform hour density for atypical days.

    Atypical days simulate disruptions to routine: vacations, illness,
    social events. Activity can happen at any hour with equal probability,
    with a mild preference for waking hours (6h-23h) to remain realistic.

    Returns
    -------
    np.ndarray
        Shape (24,), uniform-ish probability density over hours.
    """
    density = np.ones(24, dtype=np.float64)
    # Reduce the very small hours (0-5) to ~30% probability, since most
    # people are still asleep even on atypical days.
    density[0:6] *= 0.3
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
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """For each day, sample whether the player is active and whether atypical.

    Returns two boolean arrays aligned with the day range:
    - `active`: True if the player generates events that day
    - `atypical`: True if the day is an "off-pattern" day (vacations, etc.)

    Atypical flag is independent of activity; an inactive day cannot be
    atypical because there are no events to deviate.

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
    tuple[np.ndarray, np.ndarray]
        (active, atypical), both boolean arrays of shape (n_days,).
    """
    activity_probs = np.empty(n_days, dtype=np.float64)
    for i in range(n_days):
        date = start_date + timedelta(days=i)
        activity_probs[i] = (
            profile.weekend_activity if is_weekend(date) else profile.weekday_activity
        )
    active = rng.uniform(0.0, 1.0, size=n_days) < activity_probs

    # Atypical days are uncommon. We sample independently; players that are
    # inactive that day cannot be "atypical" since there are no events.
    atypical = rng.uniform(0.0, 1.0, size=n_days) < ATYPICAL_DAY_PROBABILITY
    atypical = atypical & active

    return active, atypical


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
    atypical: bool = False,
) -> npt.NDArray[np.datetime64]:
    """Sample timestamps for `n_events` events on a given day.

    On normal days, hours follow the archetype-specific hour density.
    On atypical days, hours follow a uniform distribution (with mild
    suppression of late-night hours).

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
    atypical : bool
        If True, use uniform hour density instead of archetype peaks.

    Returns
    -------
    np.ndarray
        Array of np.datetime64 timestamps of shape (n_events,).
    """
    if atypical:
        hour_density = build_uniform_hour_density()
    else:
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
