"""Player archetype definitions.

Each archetype defines a latent behavioral pattern that the clustering model
should be able to recover from observable events. Archetypes are NOT exposed
to the model during training; they serve as ground truth for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Archetype(StrEnum):
    """Latent player archetypes."""

    COMMUTER = "commuter"
    CASUAL_EVENING = "casual_evening"
    WEEKEND_EXPLORER = "weekend_explorer"
    HARDCORE_RAIDER = "hardcore_raider"
    LUNCH_PLAYER = "lunch_player"


@dataclass(frozen=True, slots=True)
class ArchetypeProfile:
    """Behavioral profile of a player archetype.

    Attributes
    ----------
    archetype : Archetype
        Identifier of the archetype.
    weekday_hour_peaks : tuple[int, ...]
        Hours of the day (0-23) where activity peaks on weekdays.
    weekend_hour_peaks : tuple[int, ...]
        Hours of the day (0-23) where activity peaks on weekends.
    weekday_activity : float
        Relative activity level on weekdays (0.0 to 1.0).
    weekend_activity : float
        Relative activity level on weekends (0.0 to 1.0).
    avg_events_per_active_day : float
        Mean number of events on a day when the player is active.
    movement_radius_km : float
        Typical distance from home where the player generates events.
    session_duration_minutes : float
        Average duration of a play session.
    preferred_zones : tuple[str, ...]
        Zone categories where this archetype is most active.
    """

    archetype: Archetype
    weekday_hour_peaks: tuple[int, ...]
    weekend_hour_peaks: tuple[int, ...]
    weekday_activity: float
    weekend_activity: float
    avg_events_per_active_day: float
    movement_radius_km: float
    session_duration_minutes: float
    preferred_zones: tuple[str, ...]


ARCHETYPE_PROFILES: dict[Archetype, ArchetypeProfile] = {
    Archetype.COMMUTER: ArchetypeProfile(
        archetype=Archetype.COMMUTER,
        weekday_hour_peaks=(7, 8, 17, 18, 19),
        weekend_hour_peaks=(10, 11, 17),
        weekday_activity=0.95,
        weekend_activity=0.35,
        avg_events_per_active_day=20.0,
        movement_radius_km=8.0,
        session_duration_minutes=15.0,
        preferred_zones=("transit", "office", "residential"),
    ),
    Archetype.CASUAL_EVENING: ArchetypeProfile(
        archetype=Archetype.CASUAL_EVENING,
        weekday_hour_peaks=(19, 20, 21, 22),
        weekend_hour_peaks=(19, 20, 21, 22),
        weekday_activity=0.70,
        weekend_activity=0.75,
        avg_events_per_active_day=12.0,
        movement_radius_km=2.5,
        session_duration_minutes=25.0,
        preferred_zones=("residential",),
    ),
    Archetype.WEEKEND_EXPLORER: ArchetypeProfile(
        archetype=Archetype.WEEKEND_EXPLORER,
        weekday_hour_peaks=(19, 20),
        weekend_hour_peaks=(10, 11, 12, 13, 14, 15, 16, 17),
        weekday_activity=0.15,
        weekend_activity=0.95,
        avg_events_per_active_day=45.0,
        movement_radius_km=12.0,
        session_duration_minutes=90.0,
        preferred_zones=("park", "recreation", "commercial"),
    ),
    Archetype.HARDCORE_RAIDER: ArchetypeProfile(
        archetype=Archetype.HARDCORE_RAIDER,
        weekday_hour_peaks=(12, 18, 19, 20, 21, 22),
        weekend_hour_peaks=(11, 12, 13, 17, 18, 19, 20, 21),
        weekday_activity=0.98,
        weekend_activity=0.98,
        avg_events_per_active_day=80.0,
        movement_radius_km=10.0,
        session_duration_minutes=120.0,
        preferred_zones=("commercial", "raid_hotspot", "park"),
    ),
    Archetype.LUNCH_PLAYER: ArchetypeProfile(
        archetype=Archetype.LUNCH_PLAYER,
        weekday_hour_peaks=(12, 13, 14),
        weekend_hour_peaks=(13, 14),
        weekday_activity=0.90,
        weekend_activity=0.25,
        avg_events_per_active_day=8.0,
        movement_radius_km=1.5,
        session_duration_minutes=10.0,
        preferred_zones=("office", "commercial"),
    ),
}


DEFAULT_DISTRIBUTION: dict[Archetype, float] = {
    Archetype.COMMUTER: 0.25,
    Archetype.CASUAL_EVENING: 0.30,
    Archetype.WEEKEND_EXPLORER: 0.15,
    Archetype.HARDCORE_RAIDER: 0.10,
    Archetype.LUNCH_PLAYER: 0.20,
}
