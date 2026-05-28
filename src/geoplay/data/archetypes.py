"""Player archetype definitions.

Each archetype defines a latent behavioral pattern that the clustering model
should be able to recover from observable events. Archetypes are NOT exposed
to the model during training; they serve as ground truth for evaluation.

The catalog is intentionally designed with overlapping "neighbor" archetypes
(e.g., commuter / morning_player / lunch_commuter all share morning peaks)
so that clustering algorithms must rely on the full feature space — not just
hour-of-day — to separate them. This creates a realistic challenge where
ARI < 1.0 is the expected outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Archetype(StrEnum):
    """Latent player archetypes."""

    # Original 5 archetypes (well-separated behavioral extremes)
    COMMUTER = "commuter"
    CASUAL_EVENING = "casual_evening"
    WEEKEND_EXPLORER = "weekend_explorer"
    HARDCORE_RAIDER = "hardcore_raider"
    LUNCH_PLAYER = "lunch_player"

    # New 4 archetypes (neighbors that overlap with the originals)
    LUNCH_COMMUTER = "lunch_commuter"
    WEEKEND_HOMEBODY = "weekend_homebody"
    MORNING_PLAYER = "morning_player"
    NIGHT_OWL = "night_owl"


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
    # ============================================================
    # ORIGINAL 5 ARCHETYPES
    # ============================================================
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
    # ============================================================
    # NEW 4 ARCHETYPES (neighbors designed to confuse clustering)
    # ============================================================
    Archetype.LUNCH_COMMUTER: ArchetypeProfile(
        archetype=Archetype.LUNCH_COMMUTER,
        # Triple peak: morning commute + lunch + evening commute.
        # Overlaps with COMMUTER (morning/evening) and LUNCH_PLAYER (midday).
        weekday_hour_peaks=(8, 13, 18),
        weekend_hour_peaks=(11, 14),
        weekday_activity=0.90,
        weekend_activity=0.40,
        avg_events_per_active_day=18.0,
        movement_radius_km=7.0,
        session_duration_minutes=18.0,
        preferred_zones=("office", "commercial", "transit"),
    ),
    Archetype.WEEKEND_HOMEBODY: ArchetypeProfile(
        archetype=Archetype.WEEKEND_HOMEBODY,
        # Casual evening pattern on weekdays, but extra daytime activity on
        # weekends without leaving home zone. Overlaps with CASUAL_EVENING
        # (evening peaks) and WEEKEND_EXPLORER (weekend daytime).
        weekday_hour_peaks=(20, 21),
        weekend_hour_peaks=(11, 14, 20, 21),
        weekday_activity=0.65,
        weekend_activity=0.85,
        avg_events_per_active_day=14.0,
        movement_radius_km=3.0,
        session_duration_minutes=30.0,
        preferred_zones=("residential",),
    ),
    Archetype.MORNING_PLAYER: ArchetypeProfile(
        archetype=Archetype.MORNING_PLAYER,
        # Only mornings. Overlaps with COMMUTER on morning peak (7-9h).
        # Distinguished by lack of evening peak.
        weekday_hour_peaks=(6, 7, 8, 9),
        weekend_hour_peaks=(8, 9, 10),
        weekday_activity=0.85,
        weekend_activity=0.55,
        avg_events_per_active_day=15.0,
        movement_radius_km=4.0,
        session_duration_minutes=20.0,
        preferred_zones=("residential", "transit"),
    ),
    Archetype.NIGHT_OWL: ArchetypeProfile(
        archetype=Archetype.NIGHT_OWL,
        # Late-night player. Overlaps with CASUAL_EVENING at 22-23h.
        # Distinguished by activity continuing into 0-2h.
        weekday_hour_peaks=(22, 23, 0, 1),
        weekend_hour_peaks=(22, 23, 0, 1, 2),
        weekday_activity=0.75,
        weekend_activity=0.80,
        avg_events_per_active_day=20.0,
        movement_radius_km=3.5,
        session_duration_minutes=40.0,
        preferred_zones=("residential",),
    ),
}


DEFAULT_DISTRIBUTION: dict[Archetype, float] = {
    # Original 5 archetypes (collectively 70% of population).
    Archetype.COMMUTER: 0.18,
    Archetype.CASUAL_EVENING: 0.22,
    Archetype.WEEKEND_EXPLORER: 0.10,
    Archetype.HARDCORE_RAIDER: 0.07,
    Archetype.LUNCH_PLAYER: 0.13,
    # New 4 archetypes (collectively 30% of population, designed as neighbors).
    Archetype.LUNCH_COMMUTER: 0.10,
    Archetype.WEEKEND_HOMEBODY: 0.08,
    Archetype.MORNING_PLAYER: 0.07,
    Archetype.NIGHT_OWL: 0.05,
}
