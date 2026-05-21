"""Configuration for synthetic data generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from geoplay.data.archetypes import DEFAULT_DISTRIBUTION, Archetype


class GenerationConfig(BaseSettings):
    """Configuration for synthetic player and event generation.

    Attributes
    ----------
    n_players : int
        Number of players to generate.
    n_days : int
        Number of days of activity to simulate.
    seed : int
        Random seed for reproducibility.
    archetype_distribution : dict[Archetype, float]
        Proportion of each archetype in the population. Must sum to 1.0.
    geographic_center : tuple[float, float]
        (latitude, longitude) of the city center.
    geographic_radius_km : float
        Maximum distance from center where players can be located.
    start_date : datetime
        Start date of the simulation period.
    noise_level : float
        Amount of stochastic noise injected into generated patterns (0.0 to 1.0).
    output_dir : Path
        Directory where generated Parquet files are written.
    """

    n_players: int = Field(default=50_000, gt=0)
    n_days: int = Field(default=180, gt=0)
    seed: int = Field(default=42)
    archetype_distribution: dict[Archetype, float] = Field(
        default_factory=lambda: dict(DEFAULT_DISTRIBUTION)
    )
    geographic_center: tuple[float, float] = Field(default=(29.0729, -110.9559))
    geographic_radius_km: float = Field(default=15.0, gt=0)
    start_date: datetime = Field(default_factory=lambda: datetime(2025, 1, 1))
    noise_level: float = Field(default=0.15, ge=0.0, le=1.0)
    output_dir: Path = Field(default_factory=lambda: Path("data/raw"))

    @field_validator("archetype_distribution")
    @classmethod
    def distribution_must_sum_to_one(cls, v: dict[Archetype, float]) -> dict[Archetype, float]:
        """Validate that archetype proportions sum to 1.0 (within tolerance)."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"archetype_distribution must sum to 1.0, got {total:.4f}")
        return v

    @field_validator("geographic_center")
    @classmethod
    def valid_lat_lon(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate latitude and longitude ranges."""
        lat, lon = v
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude {lon} out of range [-180, 180]")
        return v

    model_config = {
        "env_prefix": "GEOPLAY_",
        "frozen": True,
    }
