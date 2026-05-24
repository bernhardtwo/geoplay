"""Helpers for H3 hexagonal indexing.

H3 (Uber's Hexagonal Hierarchical Spatial Index) maps lat/lon coordinates
to hexagonal cells at multiple resolutions. We use resolution 8 (~0.7 km²
per hex) which is appropriate for city-scale player movement analysis.

H3 resolutions reference:
    res 6: ~36 km² (city district)
    res 7: ~5 km²  (neighborhood)
    res 8: ~0.7 km² (city block)  <-- our choice
    res 9: ~0.1 km² (building)
    res 10: ~0.015 km² (room)
"""

from __future__ import annotations

import h3
import numpy as np
import numpy.typing as npt

# H3 resolution used throughout the project. Calibrated for Hermosillo's
# 15 km radius city area (results in ~1000 unique hexes within the city disk).
DEFAULT_H3_RESOLUTION = 8


def latlon_to_h3(
    lat: float,
    lon: float,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> str:
    """Convert a single (lat, lon) point to its H3 cell ID.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    resolution : int
        H3 resolution (0-15). Defaults to DEFAULT_H3_RESOLUTION (8).

    Returns
    -------
    str
        H3 cell ID as a hex string (e.g., "8841aef8e3fffff").
    """
    return h3.latlng_to_cell(lat, lon, resolution)


def latlon_arrays_to_h3(
    lats: npt.NDArray[np.float64],
    lons: npt.NDArray[np.float64],
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> npt.NDArray[np.object_]:
    """Vectorized conversion of lat/lon arrays to H3 cells.

    Parameters
    ----------
    lats : np.ndarray
        Array of latitudes in degrees.
    lons : np.ndarray
        Array of longitudes in degrees (must be same shape as lats).
    resolution : int
        H3 resolution. Defaults to DEFAULT_H3_RESOLUTION.

    Returns
    -------
    np.ndarray
        Array of H3 cell IDs as strings, same length as inputs.
    """
    if lats.shape != lons.shape:
        raise ValueError(f"lats and lons must have same shape, got {lats.shape} vs {lons.shape}")
    # h3-py 4.x exposes a vectorized API; we use a Python comprehension here
    # because it's fast enough (~500k ops/sec) and works across all versions.
    return np.array(
        [
            h3.latlng_to_cell(float(lat), float(lon), resolution)
            for lat, lon in zip(lats, lons, strict=True)
        ],
        dtype=object,
    )


def shannon_entropy(counts: npt.NDArray[np.int64] | npt.NDArray[np.float64]) -> float:
    """Compute Shannon entropy in bits over a count vector.

    Useful to measure how concentrated or dispersed a player's H3 footprint is:
        - low entropy: player visits few hexes repeatedly (focused)
        - high entropy: player visits many hexes equally (exploratory)

    Parameters
    ----------
    counts : np.ndarray
        Count or frequency vector. Negative values raise ValueError.

    Returns
    -------
    float
        Shannon entropy in bits (log base 2). Zero for empty or single-bucket input.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Avoid log(0) by filtering out zero entries.
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    return -float(np.sum(probs * np.log2(probs)))


def h3_resolution_avg_area_km2(resolution: int) -> float:
    """Return the approximate average area (km²) of an H3 cell at given resolution.

    Useful for documentation and validation. Areas are H3's published averages.
    """
    return h3.average_hexagon_area(resolution, unit="km^2")
