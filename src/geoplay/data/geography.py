"""Geographic point generation for synthetic data.

Provides utilities to generate latitude/longitude points within a circular
region, distributed in patterns that mimic real urban density (with hotspots
of activity).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Earth's radius in kilometers (mean radius).
EARTH_RADIUS_KM = 6371.0


def km_to_degrees_lat(km: float) -> float:
    """Convert a north-south distance in km to degrees of latitude.

    Latitude degrees are roughly constant: 1 degree ~= 111 km anywhere on Earth.
    """
    return km / 111.0


def km_to_degrees_lon(km: float, at_latitude: float) -> float:
    """Convert an east-west distance in km to degrees of longitude.

    Longitude degrees vary with latitude: shrink toward the poles.
    1 degree of longitude ~= 111 km * cos(latitude).
    """
    return km / (111.0 * np.cos(np.radians(at_latitude)))


def haversine_km(
    lat1: npt.NDArray[np.float64],
    lon1: npt.NDArray[np.float64],
    lat2: npt.NDArray[np.float64],
    lon2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute great-circle distance in kilometers between two sets of points.

    Vectorized: accepts arrays of equal shape and returns an array of distances.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array-like of float64
        Latitudes and longitudes in degrees.

    Returns
    -------
    np.ndarray
        Distances in kilometers, same shape as inputs.
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def sample_points_in_disk(
    center: tuple[float, float],
    radius_km: float,
    n_points: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sample n points uniformly within a disk centered at `center`.

    Uses the inverse CDF method to ensure uniform distribution in 2D
    (not biased toward the center, which a naive uniform sampling would be).

    Parameters
    ----------
    center : tuple[float, float]
        (latitude, longitude) of the disk center.
    radius_km : float
        Radius of the disk in kilometers.
    n_points : int
        Number of points to sample.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (latitudes, longitudes) arrays of shape (n_points,).
    """
    center_lat, center_lon = center

    # Sample radii with sqrt to get uniform 2D distribution.
    # If we sampled r uniformly, points would cluster toward the center.
    r = radius_km * np.sqrt(rng.uniform(0.0, 1.0, size=n_points))

    # Sample angles uniformly on [0, 2pi).
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)

    # Convert polar offset (r, theta) to lat/lon offsets.
    dlat_km = r * np.sin(theta)
    dlon_km = r * np.cos(theta)

    lats = center_lat + km_to_degrees_lat(dlat_km)
    lons = center_lon + km_to_degrees_lon(dlon_km, center_lat)

    return lats, lons


def sample_points_around_anchor(
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
    n_points: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sample n points around an anchor with Gaussian-like spread.

    Unlike sample_points_in_disk (uniform), this uses a half-normal distance,
    so most points are close to the anchor and a few are far. This better
    models real player movement around their home/work locations.

    Parameters
    ----------
    anchor_lat, anchor_lon : float
        Latitude and longitude of the anchor point.
    radius_km : float
        Standard deviation of the distance distribution.
    n_points : int
        Number of points to sample.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (latitudes, longitudes) arrays of shape (n_points,).
    """
    # Half-normal: take absolute value of normal samples.
    r = np.abs(rng.normal(loc=0.0, scale=radius_km, size=n_points))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)

    dlat_km = r * np.sin(theta)
    dlon_km = r * np.cos(theta)

    lats = anchor_lat + km_to_degrees_lat(dlat_km)
    lons = anchor_lon + km_to_degrees_lon(dlon_km, anchor_lat)

    return lats, lons
