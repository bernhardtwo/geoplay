"""Clustering algorithms for player segmentation.

Three algorithms are supported, all trained on the standardized 54-feature
matrix produced by the feature engineering pipeline:

1. HDBSCAN: density-based, finds clusters of arbitrary shape, can identify
   noise points. Primary algorithm. No need to specify k.
2. KMeans: centroid-based, baseline. Assumes spherical clusters. Needs k.
3. GaussianMixture: probabilistic, allows soft cluster assignments. Needs k.

All models are wrapped in scikit-learn-compatible interfaces with fit and
predict methods. The wrappers also expose noise labels (HDBSCAN) and
cluster probabilities (GMM) for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    """Container for the output of a clustering algorithm.

    Attributes
    ----------
    algorithm : str
        Name of the algorithm used (e.g., "hdbscan").
    labels : np.ndarray
        Cluster assignment per sample. -1 indicates noise (HDBSCAN only).
    n_clusters : int
        Number of clusters found (excluding noise for HDBSCAN).
    noise_fraction : float
        Fraction of samples labeled as noise (0.0 for KMeans and GMM).
    model : object
        The fitted underlying scikit-learn or hdbscan estimator.
    """

    algorithm: str
    labels: npt.NDArray[np.int64]
    n_clusters: int
    noise_fraction: float
    model: object


def standardize_features(
    X: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], StandardScaler]:
    """Standardize features to zero mean and unit variance.

    HDBSCAN and KMeans both rely on Euclidean distance, which is highly
    sensitive to feature scale. Without standardization, features with
    larger magnitudes (e.g., total_events, ~thousands) would dominate
    features with smaller magnitudes (e.g., active_days_ratio, ~0-1).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    tuple[np.ndarray, StandardScaler]
        Standardized feature matrix and the fitted scaler (kept for
        re-use on new data later).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def fit_hdbscan(
    X: npt.NDArray[np.float64],
    min_cluster_size: int = 500,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
) -> ClusteringResult:
    """Fit HDBSCAN on standardized features.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    min_cluster_size : int
        Minimum number of samples in a cluster. For 50k players targeting
        ~5 clusters of ~10k members each, 500 (1% of population) is a
        conservative lower bound that prevents micro-clusters.
    min_samples : int | None
        Core point density. If None, defaults to min_cluster_size, which
        produces more conservative clusters.
    cluster_selection_method : str
        "eom" (excess of mass) tends to produce balanced clusters.
        "leaf" produces more, smaller clusters.

    Returns
    -------
    ClusteringResult
    """
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1,
    )
    labels = model.fit_predict(X)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    noise_fraction = float(np.sum(labels == -1)) / len(labels)

    return ClusteringResult(
        algorithm="hdbscan",
        labels=labels.astype(np.int64),
        n_clusters=n_clusters,
        noise_fraction=noise_fraction,
        model=model,
    )


def fit_kmeans(
    X: npt.NDArray[np.float64],
    n_clusters: int = 5,
    random_state: int = 42,
) -> ClusteringResult:
    """Fit KMeans on standardized features.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    n_clusters : int
        Number of clusters to find. We set 5 to match the known number
        of archetypes (using ground truth knowledge for baseline comparison).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    ClusteringResult
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(X)

    return ClusteringResult(
        algorithm="kmeans",
        labels=labels.astype(np.int64),
        n_clusters=n_clusters,
        noise_fraction=0.0,
        model=model,
    )


def fit_gaussian_mixture(
    X: npt.NDArray[np.float64],
    n_components: int = 5,
    random_state: int = 42,
) -> ClusteringResult:
    """Fit Gaussian Mixture Model on standardized features.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    n_components : int
        Number of mixture components. Set to 5 to match archetypes.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    ClusteringResult
    """
    model = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        covariance_type="full",
        max_iter=200,
    )
    labels = model.fit_predict(X)

    return ClusteringResult(
        algorithm="gaussian_mixture",
        labels=labels.astype(np.int64),
        n_clusters=n_components,
        noise_fraction=0.0,
        model=model,
    )
