"""Evaluation metrics for clustering against ground truth archetypes.

Since we have access to the true archetype labels (from the synthetic data
generator), we can compute supervised clustering metrics that quantify how
well the predicted clusters align with the known structure.

Two families of metrics:

1. Pair-based: ARI (Adjusted Rand Index) measures the fraction of pairs
   that are correctly grouped together or apart, adjusted for chance.
   Range: -1 to 1. Above 0.7 = strong agreement.

2. Information-theoretic: NMI (Normalized Mutual Information) measures
   shared information between cluster and label distributions.
   Range: 0 to 1. Above 0.7 = strong agreement.

Plus auxiliary metrics:
- Homogeneity: do clusters contain a single archetype?
- Completeness: is each archetype in a single cluster?
- V-measure: harmonic mean of homogeneity and completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


@dataclass
class ClusterEvaluation:
    """Container for clustering evaluation against ground truth.

    Attributes
    ----------
    algorithm : str
        Name of the clustering algorithm being evaluated.
    n_predicted_clusters : int
        Number of predicted clusters (excluding noise).
    n_true_classes : int
        Number of ground truth archetype classes.
    ari : float
        Adjusted Rand Index. -1 to 1, higher is better. >0.7 = strong.
    nmi : float
        Normalized Mutual Information. 0 to 1, higher is better.
    homogeneity : float
        Each cluster contains a single archetype. 0 to 1.
    completeness : float
        Each archetype is in a single cluster. 0 to 1.
    v_measure : float
        Harmonic mean of homogeneity and completeness. 0 to 1.
    noise_fraction : float
        Fraction of samples labeled as noise (HDBSCAN only, 0 otherwise).
    confusion : pd.DataFrame
        Contingency table: rows are clusters, columns are archetypes,
        values are counts.
    """

    algorithm: str
    n_predicted_clusters: int
    n_true_classes: int
    ari: float
    nmi: float
    homogeneity: float
    completeness: float
    v_measure: float
    noise_fraction: float
    confusion: pd.DataFrame = field(repr=False)

    def to_dict(self) -> dict[str, float | int | str]:
        """Return scalar metrics as a flat dictionary (for JSON export)."""
        return {
            "algorithm": self.algorithm,
            "n_predicted_clusters": self.n_predicted_clusters,
            "n_true_classes": self.n_true_classes,
            "ari": round(self.ari, 4),
            "nmi": round(self.nmi, 4),
            "homogeneity": round(self.homogeneity, 4),
            "completeness": round(self.completeness, 4),
            "v_measure": round(self.v_measure, 4),
            "noise_fraction": round(self.noise_fraction, 4),
        }


def evaluate_clustering(
    algorithm: str,
    predicted_labels: npt.NDArray[np.int64],
    true_labels: npt.NDArray[np.int64] | npt.NDArray[np.object_],
    noise_fraction: float = 0.0,
) -> ClusterEvaluation:
    """Compute clustering quality metrics against ground truth.

    Parameters
    ----------
    algorithm : str
        Name of the algorithm being evaluated (for reporting).
    predicted_labels : np.ndarray
        Cluster labels predicted by the algorithm. -1 indicates noise.
    true_labels : np.ndarray
        Ground truth class labels (archetypes as strings or integers).
    noise_fraction : float
        Fraction of samples labeled as noise. Pass through from the
        ClusteringResult.

    Returns
    -------
    ClusterEvaluation
    """
    if len(predicted_labels) != len(true_labels):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted_labels)}, " f"true={len(true_labels)}"
        )

    # Convert true_labels to a numerical representation for metric functions
    # that require it (most sklearn metrics handle strings via factorize internally).
    unique_predicted = set(predicted_labels)
    n_predicted = len(unique_predicted - {-1})
    n_true = len(set(true_labels))

    # Compute metrics. sklearn handles -1 (noise) as a valid label, treating
    # noise points as belonging to their own "cluster" for metric purposes.
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)

    # Confusion matrix as a DataFrame (clusters x archetypes).
    confusion = pd.crosstab(
        pd.Series(predicted_labels, name="cluster"),
        pd.Series(true_labels, name="archetype"),
    )

    return ClusterEvaluation(
        algorithm=algorithm,
        n_predicted_clusters=n_predicted,
        n_true_classes=n_true,
        ari=float(ari),
        nmi=float(nmi),
        homogeneity=float(homogeneity),
        completeness=float(completeness),
        v_measure=float(v_measure),
        noise_fraction=noise_fraction,
        confusion=confusion,
    )


def cluster_purity(confusion: pd.DataFrame) -> pd.Series:
    """For each cluster, compute purity (proportion of dominant archetype).

    A pure cluster (purity=1.0) contains samples from a single archetype.
    Useful for interpreting which archetype each cluster represents.

    Parameters
    ----------
    confusion : pd.DataFrame
        Confusion matrix from ClusterEvaluation. Rows are clusters.

    Returns
    -------
    pd.Series
        Purity per cluster (indexed by cluster id).
    """
    cluster_sizes = confusion.sum(axis=1)
    dominant_class_size = confusion.max(axis=1)
    return (dominant_class_size / cluster_sizes).rename("purity")


def cluster_dominant_archetype(confusion: pd.DataFrame) -> pd.Series:
    """For each cluster, identify the dominant archetype.

    Parameters
    ----------
    confusion : pd.DataFrame
        Confusion matrix from ClusterEvaluation.

    Returns
    -------
    pd.Series
        Dominant archetype name per cluster.
    """
    return confusion.idxmax(axis=1).rename("dominant_archetype")
