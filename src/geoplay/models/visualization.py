"""Visualization helpers for clustering results.

Three main visualization types:

1. PCA scatter plots: project the feature space to 2D using PCA (a linear
   projection with interpretable axes). Points are colored by cluster with
   centroids marked. Clusters appear as overlapping clouds, which reflects
   the true geometry more faithfully than UMAP for closely-related groups.

2. UMAP scatter plots: project to 2D using UMAP (a non-linear method that
   exaggerates separation between groups for visibility). Useful to confirm
   that distinct groups exist, but tends to render clusters as isolated
   islands even when they partially overlap in the original space.

3. Confusion matrix heatmaps: visualize cluster-archetype overlap.

Both PCA and UMAP are used ONLY for visualization. The clustering itself is
performed on the full 54-dimensional standardized feature space.

Showing both projections side by side is deliberate: PCA shows honest linear
structure (with overlap), UMAP shows group separability. Neither alone tells
the whole story.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA


def compute_umap_projection(
    X: npt.NDArray[np.float64],
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> npt.NDArray[np.float64]:
    """Project a high-dimensional feature matrix to 2D via UMAP.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix of shape (n_samples, n_features).
    n_neighbors : int
        Local connectivity. Higher values preserve global structure,
        lower values preserve local structure. 30 is a balanced default
        for ~50k points.
    min_dist : float
        Minimum distance between embedded points. Lower = tighter clusters.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        2D embedding of shape (n_samples, 2).
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1,  # required for deterministic output with random_state
    )
    return reducer.fit_transform(X)


def compute_pca_projection(
    X: npt.NDArray[np.float64],
    n_components: int = 2,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Project a high-dimensional feature matrix to 2D via PCA.

    Unlike UMAP, PCA is a linear projection: the axes are real principal
    components with an associated explained-variance ratio. Clusters appear
    as overlapping clouds rather than isolated islands, which more faithfully
    represents the true geometry for closely-related groups.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix of shape (n_samples, n_features).
    n_components : int
        Number of principal components to keep (2 for scatter plots).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (coords, explained_variance_ratio): the 2D projection of shape
        (n_samples, n_components), and the fraction of variance explained
        by each component (shape (n_components,)).
    """
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def plot_pca_scatter_with_centroids(
    coords: npt.NDArray[np.float64],
    labels: npt.NDArray | pd.Series,
    explained_variance: npt.NDArray[np.float64],
    title: str,
    output_path: Path,
    point_size: float = 4.0,
    alpha: float = 0.4,
    figsize: tuple[float, float] = (11, 8),
) -> None:
    """Plot a PCA scatter (PC1 vs PC2) colored by label, with centroids.

    Each cluster is drawn as a translucent cloud of points, with its centroid
    marked as a large black dot. Noise points (label -1) are drawn in gray
    without a centroid. The axes report the explained-variance ratio.

    Parameters
    ----------
    coords : np.ndarray
        2D PCA coordinates from compute_pca_projection.
    labels : np.ndarray or pd.Series
        Cluster labels (with -1 for noise) or archetype names.
    explained_variance : np.ndarray
        Explained-variance ratio per component (from compute_pca_projection).
    title : str
        Plot title.
    output_path : Path
        Where to save the PNG.
    point_size : float
        Marker size for data points.
    alpha : float
        Marker transparency.
    figsize : tuple[float, float]
        Figure size in inches.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    labels_array = np.asarray(labels)
    unique_labels = sorted(set(labels_array), key=lambda x: (str(x) == "-1", str(x)))
    palette = sns.color_palette("tab10", n_colors=max(len(unique_labels), 10))

    for idx, label in enumerate(unique_labels):
        mask = labels_array == label
        is_noise = str(label) == "-1"
        color = "lightgray" if is_noise else palette[idx % 10]
        label_str = "noise" if is_noise else str(label)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            color=color,
            label=f"{label_str} (n={int(mask.sum()):,})",
            edgecolors="none",
        )
        # Draw centroid for real clusters (not noise).
        if not is_noise:
            cx = coords[mask, 0].mean()
            cy = coords[mask, 1].mean()
            ax.scatter(
                cx,
                cy,
                s=200,
                color="black",
                edgecolors="white",
                linewidths=1.5,
                zorder=10,
            )

    var1 = explained_variance[0] * 100
    var2 = explained_variance[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        markerscale=4,
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap_scatter(
    coords: npt.NDArray[np.float64],
    labels: npt.NDArray | pd.Series,
    title: str,
    output_path: Path,
    palette: str = "tab10",
    point_size: float = 2.0,
    alpha: float = 0.5,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Plot a UMAP scatter plot colored by labels and save to disk.

    Parameters
    ----------
    coords : np.ndarray
        2D coordinates from compute_umap_projection.
    labels : np.ndarray or pd.Series
        Labels to color points by. Can be cluster ids (with -1 for noise)
        or archetype names (strings).
    title : str
        Plot title.
    output_path : Path
        Where to save the PNG file.
    palette : str
        Seaborn color palette name.
    point_size : float
        Marker size.
    alpha : float
        Marker transparency.
    figsize : tuple[float, float]
        Figure size in inches.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    labels_array = np.asarray(labels)
    unique_labels = sorted(set(labels_array), key=lambda x: (str(x) == "-1", str(x)))

    colors = sns.color_palette(palette, n_colors=len(unique_labels))

    for label, color in zip(unique_labels, colors, strict=True):
        mask = labels_array == label
        label_str = "noise" if str(label) == "-1" else str(label)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[color],
            label=f"{label_str} (n={mask.sum():,})",
            edgecolors="none",
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        markerscale=4,
        fontsize=9,
        frameon=True,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_heatmap(
    confusion: pd.DataFrame,
    title: str,
    output_path: Path,
    figsize: tuple[float, float] = (10, 6),
    normalize: bool = True,
) -> None:
    """Plot a confusion matrix heatmap (cluster vs archetype).

    Parameters
    ----------
    confusion : pd.DataFrame
        Confusion matrix from evaluation. Rows are clusters.
    title : str
        Plot title.
    output_path : Path
        Where to save the PNG file.
    figsize : tuple[float, float]
        Figure size in inches.
    normalize : bool
        If True, normalize counts so each row sums to 1.0 (cluster purity).
        If False, show raw counts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        # Row-normalize: each cluster row sums to 1.0.
        row_sums = confusion.sum(axis=1).replace(0, 1)
        data = confusion.div(row_sums, axis=0)
        fmt = ".2f"
        cbar_label = "Proportion within cluster"
    else:
        data = confusion
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar_kws={"label": cbar_label},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("True archetype")
    ax.set_ylabel("Predicted cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side_umap(
    coords: npt.NDArray[np.float64],
    predicted_labels: npt.NDArray,
    true_labels: npt.NDArray | pd.Series,
    algorithm: str,
    output_path: Path,
    figsize: tuple[float, float] = (16, 7),
) -> None:
    """Side-by-side UMAP plot: predicted clusters vs true archetypes.

    The most informative visualization for clustering portfolios: it lets
    the viewer immediately compare whether the predicted structure
    matches the known structure.

    Parameters
    ----------
    coords : np.ndarray
        2D UMAP coordinates.
    predicted_labels : np.ndarray
        Cluster labels from the algorithm.
    true_labels : np.ndarray or pd.Series
        Ground truth archetype labels.
    algorithm : str
        Name of the algorithm (for the title).
    output_path : Path
        Where to save the PNG.
    figsize : tuple[float, float]
        Figure size in inches.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    predicted_array = np.asarray(predicted_labels)
    true_array = np.asarray(true_labels)

    # Left: predicted clusters.
    for label in sorted(set(predicted_array), key=lambda x: (x == -1, x)):
        mask = predicted_array == label
        label_str = "noise" if label == -1 else f"cluster {label}"
        ax_left.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2.0,
            alpha=0.5,
            label=f"{label_str} (n={mask.sum():,})",
            edgecolors="none",
        )
    ax_left.set_title(f"{algorithm} predicted clusters", fontsize=12)
    ax_left.set_xlabel("UMAP dimension 1")
    ax_left.set_ylabel("UMAP dimension 2")
    ax_left.legend(loc="best", markerscale=4, fontsize=8, frameon=True)

    # Right: true archetypes.
    for label in sorted(set(true_array)):
        mask = true_array == label
        ax_right.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2.0,
            alpha=0.5,
            label=f"{label} (n={mask.sum():,})",
            edgecolors="none",
        )
    ax_right.set_title("True archetypes (ground truth)", fontsize=12)
    ax_right.set_xlabel("UMAP dimension 1")
    ax_right.set_ylabel("UMAP dimension 2")
    ax_right.legend(loc="best", markerscale=4, fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
