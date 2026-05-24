"""Clustering pipeline orchestration.

End-to-end pipeline that loads the feature matrix, fits all three clustering
algorithms, evaluates them against the ground truth archetypes, and produces
visualizations and metric exports.

Outputs:
    data/processed/clusters/clusters.parquet   - per-player cluster assignments
    data/processed/clusters/metrics.json       - evaluation metrics summary
    docs/figures/clusters_<algo>_side_by_side.png
    docs/figures/clusters_<algo>_confusion.png
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoplay.models.clustering import (
    ClusteringResult,
    fit_gaussian_mixture,
    fit_hdbscan,
    fit_kmeans,
    standardize_features,
)
from geoplay.models.evaluation import (
    ClusterEvaluation,
    cluster_dominant_archetype,
    cluster_purity,
    evaluate_clustering,
)
from geoplay.models.visualization import (
    compute_umap_projection,
    plot_confusion_heatmap,
    plot_side_by_side_umap,
)

# Columns in features.parquet that are NOT features (identifiers + labels).
NON_FEATURE_COLUMNS = ("player_id", "archetype")


@dataclass
class PipelineOutput:
    """Container for the full clustering pipeline output.

    Attributes
    ----------
    clusters : pd.DataFrame
        Per-player cluster assignments from all algorithms. Columns:
        player_id, archetype, hdbscan_cluster, kmeans_cluster, gmm_cluster.
    evaluations : dict[str, ClusterEvaluation]
        Evaluation result per algorithm.
    umap_coords : np.ndarray
        2D UMAP projection of the feature matrix (for visualizations).
    """

    clusters: pd.DataFrame
    evaluations: dict[str, ClusterEvaluation]
    umap_coords: np.ndarray


def _load_features(features_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the features Parquet into (full DataFrame, feature matrix, archetype array)."""
    features = pd.read_parquet(features_path)

    missing = set(NON_FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(
            f"features Parquet missing required columns: {missing}. "
            f"Got columns: {list(features.columns)[:5]}..."
        )

    feature_cols = [c for c in features.columns if c not in NON_FEATURE_COLUMNS]
    X = features[feature_cols].to_numpy(dtype=np.float64)
    archetypes = features["archetype"].to_numpy()

    return features, X, archetypes


def run_clustering_pipeline(
    features_path: Path,
    output_dir: Path,
    figures_dir: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineOutput:
    """Run the full clustering pipeline end-to-end.

    Parameters
    ----------
    features_path : Path
        Path to data/processed/features.parquet.
    output_dir : Path
        Where to write clusters.parquet and metrics.json.
    figures_dir : Path
        Where to write visualization PNGs.
    progress_callback : callable | None
        Optional callback(message) -> None for UI updates.

    Returns
    -------
    PipelineOutput
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if progress_callback is not None:
            progress_callback(msg)

    # 1. Load features.
    log("Loading features.parquet")
    features, X, archetypes = _load_features(features_path)
    log(f"  Loaded {len(features):,} players × {X.shape[1]} features")

    # 2. Standardize.
    log("Standardizing features")
    X_scaled, _ = standardize_features(X)

    # 3. Fit all three clustering algorithms.
    log("Fitting HDBSCAN")
    t0 = time.time()
    result_hdb = fit_hdbscan(X_scaled, min_cluster_size=500)
    log(
        f"  HDBSCAN done in {time.time()-t0:.1f}s "
        f"(n_clusters={result_hdb.n_clusters}, noise={result_hdb.noise_fraction:.1%})"
    )

    log("Fitting KMeans (k=5)")
    t0 = time.time()
    result_km = fit_kmeans(X_scaled, n_clusters=5)
    log(f"  KMeans done in {time.time()-t0:.1f}s (n_clusters={result_km.n_clusters})")

    log("Fitting GaussianMixture (k=5)")
    t0 = time.time()
    result_gmm = fit_gaussian_mixture(X_scaled, n_components=5)
    log(f"  GMM done in {time.time()-t0:.1f}s (n_clusters={result_gmm.n_clusters})")

    results: dict[str, ClusteringResult] = {
        "hdbscan": result_hdb,
        "kmeans": result_km,
        "gaussian_mixture": result_gmm,
    }

    # 4. Evaluate each.
    log("Evaluating clustering quality")
    evaluations: dict[str, ClusterEvaluation] = {}
    for algo_name, result in results.items():
        eval_result = evaluate_clustering(
            algorithm=algo_name,
            predicted_labels=result.labels,
            true_labels=archetypes,
            noise_fraction=result.noise_fraction,
        )
        evaluations[algo_name] = eval_result
        log(
            f"  {algo_name:20s}  ARI={eval_result.ari:.4f}  NMI={eval_result.nmi:.4f}  "
            f"V-measure={eval_result.v_measure:.4f}"
        )

    # 5. Build the per-player output DataFrame.
    clusters = features[["player_id", "archetype"]].copy()
    clusters["hdbscan_cluster"] = result_hdb.labels
    clusters["kmeans_cluster"] = result_km.labels
    clusters["gmm_cluster"] = result_gmm.labels

    clusters_path = output_dir / "clusters.parquet"
    clusters.to_parquet(clusters_path, engine="pyarrow", index=False)
    log(f"  Wrote {clusters_path} ({clusters_path.stat().st_size / 1024**2:.2f} MB)")

    # 6. Compute UMAP projection (once, for all visualizations).
    log("Computing UMAP 2D projection")
    t0 = time.time()
    umap_coords = compute_umap_projection(X_scaled)
    log(f"  UMAP done in {time.time()-t0:.1f}s")

    # 7. Generate visualizations per algorithm.
    log("Generating visualizations")
    for algo_name, result in results.items():
        eval_result = evaluations[algo_name]

        side_by_side_path = figures_dir / f"clusters_{algo_name}_side_by_side.png"
        plot_side_by_side_umap(
            coords=umap_coords,
            predicted_labels=result.labels,
            true_labels=archetypes,
            algorithm=algo_name,
            output_path=side_by_side_path,
        )

        confusion_path = figures_dir / f"clusters_{algo_name}_confusion.png"
        plot_confusion_heatmap(
            confusion=eval_result.confusion,
            title=f"{algo_name}: cluster vs archetype (row-normalized)",
            output_path=confusion_path,
        )

        log(f"  {algo_name}: side_by_side.png + confusion.png written")

    # 8. Write metrics summary.
    metrics_summary = {
        "n_players": len(features),
        "n_features": X.shape[1],
        "n_true_archetypes": len(set(archetypes)),
        "algorithms": [eval_result.to_dict() for eval_result in evaluations.values()],
        "cluster_purities": {},
        "cluster_dominant_archetypes": {},
    }
    for algo_name, eval_result in evaluations.items():
        purities = cluster_purity(eval_result.confusion).round(4).to_dict()
        dominants = cluster_dominant_archetype(eval_result.confusion).to_dict()
        metrics_summary["cluster_purities"][algo_name] = {str(k): v for k, v in purities.items()}
        metrics_summary["cluster_dominant_archetypes"][algo_name] = {
            str(k): v for k, v in dominants.items()
        }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_summary, f, indent=2)
    log(f"  Wrote {metrics_path}")

    return PipelineOutput(
        clusters=clusters,
        evaluations=evaluations,
        umap_coords=umap_coords,
    )
