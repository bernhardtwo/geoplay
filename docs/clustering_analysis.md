# Clustering Analysis

This document analyses the results of running three clustering algorithms
(HDBSCAN, KMeans, and Gaussian Mixture Model) on the 50,000-player feature
matrix produced by the feature engineering pipeline. The analysis covers
both quantitative metrics and qualitative interpretation of failure modes.

## Setup

- **Input:** `data/processed/features.parquet` — 50,000 players × 54
  standardized features across three families (temporal, spatial, behavioral).
- **Ground truth:** 5 behavioral archetypes assigned by the synthetic data
  generator (commuter, casual_evening, hardcore_raider, lunch_player,
  weekend_explorer).
- **Algorithms:** HDBSCAN (no `k` specified), KMeans (`k=5`), Gaussian
  Mixture Model (`n_components=5`).
- **Preprocessing:** `StandardScaler` to zero mean and unit variance per
  feature. No dimensionality reduction (UMAP is used only for 2D
  visualization, not for clustering itself).

## Summary of Results

| Algorithm | ARI    | NMI    | V-measure | Noise fraction | Clusters found |
|-----------|--------|--------|-----------|----------------|----------------|
| HDBSCAN   | 1.0000 | 1.0000 | 1.0000    | 0.0%           | 5              |
| KMeans    | 1.0000 | 1.0000 | 1.0000    | 0.0%           | 5              |
| GMM       | 0.7915 | 0.8825 | 0.8825    | 0.0%           | 5              |

ARI and NMI are both bounded between 0 and 1 (higher is better). HDBSCAN
and KMeans achieve perfect agreement with the ground truth. GMM does not.

The remainder of this document focuses on **why GMM underperforms** despite
being a more flexible model than KMeans, and what the failure tells us
about the structure of the data.

## HDBSCAN: the headline result

HDBSCAN recovers all five archetypes perfectly **without being told how
many clusters to look for**. The confusion matrix:

| cluster | casual_evening | commuter | hardcore_raider | lunch_player | weekend_explorer |
|---------|----------------|----------|-----------------|--------------|------------------|
| 0       | 0              | 12,573   | 0               | 0            | 0                |
| 1       | 0              | 0        | 0               | 9,998        | 0                |
| 2       | 0              | 0        | 0               | 0            | 7,427            |
| 3       | 14,989         | 0        | 0               | 0            | 0                |
| 4       | 0              | 0        | 5,013           | 0            | 0                |

Each predicted cluster contains exactly one archetype, with zero noise
points and zero confusion. This is a strong validation of the feature
engineering pipeline: the 54-dimensional feature space cleanly separates
the five archetypes into density-distinct regions.

See `docs/figures/clusters_hdbscan_side_by_side.png` for the UMAP
visualization (predicted clusters on the left, ground truth on the right —
the two are visually identical).

In a real-world scenario where the number of archetypes is unknown,
HDBSCAN's ability to discover the correct `k` automatically is a
significant operational advantage over KMeans or GMM.

## KMeans: a strong baseline

KMeans with `k=5` also achieves perfect agreement. This is expected:
with the correct number of clusters specified, and with feature values
that produce well-separated centroids, KMeans converges to the ground
truth structure on the first attempt.

KMeans depends on the user supplying `k`. In production this would
typically be chosen via silhouette analysis or the elbow method —
extra steps that HDBSCAN does not require.

## GMM: where things get interesting

GMM is the only algorithm that fails to recover the ground truth.
Its confusion matrix:

| cluster | casual_evening | commuter | hardcore_raider | lunch_player | weekend_explorer |
|---------|----------------|----------|-----------------|--------------|------------------|
| 0       | 0              | 12,573   | 0               | 0            | 0                |
| 1       | 0              | 0        | 0               | 0            | 7,427            |
| 2       | 0              | 0        | 0               | 3,242        | 0                |
| 3       | **14,989**     | 0        | **5,013**       | 0            | 0                |
| 4       | 0              | 0        | 0               | **6,756**    | 0                |

Two failures stand out:

### Failure 1: lunch_player split across two clusters

GMM places 3,242 lunch_players in cluster 2 and 6,756 in cluster 4.
The split is not random — these are likely two sub-populations within
lunch_player (e.g., players who consistently log events around 12h vs
players around 14h). The synthetic generator samples each player's peak
hour from a Gaussian centered at 13h, which naturally produces a bimodal
distribution when sliced finely enough. GMM, fitting Gaussian components,
identifies these two sub-modes as separate clusters.

This is a case of GMM **over-resolving** the data. The model is technically
correct in noticing the substructure, but it does so at the cost of
splitting a population we want to keep together.

### Failure 2: casual_evening and hardcore_raider merged into one cluster

The more serious failure: 14,989 casual_evening players and 5,013
hardcore_raider players end up together in cluster 3. These are two
distinct archetypes that GMM cannot separate.

Why does this happen? Both archetypes overlap on several feature axes:

| Feature                       | casual_evening | hardcore_raider |
|-------------------------------|----------------|-----------------|
| hour_concentration            | 0.89           | 0.60            |
| weekday_ratio                 | 0.70           | 0.71            |
| evening_ratio                 | 0.90           | 0.63            |
| distance_from_home_mean       | 2.97           | 5.28            |
| mean_events_per_session       | 2.28           | 10.24           |
| mean_session_duration_minutes | 15.88          | 68.44           |
| active_days_ratio             | 0.71           | 0.98            |

casual_evening and hardcore_raider have similar `weekday_ratio` and
`evening_ratio` values, but they differ sharply on session structure
(`mean_events_per_session` is 4.5× higher for hardcore_raider) and
session duration (~4× higher). HDBSCAN and KMeans both pick this up.
GMM does not — at least not with the freedom it had during fitting.

The underlying cause is GMM's parametric assumption: it assumes each
cluster is a Gaussian distribution with a particular covariance
structure. In `covariance_type="full"`, each component can have an
arbitrary elliptical shape, which is the most flexible variant.
But two overlapping ellipses can still be merged if a single larger
ellipse explains the combined density better according to the
likelihood-maximization criterion that drives EM.

In other words, GMM optimized a different objective (data likelihood)
than the one we care about (matching ground truth labels). When the
two ellipses overlap enough in the projection onto certain features,
GMM prefers a single fat Gaussian over two thinner ones.

## What this tells us about algorithm selection

A common assumption in machine learning is that "more flexible models
are always better." This dataset is a counterexample.

- **KMeans** is the simplest of the three, with the most restrictive
  assumption (spherical clusters). On this dataset, it works perfectly.
- **HDBSCAN** is non-parametric and makes no assumption about cluster
  shape. It also works perfectly, with the bonus of discovering `k`
  automatically.
- **GMM** is more flexible than KMeans (allows arbitrary elliptical
  shapes) and more constrained than HDBSCAN (assumes Gaussianity).
  It fails because the ground truth structure is not well-described
  by Gaussians.

The lesson: model selection should match the structure of the data,
not the prestige of the algorithm. A simpler model that matches the
assumptions of the problem can outperform a more complex one whose
assumptions are violated.

## Recommendation for production

For player segmentation in this setting, HDBSCAN is the recommended
algorithm:

1. It recovers the ground truth structure exactly when one exists.
2. It does not require specifying the number of clusters in advance,
   which matters when archetypes might shift over time as the user
   population grows.
3. It identifies noise points (players that do not fit any cluster),
   which is operationally useful for flagging atypical behavior.

KMeans is a viable fallback if `k` is known with confidence and if the
team prefers a simpler model that is easier to interpret to non-technical
stakeholders.

GMM is not recommended unless a downstream task benefits from soft
cluster assignments (probabilities rather than hard labels), in which
case its failure modes on this dataset should be documented and
mitigated through feature engineering or model regularization.

## Reproducibility

All results in this document can be reproduced by running the pipeline
module directly:

```bash
uv run python -c "
from pathlib import Path
from geoplay.models.pipeline import run_clustering_pipeline

run_clustering_pipeline(
    features_path=Path('data/processed/features.parquet'),
    output_dir=Path('data/processed/clusters'),
    figures_dir=Path('docs/figures'),
    progress_callback=print,
)
"
```

This regenerates `clusters.parquet`, `metrics.json`, and the figures
referenced above. A wrapper CLI command (`geoplay cluster fit`) is on
the roadmap.

## Figures

- `docs/figures/clusters_hdbscan_side_by_side.png`
- `docs/figures/clusters_hdbscan_confusion.png`
- `docs/figures/clusters_kmeans_side_by_side.png`
- `docs/figures/clusters_kmeans_confusion.png`
- `docs/figures/clusters_gaussian_mixture_side_by_side.png`
- `docs/figures/clusters_gaussian_mixture_confusion.png`
