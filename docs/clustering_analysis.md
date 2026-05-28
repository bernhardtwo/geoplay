# Clustering Analysis

This document analyses the results of running three clustering algorithms
(HDBSCAN, KMeans, and Gaussian Mixture Model) on a 50,000-player feature
matrix. The dataset is designed to be realistic rather than trivially
separable: it contains nine behavioral archetypes, four of which are
deliberate "neighbors" that overlap with the others, plus injected noise
to mimic real player variability.

## Setup

- **Input:** `data/processed/features.parquet` — 50,000 players × 54
  standardized features across three families (temporal, spatial, behavioral).
- **Ground truth:** 9 behavioral archetypes assigned by the synthetic data
  generator. Five are well-separated behavioral extremes (commuter,
  casual_evening, weekend_explorer, hardcore_raider, lunch_player). Four are
  neighbors designed to overlap with the originals:
  - `lunch_commuter` — morning + lunch + evening peaks (overlaps commuter and lunch_player)
  - `weekend_homebody` — evening play plus weekend daytime (overlaps casual_evening and weekend_explorer)
  - `morning_player` — mornings only (overlaps commuter on the morning peak)
  - `night_owl` — late-night play into the small hours (overlaps casual_evening at 22-23h)
- **Algorithms:** HDBSCAN (no `k` specified), KMeans (`k=9`), Gaussian
  Mixture Model (`n_components=9`).
- **Preprocessing:** `StandardScaler` to zero mean and unit variance per
  feature. No dimensionality reduction is applied before clustering; PCA and
  UMAP are used only for visualization.

### Noise model

Two sources of noise make the archetypes realistically fuzzy rather than
robotically consistent:

1. **Intra-archetype dispersion.** The Gaussian bumps around each archetype's
   peak hours use a standard deviation of 2.5 hours (up from a tighter 1.5),
   so players within an archetype vary in exactly when they play.
2. **Atypical days.** Each active day has an 8% chance of being "atypical":
   the player abandons their archetype pattern and plays at uniformly random
   hours, simulating vacations, illness, or social disruptions.

## Summary of Results

| Algorithm | ARI    | NMI    | V-measure | Clusters found | Noise |
|-----------|--------|--------|-----------|----------------|-------|
| HDBSCAN   | 0.8640 | 0.9546 | 0.9546    | 8              | 0.1%  |
| KMeans    | 0.8028 | 0.9094 | 0.9094    | 9 (forced)     | 0.0%  |
| GMM       | 0.5745 | 0.7919 | 0.7919    | 9 (forced)     | 0.0%  |

ARI and NMI are both bounded between 0 and 1 (higher is better). Unlike a
trivially separable dataset where all algorithms score 1.0, here the three
algorithms separate clearly: HDBSCAN leads, KMeans follows, GMM trails.
The rest of this document explains why.

## Dimensionality: the data is lower-dimensional than it looks

Before discussing clustering, it is worth understanding the geometry of the
feature space. A PCA decomposition of the 54 standardized features shows that
the data has a low intrinsic dimensionality:

| Principal components | Cumulative variance explained |
|----------------------|-------------------------------|
| 2                    | 55.6%                         |
| 4                    | 80.7%                         |
| 6                    | 91.8%                         |
| 8                    | 96.0%                         |
| 16                   | 99.0%                         |

Six components capture almost 92% of the variance. The 54 features are highly
correlated — the 24 hour-density features alone encode only a few underlying
"time-of-day patterns" repeated with variation. This matters for two reasons:
it explains why clustering on the raw 54 dimensions works without dimensionality
reduction (the signal is concentrated), and it shapes how the results should be
visualized.

## A note on visualization: PCA vs UMAP

Cluster scatter plots can mislead if the projection method is not understood.
This project generates both PCA and UMAP projections deliberately, because
each tells a different and incomplete story.

**PCA** is a linear projection. Its axes are real principal components with an
associated variance ratio, and distances are preserved up to the projection.
Clusters appear as overlapping clouds, which honestly reflects that
neighboring archetypes share regions of feature space. The trade-off is that
the first two components capture only 55.6% of variance, so groups that are
separated along components 3-6 can look artificially merged in a 2D PCA plot.

**UMAP** is a non-linear method that optimizes for local neighborhood
preservation. It tends to render clusters as well-separated islands, even when
those clusters partially overlap in the original space, because it actively
pushes groups apart for visual clarity. The trade-off is that the axes are not
interpretable and global distances are distorted.

Neither projection is "the truth." A reader who sees only the UMAP plot might
conclude the clusters are cleaner than they are; a reader who sees only the
2D PCA plot might conclude they are messier. Showing both, with this caveat,
is the honest presentation. See `archetypes_ground_truth_pca.png` for the
PCA view of the true archetypes and `clusters_hdbscan_side_by_side.png` for
the UMAP view.

## HDBSCAN: the headline result

HDBSCAN achieves the best agreement (ARI=0.8640) **without being told how many
clusters to look for**. It discovers 8 clusters where the ground truth has 9.
The single "error" is informative: it merges two archetypes into one cluster.

| cluster | dominant archetype          | notes                          |
|---------|-----------------------------|--------------------------------|
| 0       | night_owl (2,417)           | clean                          |
| 1       | morning_player (3,586)      | clean                          |
| 2       | hardcore_raider (3,437)     | clean                          |
| 3       | weekend_explorer (5,111)    | clean                          |
| 4       | casual_evening + weekend_homebody | **merged** (11,023 + 3,995) |
| 5       | lunch_player (6,401)        | clean                          |
| 6       | commuter (9,011)            | clean                          |
| 7       | lunch_commuter (4,948)      | clean                          |
| -1      | noise (71)                  | mostly ambiguous lunch_commuter |

The merge of casual_evening and weekend_homebody is the correct call given how
similar they are. Their feature means are nearly identical: hour_concentration
0.71 vs 0.56, weekday_ratio 0.70 vs 0.65, evening_ratio 0.69 vs 0.61, and
distance_from_home 2.96 vs 3.07. There is no density valley between them, so
HDBSCAN treats them as one group rather than inventing a boundary.

Notably, HDBSCAN successfully separates commuter from lunch_commuter and from
morning_player — the other neighbor pairs we worried might collapse. It also
flags 71 ambiguous players as noise instead of forcing them into a cluster.
In a real setting where the number of segments is unknown, automatically
discovering the right number and refusing to over-split is a major advantage.

## KMeans: the cost of a fixed k

KMeans (ARI=0.8028) is forced to produce exactly 9 clusters. Because
casual_evening and weekend_homebody do not form two natural groups, KMeans
cannot merge them — but it still needs 9 clusters to fill. Its solution is to
split both archetypes and scatter them across two mixed clusters:

| cluster | composition                              |
|---------|------------------------------------------|
| 2       | casual_evening 5,690 + weekend_homebody 2,060 |
| 7       | casual_evening 5,333 + weekend_homebody 1,935 |

Rather than cleanly merging the two similar archetypes (as HDBSCAN does) or
keeping them separate, KMeans carves them into two arbitrary partitions. The
remaining seven archetypes are recovered cleanly. This illustrates the cost of
committing to a fixed `k` when the true structure does not match it.

## GMM: the weakest performer

GMM (ARI=0.5745) fares worst, with two distinct failure modes occurring at once:

- **Over-merging:** cluster 0 absorbs three evening-type archetypes —
  casual_evening (11,023), night_owl (2,417), and weekend_homebody (3,995) —
  into a single component.
- **Over-splitting:** commuter is split across two clusters and mixed with
  lunch_commuter; lunch_player is fragmented into two pieces; weekend_explorer
  is fragmented into two pieces.

GMM assumes each cluster is a Gaussian with a particular covariance structure.
When two archetypes overlap, the expectation-maximization objective (maximize
data likelihood) prefers a single broad Gaussian over two thinner ones, which
causes over-merging. Simultaneously, archetypes with internal substructure get
split into multiple Gaussians. The algorithm optimizes likelihood, not
agreement with the ground-truth labels, and the mismatch shows.

## What this tells us about algorithm selection

A common assumption is that more flexible models are always better. These
results are a counterexample, and the ranking is instructive.

- **HDBSCAN** makes no assumption about the number or shape of clusters. It
  wins by discovering the right number of groups and merging only what is
  genuinely inseparable.
- **KMeans** is simple and assumes spherical clusters of a known count. It
  does well on the seven clean archetypes but is forced into artificial
  partitions for the ambiguous pair.
- **GMM** is the most parametric of the three. Its Gaussian assumption is the
  worst fit for this data, producing both over-merging and over-splitting.

Model selection should match the structure of the data, not the sophistication
of the algorithm. Here, the least assumption-laden method wins.

## Recommendation for production

For player segmentation in this setting, HDBSCAN is recommended:

1. It does not require committing to a number of clusters in advance, which
   matters when the segment count may shift as the population grows.
2. It refuses to over-split, merging archetypes that are genuinely
   indistinguishable instead of inventing boundaries.
3. It flags ambiguous players as noise, which is operationally useful for
   identifying users who do not fit any clean segment.

KMeans is a reasonable fallback when the segment count is known and a simpler,
more interpretable model is preferred. GMM is not recommended here unless soft
(probabilistic) assignments are specifically required, in which case its
failure modes on overlapping groups must be mitigated.

## Reproducibility

All results in this document can be reproduced by running the pipeline module
directly:

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

This regenerates `clusters.parquet`, `metrics.json`, and all figures. A
wrapper CLI command (`geoplay cluster fit`) is on the roadmap.

## Figures

PCA scatter plots (interpretable axes, overlapping clouds, centroids marked):
- `docs/figures/archetypes_ground_truth_pca.png`
- `docs/figures/clusters_hdbscan_pca.png`
- `docs/figures/clusters_kmeans_pca.png`
- `docs/figures/clusters_gaussian_mixture_pca.png`

UMAP scatter plots (predicted vs ground truth, group separability):
- `docs/figures/clusters_hdbscan_side_by_side.png`
- `docs/figures/clusters_kmeans_side_by_side.png`
- `docs/figures/clusters_gaussian_mixture_side_by_side.png`

Confusion matrix heatmaps (row-normalized):
- `docs/figures/clusters_hdbscan_confusion.png`
- `docs/figures/clusters_kmeans_confusion.png`
- `docs/figures/clusters_gaussian_mixture_confusion.png`
