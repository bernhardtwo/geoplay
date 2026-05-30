<div align="center">

# **GEOPLAY RECOMMENDER**

### *Location-based recommendation system from synthetic geospatial data*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-2C7BBF?style=flat-square)](https://lightgbm.readthedocs.io/)
[![H3](https://img.shields.io/badge/H3-4.4-7B68EE?style=flat-square)](https://h3geo.org/)
[![Ruff](https://img.shields.io/badge/Ruff-strict-D7FF00?style=flat-square)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/)

</div>

---

## Overview

End-to-end machine learning system for a synthetic location-based game (Pokémon GO style). The pipeline generates realistic player behavior data, segments players into behavioral archetypes via clustering, and ranks geographic hexes (H3) by visit probability using learning-to-rank.

Built as a portfolio project demonstrating production-grade ML engineering: streaming feature extraction over hundreds of millions of events, leak-free temporal splits, industrial ranking objectives, hyperparameter tuning, and honest analysis of model behavior.

## Pipeline status

| # | Phase | Status |
|---|---|---|
| 1 | Project scaffolding (uv, ruff, mypy, pre-commit, CI) | ✅ Done |
| 2 | Synthetic data generation (146M events, 9 archetypes) | ✅ Done |
| 3 | Exploratory data analysis | ✅ Done |
| 4 | Feature engineering (54 features, streaming H3) | ✅ Done |
| 5 | Player segmentation (HDBSCAN / KMeans / GMM) | ✅ Done |
| 6 | LightGBM ranker (LambdaRank, NDCG@10) | ✅ Done |
| 7 | MLflow experiment tracking | ⏳ Planned |
| 8 | FastAPI serving layer | ⏳ Planned |
| 9 | CI/CD with GitHub Actions | ⏳ Planned |
| 10 | Containerization with Docker | ⏳ Planned |

**Progress: 6 / 10 phases.**

## Key results

### Clustering (player segmentation)

Three algorithms on 50,000 players × 54 features, 9 behavioral archetypes:

| Algorithm | ARI | NMI | Clusters found |
|-----------|-----|-----|----------------|
| **HDBSCAN** | **0.8640** | **0.9546** | 8 (discovered automatically) |
| KMeans (k=9) | 0.8028 | 0.9094 | 9 (forced) |
| Gaussian Mixture (k=9) | 0.5745 | 0.7919 | 9 (forced) |

HDBSCAN wins by discovering 8 clusters and merging two archetypes (`casual_evening` and `weekend_homebody`) that the synthetic generator designed to be nearly indistinguishable. Full analysis with PCA dimensionality study (6 components capture 92% of variance) in [`docs/clustering_analysis.md`](docs/clustering_analysis.md).

### Ranking (hex recommendation)

LightGBM Ranker with LambdaRank objective. Evaluated on 566,297 test queries:

| Metric | Model | Random | Lift |
|--------|-------|--------|------|
| NDCG@10 | **0.6337** | 0.2675 | +137% |
| NDCG@20 | 0.7195 | 0.3747 | +92% |
| MAP | 0.5927 | 0.2711 | +119% |
| MRR | **0.8679** | 0.3866 | +124% |
| Precision@5 | 0.5447 | 0.1864 | +192% |

MRR of 0.87 means the first relevant hex is on average at position 1.15 in the ranked list. Full analysis with feature engineering decisions, hyperparameter tuning results, and the plateau diagnosis in [`docs/ranking_analysis.md`](docs/ranking_analysis.md).

## Architecture

```
geoplay/
├── data/
│   ├── raw/                         # Synthetic players + 50 event partitions
│   └── processed/                   # Features, clusters, ranking outputs
├── docs/
│   ├── clustering_analysis.md       # 9-archetype clustering with PCA + UMAP
│   ├── ranking_analysis.md          # LightGBM ranker analysis
│   └── figures/                     # 10 visualizations (PCA + UMAP + confusion)
├── src/geoplay/
│   ├── data/                        # Synthetic data generation
│   │   ├── archetypes.py            # 9 behavioral archetypes
│   │   ├── geography.py             # Haversine + bounded random points
│   │   ├── temporal.py              # Hour-of-day modeling, atypical days
│   │   └── events.py                # Event simulation
│   ├── features/
│   │   ├── h3_utils.py              # H3 vectorized helpers
│   │   ├── temporal_features.py     # Per-hour, per-day patterns
│   │   ├── spatial_features.py      # Hex entropy, movement radius
│   │   └── pipeline.py              # Streaming orchestrator
│   ├── models/
│   │   ├── clustering.py            # HDBSCAN + KMeans + GMM
│   │   ├── evaluation.py            # ARI, NMI, V-measure, confusion
│   │   ├── visualization.py         # PCA + UMAP scatter
│   │   └── pipeline.py              # Clustering pipeline
│   └── ranking/                     # ← Phase 6
│       ├── dataset.py               # (query, hex, label) pairs
│       ├── player_features.py       # 10 query-side features
│       ├── hex_features.py          # 3 item-side features
│       ├── pair_features.py         # 2 player-hex features
│       ├── features.py              # Final feature assembly
│       ├── evaluation.py            # NDCG, MAP, MRR, Precision@K
│       ├── model.py                 # LGBMRanker wrapper
│       └── tuning.py                # Random search
└── tests/                           # pytest unit tests
```

## Data flow

```
players.parquet (50k)            archetypes (9)
        │                              │
        ▼                              │
events.parquet (146M, 50 parts)        │
        │                              │
        ├─────► feature pipeline ──────┴─►  features.parquet (50k × 54)
        │                                          │
        │                                          ▼
        │                                  clustering pipeline
        │                                          │
        │                                          ▼
        │                                  clusters.parquet + metrics
        │
        └─────► ranking pipeline (Phase 6)
                    │
                    ├─► train.parquet / test.parquet (raw pairs)
                    ├─► player/hex/pair features (train-cutoff only)
                    ├─► train_enriched / test_enriched (17 features)
                    ├─► tuning (random search, 20 trials)
                    └─► model.txt + metrics.json + feature_importance.csv
```

## Quick start

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- ~16 GB RAM (24 GB recommended)
- ~5 GB disk for raw events + processed outputs

### Install

```bash
git clone git@github.com:bernhardtwo/geoplay.git
cd geoplay
uv sync
```

### Run the pipelines

Generate synthetic data (180 days, 50k players, 146M events):

```bash
uv run python -m geoplay.data.generate \
    --n-players 50000 \
    --n-days 180 \
    --output-dir data/raw
```

Build features (streaming over 50 partitions):

```bash
uv run python -m geoplay.features.pipeline \
    --events-dir data/raw/events \
    --output data/processed/features.parquet
```

Run clustering:

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

Run ranking pipeline (each module invoked in sequence — see [`docs/ranking_analysis.md`](docs/ranking_analysis.md) for the full command list).

### Development

```bash
# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type-check
uv run mypy src/

# Run tests
uv run pytest -v

# Pre-commit hooks
uv run pre-commit run --all-files
```

## Engineering principles

This project intentionally demonstrates production practices that distinguish ML engineering from notebook experimentation:

- **Streaming feature extraction over partitioned data.** Processing 146M events without loading them all into memory; per-partition aggregates merged incrementally.
- **Leak-free temporal splits.** Player features for the ranker are computed strictly over the train period, not over the full timeline (which would leak future activity counts into training).
- **Hard negative mining.** Ranking negatives are drawn from each player's visited-hex universe, not random global hexes. This forces the model to learn real discriminative patterns, not trivial geographic filters.
- **Hyperparameter tuning with realistic budgets.** Random search on a stratified subsample, not exhaustive CV. The actual industrial choice for large datasets.
- **Honest analysis of model behavior.** The ranking documentation discusses the plateau the model hits at NDCG@10 ≈ 0.63 and explains the three contributing causes (irreducible noise, feature saturation, cap effects). No score-chasing.
- **Discarded features documented.** Two features were implemented, evaluated, and removed when found uninformative. Both removals are explained in the module docstrings so future readers understand the design choices.

## Technical highlights

- **Synthetic data generator** with 9 archetypes designed to overlap deliberately, plus injected noise (8% atypical days, σ=2.5 on hour distribution). The result is a clustering task with realistic ARI in [0.5, 0.9], not trivially perfect separation.
- **H3 spatial indexing** at resolution 8 (~0.74 km² hexes). Vectorized lat/lon → hex conversion.
- **PCA + UMAP dual visualization** for clusters, with explicit explanation in docs about why both are shown (PCA reflects honest overlap; UMAP exaggerates separation).
- **LambdaRank with industrial parameters** tuned via random search over 1,728-combination space.
- **17 carefully designed features** across query / item / pair / contextual families, with empirical distribution analysis per feature class.

## Documentation

- **[`docs/clustering_analysis.md`](docs/clustering_analysis.md)** — Detailed analysis of the player segmentation phase. Covers archetype design, noise model, PCA dimensionality study, algorithm comparison, and why HDBSCAN wins.
- **[`docs/ranking_analysis.md`](docs/ranking_analysis.md)** — Detailed analysis of the ranking phase. Covers problem formulation, feature engineering decisions (including features that were eliminated), tuning methodology, the plateau diagnosis, and feature importance.

## Roadmap

Phases 7-10 are planned and not yet implemented:

- **MLflow experiment tracking** — wrap training in MLflow runs for systematic experiment management
- **FastAPI serving layer** — expose `/rank` endpoint with cached model loading
- **CI/CD pipeline** — GitHub Actions for tests + lint + model regression on every push
- **Docker containerization** — Dockerfile + docker-compose for reproducible deployment

## License

MIT.

---

<div align="center">

*Built by [Bernardo Vega](https://github.com/bernhardtwo) — full-stack developer transitioning to ML engineering.*

</div>
