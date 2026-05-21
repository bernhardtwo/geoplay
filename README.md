# GeoPlay

> Geo-contextual player segmentation and content recommendation system for location-based mobile games.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Overview

GeoPlay is a production-grade machine learning system that personalizes content delivery in location-based mobile games (e.g., Pokémon GO-style experiences). It combines unsupervised player segmentation with contextual ranking to decide what content to show, where, and when based on player behavior, geospatial context, and temporal patterns.

This is a portfolio project designed to demonstrate end-to-end ML engineering: data generation, feature engineering with geospatial indexing (H3), clustering (HDBSCAN), gradient-boosted ranking (LightGBM), experiment tracking (MLflow), and production serving (FastAPI).

## Architecture

The system pipeline:

1. **Player identification** — HDBSCAN clustering produces a behavioral segment (e.g., "nocturnal explorer", "weekend social").
2. **Spatial context** — H3 hexagonal indexing converts raw lat/lon into zones with semantic properties.
3. **Temporal context** — Cyclical features (sin/cos of hour and day) capture time-of-day and day-of-week patterns.
4. **Content ranking** — LightGBM ranks candidate content items given the player segment and context.
5. **Feedback loop** — Player interactions update the model.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Package manager | uv |
| Data | NumPy, pandas, PyArrow |
| Geospatial | H3 (Uber's hexagonal indexing) |
| ML | scikit-learn, HDBSCAN, LightGBM |
| Experiment tracking | MLflow |
| Serving | FastAPI + Uvicorn |
| CLI | Typer + Rich |
| Configuration | Pydantic Settings |
| Logging | structlog |
| Quality | Ruff, mypy (strict), pytest |

## Project Status

Work in progress. Roadmap:

- [x] Project scaffolding and tooling setup
- [ ] Synthetic data generation
- [ ] Exploratory data analysis
- [ ] Feature engineering (H3 + temporal)
- [ ] Player segmentation with HDBSCAN
- [ ] Content ranking with LightGBM
- [ ] MLflow experiment tracking
- [ ] FastAPI serving layer
- [ ] CI/CD with GitHub Actions
- [ ] Containerization with Docker

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
git clone git@github.com:bernhardtwo/geoplay.git
cd geoplay
uv sync --all-extras
```

### Running tests

```bash
uv run pytest
```

### Linting and type checking

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
```

## Project Structure

The repository follows a src-layout with separation by domain:

- `src/geoplay/config/` — Pydantic Settings
- `src/geoplay/data/` — Data generation and loading
- `src/geoplay/features/` — H3, temporal, behavioral features
- `src/geoplay/models/` — Clustering and ranking models
- `src/geoplay/api/` — FastAPI endpoints
- `src/geoplay/cli/` — Typer CLI
- `src/geoplay/utils/` — Shared utilities
- `tests/unit/` — Unit tests
- `tests/integration/` — Integration tests
- `data/` — Datasets (gitignored)
- `notebooks/` — Exploratory analysis
- `docs/` — Documentation

## Why This Project?

Real-world ML systems are not Jupyter notebooks. They require reproducibility through locked dependencies and deterministic data pipelines, observability via structured logging and experiment tracking, quality enforced by type checking, testing, and linting, and serving layers that scale beyond a notebook cell.

This project intentionally adopts production patterns from day one.

## License

MIT — see [LICENSE](LICENSE) for details.

## Author

**Bernardo Vega** — Software developer based in Hermosillo, Mexico.

GitHub: [@bernhardtwo](https://github.com/bernhardtwo)
