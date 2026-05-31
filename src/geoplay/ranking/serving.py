"""Serving layer for the ranking model.

The serving layer answers a request (player_id, day_of_week, period) by
ranking the hexes the player already knows. It deliberately does not score
hexes the player has never visited: every training pair had at least one
prior visit (negatives are sampled from the player's own universe), so a
zero-visit hex is outside the model's training distribution. This service
re-ranks the known universe; discovery of new hexes is a different model.

Parts:

- prepare_serving_artifacts: a one-time build step. The player, hex, and
  pair features do not depend on the query window, so they are all present
  in train_enriched. We deduplicate it to one row per (player, hex), store
  the period categories (their order drives LightGBM's categorical codes),
  and register the final model in the MLflow Model Registry.
- export_model: dumps the registered model to a self-contained directory so
  a container can serve it without the registry (mlflow.db, mlartifacts).
- RankingService: loads the artifacts once and serves ranking requests. The
  model source is configurable: a local path (production / container) or,
  when omitted, the latest registry version (local development).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENT = "geoplay-ranking"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_REGISTERED_MODEL = "geoplay-ranker"
CONTEXTUAL_FEATURES = ("day_of_week", "period")


def _load_feature_columns(model_params_path: Path) -> tuple[list[str], list[str]]:
    """Read feature and categorical column names from a saved params.json."""
    with model_params_path.open() as f:
        data: dict[str, Any] = json.load(f)
    feature_names = [str(c) for c in data["feature_names"]]
    categorical = [str(c) for c in data["categorical_features"]]
    return feature_names, categorical


def _latest_version(client: MlflowClient, registered_model_name: str) -> str:
    """Return the highest registered version number as a string."""
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions for '{registered_model_name}'. "
            "Run prepare_serving_artifacts first."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return str(latest.version)


def prepare_serving_artifacts(
    train_enriched_path: Path,
    model_params_path: Path,
    output_dir: Path,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL,
) -> None:
    """Build the artifacts the serving layer needs and register the model.

    Writes to output_dir:
      serving_features.parquet  one row per (player, hex) with the 15
                                query-window-independent features
      serving_meta.json         feature column order, categorical columns,
                                and the period categories in training order

    Then registers the final_model run's logged model in the MLflow Model
    Registry under registered_model_name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns, categorical_columns = _load_feature_columns(model_params_path)
    static_features = [c for c in feature_columns if c not in CONTEXTUAL_FEATURES]

    print("Deriving per-(player, hex) feature table from train_enriched...")
    columns_to_read = ["player_id", "hex_id", *static_features]
    enriched = pd.read_parquet(train_enriched_path, columns=columns_to_read)
    serving_features = enriched.drop_duplicates(subset=["player_id", "hex_id"]).reset_index(
        drop=True
    )
    serving_features.to_parquet(output_dir / "serving_features.parquet", index=False)
    print(
        f"  {len(serving_features):,} unique (player, hex) pairs "
        f"for {serving_features['player_id'].nunique():,} players"
    )

    print("Reading period categories (order matters for categorical codes)...")
    period_series = pd.read_parquet(train_enriched_path, columns=["period"])["period"]
    period_categories = [str(c) for c in period_series.cat.categories]

    meta = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "static_features": static_features,
        "contextual_features": list(CONTEXTUAL_FEATURES),
        "period_categories": period_categories,
    }
    with (output_dir / "serving_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"  period categories: {period_categories}")

    print("Registering final model in the MLflow Model Registry...")
    mlflow.set_tracking_uri(tracking_uri)
    runs = cast(pd.DataFrame, mlflow.search_runs(experiment_names=[experiment_name]))
    final_runs = runs[runs["tags.mlflow.runName"] == "final_model"]
    if len(final_runs) == 0:
        raise RuntimeError("No 'final_model' run found. Run log_final_model first.")
    run_id = str(final_runs.iloc[0]["run_id"])
    version = mlflow.register_model(f"runs:/{run_id}/model", registered_model_name)
    print(f"  registered '{registered_model_name}' version {version.version} " f"from run {run_id}")


def export_model(
    output_dir: Path,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> None:
    """Export the latest registered model to a self-contained directory.

    A container serves from this directory and never touches the MLflow
    registry (mlflow.db, mlartifacts) at runtime. This separates "where
    models are developed and versioned" from "what is served".
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    version = _latest_version(client, registered_model_name)
    model = mlflow.lightgbm.load_model(f"models:/{registered_model_name}/{version}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    mlflow.lightgbm.save_model(model, str(output_dir))
    print(f"Exported '{registered_model_name}' v{version} to {output_dir}")


@dataclass
class RankResult:
    """Result of a ranking request.

    Attributes
    ----------
    n_candidates : int
        Size of the player's known universe (all candidate hexes scored).
    ranked : list[tuple[str, float]]
        Top-k (hex_id, score) pairs, highest score first.
    """

    n_candidates: int
    ranked: list[tuple[str, float]]


class RankingService:
    """Loads serving artifacts once and ranks a player's known hexes.

    The model source is configurable. Pass model_uri to load a specific
    location (e.g. a local exported model in a container); leave it None to
    resolve the latest version from the MLflow registry (local development).
    """

    def __init__(
        self,
        serving_dir: Path,
        model_uri: str | None = None,
        registered_model_name: str = DEFAULT_REGISTERED_MODEL,
        tracking_uri: str = DEFAULT_TRACKING_URI,
    ) -> None:
        with (serving_dir / "serving_meta.json").open() as f:
            meta: dict[str, Any] = json.load(f)
        self.feature_columns: list[str] = [str(c) for c in meta["feature_columns"]]
        self.period_categories: list[str] = [str(c) for c in meta["period_categories"]]

        features = pd.read_parquet(serving_dir / "serving_features.parquet")
        # Index by player for fast per-player candidate lookup.
        self._features = features.set_index("player_id", drop=False).sort_index()

        if model_uri is None:
            # Resolve from the registry (local / development use).
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            version = _latest_version(client, registered_model_name)
            model_uri = f"models:/{registered_model_name}/{version}"
        self.model: Any = mlflow.lightgbm.load_model(model_uri)

    def known_players(self) -> int:
        """Number of distinct players in the served universe."""
        return int(self._features.index.nunique())

    def rank(
        self,
        player_id: str,
        day_of_week: int,
        period: str,
        top_k: int = 10,
    ) -> RankResult:
        """Rank a player's known hexes for a (day_of_week, period) window.

        Raises KeyError if the player is not in the served universe.
        """
        if period not in self.period_categories:
            raise ValueError(f"unknown period: {period}")

        try:
            candidates = self._features.loc[[player_id]].reset_index(drop=True)
        except KeyError as exc:
            raise KeyError(player_id) from exc

        candidates["day_of_week"] = day_of_week
        candidates["period"] = pd.Categorical(
            [period] * len(candidates), categories=self.period_categories
        )

        scores = self.model.predict(candidates[self.feature_columns])
        candidates["score"] = scores
        top = candidates.nlargest(top_k, "score")
        ranked = [(str(h), float(s)) for h, s in zip(top["hex_id"], top["score"], strict=True)]
        return RankResult(n_candidates=len(candidates), ranked=ranked)
