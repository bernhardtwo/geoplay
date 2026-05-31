"""MLflow experiment tracking for the ranking pipeline.

This module is the tracking layer. It orchestrates the existing modeling
functions (fit_ranker, random_search, evaluate_predictions) and logs runs,
parameters, metrics, and artifacts to MLflow. The modeling code stays free
of any MLflow dependency: tracking is composed on top of it, not baked in.

Two entry points:

- log_tuning_experiment: runs random search inside a parent MLflow run,
  logging each trial as a nested run. It reuses random_search's existing
  progress_callback hook, so the search code itself is never modified.
- log_final_model: trains the final ranker with a fixed set of parameters
  on the full train set, evaluates on the test set, and logs parameters,
  metrics, the model, and feature importance.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from geoplay.ranking.evaluation import evaluate_predictions
from geoplay.ranking.model import fit_ranker, predict_scores, save_model
from geoplay.ranking.tuning import Trial, random_search, subsample_train_val

DEFAULT_EXPERIMENT = "geoplay-ranking"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _load_columns(model_params_path: Path) -> tuple[list[str], list[str]]:
    """Read feature and categorical column names from a saved params.json.

    params.json is written by model.save_model and is the source of truth
    for the exact columns (and their order) the model was trained on.
    """
    with model_params_path.open() as f:
        data: dict[str, Any] = json.load(f)
    feature_names = [str(c) for c in data["feature_names"]]
    categorical = [str(c) for c in data["categorical_features"]]
    return feature_names, categorical


def _load_best_params(best_params_path: Path) -> dict[str, Any]:
    """Read the best hyperparameter set from a saved best_params.json."""
    with best_params_path.open() as f:
        data: dict[str, Any] = json.load(f)
    return data


def _split_by_groups(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a ranking dataset into train/val by whole queries.

    Splitting at the row level would break query groups, so we partition
    the unique group_ids and assign whole queries to each side. This keeps
    the test set untouched: trials are validated on an internal holdout of
    the train period, not on the held-out test period.
    """
    rng = np.random.default_rng(seed)
    unique_groups = df["group_id"].unique()
    rng.shuffle(unique_groups)
    n_val = int(len(unique_groups) * val_fraction)
    val_groups = set(unique_groups[:n_val].tolist())
    val_mask = df["group_id"].isin(val_groups)
    return df[~val_mask].copy(), df[val_mask].copy()


def log_tuning_experiment(
    train_enriched_path: Path,
    model_params_path: Path,
    n_trials: int = 20,
    train_subsample_fraction: float = 0.10,
    val_fraction: float = 0.20,
    seed: int = 42,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> None:
    """Run random search under MLflow tracking.

    Opens a parent run for the search and logs every trial as a nested run.
    Validation uses an internal holdout of the train set, so the test set is
    never seen during hyperparameter selection.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    feature_columns, categorical_columns = _load_columns(model_params_path)

    train_full = pd.read_parquet(train_enriched_path)
    train_part, val_part = _split_by_groups(train_full, val_fraction, seed)
    train_sub, val_sub = subsample_train_val(
        train_part,
        val_part,
        train_fraction=train_subsample_fraction,
        val_fraction=1.0,
        seed=seed,
    )

    def _log_trial(_idx: int, total: int, trial: Trial) -> None:
        with mlflow.start_run(run_name=f"trial_{trial.trial_id:02d}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("ndcg_10", trial.ndcg_10)
            mlflow.log_metric("fit_seconds", trial.fit_seconds)
            mlflow.log_metric("eval_seconds", trial.eval_seconds)
        print(
            f"  trial {trial.trial_id:>2}/{total}  "
            f"ndcg@10={trial.ndcg_10:.4f}  ({trial.fit_seconds:.1f}s fit)"
        )

    with mlflow.start_run(run_name="random_search"):
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "train_subsample_fraction": train_subsample_fraction,
                "val_fraction": val_fraction,
                "seed": seed,
                "n_train_queries": int(train_sub["group_id"].nunique()),
                "n_val_queries": int(val_sub["group_id"].nunique()),
            }
        )

        result = random_search(
            train_df=train_sub,
            val_df=val_sub,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            n_trials=n_trials,
            seed=seed,
            progress_callback=_log_trial,
        )

        mlflow.log_metric("best_ndcg_10", result.best_trial.ndcg_10)
        mlflow.log_param("best_trial_id", result.best_trial.trial_id)
        mlflow.log_params({f"best_{k}": v for k, v in result.best_trial.params.items()})

    print(
        f"\nBest trial: {result.best_trial.trial_id}  " f"ndcg@10={result.best_trial.ndcg_10:.4f}"
    )


def log_final_model(
    train_enriched_path: Path,
    test_enriched_path: Path,
    best_params_path: Path,
    model_params_path: Path,
    model_output_dir: Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> None:
    """Train and log the final ranker under MLflow tracking.

    Trains on the full train set with the selected hyperparameters, evaluates
    on the test set, and logs parameters, all ranking metrics, the model, and
    the feature importance table. Because the parameters and random_state are
    fixed, this reproduces the documented results deterministically.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    feature_columns, categorical_columns = _load_columns(model_params_path)
    best = _load_best_params(best_params_path)
    params: dict[str, Any] = dict(best["params"])
    n_estimators = int(params.pop("n_estimators"))

    train_df = pd.read_parquet(train_enriched_path)
    test_df = pd.read_parquet(test_enriched_path)

    with mlflow.start_run(run_name="final_model"):
        mlflow.log_params(best["params"])
        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_param("n_test_rows", len(test_df))

        training = fit_ranker(
            train_df=train_df,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            n_estimators=n_estimators,
            params=params,
        )

        scores = predict_scores(training.model, test_df, feature_columns)
        metrics = evaluate_predictions(
            labels=test_df["label"].to_numpy(),
            predictions=scores,
            group_ids=test_df["group_id"].to_numpy(),
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.to_dict().items()})

        signature = infer_signature(test_df[feature_columns].head(50), scores[:50])
        mlflow.lightgbm.log_model(training.model, name="model", signature=signature)

        with tempfile.TemporaryDirectory() as tmp:
            fi_path = Path(tmp) / "feature_importance.csv"
            training.feature_importance.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path))

        if model_output_dir is not None:
            save_model(training, model_output_dir)

    print("Final model metrics:")
    print(json.dumps(metrics.to_dict(), indent=2))
