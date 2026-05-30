"""Random search hyperparameter tuning for the LightGBM ranker.

Strategy: subsample the training set to a configurable fraction, then
fit one model per random hyperparameter combination and score it on a
held-out validation slice using NDCG@10. The best configuration is
returned and a CSV log of all trials is written.

This is a common pragmatic approach for large datasets where full
cross-validation would take 10+ hours. Random search over 20 trials on
a 10% subsample typically finds parameters within 1-2% of the grid
optimum at ~5% of the compute cost.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geoplay.ranking.evaluation import evaluate_predictions
from geoplay.ranking.model import fit_ranker, predict_scores

# Search space for random search. Each parameter is sampled independently
# from these candidate values. The total search space is large enough that
# 20 random trials provide good coverage.
DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300, 500],
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "min_child_samples": [20, 50, 100],
}


@dataclass
class Trial:
    """One random search trial: a hyperparameter sample and its NDCG@10."""

    trial_id: int
    params: dict[str, Any]
    ndcg_10: float
    fit_seconds: float
    eval_seconds: float


@dataclass
class TuningResult:
    """Container for the full tuning run.

    Attributes
    ----------
    trials : list[Trial]
        Every trial attempted, in order.
    best_trial : Trial
        Trial with the highest NDCG@10.
    search_space : dict
        The space that was searched.
    n_train_queries : int
        Number of queries in the train subsample.
    n_val_queries : int
        Number of queries in the validation subsample.
    """

    trials: list[Trial]
    best_trial: Trial
    search_space: dict[str, list[Any]]
    n_train_queries: int
    n_val_queries: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return all trials as a DataFrame for easy inspection."""
        rows = []
        for t in self.trials:
            row = {
                "trial_id": t.trial_id,
                "ndcg_10": round(t.ndcg_10, 4),
                "fit_seconds": round(t.fit_seconds, 1),
                "eval_seconds": round(t.eval_seconds, 1),
            }
            row.update(t.params)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("ndcg_10", ascending=False)


def _sample_params(
    search_space: dict[str, list[Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample one hyperparameter combination uniformly from the search space."""
    return {key: rng.choice(values).item() for key, values in search_space.items()}


def _subsample_by_groups(
    df: pd.DataFrame,
    n_groups: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Subsample a learning-to-rank dataset by selecting whole queries.

    Subsampling at the row level would break query groups. Subsampling
    at the group level keeps queries intact, which is required for
    LightGBM's ranking objective.
    """
    unique_groups = df["group_id"].unique()
    if len(unique_groups) <= n_groups:
        return df.copy()
    selected = rng.choice(unique_groups, size=n_groups, replace=False)
    return df[df["group_id"].isin(selected)].copy()


def random_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    n_trials: int = 20,
    search_space: dict[str, list[Any]] | None = None,
    seed: int = 42,
    progress_callback: Callable[[int, int, Trial], None] | None = None,
) -> TuningResult:
    """Run random search over hyperparameters.

    Each trial fits a fresh LightGBM ranker with sampled hyperparameters
    on train_df, predicts on val_df, and computes NDCG@10.

    Parameters
    ----------
    train_df : pd.DataFrame
        Already-subsampled training data with feature_columns + label + group_id.
    val_df : pd.DataFrame
        Validation slice with the same schema.
    feature_columns : list[str]
        Column order to pass to LightGBM.
    categorical_columns : list[str]
        Subset of feature_columns treated as categorical.
    n_trials : int
        Number of random trials.
    search_space : dict | None
        Parameter search space. Defaults to DEFAULT_SEARCH_SPACE.
    seed : int
        Random seed.
    progress_callback : callable | None
        Optional callback(trial_idx, total, trial) for progress.

    Returns
    -------
    TuningResult
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    rng = np.random.default_rng(seed)
    trials: list[Trial] = []

    val_labels = val_df["label"].to_numpy()
    val_groups_array = val_df["group_id"].to_numpy()

    for trial_idx in range(1, n_trials + 1):
        sampled = _sample_params(search_space, rng)
        n_estimators = sampled.pop("n_estimators")

        t_fit = time.time()
        result = fit_ranker(
            train_df=train_df,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            n_estimators=n_estimators,
            params=sampled,
        )
        fit_seconds = time.time() - t_fit

        t_eval = time.time()
        scores = predict_scores(result.model, val_df, feature_columns)
        metrics = evaluate_predictions(
            labels=val_labels,
            predictions=scores,
            group_ids=val_groups_array,
        )
        eval_seconds = time.time() - t_eval

        full_params = dict(sampled)
        full_params["n_estimators"] = n_estimators

        trial = Trial(
            trial_id=trial_idx,
            params=full_params,
            ndcg_10=metrics.ndcg_10,
            fit_seconds=fit_seconds,
            eval_seconds=eval_seconds,
        )
        trials.append(trial)

        if progress_callback is not None:
            progress_callback(trial_idx, n_trials, trial)

    best_trial = max(trials, key=lambda t: t.ndcg_10)
    n_train_queries = train_df["group_id"].nunique()
    n_val_queries = val_df["group_id"].nunique()

    return TuningResult(
        trials=trials,
        best_trial=best_trial,
        search_space=search_space,
        n_train_queries=n_train_queries,
        n_val_queries=n_val_queries,
    )


def save_tuning_result(
    result: TuningResult,
    output_dir: Path,
) -> None:
    """Persist tuning results to disk.

    Writes:
        tuning_trials.csv   - all trials with their parameters and scores
        best_params.json    - the best parameter combination
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = result.to_dataframe()
    df.to_csv(output_dir / "tuning_trials.csv", index=False)

    with (output_dir / "best_params.json").open("w") as f:
        json.dump(
            {
                "params": result.best_trial.params,
                "ndcg_10": round(result.best_trial.ndcg_10, 4),
                "trial_id": result.best_trial.trial_id,
                "n_train_queries": result.n_train_queries,
                "n_val_queries": result.n_val_queries,
                "n_trials": len(result.trials),
            },
            f,
            indent=2,
        )


def subsample_train_val(
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subsample train and val datasets by groups.

    Convenience wrapper around _subsample_by_groups used by the tuning CLI.

    Parameters
    ----------
    train_full : pd.DataFrame
        Full enriched train dataset.
    val_full : pd.DataFrame
        Full enriched validation/test dataset.
    train_fraction : float
        Fraction of train queries to keep.
    val_fraction : float
        Fraction of val queries to keep.
    seed : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_subsample, val_subsample)
    """
    rng = np.random.default_rng(seed)
    n_train = int(train_full["group_id"].nunique() * train_fraction)
    n_val = int(val_full["group_id"].nunique() * val_fraction)
    train_sub = _subsample_by_groups(train_full, n_train, rng)
    val_sub = _subsample_by_groups(val_full, n_val, rng)
    return train_sub, val_sub
