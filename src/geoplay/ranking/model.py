"""LightGBM ranker wrapper for the geoplay recommendation task.

Wraps lightgbm.LGBMRanker with a clean fit/predict interface that
handles group sizes (required for ranking), categorical features
(passed as pandas categoricals), and model persistence.

The default hyperparameters are sensible starting points for the
geoplay dataset. Use the tuning module to search over a grid of
alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

# Default LightGBM hyperparameters. lambdarank is the industrial standard
# for learning-to-rank with binary or graded relevance labels.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10, 20],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


@dataclass
class TrainingResult:
    """Container for the output of a model training run.

    Attributes
    ----------
    model : lgb.LGBMRanker
        The fitted LightGBM ranker.
    feature_names : list[str]
        Names of the features used during training (column order matters).
    categorical_features : list[str]
        Names of the categorical features.
    params : dict
        Hyperparameters used.
    best_iteration : int
        Best boosting iteration (only meaningful if early stopping was used).
    feature_importance : pd.DataFrame
        Per-feature importance scores: gain and split count.
    """

    model: lgb.LGBMRanker
    feature_names: list[str]
    categorical_features: list[str]
    params: dict[str, Any]
    best_iteration: int
    feature_importance: pd.DataFrame = field(repr=False)


def _compute_group_sizes(group_ids: npt.NDArray) -> npt.NDArray[np.int64]:
    """Compute the size of each contiguous group of identical group_ids.

    LightGBM expects a list of group sizes (not the per-row group_id).
    For example, group_ids [0,0,0,1,1,2,2,2,2] -> [3, 2, 4].

    Rows are assumed to be already sorted by group_id (consecutive rows
    with the same group_id belong to the same query).

    Parameters
    ----------
    group_ids : np.ndarray
        Per-row group identifiers.

    Returns
    -------
    np.ndarray
        Size of each group, in order.
    """
    ids = np.asarray(group_ids)
    if len(ids) == 0:
        return np.array([], dtype=np.int64)

    # Find indices where group_id changes.
    changes = np.concatenate([[True], ids[1:] != ids[:-1]])
    group_starts = np.where(changes)[0]
    group_sizes = np.diff(np.concatenate([group_starts, [len(ids)]]))
    return group_sizes.astype(np.int64)


def fit_ranker(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    n_estimators: int = 200,
    params: dict[str, Any] | None = None,
    eval_df: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = None,
) -> TrainingResult:
    """Fit an LGBMRanker on the train dataset.

    The input must be sorted by group_id so that contiguous rows form
    queries. This is enforced internally as a safety check.

    Parameters
    ----------
    train_df : pd.DataFrame
        Enriched training dataset. Must include feature_columns plus
        'label' and 'group_id' columns.
    feature_columns : list[str]
        Ordered names of the columns to use as features.
    categorical_columns : list[str]
        Subset of feature_columns that should be treated as categorical
        by LightGBM.
    n_estimators : int
        Number of boosting iterations.
    params : dict | None
        LightGBM parameters. If None, uses DEFAULT_PARAMS.
    eval_df : pd.DataFrame | None
        Optional evaluation dataset for early stopping and metric tracking.
    early_stopping_rounds : int | None
        If set with eval_df, stop training when the metric stops improving
        for this many rounds.

    Returns
    -------
    TrainingResult
    """
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        merged = dict(DEFAULT_PARAMS)
        merged.update(params)
        params = merged

    # Sort by group_id to ensure contiguous queries.
    train_df = train_df.sort_values("group_id", kind="stable").reset_index(drop=True)
    train_x = train_df[feature_columns]
    train_y = train_df["label"].to_numpy()
    train_group = _compute_group_sizes(train_df["group_id"].to_numpy())

    fit_kwargs: dict[str, Any] = {}
    if eval_df is not None:
        eval_df = eval_df.sort_values("group_id", kind="stable").reset_index(drop=True)
        eval_x = eval_df[feature_columns]
        eval_y = eval_df["label"].to_numpy()
        eval_group = _compute_group_sizes(eval_df["group_id"].to_numpy())
        fit_kwargs["eval_set"] = [(eval_x, eval_y)]
        fit_kwargs["eval_group"] = [eval_group]
        if early_stopping_rounds is not None:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            ]

    model = lgb.LGBMRanker(n_estimators=n_estimators, **params)
    model.fit(
        train_x,
        train_y,
        group=train_group,
        categorical_feature=categorical_columns,
        **fit_kwargs,
    )

    best_iter = int(model.best_iteration_) if model.best_iteration_ is not None else n_estimators

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    return TrainingResult(
        model=model,
        feature_names=list(feature_columns),
        categorical_features=list(categorical_columns),
        params=params,
        best_iteration=best_iter,
        feature_importance=importance_df,
    )


def predict_scores(
    model: lgb.LGBMRanker,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> npt.NDArray[np.float64]:
    """Run inference on a dataframe and return per-row ranking scores.

    Parameters
    ----------
    model : lgb.LGBMRanker
        A fitted model.
    df : pd.DataFrame
        Dataframe with the same feature_columns used during training.
    feature_columns : list[str]
        Column order must match training exactly.

    Returns
    -------
    np.ndarray
        Per-row ranking scores. Higher = more relevant.
    """
    return np.asarray(model.predict(df[feature_columns]), dtype=np.float64)


def save_model(result: TrainingResult, output_dir: Path) -> None:
    """Persist the trained model and its metadata to disk.

    Writes three files:
        model.txt          - LightGBM native serialization
        params.json        - hyperparameters (human-readable)
        feature_importance.csv - per-feature gain and split counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result.model.booster_.save_model(str(output_dir / "model.txt"))

    import json

    with (output_dir / "params.json").open("w") as f:
        json.dump(
            {
                "params": result.params,
                "feature_names": result.feature_names,
                "categorical_features": result.categorical_features,
                "best_iteration": result.best_iteration,
            },
            f,
            indent=2,
        )

    result.feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
