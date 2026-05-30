"""Orchestrator that assembles the final training and test feature matrices.

The pipeline:

    train.parquet, test.parquet                (raw query-hex pairs with labels)
                  +
    player_features_train.parquet               (10 query-side features)
                  +
    hex_features_train.parquet                  (3 item-side features)
                  +
    pair lookups (player_hex_visits, homes,     (2 pair-side features)
                  hex centroids)
                  =
    train_enriched.parquet, test_enriched.parquet

Categorical features (player_archetype, period) are left as pandas
`category` dtype, which LightGBM consumes natively as `categorical_feature`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoplay.ranking.pair_features import (
    PairFeaturesLookups,
    add_pair_features,
)

# Columns that identify a row (not features themselves).
ID_COLUMNS = ("player_id", "hex_id", "group_id", "label")

# Categorical features (LightGBM consumes them as `categorical_feature`).
CATEGORICAL_FEATURES = ("player_archetype", "period")

# All numeric feature names (used to validate the final schema).
NUMERIC_FEATURES = (
    # Query (player)
    "player_total_events_log",
    "player_unique_sessions",
    "player_events_per_session_mean",
    "player_active_days_ratio",
    "player_weekday_ratio",
    "player_evening_ratio",
    "player_hour_concentration",
    "player_distance_from_home_mean_km",
    "player_unique_hexes_count_log",
    # Item (hex)
    "hex_global_popularity",
    "hex_total_visits",
    "hex_n_unique_archetypes",
    # Pair (player, hex)
    "pair_past_visits",
    "pair_distance_from_home_km",
    # Contextual numeric
    "day_of_week",
)


@dataclass
class EnrichedDatasets:
    """Container for enriched train and test datasets.

    Attributes
    ----------
    train : pd.DataFrame
        Train pairs with all feature columns joined.
    test : pd.DataFrame
        Test pairs with all feature columns joined.
    feature_columns : list[str]
        Ordered list of column names that are features (numeric + categorical).
    categorical_columns : list[str]
        Names of categorical features (subset of feature_columns).
    """

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]


def _join_features(
    pairs: pd.DataFrame,
    player_features: pd.DataFrame,
    hex_features: pd.DataFrame,
    pair_lookups: PairFeaturesLookups,
) -> pd.DataFrame:
    """Join query, item, and pair features onto a pairs DataFrame.

    Performs the joins in a specific order:
      1. Pair features via lookups (in-place column addition).
      2. Player features via merge on player_id.
      3. Hex features via merge on hex_id.

    Parameters
    ----------
    pairs : pd.DataFrame
        Raw (player_id, day_of_week, period, hex_id, label, group_id) rows.
    player_features : pd.DataFrame
        Output of compute_player_features. Must have player_id column.
    hex_features : pd.DataFrame
        Output of compute_hex_features. Must have hex_id column.
    pair_lookups : PairFeaturesLookups
        Lookups built by build_pair_features_lookups.

    Returns
    -------
    pd.DataFrame
        Pairs with all features joined.
    """
    enriched = add_pair_features(pairs, pair_lookups)

    enriched = enriched.merge(player_features, on="player_id", how="left")
    enriched = enriched.merge(hex_features, on="hex_id", how="left")

    return enriched


def _set_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical feature columns to pandas `category` dtype.

    LightGBM's Python API consumes pandas categoricals natively when
    passed via the `categorical_feature` parameter, which is more
    efficient than one-hot encoding for high-cardinality categoricals.
    """
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def build_enriched_datasets(
    train_pairs_path: Path,
    test_pairs_path: Path,
    player_features_path: Path,
    hex_features_path: Path,
    pair_lookups: PairFeaturesLookups,
    progress_callback: Callable[[str], None] | None = None,
) -> EnrichedDatasets:
    """Assemble enriched train and test datasets ready for LightGBM.

    Parameters
    ----------
    train_pairs_path : Path
        Path to train.parquet from dataset.build_train_test_datasets.
    test_pairs_path : Path
        Path to test.parquet.
    player_features_path : Path
        Path to player_features_train.parquet (used for both train and
        test, since the cutoff already prevents leakage).
    hex_features_path : Path
        Path to hex_features_train.parquet.
    pair_lookups : PairFeaturesLookups
        Pair lookups built by build_pair_features_lookups.
    progress_callback : callable | None
        Optional callback for progress.

    Returns
    -------
    EnrichedDatasets
    """

    def log(msg: str) -> None:
        if progress_callback is not None:
            progress_callback(msg)

    t0 = time.time()

    log("Loading raw pair datasets")
    train = pd.read_parquet(train_pairs_path)
    test = pd.read_parquet(test_pairs_path)
    log(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")

    log("Loading player and hex features")
    player_features = pd.read_parquet(player_features_path)
    hex_features = pd.read_parquet(hex_features_path)
    log(f"  Players: {len(player_features):,} | Hexes: {len(hex_features):,}")

    log("Joining features onto train")
    train_enriched = _join_features(train, player_features, hex_features, pair_lookups)
    log(f"  Train enriched in {(time.time()-t0)/60:.1f} min")

    t1 = time.time()
    log("Joining features onto test")
    test_enriched = _join_features(test, player_features, hex_features, pair_lookups)
    log(f"  Test enriched in {(time.time()-t1)/60:.1f} min")

    log("Setting categorical dtypes")
    train_enriched = _set_categorical_dtypes(train_enriched)
    test_enriched = _set_categorical_dtypes(test_enriched)

    # Validate: every numeric feature should be present.
    missing_train = set(NUMERIC_FEATURES) - set(train_enriched.columns)
    missing_test = set(NUMERIC_FEATURES) - set(test_enriched.columns)
    if missing_train:
        raise ValueError(f"Train missing numeric features: {missing_train}")
    if missing_test:
        raise ValueError(f"Test missing numeric features: {missing_test}")

    feature_columns = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    categorical_columns = list(CATEGORICAL_FEATURES)

    log(f"Train: {len(train_enriched):,} rows x {len(feature_columns)} features")
    log(f"Test:  {len(test_enriched):,} rows x {len(feature_columns)} features")
    log(f"Total elapsed: {(time.time()-t0)/60:.1f} min")

    return EnrichedDatasets(
        train=train_enriched,
        test=test_enriched,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
