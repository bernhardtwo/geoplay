"""Evaluation metrics for learning-to-rank.

Implements the standard family of ranking metrics, all computed per query
(group) and then averaged across queries:

- NDCG@K (Normalized Discounted Cumulative Gain): the dominant metric in
  industrial ranking systems. Rewards placing relevant items at the top,
  with logarithmic decay. Bounded between 0 and 1.
- MAP (Mean Average Precision): average precision across all retrieved
  items, averaged across queries. Sensitive to the order of all relevant
  items, not just the top K.
- MRR (Mean Reciprocal Rank): 1 / rank of the first relevant item.
  Useful when only the top match matters.
- Precision@K: fraction of top-K predictions that are relevant.

All metrics work with binary relevance labels (0/1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class RankingMetrics:
    """Container for a complete set of ranking metrics.

    All scores are means across queries.
    """

    ndcg_10: float
    ndcg_20: float
    mean_average_precision: float
    mean_reciprocal_rank: float
    precision_at_5: float
    precision_at_10: float
    n_queries_evaluated: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ndcg_10": round(self.ndcg_10, 4),
            "ndcg_20": round(self.ndcg_20, 4),
            "map": round(self.mean_average_precision, 4),
            "mrr": round(self.mean_reciprocal_rank, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "precision_at_10": round(self.precision_at_10, 4),
            "n_queries_evaluated": self.n_queries_evaluated,
        }


def _dcg_at_k(relevances: npt.NDArray[np.float64], k: int) -> float:
    """Compute Discounted Cumulative Gain at position k.

    DCG@k = sum_{i=1..k} rel_i / log2(i + 1)
    """
    k = min(k, len(relevances))
    if k == 0:
        return 0.0
    rels = relevances[:k]
    discounts = np.log2(np.arange(2, k + 2))  # log2(2), log2(3), ...
    return float(np.sum(rels / discounts))


def _ndcg_at_k(predicted_relevances: npt.NDArray[np.float64], k: int) -> float:
    """Compute Normalized DCG at position k.

    Sorts ideal relevances descending to get IDCG. Returns 0 when IDCG is 0
    (no relevant items in this query) which keeps the per-query mean
    well-defined.
    """
    dcg = _dcg_at_k(predicted_relevances, k)
    ideal = np.sort(predicted_relevances)[::-1]
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _average_precision(relevances: npt.NDArray[np.float64]) -> float:
    """Compute Average Precision for a single query.

    AP = (1/R) * sum_{i: rel_i=1} precision@i
    where R is the total number of relevant items in the query.
    """
    if relevances.sum() == 0:
        return 0.0
    cumulative_hits = np.cumsum(relevances)
    precisions_at_i = cumulative_hits / np.arange(1, len(relevances) + 1)
    relevant_mask = relevances > 0
    return float(precisions_at_i[relevant_mask].mean())


def _reciprocal_rank(relevances: npt.NDArray[np.float64]) -> float:
    """Compute Reciprocal Rank for a single query.

    Returns 1 / rank_of_first_relevant_item. Returns 0 if no relevant item.
    """
    relevant_positions = np.where(relevances > 0)[0]
    if len(relevant_positions) == 0:
        return 0.0
    return float(1.0 / (relevant_positions[0] + 1))


def _precision_at_k(relevances: npt.NDArray[np.float64], k: int) -> float:
    """Compute Precision at position k."""
    k = min(k, len(relevances))
    if k == 0:
        return 0.0
    return float(relevances[:k].sum() / k)


def evaluate_predictions(
    labels: npt.NDArray[np.int64] | pd.Series,
    predictions: npt.NDArray[np.float64] | pd.Series,
    group_ids: npt.NDArray[np.int64] | pd.Series,
) -> RankingMetrics:
    """Compute all ranking metrics on a set of (label, score, group_id) rows.

    For each unique group_id (query), sorts items by predicted score
    descending and computes per-query metrics, then averages across queries.

    Parameters
    ----------
    labels : np.ndarray
        Binary relevance labels (0/1).
    predictions : np.ndarray
        Predicted ranking scores from the model. Higher = more relevant.
    group_ids : np.ndarray
        Query identifiers grouping rows that belong to the same query.

    Returns
    -------
    RankingMetrics
    """
    df = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.float64),
            "score": np.asarray(predictions, dtype=np.float64),
            "group_id": np.asarray(group_ids),
        }
    )

    ndcg_10_scores: list[float] = []
    ndcg_20_scores: list[float] = []
    ap_scores: list[float] = []
    rr_scores: list[float] = []
    p5_scores: list[float] = []
    p10_scores: list[float] = []

    for _, group in df.groupby("group_id", sort=False):
        # Sort items in this query by predicted score (descending).
        sorted_group = group.sort_values("score", ascending=False)
        relevances = sorted_group["label"].to_numpy()

        ndcg_10_scores.append(_ndcg_at_k(relevances, 10))
        ndcg_20_scores.append(_ndcg_at_k(relevances, 20))
        ap_scores.append(_average_precision(relevances))
        rr_scores.append(_reciprocal_rank(relevances))
        p5_scores.append(_precision_at_k(relevances, 5))
        p10_scores.append(_precision_at_k(relevances, 10))

    return RankingMetrics(
        ndcg_10=float(np.mean(ndcg_10_scores)),
        ndcg_20=float(np.mean(ndcg_20_scores)),
        mean_average_precision=float(np.mean(ap_scores)),
        mean_reciprocal_rank=float(np.mean(rr_scores)),
        precision_at_5=float(np.mean(p5_scores)),
        precision_at_10=float(np.mean(p10_scores)),
        n_queries_evaluated=len(ap_scores),
    )
