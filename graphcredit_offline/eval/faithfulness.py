from __future__ import annotations

from graphcredit_offline.rewards.counterfactual import CounterfactualResult


def counterfactual_faithfulness(top_results: list[CounterfactualResult], random_results: list[CounterfactualResult]) -> float:
    """Compare average value drop for top-credit nodes against random nodes."""

    return _avg_drop(top_results) - _avg_drop(random_results)


def _avg_drop(results: list[CounterfactualResult]) -> float:
    if not results:
        return 0.0
    return sum(result.original_value - result.masked_value for result in results) / len(results)
