from __future__ import annotations

from graphcredit_offline.rewards.fusion import RewardBreakdown


def harmful_node_ratio(breakdowns: list[RewardBreakdown]) -> float:
    """Fraction of nodes with negative counterfactual credit."""

    if not breakdowns:
        return 0.0
    return sum(1 for item in breakdowns if item.counterfactual_credit < 0.0) / len(breakdowns)


def redundant_node_ratio(breakdowns: list[RewardBreakdown]) -> float:
    """Fraction of nodes flagged as redundant."""

    if not breakdowns:
        return 0.0
    return sum(1 for item in breakdowns if item.redundancy_penalty > 0.0) / len(breakdowns)


def reward_variance(breakdowns: list[RewardBreakdown]) -> float:
    """Population variance of fused node rewards."""

    if not breakdowns:
        return 0.0
    values = [item.node_reward for item in breakdowns]
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)
