from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphcredit_offline.rewards.process_scorers import clip01


@dataclass
class RewardBreakdown:
    """All terms used to produce a node reward."""

    node_id: str
    global_reward: float
    process_reward: float
    counterfactual_credit: float
    downstream_usage_score: float
    cost_penalty: float
    redundancy_penalty: float
    node_reward: float
    details: dict[str, Any] = field(default_factory=dict)


def fuse_node_reward(
    node_id: str,
    global_reward: float,
    process_reward: float,
    counterfactual_credit: float,
    downstream_usage: float,
    cost: float,
    redundancy: float,
    weights: dict[str, float] | None = None,
    negative_credit_clip: float = -0.5,
    allow_failed_positive: bool = True,
) -> RewardBreakdown:
    """Fuse reward components into one trainable scalar."""

    weights = weights or {}
    clipped_credit = max(float(counterfactual_credit), float(negative_credit_clip))
    node_reward = (
        weights.get("alpha_global", 0.70) * float(global_reward)
        + weights.get("beta_process", 0.10) * float(process_reward)
        + weights.get("gamma_counterfactual", 0.10) * clipped_credit
        + weights.get("delta_downstream_usage", 0.05) * float(downstream_usage)
        - weights.get("eta_cost", 0.025) * float(cost)
        - weights.get("zeta_redundancy", 0.025) * float(redundancy)
    )
    if float(global_reward) <= 0.0:
        node_reward = node_reward - weights.get("failure_penalty", 0.10)
        if not allow_failed_positive:
            node_reward = min(node_reward, 0.0)
    return RewardBreakdown(
        node_id=node_id,
        global_reward=float(global_reward),
        process_reward=clip01(process_reward),
        counterfactual_credit=clipped_credit,
        downstream_usage_score=clip01(downstream_usage),
        cost_penalty=clip01(cost),
        redundancy_penalty=clip01(redundancy),
        node_reward=float(node_reward),
        details={"weights": dict(weights), "raw_counterfactual_credit": float(counterfactual_credit)},
    )
