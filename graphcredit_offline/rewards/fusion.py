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
    process = clip01(process_reward)
    usage = clip01(downstream_usage)
    cost_value = clip01(cost)
    redundancy_value = clip01(redundancy)
    beta = weights.get("beta_process", 0.10)
    gamma = weights.get("gamma_counterfactual", 0.10)
    delta = weights.get("delta_downstream_usage", 0.05)
    eta = weights.get("eta_cost", 0.025)
    zeta = weights.get("zeta_redundancy", 0.025)
    cost_term = eta * cost_value + zeta * redundancy_value
    local_positive = beta * process + gamma * max(clipped_credit, 0.0) + delta * usage
    local_harm = gamma * min(clipped_credit, 0.0)

    if float(global_reward) > 0.0:
        node_reward = local_positive + local_harm - cost_term
    else:
        failed_positive = gamma * max(clipped_credit, 0.0)
        node_reward = failed_positive + local_harm - cost_term - weights.get("failure_penalty", 0.10)
        if not allow_failed_positive and failed_positive <= 0.0:
            node_reward = min(node_reward, 0.0)
        elif not allow_failed_positive:
            node_reward = min(node_reward, weights.get("failed_positive_cap", 0.05))
    return RewardBreakdown(
        node_id=node_id,
        global_reward=float(global_reward),
        process_reward=process,
        counterfactual_credit=clipped_credit,
        downstream_usage_score=usage,
        cost_penalty=cost_value,
        redundancy_penalty=redundancy_value,
        node_reward=float(node_reward),
        details={"weights": dict(weights), "raw_counterfactual_credit": float(counterfactual_credit)},
    )
