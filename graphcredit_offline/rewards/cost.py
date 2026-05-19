from __future__ import annotations

from graphcredit_offline.core.graph_builder import lexical_overlap
from graphcredit_offline.core.schema import EventGraph, EventNode
from graphcredit_offline.rewards.process_scorers import clip01


def cost_penalty(node: EventNode, token_budget: int = 512, min_useful_tokens: int = 8) -> float:
    """Penalize empty, too-short, long, expensive, or tool-heavy events."""

    output_tokens = len((node.output_content or "").split())
    if output_tokens < min_useful_tokens:
        brevity_cost = 1.0
    else:
        brevity_cost = 0.0
    over_budget_cost = max(output_tokens - max(token_budget, 1), 0) / max(token_budget, 1)
    tool_cost = 1.0 if node.node_type == "tool_call" else 0.0
    return clip01(brevity_cost + 0.7 * over_budget_cost + 0.3 * tool_cost)


def redundancy_penalty(graph: EventGraph, node: EventNode, counterfactual_credit: float, usage_score: float, cost: float) -> float:
    """Flag costly nodes whose removal appears to change little."""

    duplicate = any(
        other.node_id != node.node_id
        and other.agent_id == node.agent_id
        and other.time_step <= node.time_step
        and lexical_overlap(other.output_content, node.output_content) > 0.85
        for other in graph.nodes
    )
    if abs(counterfactual_credit) <= 0.05 and usage_score < 0.2 and (cost > 0.5 or duplicate):
        return 1.0
    return 0.0
