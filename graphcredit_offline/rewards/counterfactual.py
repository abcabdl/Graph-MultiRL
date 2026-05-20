from __future__ import annotations

import copy
from dataclasses import dataclass

from graphcredit_offline.core.graph_builder import extract_math_answer, lexical_overlap
from graphcredit_offline.core.schema import EventGraph, EventNode
from graphcredit_offline.rewards.downstream_usage import downstream_usage_score
from graphcredit_offline.rewards.process_scorers import clip01
from graphcredit_offline.rewards.verifier_diagnostics import math_verifier_diagnostics


@dataclass
class CounterfactualResult:
    """Offline proxy for node marginal contribution."""

    node_id: str
    original_value: float
    masked_value: float
    credit: float
    reason: str


MASK_BY_NODE_TYPE = {
    "agent_message": "[MESSAGE_REMOVED]",
    "agent_action": "[ACTION_REMOVED]",
    "tool_call": "[TOOL_CALL_REMOVED]",
    "tool_result": "[TOOL_RESULT_UNAVAILABLE]",
    "router_decision": "[ROUTER_DECISION_REMOVED]",
    "verifier_judgment": "[NEUTRAL_VERIFIER_JUDGMENT]",
    "memory_write": "[MEMORY_WRITE_REMOVED]",
    "memory_read": "[MEMORY_READ_EMPTY]",
}


def offline_masking_credit(graph: EventGraph, node: EventNode, process_reward: float = 0.0, task_type: str | None = None) -> CounterfactualResult:
    """Estimate SHARP-style marginal contribution on a statically masked graph.

    This follows the form R(original_graph) - R(masked_graph) while avoiding
    LLM/environment reruns. The masked graph value is recomputed by task-specific
    offline rules over graph support, evidence, router decisions, and verifier
    judgments.
    """

    original_value = float(graph.final_reward or node.final_reward or 0.0)
    if node.node_type == "final_answer":
        return CounterfactualResult(node.node_id, original_value, original_value, 0.0, "final_answer is evaluated, not masked")
    masked_graph = mask_node(graph, node.node_id)
    masked_value = evaluate_masked_graph(masked_graph, original_graph=graph, masked_node=node, process_reward=process_reward, task_type=task_type)
    credit = original_value - masked_value
    return CounterfactualResult(
        node_id=node.node_id,
        original_value=original_value,
        masked_value=masked_value,
        credit=credit,
        reason="static masked graph value difference: R(original_graph) - R(masked_graph)",
    )


def mask_node(graph: EventGraph, node_id: str) -> EventGraph:
    """Return a graph where one node is masked and its outgoing support edges are removed."""

    masked_graph = copy.deepcopy(graph)
    for node in masked_graph.nodes:
        if node.node_id == node_id:
            replacement = MASK_BY_NODE_TYPE.get(node.node_type, "[NODE_REMOVED]")
            node.metadata["masked"] = True
            node.metadata["original_output_content"] = node.output_content
            node.output_content = replacement
            node.tool_result = None
            node.tool_success = False if node.node_type in {"tool_call", "tool_result"} else node.tool_success
            break
    masked_graph.edges = [
        edge
        for edge in masked_graph.edges
        if not (edge.source_node_id == node_id and edge.edge_type in {"communication_edge", "evidence_edge", "control_edge", "artifact_edge", "tool_edge"})
    ]
    masked_graph.metadata["masked_node_id"] = node_id
    return masked_graph


def evaluate_masked_graph(
    masked_graph: EventGraph,
    original_graph: EventGraph,
    masked_node: EventNode,
    process_reward: float = 0.0,
    task_type: str | None = None,
) -> float:
    """Evaluate final value of a statically masked graph."""

    task = (task_type or original_graph.task_type or "").lower()
    if task == "search":
        return _evaluate_search_masked_graph(masked_graph, original_graph, masked_node, process_reward)
    return _evaluate_math_masked_graph(masked_graph, original_graph, masked_node, process_reward)


def _evaluate_math_masked_graph(masked_graph: EventGraph, original_graph: EventGraph, masked_node: EventNode, process_reward: float) -> float:
    original_value = float(original_graph.final_reward or masked_node.final_reward or 0.0)
    if masked_node.node_type == "verifier_judgment":
        diagnostics = math_verifier_diagnostics(original_graph, masked_node)
        return original_value - diagnostics.verifier_reward

    support = _final_answer_support(masked_graph)
    usage_before = downstream_usage_score(original_graph, masked_node)
    if original_value > 0:
        if masked_node.node_type in {"agent_message", "agent_action"}:
            support_floor = 0.25 if _has_unmasked_solver_support(masked_graph, original_graph.final_answer) else 0.0
            return clip01(max(support, support_floor))
        return clip01(max(support, 1.0 - 0.7 * usage_before))

    harmful = clip01(1.0 - process_reward)
    if masked_node.node_type in {"agent_message", "agent_action"}:
        has_boxed_answer = extract_math_answer(masked_node.output_content) is not None
        misleading_answer = 0.35 if has_boxed_answer else 0.0
        if has_boxed_answer or usage_before > 0.1:
            return clip01(0.20 + misleading_answer + 0.25 * usage_before + 0.20 * process_reward)
    return clip01(original_value)


def _evaluate_search_masked_graph(masked_graph: EventGraph, original_graph: EventGraph, masked_node: EventNode, process_reward: float) -> float:
    original_value = float(original_graph.final_reward or masked_node.final_reward or 0.0)
    support = _final_answer_support(masked_graph)
    usage_before = downstream_usage_score(original_graph, masked_node)
    if original_value > 0:
        if masked_node.node_type == "tool_call":
            evidence_loss = max(usage_before, 1.0 - support)
            return clip01(original_value - 0.85 * evidence_loss)
        if masked_node.node_type == "router_decision":
            return clip01(0.5 + 0.5 * support)
        return clip01(max(support, original_value - 0.6 * usage_before))

    harmful = clip01(1.0 - process_reward)
    if masked_node.node_type == "router_decision" and _router_stopped(masked_node):
        return clip01(original_value + 0.5 * harmful)
    if masked_node.node_type == "tool_call" and process_reward < 0.35:
        return clip01(original_value + 0.35 * harmful)
    return clip01(original_value)


def _final_answer_support(graph: EventGraph) -> float:
    final_answer = graph.final_answer or ""
    if not final_answer:
        return 0.0
    supporting_nodes = [node for node in graph.nodes if node.node_type != "final_answer" and not node.metadata.get("masked", False)]
    if not supporting_nodes:
        return 0.0
    if graph.task_type.lower() == "math":
        target_answer = extract_math_answer(final_answer)
        if target_answer is not None:
            answer_match = any(
                node.node_type in {"agent_message", "agent_action"}
                and extract_math_answer(node.output_content) == target_answer
                for node in supporting_nodes
            )
            if answer_match:
                return 1.0
            overlap = max(
                (lexical_overlap(node.output_content, final_answer) for node in supporting_nodes if node.node_type in {"agent_message", "agent_action"}),
                default=0.0,
            )
            return clip01(0.3 * overlap)
    evidence_edge_sources = {edge.source_node_id for edge in graph.edges if edge.edge_type == "evidence_edge"}
    overlap = max((lexical_overlap(node.output_content, final_answer) for node in supporting_nodes), default=0.0)
    edge_support = 1.0 if evidence_edge_sources else 0.0
    return clip01(0.6 * overlap + 0.4 * edge_support)


def _has_unmasked_solver_support(graph: EventGraph, final_answer: str | None = None) -> bool:
    target_answer = extract_math_answer(final_answer)
    return any(
        node.node_type in {"agent_message", "agent_action"}
        and not node.metadata.get("masked", False)
        and (
            (target_answer is not None and extract_math_answer(node.output_content) == target_answer)
            or (target_answer is None and "\\boxed" in node.output_content)
        )
        for node in graph.nodes
    )


def _router_stopped(node: EventNode) -> bool:
    return "<verify>yes</verify>" in (node.output_content or "").lower()
