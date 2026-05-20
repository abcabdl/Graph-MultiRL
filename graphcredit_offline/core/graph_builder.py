from __future__ import annotations

import re

from graphcredit_offline.core.schema import EventEdge, EventGraph, EventNode


_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_EMPTY_BOXED_RE = re.compile(r"\\boxed\s*\{\s*\}")


def infer_node_type(agent_id: str | None, output_content: str = "", orchestra_type: str | None = None) -> str:
    """Infer a stable node type from Dr. MAS agent metadata and response tags."""

    agent = (agent_id or "").lower()
    output = (output_content or "").lower()
    if "search agent" in agent or "<search>" in output:
        return "tool_call"
    if "answer agent" in agent or "<answer>" in output:
        return "final_answer"
    if "verifier agent" in agent:
        return "router_decision" if orchestra_type == "search" else "verifier_judgment"
    if "solver agent" in agent:
        return "agent_message"
    return "agent_action"


def build_event_graph(
    trajectory_id: str,
    nodes: list[EventNode],
    task_prompt: str = "",
    task_type: str = "unknown",
    task_id: str | None = None,
    final_reward: float | None = None,
) -> EventGraph:
    """Build a lightweight event graph from ordered rollout nodes."""

    unique_nodes: dict[str, EventNode] = {}
    for node in nodes:
        unique_nodes.setdefault(node.node_id, node)
    sorted_nodes = sorted(unique_nodes.values(), key=_node_order_key)
    edges: list[EventEdge] = []

    def add_edge(source_node_id: str, target_node_id: str, edge_type: str, edge_id: str) -> None:
        if source_node_id == target_node_id:
            return
        edge_key = (edge_type, source_node_id, target_node_id)
        if edge_key in seen_edges:
            return
        seen_edges.add(edge_key)
        edges.append(
            EventEdge(
                edge_id=edge_id,
                trajectory_id=trajectory_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=edge_type,
            )
        )

    seen_edges: set[tuple[str, str, str]] = set()
    for idx in range(len(sorted_nodes) - 1):
        add_edge(
            sorted_nodes[idx].node_id,
            sorted_nodes[idx + 1].node_id,
            "temporal_edge",
            f"{trajectory_id}:temporal:{idx}",
        )
    for src in sorted_nodes:
        for dst in sorted_nodes:
            if src.node_id == dst.node_id or src.time_step > dst.time_step:
                continue
            if src.output_content and src.output_content[:80] in dst.input_context:
                add_edge(
                    src.node_id,
                    dst.node_id,
                    "communication_edge",
                    f"{trajectory_id}:communication:{src.node_id}:{dst.node_id}",
                )
            if dst.node_type == "final_answer" and lexical_overlap(src.output_content, dst.output_content) > 0.15:
                add_edge(
                    src.node_id,
                    dst.node_id,
                    "evidence_edge",
                    f"{trajectory_id}:evidence:{src.node_id}:{dst.node_id}",
                )
    final_answer = _select_final_answer(sorted_nodes, task_type)
    return EventGraph(
        trajectory_id=trajectory_id,
        task_id=task_id or trajectory_id,
        task_type=task_type,
        task_prompt=task_prompt,
        nodes=sorted_nodes,
        edges=edges,
        final_answer=final_answer,
        final_reward=final_reward,
    )


def lexical_overlap(left: str, right: str) -> float:
    """Compute simple token overlap without external dependencies."""

    left_tokens = {token for token in _tokens(left) if len(token) > 2}
    right_tokens = {token for token in _tokens(right) if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(text: str) -> list[str]:
    return [token.strip(".,;:!?()[]{}<>\"'`").lower() for token in (text or "").split()]


def node_order_key(node: EventNode) -> tuple[int, int, str]:
    return _node_order_key(node)


def _node_order_key(node: EventNode) -> tuple[int, int, str]:
    sample_index = node.metadata.get("sample_index", None)
    try:
        order_index = int(sample_index)
    except (TypeError, ValueError):
        order_index = 0
    return (int(node.time_step), order_index, node.node_id)


def extract_math_answer(text: str | None) -> str | None:
    """Extract the last boxed answer from a math response."""

    matches = _BOXED_RE.findall(text or "")
    if not matches:
        return None
    return _normalize_answer(matches[-1])


def _select_final_answer(nodes: list[EventNode], task_type: str) -> str | None:
    if task_type.lower() == "math":
        for node in reversed(nodes):
            if node.node_type == "final_answer":
                answer = extract_math_answer(node.output_content)
                if answer is not None:
                    return f"\\boxed{{{answer}}}"
        for node in reversed(nodes):
            if node.node_type in {"agent_message", "agent_action"}:
                answer = extract_math_answer(node.output_content)
                if answer is not None:
                    return f"\\boxed{{{answer}}}"
        for node in reversed(nodes):
            if node.node_type not in {"verifier_judgment", "router_decision"} and _has_meaningful_math_output(node.output_content):
                return node.output_content
        return None

    final_answer = next((node.output_content for node in reversed(nodes) if node.node_type == "final_answer"), None)
    if final_answer is None and nodes:
        final_answer = nodes[-1].output_content
    return final_answer


def _normalize_answer(answer: str) -> str:
    return " ".join((answer or "").replace("$", "").strip().split()).lower()


def _has_meaningful_math_output(text: str | None) -> bool:
    output = (text or "").strip()
    return bool(output) and not _EMPTY_BOXED_RE.fullmatch(output)
