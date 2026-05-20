from __future__ import annotations

from dataclasses import dataclass

from graphcredit_offline.core.graph_builder import extract_math_answer, node_order_key
from graphcredit_offline.core.schema import EventGraph, EventNode


@dataclass
class VerifierDiagnostics:
    """Offline signals for judging whether a verifier helped correction."""

    verdict: str
    explained: bool
    previous_answer: str | None
    next_answer: str | None
    final_answer: str | None
    before_correct: bool
    after_correct: bool
    answer_changed: bool
    correction_gain: float
    verifier_reward: float
    reason: str


def verifier_verdict(output: str | None) -> str:
    text = (output or "").lower()
    if "<verify>approve</verify>" in text:
        return "approve"
    if "<verify>reject</verify>" in text:
        return "reject"
    return "other"


def verifier_explained(output: str | None) -> bool:
    text = output or ""
    cleaned = text.lower().replace("<verify>approve</verify>", "").replace("<verify>reject</verify>", "").strip()
    return len(cleaned.split()) >= 4


def math_verifier_diagnostics(graph: EventGraph, node: EventNode) -> VerifierDiagnostics:
    verdict = verifier_verdict(node.output_content)
    explained = verifier_explained(node.output_content)
    previous_solver = _nearest_solver_before(graph, node)
    next_solver = _nearest_solver_after(graph, node)
    previous_answer = extract_math_answer(previous_solver.output_content) if previous_solver else None
    next_answer = extract_math_answer(next_solver.output_content) if next_solver else None
    final_answer = extract_math_answer(graph.final_answer)
    final_success = float(graph.final_reward or node.final_reward or 0.0) > 0.0
    before_correct = bool(final_success and previous_answer is not None and previous_answer == final_answer)
    after_correct = bool(final_success and (next_answer == final_answer or (next_answer is None and previous_answer == final_answer)))
    answer_changed = bool(previous_answer is not None and next_answer is not None and previous_answer != next_answer)
    correction_gain = float(after_correct) - float(before_correct)

    reward = 0.0
    reason = "verifier did not provide an actionable verified correction"
    if verdict == "approve" and before_correct and explained:
        reward = 0.4
        reason = "verifier approved a correct prior solver answer"
    elif verdict == "reject" and correction_gain > 0.0 and explained:
        reward = 0.5
        reason = "verifier rejected a wrong prior answer and the next answer became correct"
    elif verdict == "reject" and before_correct:
        reward = -0.4
        reason = "verifier rejected a correct prior solver answer"
    elif verdict == "approve" and not before_correct:
        reward = -0.5
        reason = "verifier approved an unverified or wrong prior solver answer"
    elif verdict == "reject" and not final_success and explained and (previous_answer is not None or answer_changed):
        reward = 0.1
        reason = "verifier rejected a failed trajectory answer but did not produce a verified correction"
    elif verdict == "reject" and explained:
        reward = 0.0
        reason = "verifier rejection had explanation but no verified correction gain"

    return VerifierDiagnostics(
        verdict=verdict,
        explained=explained,
        previous_answer=previous_answer,
        next_answer=next_answer,
        final_answer=final_answer,
        before_correct=before_correct,
        after_correct=after_correct,
        answer_changed=answer_changed,
        correction_gain=correction_gain,
        verifier_reward=reward,
        reason=reason,
    )


def _nearest_solver_before(graph: EventGraph, node: EventNode) -> EventNode | None:
    target_key = node_order_key(node)
    candidates = [
        item
        for item in graph.nodes
        if item.node_id != node.node_id
        and node_order_key(item) < target_key
        and item.node_type in {"agent_message", "agent_action"}
    ]
    return max(candidates, key=node_order_key, default=None)


def _nearest_solver_after(graph: EventGraph, node: EventNode) -> EventNode | None:
    target_key = node_order_key(node)
    candidates = [
        item
        for item in graph.nodes
        if item.node_id != node.node_id
        and node_order_key(item) > target_key
        and item.node_type in {"agent_message", "agent_action"}
    ]
    return min(candidates, key=node_order_key, default=None)
