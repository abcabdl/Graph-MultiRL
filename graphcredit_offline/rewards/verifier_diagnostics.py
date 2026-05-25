from __future__ import annotations

from dataclasses import dataclass
import re

from graphcredit_offline.core.graph_builder import extract_math_answer, is_solver_node_type, node_order_key
from graphcredit_offline.core.schema import EventGraph, EventNode
from graphcredit_offline.core.verify_tags import VERIFY_TAG_RE, last_verifier_verdict, verify_tags

_TEXT_SAYS_CORRECT_RE = re.compile(
    r"\b(solution|answer|calculation|reasoning|steps?)\s+(is|are|looks?|seems?)\s+(actually\s+)?correct\b|"
    r"\b(final answer|conclusion)\s+(is|are)\s+correct\b|"
    r"\bshould\s+be\s+approved\b",
    re.IGNORECASE,
)
_TEXT_SAYS_ERROR_RE = re.compile(
    r"\b(contains|has|made)\s+(an?\s+)?(error|mistake|flaw)\b|"
    r"\bincorrect\b|\bflawed\b|\bshould\s+be\s+rejected\b",
    re.IGNORECASE,
)


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
    verify_tag_count: int = 0
    format_valid: bool = False
    contradiction: bool = False


def verifier_verdict(output: str | None) -> str:
    """Return the last explicit verifier verdict.

    Models often produce a tentative tag early and then revise themselves later.
    The final tag is the one closest to the actual decision, while multi-tag
    output is tracked separately and penalized as a format issue.
    """

    return last_verifier_verdict(output)


def verifier_format_stats(output: str | None) -> tuple[int, bool, bool]:
    tags = verify_tags(output)
    tag_count = len(tags)
    verdict = tags[-1] if tags else "other"
    text = output or ""
    contradiction = bool(
        (verdict == "reject" and _TEXT_SAYS_CORRECT_RE.search(text))
        or (verdict == "approve" and _TEXT_SAYS_ERROR_RE.search(text))
    )
    return tag_count, tag_count == 1, contradiction


def verifier_explained(output: str | None) -> bool:
    text = output or ""
    cleaned = VERIFY_TAG_RE.sub("", text.lower()).strip()
    return len(cleaned.split()) >= 4


def math_verifier_diagnostics(graph: EventGraph, node: EventNode) -> VerifierDiagnostics:
    verdict = verifier_verdict(node.output_content)
    explained = verifier_explained(node.output_content)
    verify_tag_count, format_valid, contradiction = verifier_format_stats(node.output_content)
    previous_solver = _nearest_solver_before(graph, node)
    next_solver = _nearest_solver_after(graph, node)
    previous_answer = extract_math_answer(previous_solver.output_content) if previous_solver else None
    next_answer = extract_math_answer(next_solver.output_content) if next_solver else None
    verifier_answer = extract_math_answer(node.output_content)
    final_answer = extract_math_answer(graph.final_answer)
    final_success = float(graph.final_reward or node.final_reward or 0.0) > 0.0
    before_correct = bool(final_success and previous_answer is not None and previous_answer == final_answer)
    after_correct = bool(
        final_success
        and (
            next_answer == final_answer
            or verifier_answer == final_answer
            or (next_answer is None and previous_answer == final_answer)
        )
    )
    answer_changed = bool(previous_answer is not None and next_answer is not None and previous_answer != next_answer)
    correction_gain = float(after_correct) - float(before_correct)

    reward = 0.0
    reason = "verifier did not provide an actionable verified correction"
    if verdict == "approve" and before_correct:
        reward = 0.60 if explained else 0.45
        reason = "verifier approved a correct prior solver answer"
    elif verdict == "reject" and correction_gain > 0.0 and explained:
        reward = 0.60 if after_correct else 0.45
        reason = "verifier rejected a wrong prior answer and the next answer became correct"
    elif verdict == "reject" and before_correct:
        reward = -0.80
        reason = "verifier rejected a correct prior solver answer"
    elif verdict == "approve" and not before_correct:
        reward = -0.8
        reason = "verifier approved an unverified or wrong prior solver answer"
    elif verdict == "reject" and not final_success and explained and (previous_answer is not None or answer_changed):
        reward = 0.15 if answer_changed else 0.08
        reason = "verifier rejected a failed trajectory answer but did not produce a verified correction"
    elif verdict == "reject" and explained:
        reward = 0.02 if answer_changed else 0.0
        reason = "verifier rejection had explanation but no verified correction gain"
    elif verdict == "other":
        reward = -0.25
        reason = "verifier did not emit a valid verify tag"

    if verify_tag_count != 1:
        reward = min(reward, 0.0) - 0.25
        reason = f"{reason}; invalid verify tag count"
    if contradiction:
        reward = min(reward, 0.0) - 0.35
        reason = f"{reason}; verdict contradicts explanation text"

    return VerifierDiagnostics(
        verdict=verdict,
        explained=explained,
        previous_answer=previous_answer,
        next_answer=verifier_answer or next_answer,
        final_answer=final_answer,
        before_correct=before_correct,
        after_correct=after_correct,
        answer_changed=answer_changed,
        correction_gain=correction_gain,
        verifier_reward=reward,
        reason=reason,
        verify_tag_count=verify_tag_count,
        format_valid=format_valid,
        contradiction=contradiction,
    )


def _nearest_solver_before(graph: EventGraph, node: EventNode) -> EventNode | None:
    target_key = node_order_key(node)
    candidates = [
        item
        for item in graph.nodes
        if item.node_id != node.node_id
        and node_order_key(item) < target_key
        and is_solver_node_type(item.node_type)
    ]
    return max(candidates, key=node_order_key, default=None)


def _nearest_solver_after(graph: EventGraph, node: EventNode) -> EventNode | None:
    target_key = node_order_key(node)
    candidates = [
        item
        for item in graph.nodes
        if item.node_id != node.node_id
        and node_order_key(item) > target_key
        and is_solver_node_type(item.node_type)
    ]
    return min(candidates, key=node_order_key, default=None)
