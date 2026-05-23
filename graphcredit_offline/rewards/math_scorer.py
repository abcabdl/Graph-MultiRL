from __future__ import annotations

import re

from graphcredit_offline.core.graph_builder import extract_math_answer, is_answer_node_type, is_solver_reasoning_node_type, is_verifier_node_type
from graphcredit_offline.core.schema import EventGraph, EventNode
from graphcredit_offline.rewards.format_scorer import format_score
from graphcredit_offline.rewards.process_scorers import ProcessScore, clip01
from graphcredit_offline.rewards.verifier_diagnostics import math_verifier_diagnostics


_EMPTY_BOXED_RE = re.compile(r"\\boxed\s*\{\s*\}")


class MathProcessScorer:
    """Rule-based process scoring for Dr. MAS math rollouts."""

    def score(self, graph: EventGraph, node: EventNode) -> ProcessScore:
        final_reward = float(graph.final_reward or node.final_reward or 0.0)
        output = node.output_content or ""
        fmt = format_score(node.node_type, output)
        word_count = len(output.split())
        length_score = _length_score(word_count, node.node_type)
        non_repetitive = 0.0 if _has_repetition(output) else 1.0

        if _is_empty_boxed(output):
            return ProcessScore(node_id=node.node_id, score=0.0, reason="empty boxed answer receives no process reward")

        if is_answer_node_type(node.node_type):
            score = 0.78 * final_reward + 0.16 * fmt + 0.06 * length_score
            reason = "final answer combines correctness, answer format, and mild length support"
        elif is_verifier_node_type(node.node_type):
            diagnostics = math_verifier_diagnostics(graph, node)
            if node.node_type == "router_decision":
                score = max(0.0, diagnostics.verifier_reward)
            else:
                score = max(0.0, diagnostics.verifier_reward)
            if diagnostics.verdict == "reject" and diagnostics.explained and diagnostics.after_correct and not diagnostics.before_correct:
                score = min(1.0, score + 0.15)
            if diagnostics.verdict == "approve" and not diagnostics.before_correct:
                score = max(0.0, score - 0.15)
            reason = diagnostics.reason
        else:
            if _is_too_short(output) and final_reward <= 0:
                score = 0.0
                reason = "too-short incorrect solver output receives no process reward"
            else:
                has_math = _has_math_structure(output)
                if final_reward <= 0:
                    has_boxed_answer = extract_math_answer(output) is not None
                    score = 0.10 * fmt + 0.12 * length_score + 0.10 * non_repetitive + 0.18 * float(has_math)
                    score = min(score, 0.20 if has_boxed_answer else 0.25)
                    reason = "failed solver output receives only capped local structure credit"
                else:
                    score = 0.12 * fmt + 0.18 * length_score + 0.15 * non_repetitive + 0.20 * final_reward + 0.35 * float(has_math)
                    if is_solver_reasoning_node_type(node.node_type):
                        score += 0.05 * min(word_count / 120.0, 1.0)
                    reason = "solver score uses nontrivial math structure, final success, and reasoning support"
        return ProcessScore(node_id=node.node_id, score=clip01(score), reason=reason)


def _has_repetition(text: str) -> bool:
    words = [word.lower() for word in (text or "").split()]
    if len(words) < 20:
        return False
    return len(set(words)) / len(words) < 0.35


def _is_empty_boxed(text: str) -> bool:
    return bool(_EMPTY_BOXED_RE.search(text or ""))


def _is_too_short(text: str, min_words: int = 8) -> bool:
    return len((text or "").split()) < min_words


def _length_score(word_count: int, node_type: str) -> float:
    if word_count < 8:
        return 0.0
    if node_type in {"solver_reasoning", "agent_message", "agent_action"}:
        if word_count < 24:
            return 0.25
        if word_count <= 256:
            return 1.0
        if word_count <= 512:
            return 0.6
        return 0.2
    if word_count <= 512:
        return 1.0
    return 0.2


def _has_math_structure(text: str) -> bool:
    output = text or ""
    has_answer = extract_math_answer(output) is not None
    has_work = any(marker in output for marker in ["=", "$", "\\frac", "\\sqrt", "therefore", "Thus", "So"])
    return has_answer and has_work and not _is_too_short(output)
