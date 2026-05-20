from __future__ import annotations

import re

from graphcredit_offline.core.graph_builder import extract_math_answer
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
        length_score = _length_score(word_count)
        non_repetitive = 0.0 if _has_repetition(output) else 1.0

        if _is_empty_boxed(output):
            return ProcessScore(node_id=node.node_id, score=0.0, reason="empty boxed answer receives no process reward")

        if node.node_type == "final_answer":
            score = 0.8 * final_reward + 0.2 * fmt
            reason = "final answer combines correctness and required answer format"
        elif node.node_type == "verifier_judgment":
            diagnostics = math_verifier_diagnostics(graph, node)
            score = max(0.0, diagnostics.verifier_reward)
            reason = diagnostics.reason
        else:
            if _is_too_short(output) and final_reward <= 0:
                score = 0.0
                reason = "too-short incorrect solver output receives no process reward"
            else:
                has_math = _has_math_structure(output)
                if final_reward <= 0:
                    has_boxed_answer = extract_math_answer(output) is not None
                    score = 0.10 * fmt + 0.10 * length_score + 0.10 * non_repetitive + 0.15 * float(has_math)
                    score = min(score, 0.20 if has_boxed_answer else 0.25)
                    reason = "failed solver output receives only capped local structure credit"
                else:
                    score = 0.15 * fmt + 0.2 * length_score + 0.2 * non_repetitive + 0.2 * final_reward + 0.25 * float(has_math)
                    reason = "solver score uses nontrivial math structure and final success"
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


def _length_score(word_count: int) -> float:
    if word_count < 8:
        return 0.0
    if word_count <= 512:
        return 1.0
    return 0.2


def _has_math_structure(text: str) -> bool:
    output = text or ""
    has_answer = extract_math_answer(output) is not None
    has_work = any(marker in output for marker in ["=", "$", "\\frac", "\\sqrt", "therefore", "Thus", "So"])
    return has_answer and has_work and not _is_too_short(output)
