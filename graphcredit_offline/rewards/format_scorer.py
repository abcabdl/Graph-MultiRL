from __future__ import annotations

from graphcredit_offline.core.verify_tags import verify_tags


def has_balanced_tag(text: str, start_tag: str, end_tag: str) -> bool:
    """Return true when both required tags appear in order."""

    start = (text or "").find(start_tag)
    end = (text or "").find(end_tag)
    return start >= 0 and end > start


def format_score(node_type: str, output: str) -> float:
    """Score common Dr. MAS output formats."""

    text = output or ""
    if node_type == "tool_call":
        return 1.0 if has_balanced_tag(text, "<search>", "</search>") else 0.0
    if node_type == "router_decision":
        low = text.lower()
        return 1.0 if "<verify>yes</verify>" in low or "<verify>no</verify>" in low else 0.0
    if node_type in {"verifier_judgment", "verifier_check", "verifier_correction"}:
        tag_count = len(verify_tags(text))
        return 1.0 if tag_count == 1 else 0.0
    if node_type in {"final_answer", "solver_final_answer"}:
        return 1.0 if ("<answer>" in text and "</answer>" in text) or "\\boxed" in text else 0.5
    if node_type in {"solver_reasoning", "agent_message", "agent_action"}:
        return 1.0 if text.strip() else 0.0
    return 1.0 if text.strip() else 0.0
