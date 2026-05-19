from __future__ import annotations

from graphcredit_offline.rewards.format_scorer import format_score


def generic_tool_call_score(output: str, tool_success: bool | None = None) -> float:
    """Score a tool call without calling external services."""

    valid_name = 1.0 if "<search>" in (output or "").lower() else 0.5
    valid_arguments = format_score("tool_call", output)
    success = 0.5 if tool_success is None else float(tool_success)
    return 0.4 * valid_name + 0.4 * valid_arguments + 0.2 * success
