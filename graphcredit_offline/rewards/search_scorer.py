from __future__ import annotations

from graphcredit_offline.core.graph_builder import lexical_overlap
from graphcredit_offline.core.schema import EventGraph, EventNode
from graphcredit_offline.rewards.format_scorer import format_score
from graphcredit_offline.rewards.process_scorers import ProcessScore, clip01


class SearchProcessScorer:
    """Offline process scoring for search/router/answer agents."""

    def score(self, graph: EventGraph, node: EventNode) -> ProcessScore:
        final_reward = float(graph.final_reward or node.final_reward or 0.0)
        output = node.output_content or ""
        fmt = format_score(node.node_type, output)
        if node.node_type == "tool_call":
            query = _between(output, "<search>", "</search>") or output
            specificity = 1.0 if 3 <= len(query.split()) <= 18 else 0.4
            entity_overlap = lexical_overlap(graph.task_prompt, query)
            repeated = any(
                other.node_id != node.node_id and other.node_type == "tool_call" and lexical_overlap(other.output_content, output) > 0.85
                for other in graph.nodes
            )
            score = 0.3 * fmt + 0.25 * specificity + 0.25 * entity_overlap + 0.2 * (0.0 if repeated else 1.0)
            reason = "search query score uses format, specificity, task-entity coverage, and non-redundancy"
        elif node.node_type == "router_decision":
            says_yes = "<verify>yes</verify>" in output.lower()
            score = 1.0 if says_yes and final_reward > 0 else 0.8 if (not says_yes and final_reward <= 0) else 0.25
            reason = "router score checks whether stop/continue decision agrees with final outcome"
        elif node.node_type == "final_answer":
            support = max((lexical_overlap(prev.output_content, output) for prev in graph.nodes if prev.node_id != node.node_id), default=0.0)
            score = 0.7 * final_reward + 0.2 * fmt + 0.1 * support
            reason = "answer score combines final correctness, answer format, and evidence overlap"
        else:
            score = 0.5 * fmt + 0.5 * final_reward
            reason = "generic search node score uses format and final outcome"
        return ProcessScore(node_id=node.node_id, score=clip01(score), reason=reason)


def _between(text: str, start: str, end: str) -> str | None:
    low = text.lower()
    start_idx = low.find(start)
    end_idx = low.find(end)
    if start_idx < 0 or end_idx <= start_idx:
        return None
    return text[start_idx + len(start) : end_idx].strip()
