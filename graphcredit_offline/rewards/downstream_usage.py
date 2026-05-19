from __future__ import annotations

from graphcredit_offline.core.graph_builder import lexical_overlap
from graphcredit_offline.core.schema import EventGraph, EventNode
from graphcredit_offline.rewards.process_scorers import clip01


def downstream_usage_score(graph: EventGraph, node: EventNode) -> float:
    """Estimate how much a node is consumed downstream."""

    downstream = [other for other in graph.nodes if other.time_step >= node.time_step and other.node_id != node.node_id]
    if not downstream:
        return 0.0
    direct_reference = max((1.0 if node.node_id in other.input_context else 0.0 for other in downstream), default=0.0)
    path_score = min(
        sum(1 for edge in graph.edges if edge.source_node_id == node.node_id and edge.edge_type in {"communication_edge", "evidence_edge"}) / 3.0,
        1.0,
    )
    final_overlap = lexical_overlap(node.output_content, graph.final_answer or "")
    return clip01(0.4 * direct_reference + 0.3 * path_score + 0.3 * final_overlap)
