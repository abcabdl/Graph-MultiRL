from __future__ import annotations

from graphcredit_offline.core.graph_builder import infer_node_type


def make_node_id(trajectory_id: str, agent_id: str | None, env_step: int, local_index: int) -> str:
    """Create a deterministic node id from rollout metadata."""

    safe_agent = (agent_id or "agent").replace(" ", "_").replace("/", "_")
    return f"{trajectory_id}:{env_step}:{safe_agent}:{local_index}"


def annotate_rollout_item(item: dict, local_index: int, orchestra_type: str | None = None) -> dict:
    """Add GraphCredit metadata to one Dr. MAS rollout item."""

    trajectory_id = str(item.get("traj_uid", item.get("uid", "trajectory")))
    agent_id = item.get("agent_id")
    env_step = int(item.get("env_step", 0))
    output_content = str(item.get("graphcredit_output", ""))
    node_type = item.get("node_type") or infer_node_type(agent_id, output_content, orchestra_type)
    item["node_type"] = node_type
    item["graphcredit_node_id"] = item.get("graphcredit_node_id") or make_node_id(trajectory_id, agent_id, env_step, local_index)
    item["graphcredit_trainable"] = node_type != "tool_result"
    return item
