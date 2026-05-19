from __future__ import annotations

import numpy as np


def build_graphcredit_group_index(data, mode: str = "agent_node_type") -> np.ndarray:
    """Build grouping ids for GraphCredit advantage normalization."""

    uid = data.non_tensor_batch["uid"]
    agent_ids = data.non_tensor_batch.get("agent_id", np.array(["agent"] * len(uid), dtype=object))
    node_types = data.non_tensor_batch.get("node_type", np.array(["agent_action"] * len(uid), dtype=object))
    if mode == "agent":
        return np.array([f"{u}_{agent}" for u, agent in zip(uid, agent_ids, strict=True)], dtype=object)
    if mode == "node_type":
        return np.array([f"{u}_{node_type}" for u, node_type in zip(uid, node_types, strict=True)], dtype=object)
    if mode == "global":
        return uid
    return np.array([f"{u}_{agent}_{node_type}" for u, agent, node_type in zip(uid, agent_ids, node_types, strict=True)], dtype=object)
