from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from graphcredit_offline.core.schema import EventEdge, EventGraph, EventNode


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert graph dataclasses to plain dictionaries."""

    return asdict(obj)


def graph_to_json(graph: EventGraph) -> str:
    """Serialize an event graph to a stable JSON string."""

    return json.dumps(dataclass_to_dict(graph), ensure_ascii=False, sort_keys=True)


def graph_from_dict(data: dict[str, Any]) -> EventGraph:
    """Deserialize an event graph from a dictionary."""

    return EventGraph(
        trajectory_id=data["trajectory_id"],
        task_id=data.get("task_id", data["trajectory_id"]),
        task_type=data.get("task_type", "unknown"),
        task_prompt=data.get("task_prompt", ""),
        nodes=[EventNode(**node) for node in data.get("nodes", [])],
        edges=[EventEdge(**edge) for edge in data.get("edges", [])],
        final_answer=data.get("final_answer"),
        final_reward=data.get("final_reward"),
        ground_truth=data.get("ground_truth"),
        metadata=data.get("metadata", {}),
    )


def append_graph_jsonl(path: str | Path, graph: EventGraph) -> None:
    """Append one graph to a JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(graph_to_json(graph) + "\n")
