from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventNode:
    """A trainable or observational event in a multi-agent rollout."""

    node_id: str
    trajectory_id: str
    agent_id: str | None
    role: str | None
    node_type: str
    time_step: int
    input_context: str
    output_content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: str | None = None
    tool_success: bool | None = None
    token_span_start: int | None = None
    token_span_end: int | None = None
    logprob_old: float | None = None
    downstream_consumers: list[str] = field(default_factory=list)
    upstream_dependencies: list[str] = field(default_factory=list)
    final_reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventEdge:
    """A dependency edge between two rollout events."""

    edge_id: str
    trajectory_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventGraph:
    """A graph view of one multi-agent trajectory."""

    trajectory_id: str
    task_id: str
    task_type: str
    task_prompt: str
    nodes: list[EventNode]
    edges: list[EventEdge]
    final_answer: str | None = None
    final_reward: float | None = None
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
