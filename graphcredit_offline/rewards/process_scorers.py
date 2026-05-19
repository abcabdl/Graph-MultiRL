from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from graphcredit_offline.core.schema import EventGraph, EventNode


@dataclass
class ProcessScore:
    """A bounded, explainable process score for one node."""

    node_id: str
    score: float
    reason: str
    failure_type: str = "none"
    details: dict[str, Any] = field(default_factory=dict)


class ProcessScorer(Protocol):
    def score(self, graph: EventGraph, node: EventNode) -> ProcessScore:
        ...


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
