from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizedReward:
    """Normalized node reward with bucket diagnostics."""

    node_id: str
    bucket: str
    raw_reward: float
    advantage: float
    mean: float
    std: float


def normalize_node_rewards(
    node_ids: list[str],
    rewards: list[float],
    agent_ids: list[str | None],
    node_types: list[str | None],
    mode: str = "agent_node_type",
    min_bucket_size: int = 4,
    min_std: float = 1.0e-3,
    epsilon: float = 1.0e-6,
) -> list[NormalizedReward]:
    """Normalize node rewards by progressively broader buckets."""

    primary = [_bucket(agent, node_type, mode) for agent, node_type in zip(agent_ids, node_types, strict=True)]
    fallbacks = [
        [str(agent or "unknown_agent") for agent in agent_ids],
        [str(node_type or "unknown_type") for node_type in node_types],
        ["global" for _ in rewards],
    ]
    selected = _select_buckets(primary, fallbacks, rewards, min_bucket_size)
    stats = _bucket_stats(selected, rewards, min_std)
    normalized = []
    for node_id, reward, bucket in zip(node_ids, rewards, selected, strict=True):
        mean, std = stats[bucket]
        normalized.append(
            NormalizedReward(
                node_id=node_id,
                bucket=bucket,
                raw_reward=float(reward),
                advantage=(float(reward) - mean) / (std + epsilon),
                mean=mean,
                std=std,
            )
        )
    return normalized


def _bucket(agent_id: str | None, node_type: str | None, mode: str) -> str:
    agent = str(agent_id or "unknown_agent")
    ntype = str(node_type or "unknown_type")
    if mode == "agent":
        return agent
    if mode == "node_type":
        return ntype
    if mode == "global":
        return "global"
    return f"{agent}:{ntype}"


def _select_buckets(primary: list[str], fallbacks: list[list[str]], rewards: list[float], min_bucket_size: int) -> list[str]:
    counts = defaultdict(int)
    for bucket in primary:
        counts[bucket] += 1
    selected = list(primary)
    for idx, bucket in enumerate(primary):
        if counts[bucket] >= min_bucket_size:
            continue
        for fallback in fallbacks:
            candidate = fallback[idx]
            candidate_count = sum(1 for item in fallback if item == candidate)
            if candidate_count >= min_bucket_size:
                selected[idx] = candidate
                break
        else:
            selected[idx] = "global"
    return selected


def _bucket_stats(buckets: list[str], rewards: list[float], min_std: float) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for bucket, reward in zip(buckets, rewards, strict=True):
        grouped[bucket].append(float(reward))
    stats = {}
    for bucket, values in grouped.items():
        arr = np.asarray(values, dtype=np.float32)
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std(ddof=0)) if len(arr) > 1 else 1.0
        stats[bucket] = (mean, max(std, min_std))
    return stats
