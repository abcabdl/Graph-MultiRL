from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from graphcredit_offline.core.graph_builder import build_event_graph, infer_node_type
from graphcredit_offline.core.schema import EventNode
from graphcredit_offline.core.serialization import append_graph_jsonl
from graphcredit_offline.eval.metrics import harmful_node_ratio, redundant_node_ratio, reward_variance
from graphcredit_offline.rewards.cost import cost_penalty, redundancy_penalty
from graphcredit_offline.rewards.counterfactual import offline_masking_credit
from graphcredit_offline.rewards.downstream_usage import downstream_usage_score
from graphcredit_offline.rewards.fusion import RewardBreakdown, fuse_node_reward
from graphcredit_offline.rewards.math_scorer import MathProcessScorer
from graphcredit_offline.rewards.search_scorer import SearchProcessScorer
from graphcredit_offline.rl.grouping import build_graphcredit_group_index


def graphcredit_enabled(config: Any) -> bool:
    """Read GraphCredit enable flag without requiring the config section."""

    return bool(config.get("graphcredit", {}).get("enabled", False))


def apply_graphcredit_rewards(
    data,
    reward_tensor: torch.Tensor,
    config: Any,
    tokenizers: dict[str, Any] | None = None,
    global_step: int | None = None,
) -> tuple[torch.Tensor, dict[str, list[Any]], dict[str, float]]:
    """Compute offline node-level rewards and return a replacement reward tensor."""

    gc_config = config.get("graphcredit", {})
    if not gc_config.get("enabled", False):
        return reward_tensor, {}, {}

    task_type = str(gc_config.get("task_type", config.agent.get("orchestra_type", "unknown")))
    orchestra_type = str(config.agent.get("orchestra_type", task_type))
    weights = dict(gc_config.get("reward_weights", {}))
    role_weights = dict(gc_config.get("role_reward_weights", {}))
    token_budget = int(gc_config.get("cost", {}).get("token_budget", 512))
    negative_clip = float(gc_config.get("counterfactual", {}).get("negative_credit_clip", -0.5))
    failed_negative_clip = float(gc_config.get("counterfactual", {}).get("failed_negative_credit_clip", negative_clip))
    save_graphs = bool(gc_config.get("logging", {}).get("save_event_graphs", False))
    graph_path = gc_config.get("logging", {}).get("output_path", "outputs/graphcredit_event_graphs.jsonl")
    save_node_rewards = bool(gc_config.get("logging", {}).get("save_node_rewards", False))
    node_reward_path = gc_config.get("logging", {}).get("node_reward_output_path", "outputs/graphcredit_node_rewards.jsonl")

    decoded = _decode_batch(data, tokenizers)
    graph_by_traj = _build_graphs(data, decoded, reward_tensor, task_type, orchestra_type)
    scorer = SearchProcessScorer() if task_type == "search" else MathProcessScorer()

    new_reward = torch.zeros_like(reward_tensor)
    extras: dict[str, list[Any]] = {
        "graphcredit_node_reward": [],
        "graphcredit_process_reward": [],
        "graphcredit_counterfactual_credit": [],
        "graphcredit_downstream_usage": [],
        "graphcredit_cost_penalty": [],
        "graphcredit_redundancy_penalty": [],
        "graphcredit_node_type": [],
        "graphcredit_node_id": [],
        "graphcredit_breakdown_json": [],
    }
    breakdowns: list[RewardBreakdown] = []
    node_reward_records: list[dict[str, Any]] = []
    for i in range(len(data)):
        item = data[i]
        nt = item.non_tensor_batch
        node_id = str(nt.get("graphcredit_node_id", f"node:{i}"))
        traj_uid = str(nt.get("traj_uid", nt.get("uid", "trajectory")))
        graph = graph_by_traj[traj_uid]
        node = next(node for node in graph.nodes if node.node_id == node_id)
        process = scorer.score(graph, node)
        usage = downstream_usage_score(graph, node)
        cf = offline_masking_credit(graph, node, process.score, task_type=task_type)
        cost = cost_penalty(node, token_budget=token_budget)
        redundant = redundancy_penalty(graph, node, cf.credit, usage, cost)
        global_reward = float(graph.final_reward or 0.0)
        node_weights = _weights_for_node(weights, role_weights, node)
        training_cf_credit = cf.credit
        if global_reward <= 0.0:
            training_cf_credit = max(training_cf_credit, failed_negative_clip)
        breakdown = fuse_node_reward(
            node_id=node.node_id,
            global_reward=global_reward,
            process_reward=process.score,
            counterfactual_credit=training_cf_credit,
            downstream_usage=usage,
            cost=cost,
            redundancy=redundant,
            weights=node_weights,
            negative_credit_clip=negative_clip,
            allow_failed_positive=bool(gc_config.get("allow_failed_positive_node_reward", True)),
        )
        breakdown.details.update(
            {
                "process_reason": process.reason,
                "counterfactual_reason": cf.reason,
                "raw_counterfactual_credit": cf.credit,
                "training_counterfactual_credit": training_cf_credit,
            }
        )
        breakdowns.append(breakdown)
        valid_response_length = int(item.batch["attention_mask"][item.batch["prompts"].shape[-1] :].sum().item())
        if valid_response_length > 0:
            new_reward[i, valid_response_length - 1] = torch.tensor(breakdown.node_reward, dtype=torch.float32, device=new_reward.device)
        extras["graphcredit_node_reward"].append(breakdown.node_reward)
        extras["graphcredit_process_reward"].append(breakdown.process_reward)
        extras["graphcredit_counterfactual_credit"].append(breakdown.counterfactual_credit)
        extras["graphcredit_downstream_usage"].append(breakdown.downstream_usage_score)
        extras["graphcredit_cost_penalty"].append(breakdown.cost_penalty)
        extras["graphcredit_redundancy_penalty"].append(breakdown.redundancy_penalty)
        extras["graphcredit_node_type"].append(node.node_type)
        extras["graphcredit_node_id"].append(node.node_id)
        extras["graphcredit_breakdown_json"].append(json.dumps(asdict(breakdown), ensure_ascii=False, sort_keys=True))
        node_reward_records.append(
            {
                "global_step": global_step,
                "sample_index": i,
                "uid": str(nt.get("uid", "")),
                "traj_uid": traj_uid,
                "trajectory_id": graph.trajectory_id,
                "agent_id": node.agent_id,
                "role": node.role,
                "node_id": node.node_id,
                "node_type": node.node_type,
                "time_step": node.time_step,
                "wg_id": str(nt.get("wg_id", "")),
                "global_reward": breakdown.global_reward,
                "node_reward": breakdown.node_reward,
                "process_reward": breakdown.process_reward,
                "counterfactual_credit": breakdown.counterfactual_credit,
                "raw_counterfactual_credit": breakdown.details.get("raw_counterfactual_credit"),
                "downstream_usage_score": breakdown.downstream_usage_score,
                "cost_penalty": breakdown.cost_penalty,
                "redundancy_penalty": breakdown.redundancy_penalty,
                "process_reason": breakdown.details.get("process_reason", ""),
                "counterfactual_reason": breakdown.details.get("counterfactual_reason", ""),
                "final_answer": graph.final_answer,
                "output_content": node.output_content,
            }
        )

    if save_graphs:
        for graph in graph_by_traj.values():
            append_graph_jsonl(graph_path, graph)
    if save_node_rewards:
        _append_jsonl(node_reward_path, node_reward_records)

    metrics = {
        "graphcredit/reward_variance": reward_variance(breakdowns),
        "graphcredit/harmful_node_ratio": harmful_node_ratio(breakdowns),
        "graphcredit/redundant_node_ratio": redundant_node_ratio(breakdowns),
        "graphcredit/node_reward_mean": float(np.mean([item.node_reward for item in breakdowns])) if breakdowns else 0.0,
    }
    return new_reward, extras, metrics


def _append_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _weights_for_node(base_weights: dict[str, float], role_weights: dict[str, Any], node: EventNode) -> dict[str, float]:
    """Merge global reward weights with optional role/node-type overrides."""

    merged = dict(base_weights)
    candidates = [
        str(node.role or ""),
        str(node.agent_id or ""),
        str(node.node_type or ""),
    ]
    normalized_candidates = {_normalize_key(item) for item in candidates if item}
    for key, override in role_weights.items():
        if _normalize_key(str(key)) in normalized_candidates and isinstance(override, dict):
            merged.update({str(k): float(v) for k, v in override.items()})
    return merged


def _normalize_key(value: str) -> str:
    normalized = value.lower().replace(" ", "_").replace("-", "_")
    if normalized.endswith("_agent"):
        normalized = normalized[: -len("_agent")]
    return normalized


def _decode_batch(data, tokenizers: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    decoded = []
    for i in range(len(data)):
        item = data[i]
        wg_id = item.non_tensor_batch.get("wg_id", None)
        tokenizer = tokenizers.get(wg_id) if isinstance(tokenizers, dict) else None
        if tokenizer is None:
            decoded.append(("", str(item.non_tensor_batch.get("graphcredit_output", ""))))
            continue
        prompt_length = item.batch["prompts"].shape[-1]
        valid_prompt_length = int(item.batch["attention_mask"][:prompt_length].sum().item())
        valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
        prompt = tokenizer.decode(item.batch["prompts"][-valid_prompt_length:], skip_special_tokens=False)
        response = tokenizer.decode(item.batch["responses"][:valid_response_length], skip_special_tokens=True)
        decoded.append((prompt, response))
    return decoded


def _build_graphs(data, decoded: list[tuple[str, str]], reward_tensor: torch.Tensor, task_type: str, orchestra_type: str) -> dict[str, Any]:
    by_traj: dict[str, list[EventNode]] = {}
    prompts: dict[str, str] = {}
    final_rewards: dict[str, float] = {}
    for i in range(len(data)):
        item = data[i]
        nt = item.non_tensor_batch
        prompt, response = decoded[i]
        traj_uid = str(nt.get("traj_uid", nt.get("uid", "trajectory")))
        agent_id = nt.get("agent_id", None)
        env_step = int(nt.get("env_step", 0))
        node_type = str(nt.get("node_type", infer_node_type(agent_id, response, orchestra_type)))
        node_id = str(nt.get("graphcredit_node_id", f"{traj_uid}:{env_step}:{agent_id}:{i}"))
        final_reward = float(nt.get("episode_rewards", reward_tensor[i].sum().detach().cpu().item()))
        prompts.setdefault(traj_uid, prompt)
        final_rewards[traj_uid] = final_reward
        by_traj.setdefault(traj_uid, []).append(
            EventNode(
                node_id=node_id,
                trajectory_id=traj_uid,
                agent_id=str(agent_id) if agent_id is not None else None,
                role=str(agent_id) if agent_id is not None else None,
                node_type=node_type,
                time_step=env_step,
                input_context=prompt,
                output_content=response,
                final_reward=final_reward,
                metadata={"sample_index": i, "wg_id": str(nt.get("wg_id", ""))},
            )
        )
    return {
        traj: build_event_graph(
            trajectory_id=traj,
            nodes=nodes,
            task_prompt=prompts.get(traj, ""),
            task_type=task_type,
            final_reward=final_rewards.get(traj, 0.0),
        )
        for traj, nodes in by_traj.items()
    }
