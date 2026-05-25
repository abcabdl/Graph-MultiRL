from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from graphcredit_offline.core.graph_builder import build_event_graph, infer_node_type, refine_node_type
from graphcredit_offline.core.graph_builder import is_solver_node_type, is_verifier_node_type
from graphcredit_offline.core.schema import EventNode
from graphcredit_offline.core.serialization import append_graph_jsonl
from graphcredit_offline.core.text_sanitize import sanitize_text
from graphcredit_offline.eval.metrics import harmful_node_ratio, redundant_node_ratio, reward_variance
from graphcredit_offline.rewards.cost import cost_penalty, redundancy_penalty
from graphcredit_offline.rewards.counterfactual import offline_masking_credit
from graphcredit_offline.rewards.downstream_usage import downstream_usage_score
from graphcredit_offline.rewards.fusion import RewardBreakdown, fuse_node_reward
from graphcredit_offline.rewards.math_scorer import MathProcessScorer
from graphcredit_offline.rewards.search_scorer import SearchProcessScorer
from graphcredit_offline.rewards.verifier_diagnostics import math_verifier_diagnostics
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

    gc_config = _to_plain_container(gc_config)
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
    reward_mode = str(gc_config.get("reward_mode", "fusion"))

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
    pending_items: list[dict[str, Any]] = []
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
        global_reward = float(graph.final_reward or 0.0)
        cost = _failed_math_solver_cost_floor(task_type, global_reward, node, cost)
        redundant = redundancy_penalty(graph, node, cf.credit, usage, cost)
        node_weights = _weights_for_node(weights, role_weights, node)
        verifier_diagnostics = math_verifier_diagnostics(graph, node) if task_type == "math" and node.node_type in {"verifier_judgment", "verifier_check", "verifier_correction"} else None
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
                "applied_weights": dict(node_weights),
                "verifier_diagnostics": asdict(verifier_diagnostics) if verifier_diagnostics is not None else {},
            }
        )
        breakdowns.append(breakdown)
        valid_response_length = int(item.batch["attention_mask"][item.batch["prompts"].shape[-1] :].sum().item())
        pending_items.append(
            {
                "index": i,
                "valid_response_length": valid_response_length,
                "node": node,
                "graph": graph,
                "nt": nt,
                "breakdown": breakdown,
            }
        )

    if reward_mode == "outcome_redistribution":
        _apply_outcome_redistribution(pending_items, gc_config)
    elif reward_mode != "fusion":
        raise ValueError(f"Unsupported graphcredit.reward_mode={reward_mode!r}. Expected 'fusion' or 'outcome_redistribution'.")

    for pending in pending_items:
        i = pending["index"]
        valid_response_length = pending["valid_response_length"]
        node = pending["node"]
        graph = pending["graph"]
        nt = pending["nt"]
        breakdown = pending["breakdown"]
        traj_uid = str(nt.get("traj_uid", nt.get("uid", "trajectory")))
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
                "applied_weights": breakdown.details.get("applied_weights", {}),
                "verifier_diagnostics": breakdown.details.get("verifier_diagnostics", {}),
                "final_answer": graph.final_answer,
                "output_content": sanitize_text(node.output_content),
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
    metrics.update(_outcome_reward_metrics(pending_items, reward_mode))
    return new_reward, extras, metrics


def _apply_outcome_redistribution(pending_items: list[dict[str, Any]], gc_config: dict[str, Any]) -> None:
    """Conservatively redistribute final outcome reward across eligible graph nodes.

    This mode keeps final correctness as the reward anchor: failed trajectories do
    not receive positive node rewards, and successful trajectories only split the
    observed final reward among selected roles. Process/counterfactual scores are
    used only as allocation weights, not as independent reward sources.
    """

    outcome_cfg = _to_plain_container(gc_config.get("outcome_redistribution", {}))
    train_roles = outcome_cfg.get("train_roles", gc_config.get("train_roles", [])) or []
    failed_node_reward = float(outcome_cfg.get("failed_node_reward", 0.0))
    success_reward_scale = float(outcome_cfg.get("success_reward_scale", 1.0))
    credit_basis = str(outcome_cfg.get("credit_basis", "counterfactual"))
    solver_min_success_share = float(outcome_cfg.get("solver_min_success_share", 0.55))
    verifier_max_success_share = float(outcome_cfg.get("verifier_max_success_share", 0.35))
    verifier_hard_penalty = bool(outcome_cfg.get("verifier_hard_penalty", True))
    verifier_negative_reward_scale = float(outcome_cfg.get("verifier_negative_reward_scale", 0.25))
    redistribute_verifier_penalty_share = bool(outcome_cfg.get("redistribute_verifier_penalty_share", True))

    by_traj: dict[str, list[dict[str, Any]]] = {}
    for pending in pending_items:
        graph = pending["graph"]
        by_traj.setdefault(graph.trajectory_id, []).append(pending)

    for traj_items in by_traj.values():
        graph = traj_items[0]["graph"]
        global_reward = float(graph.final_reward or 0.0)
        eligible = [item for item in traj_items if _role_is_trainable(item["node"], train_roles)]

        for item in traj_items:
            breakdown = item["breakdown"]
            breakdown.details["pre_redistribution_node_reward"] = breakdown.node_reward
            breakdown.details["reward_mode"] = "outcome_redistribution"
            breakdown.details["outcome_credit_basis"] = credit_basis
            breakdown.details["outcome_train_roles"] = list(train_roles)

        if global_reward <= 0.0 or not eligible:
            for item in traj_items:
                item["breakdown"].node_reward = failed_node_reward if item in eligible else 0.0
            if verifier_hard_penalty:
                _apply_verifier_hard_penalties(
                    traj_items,
                    train_roles,
                    negative_reward_scale=verifier_negative_reward_scale,
                    redistribute_positive_share=False,
                    credit_basis=credit_basis,
                )
            continue

        reward_mass = global_reward * success_reward_scale
        for item in traj_items:
            item["breakdown"].node_reward = 0.0
        shares = _allocate_success_reward(
            eligible,
            credit_basis,
            solver_min_success_share=solver_min_success_share,
            verifier_max_success_share=verifier_max_success_share,
        )
        for item, share in zip(eligible, shares, strict=True):
            item["breakdown"].node_reward = float(reward_mass * share)
            item["breakdown"].details["outcome_reward_share"] = float(share)
        if verifier_hard_penalty:
            _apply_verifier_hard_penalties(
                traj_items,
                train_roles,
                negative_reward_scale=verifier_negative_reward_scale,
                redistribute_positive_share=redistribute_verifier_penalty_share,
                credit_basis=credit_basis,
            )


def _apply_verifier_hard_penalties(
    traj_items: list[dict[str, Any]],
    train_roles: list[str],
    negative_reward_scale: float,
    redistribute_positive_share: bool,
    credit_basis: str,
) -> None:
    if negative_reward_scale <= 0.0:
        return

    penalized: list[tuple[dict[str, Any], float]] = []
    for item in traj_items:
        node = item["node"]
        if not is_verifier_node_type(node.node_type) or not _role_is_trainable(node, train_roles):
            continue
        breakdown = item["breakdown"]
        diagnostic_reward = _negative_verifier_diagnostic_reward(
            breakdown.details.get("verifier_diagnostics", {})
        )
        if diagnostic_reward >= 0.0:
            continue

        old_reward = float(breakdown.node_reward)
        penalty_reward = float(diagnostic_reward * negative_reward_scale)
        breakdown.node_reward = penalty_reward
        breakdown.details["verifier_hard_penalty_applied"] = True
        breakdown.details["pre_verifier_penalty_node_reward"] = old_reward
        breakdown.details["verifier_penalty_reward"] = penalty_reward
        breakdown.details["verifier_penalty_source_reward"] = diagnostic_reward
        penalized.append((item, max(old_reward, 0.0)))

    if not redistribute_positive_share or not penalized:
        return

    returned_mass = sum(old_positive for _, old_positive in penalized)
    if returned_mass <= 1e-8:
        return

    penalized_ids = {id(item) for item, _ in penalized}
    recipients = [
        item
        for item in traj_items
        if id(item) not in penalized_ids
        and _role_is_trainable(item["node"], train_roles)
        and float(item["breakdown"].node_reward) >= 0.0
    ]
    if not recipients:
        return

    weights = [max(float(item["breakdown"].node_reward), 0.0) for item in recipients]
    if sum(weights) <= 1e-8:
        weights = [_outcome_credit(item["breakdown"], credit_basis) for item in recipients]
    shares = _normalize_shares(weights)
    for item, share in zip(recipients, shares, strict=True):
        bonus = float(returned_mass * share)
        breakdown = item["breakdown"]
        breakdown.node_reward = float(breakdown.node_reward + bonus)
        breakdown.details["verifier_penalty_redistributed_bonus"] = float(
            breakdown.details.get("verifier_penalty_redistributed_bonus", 0.0) + bonus
        )


def _negative_verifier_diagnostic_reward(diagnostics: Any) -> float:
    diagnostics = _to_plain_container(diagnostics) or {}
    if not isinstance(diagnostics, Mapping):
        return 0.0
    try:
        verifier_reward = float(diagnostics.get("verifier_reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        verifier_reward = 0.0
    if verifier_reward < 0.0:
        return verifier_reward

    format_valid = bool(diagnostics.get("format_valid", True))
    contradiction = bool(diagnostics.get("contradiction", False))
    try:
        verify_tag_count = int(diagnostics.get("verify_tag_count", 1) or 0)
    except (TypeError, ValueError):
        verify_tag_count = 0
    if not format_valid or contradiction or verify_tag_count != 1:
        return -0.5
    return 0.0


def _outcome_credit(breakdown: RewardBreakdown, credit_basis: str) -> float:
    cf_credit = max(float(breakdown.counterfactual_credit), 0.0)
    if credit_basis == "counterfactual":
        return cf_credit
    if credit_basis == "process":
        return max(float(breakdown.process_reward), 0.0)
    if credit_basis in {"counterfactual_process", "hybrid"}:
        return cf_credit * max(float(breakdown.process_reward), 0.0)
    if credit_basis == "uniform":
        return 1.0
    raise ValueError(
        f"Unsupported graphcredit.outcome_redistribution.credit_basis={credit_basis!r}. "
        "Expected counterfactual, process, counterfactual_process, hybrid, or uniform."
    )


def _allocate_success_reward(
    eligible: list[dict[str, Any]],
    credit_basis: str,
    solver_min_success_share: float,
    verifier_max_success_share: float,
) -> list[float]:
    eps = 1e-8
    credits = [_outcome_credit(item["breakdown"], credit_basis) for item in eligible]
    if sum(credits) <= eps:
        credits = [1.0 for _ in eligible]
    shares = _normalize_shares(credits)

    solver_indices = [idx for idx, item in enumerate(eligible) if is_solver_node_type(item["node"].node_type)]
    verifier_indices = [idx for idx, item in enumerate(eligible) if is_verifier_node_type(item["node"].node_type)]

    solver_min_success_share = clip_fraction(solver_min_success_share)
    verifier_max_success_share = clip_fraction(verifier_max_success_share)
    shares = _raise_group_floor(shares, solver_indices, solver_min_success_share)
    shares = _cap_group_share(shares, verifier_indices, verifier_max_success_share)
    return _normalize_shares(shares)


def _normalize_shares(values: list[float]) -> list[float]:
    total = sum(max(float(value), 0.0) for value in values)
    if total <= 1e-8:
        return [1.0 / len(values) for values in values] if values else []
    return [max(float(value), 0.0) / total for value in values]


def _raise_group_floor(shares: list[float], indices: list[int], floor: float) -> list[float]:
    if not shares or not indices or floor <= 0.0:
        return shares
    group_sum = sum(shares[idx] for idx in indices)
    if group_sum >= floor:
        return shares
    other_indices = [idx for idx in range(len(shares)) if idx not in set(indices)]
    other_sum = sum(shares[idx] for idx in other_indices)
    new_shares = list(shares)
    if group_sum <= 1e-8:
        per_group = floor / len(indices)
        for idx in indices:
            new_shares[idx] = per_group
    else:
        scale = floor / group_sum
        for idx in indices:
            new_shares[idx] *= scale
    remaining = max(0.0, 1.0 - floor)
    if other_indices and other_sum > 1e-8:
        for idx in other_indices:
            new_shares[idx] = shares[idx] * remaining / other_sum
    elif other_indices:
        per_other = remaining / len(other_indices)
        for idx in other_indices:
            new_shares[idx] = per_other
    return new_shares


def _cap_group_share(shares: list[float], indices: list[int], cap: float) -> list[float]:
    if not shares or not indices or cap >= 1.0:
        return shares
    group_sum = sum(shares[idx] for idx in indices)
    if group_sum <= cap:
        return shares
    other_indices = [idx for idx in range(len(shares)) if idx not in set(indices)]
    other_sum = sum(shares[idx] for idx in other_indices)
    new_shares = list(shares)
    for idx in indices:
        new_shares[idx] = shares[idx] * cap / group_sum
    remaining = max(0.0, 1.0 - cap)
    if other_indices and other_sum > 1e-8:
        for idx in other_indices:
            new_shares[idx] = shares[idx] * remaining / other_sum
    elif other_indices:
        per_other = remaining / len(other_indices)
        for idx in other_indices:
            new_shares[idx] = per_other
    return new_shares


def clip_fraction(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _role_is_trainable(node: EventNode, train_roles: list[str]) -> bool:
    if not train_roles:
        return True
    trainable = {_normalize_key(str(role)) for role in train_roles}
    candidates = {
        _normalize_key(str(node.role or "")),
        _normalize_key(str(node.agent_id or "")),
        _normalize_key(str(node.node_type or "")),
    }
    return bool(trainable.intersection(candidates))


def _outcome_reward_metrics(pending_items: list[dict[str, Any]], reward_mode: str) -> dict[str, float]:
    if not pending_items:
        return {}

    rewards = [float(item["breakdown"].node_reward) for item in pending_items]
    success_rewards = [
        float(item["breakdown"].node_reward)
        for item in pending_items
        if float(item["graph"].final_reward or 0.0) > 0.0
    ]
    failed_rewards = [
        float(item["breakdown"].node_reward)
        for item in pending_items
        if float(item["graph"].final_reward or 0.0) <= 0.0
    ]
    solver_rewards = [
        float(item["breakdown"].node_reward)
        for item in pending_items
        if _normalize_key(str(item["node"].agent_id or item["node"].role or "")) == "solver"
    ]
    verifier_rewards = [
        float(item["breakdown"].node_reward)
        for item in pending_items
        if _normalize_key(str(item["node"].agent_id or item["node"].role or "")) == "verifier"
    ]
    verifier_penalty_flags = [
        bool(item["breakdown"].details.get("verifier_hard_penalty_applied", False))
        for item in pending_items
        if is_verifier_node_type(item["node"].node_type)
    ]

    metrics = {
        "graphcredit/outcome_reward_mode": 1.0 if reward_mode == "outcome_redistribution" else 0.0,
        "graphcredit/success_node_reward_mean": float(np.mean(success_rewards)) if success_rewards else 0.0,
        "graphcredit/failed_node_reward_mean": float(np.mean(failed_rewards)) if failed_rewards else 0.0,
        "graphcredit/failed_positive_node_ratio": float(np.mean([reward > 0.0 for reward in failed_rewards])) if failed_rewards else 0.0,
        "graphcredit/positive_node_ratio": float(np.mean([reward > 0.0 for reward in rewards])) if rewards else 0.0,
        "graphcredit/negative_node_ratio": float(np.mean([reward < 0.0 for reward in rewards])) if rewards else 0.0,
    }
    if solver_rewards:
        metrics["graphcredit/solver_node_reward_mean"] = float(np.mean(solver_rewards))
    if verifier_rewards:
        metrics["graphcredit/verifier_node_reward_mean"] = float(np.mean(verifier_rewards))
        metrics["graphcredit/verifier_negative_node_ratio"] = float(np.mean([reward < 0.0 for reward in verifier_rewards]))
    if verifier_penalty_flags:
        metrics["graphcredit/verifier_hard_penalty_ratio"] = float(np.mean(verifier_penalty_flags))
    return metrics


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

    merged = {str(k): float(v) for k, v in _to_plain_container(base_weights).items()}
    role_weights = _to_plain_container(role_weights)
    candidates = [
        str(node.role or ""),
        str(node.agent_id or ""),
        str(node.node_type or ""),
    ]
    normalized_candidates = {_normalize_key(item) for item in candidates if item}
    for key, override in role_weights.items():
        if _normalize_key(str(key)) in normalized_candidates and isinstance(override, Mapping):
            merged.update({str(k): float(v) for k, v in override.items()})
    return merged


def _normalize_key(value: str) -> str:
    normalized = value.lower().replace(" ", "_").replace("-", "_")
    if normalized.endswith("_agent"):
        normalized = normalized[: -len("_agent")]
    return normalized


def _failed_math_solver_cost_floor(task_type: str, global_reward: float, node: EventNode, cost: float) -> float:
    if task_type != "math" or global_reward > 0.0 or not is_solver_node_type(node.node_type):
        return cost
    output = node.output_content or ""
    if "\\boxed" not in output:
        return cost
    word_count = len(output.split())
    if word_count <= 128:
        return cost
    return max(float(cost), min(1.0, 0.25 + (word_count - 128) / 384.0))


def _to_plain_container(value: Any) -> Any:
    """Convert OmegaConf containers to plain Python containers when available."""

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {key: _to_plain_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_container(item) for item in value]
    return value


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
        prompt = sanitize_text(tokenizer.decode(item.batch["prompts"][-valid_prompt_length:], skip_special_tokens=False))
        response = sanitize_text(tokenizer.decode(item.batch["responses"][:valid_response_length], skip_special_tokens=True))
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
        node_type = refine_node_type(nt.get("node_type", infer_node_type(agent_id, response, orchestra_type)), agent_id, response, orchestra_type)
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
