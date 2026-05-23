from collections import UserDict

from graphcredit_offline.core.graph_builder import build_event_graph, refine_node_type
from graphcredit_offline.core.schema import EventNode
from graphcredit_offline.integration import _apply_outcome_redistribution, _weights_for_node
from graphcredit_offline.rewards.fusion import RewardBreakdown
from graphcredit_offline.rewards.counterfactual import offline_masking_credit
from graphcredit_offline.rewards.cost import cost_penalty, redundancy_penalty
from graphcredit_offline.rewards.downstream_usage import downstream_usage_score
from graphcredit_offline.rewards.fusion import fuse_node_reward
from graphcredit_offline.rewards.math_scorer import MathProcessScorer


def test_math_process_and_counterfactual_reward_are_bounded():
    node = EventNode(
        node_id="n1",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="x = 1 therefore \\boxed{1}",
        final_reward=1.0,
    )
    graph = build_event_graph("t1", [node], task_type="math", final_reward=1.0)

    process = MathProcessScorer().score(graph, node)
    cf = offline_masking_credit(graph, node, process.score)
    breakdown = fuse_node_reward("n1", 1.0, process.score, cf.credit, 0.5, 0.1, 0.0)

    assert 0.0 <= process.score <= 1.0
    assert -1.0 <= cf.credit <= 1.0
    assert breakdown.node_reward > 0.0


def test_masked_math_graph_credit_drops_when_solver_support_removed():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="x = 1 therefore \\boxed{1}",
        final_reward=1.0,
    )
    answer = EventNode(
        node_id="answer",
        trajectory_id="t1",
        agent_id="Answer Agent",
        role="answer",
        node_type="final_answer",
        time_step=2,
        input_context="x = 1 therefore \\boxed{1}",
        output_content="The answer is \\boxed{1}",
        final_reward=1.0,
    )
    graph = build_event_graph("t1", [solver, answer], task_type="math", final_reward=1.0)

    cf = offline_masking_credit(graph, solver, process_reward=1.0, task_type="math")

    assert cf.original_value == 1.0
    assert cf.masked_value < 1.0
    assert cf.credit > 0.0


def test_math_graph_final_answer_prefers_solver_boxed_answer_over_verifier_text():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="therefore \\boxed{16}",
        final_reward=1.0,
    )
    verifier = EventNode(
        node_id="verifier",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_judgment",
        time_step=2,
        input_context="therefore \\boxed{16}",
        output_content="<verify>approve</verify> the solution is correct",
        final_reward=1.0,
    )
    graph = build_event_graph("t1", [solver, verifier], task_type="math", final_reward=1.0)

    assert graph.final_answer == "\\boxed{16}"


def test_masked_math_bad_solver_and_bad_approve_get_negative_credit():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="invalid reasoning therefore \\boxed{999}",
        final_reward=0.0,
    )
    verifier = EventNode(
        node_id="verifier",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_judgment",
        time_step=2,
        input_context="invalid reasoning therefore \\boxed{999}",
        output_content="<verify>approve</verify>",
        final_reward=0.0,
    )
    graph = build_event_graph("t1", [solver, verifier], task_type="math", final_reward=0.0)

    solver_cf = offline_masking_credit(graph, solver, process_reward=0.8, task_type="math")
    verifier_cf = offline_masking_credit(graph, verifier, process_reward=0.1, task_type="math")

    assert solver_cf.credit < 0.0
    assert verifier_cf.credit < 0.0


def test_masked_search_router_can_receive_negative_credit_on_wrong_stop():
    router = EventNode(
        node_id="router",
        trajectory_id="s1",
        agent_id="Verifier Agent",
        role="router",
        node_type="router_decision",
        time_step=1,
        input_context="question",
        output_content="<verify>yes</verify>",
        final_reward=0.0,
    )
    answer = EventNode(
        node_id="answer",
        trajectory_id="s1",
        agent_id="Answer Agent",
        role="answer",
        node_type="final_answer",
        time_step=2,
        input_context="no evidence",
        output_content="<answer>wrong</answer>",
        final_reward=0.0,
    )
    graph = build_event_graph("s1", [router, answer], task_type="search", final_reward=0.0)

    cf = offline_masking_credit(graph, router, process_reward=0.0, task_type="search")

    assert cf.masked_value > cf.original_value
    assert cf.credit < 0.0


def test_empty_boxed_and_reject_only_receive_no_process_reward():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="\\boxed{}",
        final_reward=0.0,
    )
    verifier = EventNode(
        node_id="verifier",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_judgment",
        time_step=2,
        input_context="\\boxed{}",
        output_content="<verify>reject</verify>",
        final_reward=0.0,
    )
    graph = build_event_graph("t1", [solver, verifier], task_type="math", final_reward=0.0)
    scorer = MathProcessScorer()

    assert scorer.score(graph, solver).score == 0.0
    assert scorer.score(graph, verifier).score == 0.0


def test_explained_correct_verifier_judgment_gets_small_positive_reward():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="bad arithmetic therefore \\boxed{999}",
        final_reward=0.0,
    )
    verifier = EventNode(
        node_id="verifier",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_judgment",
        time_step=1,
        input_context="solution",
        output_content="<verify>reject</verify> the final arithmetic contradicts the equation",
        final_reward=0.0,
    )
    graph = build_event_graph("t1", [solver, verifier], task_type="math", final_reward=0.0)

    score = MathProcessScorer().score(graph, verifier).score

    assert 0.0 < score <= 0.1


def test_failed_trajectory_does_not_get_positive_reward_from_process_format_alone():
    breakdown = fuse_node_reward(
        "solver",
        global_reward=0.0,
        process_reward=0.65,
        counterfactual_credit=0.0,
        downstream_usage=0.5,
        cost=0.0,
        redundancy=0.0,
        weights={
            "alpha_global": 0.30,
            "beta_process": 0.35,
            "gamma_counterfactual": 0.05,
            "delta_downstream_usage": 0.10,
            "failure_penalty": 0.08,
        },
        allow_failed_positive=False,
    )

    assert breakdown.node_reward <= 0.0


def test_failed_trajectory_can_keep_capped_positive_verified_local_credit():
    breakdown = fuse_node_reward(
        "verifier",
        global_reward=0.0,
        process_reward=0.1,
        counterfactual_credit=0.5,
        downstream_usage=0.0,
        cost=0.0,
        redundancy=0.0,
        weights={
            "gamma_counterfactual": 0.20,
            "failure_penalty": 0.02,
            "failed_positive_cap": 0.05,
        },
        allow_failed_positive=False,
    )

    assert 0.0 < breakdown.node_reward <= 0.05


def test_outcome_redistribution_zeros_failed_local_credit():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="solver_reasoning",
        time_step=1,
        input_context="problem",
        output_content="structured but wrong solution \\boxed{999}",
        final_reward=0.0,
    )
    graph = build_event_graph("t1", [solver], task_type="math", final_reward=0.0)
    breakdown = RewardBreakdown(
        node_id="solver",
        global_reward=0.0,
        process_reward=0.8,
        counterfactual_credit=0.4,
        downstream_usage_score=0.5,
        cost_penalty=0.0,
        redundancy_penalty=0.0,
        node_reward=0.6,
    )
    pending = [{"node": solver, "graph": graph, "breakdown": breakdown}]

    _apply_outcome_redistribution(
        pending,
        {
            "outcome_redistribution": {
                "train_roles": ["Solver Agent"],
                "failed_node_reward": 0.0,
                "credit_basis": "counterfactual_process",
            }
        },
    )

    assert breakdown.node_reward == 0.0
    assert breakdown.details["pre_redistribution_node_reward"] == 0.6


def test_outcome_redistribution_sends_success_reward_to_train_role_only():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="solver_reasoning",
        time_step=1,
        input_context="problem",
        output_content="valid reasoning \\boxed{1}",
        final_reward=1.0,
    )
    verifier = EventNode(
        node_id="verifier",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_check",
        time_step=2,
        input_context="solution",
        output_content="<verify>approve</verify>",
        final_reward=1.0,
    )
    graph = build_event_graph("t1", [solver, verifier], task_type="math", final_reward=1.0)
    solver_breakdown = RewardBreakdown("solver", 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.2)
    verifier_breakdown = RewardBreakdown("verifier", 1.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.8)
    pending = [
        {"node": solver, "graph": graph, "breakdown": solver_breakdown},
        {"node": verifier, "graph": graph, "breakdown": verifier_breakdown},
    ]

    _apply_outcome_redistribution(
        pending,
        {
            "outcome_redistribution": {
                "train_roles": ["Solver Agent"],
                "success_reward_scale": 1.0,
                "credit_basis": "counterfactual_process",
            }
        },
    )

    assert solver_breakdown.node_reward == 1.0
    assert verifier_breakdown.node_reward == 0.0


def test_verifier_rejecting_correct_solver_answer_gets_negative_counterfactual_credit():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="x = 1 therefore \\boxed{1}",
        final_reward=1.0,
    )
    verifier = EventNode(
        node_id="verifier",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_judgment",
        time_step=2,
        input_context="x = 1 therefore \\boxed{1}",
        output_content="<verify>reject</verify> the final answer uses a wrong substitution",
        final_reward=1.0,
    )
    graph = build_event_graph("t1", [solver, verifier], task_type="math", final_reward=1.0)

    process = MathProcessScorer().score(graph, verifier)
    cf = offline_masking_credit(graph, verifier, process_reward=process.score, task_type="math")

    assert process.score == 0.0
    assert cf.credit < 0.0


def test_verifier_correction_gain_uses_sample_index_when_time_step_ties():
    first_solver = EventNode(
        node_id="solver1",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="first try gives \\boxed{9}",
        final_reward=1.0,
        metadata={"sample_index": 0},
    )
    verifier = EventNode(
        node_id="verifier",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_correction",
        time_step=1,
        input_context="first try gives \\boxed{9}",
        output_content="<verify>reject</verify> the arithmetic sign is inconsistent; the corrected answer should be \\boxed{1}",
        final_reward=1.0,
        metadata={"sample_index": 1},
    )
    second_solver = EventNode(
        node_id="solver2",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="revise",
        output_content="correcting the sign gives \\boxed{1}",
        final_reward=1.0,
        metadata={"sample_index": 2},
    )
    graph = build_event_graph("t1", [second_solver, verifier, first_solver], task_type="math", final_reward=1.0)

    process = MathProcessScorer().score(graph, verifier)
    cf = offline_masking_credit(graph, verifier, process_reward=process.score, task_type="math")

    assert process.score > 0.5
    assert cf.credit == 0.7


def test_failed_downstream_usage_ignores_overlap_with_wrong_final_answer():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="therefore \\boxed{999}",
        final_reward=0.0,
    )
    verifier = EventNode(
        node_id="verifier",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_judgment",
        time_step=2,
        input_context="therefore \\boxed{999}",
        output_content="<verify>approve</verify> looks fine",
        final_reward=0.0,
    )
    graph = build_event_graph("t1", [solver, verifier], task_type="math", final_reward=0.0)

    assert downstream_usage_score(graph, verifier) == 0.0


def test_role_reward_weights_accept_nested_mapping_configs():
    node = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="Solver Agent",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="solution",
    )
    cfg = UserDict(
        {
            "solver": UserDict({
                "alpha_global": 0.30,
                "beta_process": 0.35,
                "gamma_counterfactual": 0.05,
                "delta_downstream_usage": 0.10,
            }),
            "verifier": UserDict({
                "alpha_global": 0.20,
                "beta_process": 0.20,
            }),
        }
    )

    weights = _weights_for_node({"alpha_global": 0.30, "beta_process": 0.30}, cfg, node)

    assert weights["beta_process"] == 0.35
    assert weights["delta_downstream_usage"] == 0.10


def test_refine_node_type_distinguishes_solver_and_verifier_subtypes():
    reasoning = refine_node_type("agent_message", "Solver Agent", "Step 1: therefore x = 2", "math")
    final_answer = refine_node_type("agent_message", "Solver Agent", "Thus \\boxed{2}", "math")
    verifier_check = refine_node_type("verifier_judgment", "Verifier Agent", "<verify>reject</verify> explanation", "math")
    verifier_correction = refine_node_type("verifier_judgment", "Verifier Agent", "<verify>reject</verify> the correct answer should be \\boxed{2}", "math")

    assert reasoning == "solver_reasoning"
    assert final_answer == "solver_final_answer"
    assert verifier_check == "verifier_check"
    assert verifier_correction == "verifier_correction"


def test_verifier_approve_wrong_answer_is_penalized_more_than_reject_with_correction():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="solver_reasoning",
        time_step=1,
        input_context="problem",
        output_content="bad step \\boxed{999}",
        final_reward=0.0,
    )
    verifier_wrong = EventNode(
        node_id="verifier_wrong",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_check",
        time_step=2,
        input_context="bad step \\boxed{999}",
        output_content="<verify>approve</verify> looks fine",
        final_reward=0.0,
    )
    verifier_good = EventNode(
        node_id="verifier_good",
        trajectory_id="t2",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_correction",
        time_step=2,
        input_context="bad step \\boxed{999}",
        output_content="<verify>reject</verify> the correct answer should be \\boxed{1}",
        final_reward=1.0,
    )
    graph_wrong = build_event_graph("t1", [solver, verifier_wrong], task_type="math", final_reward=0.0)
    graph_good = build_event_graph("t2", [solver, verifier_good], task_type="math", final_reward=1.0)

    wrong = MathProcessScorer().score(graph_wrong, verifier_wrong).score
    good = MathProcessScorer().score(graph_good, verifier_good).score

    assert wrong < 0.05
    assert good >= wrong


def test_solver_reasoning_length_is_not_capped_too_early():
    short = EventNode(
        node_id="short",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="solver_reasoning",
        time_step=1,
        input_context="problem",
        output_content="x = 1 \\boxed{1}",
        final_reward=1.0,
    )
    long = EventNode(
        node_id="long",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="solver_reasoning",
        time_step=1,
        input_context="problem",
        output_content="Step 1: consider the equation carefully. Step 2: substitute the values. Step 3: therefore the result follows. We continue the derivation until the boxed answer is justified. x = 1 so we get \\boxed{1}",
        final_reward=1.0,
    )
    graph = build_event_graph("t1", [short, long], task_type="math", final_reward=1.0)

    short_score = MathProcessScorer().score(graph, short).score
    long_score = MathProcessScorer().score(graph, long).score

    assert long_score >= short_score


def test_too_short_solver_output_has_high_cost_penalty():
    node = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="\\boxed{}",
    )

    assert cost_penalty(node) == 1.0


def test_repeated_solver_answer_is_redundant_even_with_different_wording():
    first = EventNode(
        node_id="solver1",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="A direct computation gives \\boxed{7}",
    )
    second = EventNode(
        node_id="solver2",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=2,
        input_context="problem",
        output_content="Trying another derivation, the result is still \\boxed{7}",
    )
    graph = build_event_graph("t1", [first, second], task_type="math", final_reward=0.0)

    assert redundancy_penalty(graph, second, counterfactual_credit=0.0, usage_score=0.0, cost=0.3) == 1.0
