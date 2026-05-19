from graphcredit_offline.core.graph_builder import build_event_graph
from graphcredit_offline.core.schema import EventNode
from graphcredit_offline.rewards.counterfactual import offline_masking_credit
from graphcredit_offline.rewards.cost import cost_penalty
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
    graph = build_event_graph("t1", [verifier], task_type="math", final_reward=0.0)

    score = MathProcessScorer().score(graph, verifier).score

    assert 0.0 < score <= 0.2


def test_failed_trajectory_can_keep_small_positive_local_process_reward():
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
            "failure_penalty": 0.02,
        },
    )

    assert breakdown.node_reward > 0.0


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
