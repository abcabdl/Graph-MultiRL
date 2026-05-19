from graphcredit_offline.core.graph_builder import build_event_graph
from graphcredit_offline.core.schema import EventGraph, EventNode
from graphcredit_offline.core.serialization import graph_from_dict, graph_to_json


def test_event_graph_serializes_roundtrip():
    graph = EventGraph(
        trajectory_id="t1",
        task_id="task",
        task_type="math",
        task_prompt="solve",
        nodes=[
            EventNode(
                node_id="n1",
                trajectory_id="t1",
                agent_id="Solver Agent",
                role="solver",
                node_type="agent_message",
                time_step=1,
                input_context="x",
                output_content="y",
            )
        ],
        edges=[],
    )

    loaded = graph_from_dict(__import__("json").loads(graph_to_json(graph)))

    assert loaded.trajectory_id == "t1"
    assert loaded.nodes[0].node_type == "agent_message"


def test_event_graph_deduplicates_nodes_and_edges():
    solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="x = 1 therefore \\boxed{1}",
    )
    duplicate_solver = EventNode(
        node_id="solver",
        trajectory_id="t1",
        agent_id="Solver Agent",
        role="solver",
        node_type="agent_message",
        time_step=1,
        input_context="problem",
        output_content="x = 1 therefore \\boxed{1}",
    )
    verifier = EventNode(
        node_id="verifier",
        trajectory_id="t1",
        agent_id="Verifier Agent",
        role="verifier",
        node_type="verifier_judgment",
        time_step=2,
        input_context="x = 1 therefore \\boxed{1}",
        output_content="<verify>approve</verify> the solution is correct",
    )

    graph = build_event_graph("t1", [solver, duplicate_solver, verifier], task_type="math")
    edge_keys = {(edge.edge_type, edge.source_node_id, edge.target_node_id) for edge in graph.edges}

    assert len(graph.nodes) == 2
    assert len(edge_keys) == len(graph.edges)
    assert all(edge.source_node_id != edge.target_node_id for edge in graph.edges)
