from graphcredit_offline.rl.normalizer import normalize_node_rewards


def test_agent_node_type_normalizer_uses_bucket_stats():
    normalized = normalize_node_rewards(
        node_ids=["a", "b", "c", "d"],
        rewards=[0.0, 1.0, 0.0, 1.0],
        agent_ids=["solver", "solver", "solver", "solver"],
        node_types=["msg", "msg", "msg", "msg"],
        min_bucket_size=2,
    )

    assert {item.bucket for item in normalized} == {"solver:msg"}
    assert normalized[0].advantage < 0
    assert normalized[1].advantage > 0
