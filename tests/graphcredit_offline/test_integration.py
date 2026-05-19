import numpy as np

from graphcredit_offline.rl.grouping import build_graphcredit_group_index


class DummyData:
    def __init__(self):
        self.non_tensor_batch = {
            "uid": np.array(["u", "u"], dtype=object),
            "agent_id": np.array(["Solver Agent", "Verifier Agent"], dtype=object),
            "node_type": np.array(["agent_message", "verifier_judgment"], dtype=object),
        }


def test_graphcredit_group_index_includes_node_type():
    index = build_graphcredit_group_index(DummyData(), mode="agent_node_type")

    assert index.tolist() == ["u_Solver Agent_agent_message", "u_Verifier Agent_verifier_judgment"]
