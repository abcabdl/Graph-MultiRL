from graphcredit_offline.core.verify_tags import has_single_final_verifier_tag, last_verifier_verdict


def test_math_verifier_agent_uses_last_verdict_but_requires_single_final_tag():
    assert last_verifier_verdict("<verify>reject</verify> revised <verify>approve</verify>") == "approve"
    assert has_single_final_verifier_tag("reasoning\n<verify>approve</verify>")
    assert not has_single_final_verifier_tag("<verify>reject</verify> revised <verify>approve</verify>")
    assert not has_single_final_verifier_tag("<verify>approve</verify> trailing text")
