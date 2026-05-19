# GraphCredit Math 200 Steps v3 Result Summary

Generated from:

- `D:\graphcredit_math_200steps_v3_node_rewards.jsonl`
- `D:\graphcredit_math_200steps_v3_event_graphs.jsonl`
- `D:\graphcredit_math_200steps_v2_node_rewards.jsonl`
- `D:\graphcredit_math_200steps_v2_event_graphs.jsonl`
- `D:\eval_qwen3-4b_step_200.log`
- `D:\train_graphcredit_math_200steps_safe.log`

Note: the pasted `pid=208211 / step:100` full validation metrics were not found in the D-drive logs by `rg`. The D-drive full validation log found is `pid=504963 / step:200`, likely a different or later evaluation run. Treat the comparison below as cross-log reference unless run IDs are confirmed.

## 1. Full Validation Comparison

Pasted full validation at step 100 versus D-drive full validation at step 200:

| metric | pasted step100 | D eval step200 | abs delta | rel delta |
|---|---:|---:|---:|---:|
| pass@16 | 0.4499 | 0.6044 | +0.1545 | +34.3% |
| avg@16 | 0.2877 | 0.4345 | +0.1469 | +51.1% |
| math500/test_score | 0.4923 | 0.6604 | +0.1680 | +34.1% |
| aime24/test_score | 0.0197 | 0.0992 | +0.0795 | +403.6% |
| aime25/test_score | 0.0151 | 0.0689 | +0.0538 | +356.4% |
| olympiadbench/test_score | 0.1580 | 0.3072 | +0.1493 | +94.5% |
| amc23/test_score | 0.2657 | 0.4181 | +0.1524 | +57.4% |
| minerva/test_score | 0.2113 | 0.2694 | +0.0581 | +27.5% |

Dataset pass/avg:

| dataset | pass@16 step100 | pass@16 step200 | delta | avg@16 step100 | avg@16 step200 | delta |
|---|---:|---:|---:|---:|---:|---:|
| math500 | 0.7020 | 0.8580 | +0.1560 | 0.5164 | 0.6960 | +0.1796 |
| aime24 | 0.0667 | 0.2333 | +0.1667 | 0.0208 | 0.0979 | +0.0771 |
| aime25 | 0.1000 | 0.2333 | +0.1333 | 0.0167 | 0.0688 | +0.0521 |
| olympiadbench | 0.3363 | 0.5274 | +0.1911 | 0.1674 | 0.3265 | +0.1591 |
| amc23 | 0.5000 | 0.7500 | +0.2500 | 0.2875 | 0.4469 | +0.1594 |
| minerva | 0.3419 | 0.3897 | +0.0478 | 0.2250 | 0.2976 | +0.0726 |

Interpretation: if these two validation points are on the same training trajectory, the improvement is strong and broad-based. The largest relative gains are on AIME24/AIME25, but those start from a very low base, so absolute gains matter more.

## 2. In-Training Small Validation

From `D:\train_graphcredit_math_200steps_safe.log`, the in-training validation uses `pass@1` and a smaller set:

| step | math500_first50/test_score | aime24/test_score | aime25/test_score | pass@1 |
|---:|---:|---:|---:|---:|
| 100 | 0.748 | 0.101 | 0.056 | 0.400 |
| 200 | 0.641 | 0.136 | 0.067 | 0.373 |

Interpretation: this smaller validation is mixed. AIME improves, but Math500-first50 drops, and aggregate pass@1 slightly decreases. This conflicts with the full `pass@16` improvement, so pass@1 small validation should not be used as the main claim.

## 3. v3 Reward/Event Graph Summary

Raw structure:

| item | value |
|---|---:|
| node reward records | 9,584 |
| event graphs / trajectories | 3,200 |
| steps | 1-200 |
| trajectories per step | 16 |
| avg node reward records per trajectory | 2.995 |
| avg graph nodes | 2.5525 |
| avg graph edges | 4.5991 |

Overall node reward components:

| component | mean | std | p10 | p50 | p90 |
|---|---:|---:|---:|---:|---:|
| node_reward | 0.1386 | 0.4980 | -0.1479 | -0.1413 | 1.0341 |
| global_reward | 0.2264 | 0.4185 | 0.0000 | 0.0000 | 1.0000 |
| process_reward | 0.2393 | 0.2592 | 0.0000 | 0.1500 | 0.6400 |
| counterfactual_credit | -0.1982 | 0.4334 | -0.5000 | -0.5000 | 0.0000 |
| downstream_usage_score | 0.1323 | 0.0926 | 0.0000 | 0.1091 | 0.2094 |
| cost_penalty | 0.0670 | 0.2061 | 0.0000 | 0.0000 | 0.2037 |
| redundancy_penalty | 0.0033 | 0.0577 | 0.0000 | 0.0000 | 0.0000 |

Role breakdown:

| role | n | node_reward mean | process mean | cf mean | usage mean | cost mean | negative node ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Solver Agent | 5,780 | 0.1144 | 0.2907 | -0.2419 | 0.1750 | 0.1030 | 0.7936 |
| Verifier Agent | 3,804 | 0.1752 | 0.1611 | -0.1318 | 0.0675 | 0.0124 | 0.7432 |

Step-window trend:

| step range | traj pass | node_reward mean | negative node ratio | cf mean | process mean | cost mean |
|---|---:|---:|---:|---:|---:|---:|
| 1-50 | 0.4150 | 0.3226 | 0.6166 | -0.0825 | 0.2858 | 0.1736 |
| 51-100 | 0.3425 | 0.2428 | 0.6850 | -0.1367 | 0.2673 | 0.0618 |
| 101-150 | 0.1200 | 0.0008 | 0.8908 | -0.2813 | 0.1966 | 0.0208 |
| 151-200 | 0.1013 | -0.0107 | 0.9008 | -0.2916 | 0.2077 | 0.0126 |

Interpretation: the v3 graph-reward stream becomes much more punitive after step 100. The event-graph success rate drops from 0.3787 in steps 1-100 to 0.1106 in steps 101-200, while nearly 90% of node rewards become negative.

## 4. v2 versus v3 Reward Comparison

| run | overall traj success | overall node_reward mean | overall negative node ratio | step 1-100 traj pass | step 101-200 traj pass |
|---|---:|---:|---:|---:|---:|
| v2 | 0.1006 | 0.1174 | 0.0755 | 0.2013 | 0.0000 |
| v3 | 0.2447 | 0.1386 | 0.7736 | 0.3787 | 0.1106 |

Interpretation: v3 is better than v2 on trajectory success, especially in the second half where v2 collapses to zero. However, v3 uses a much stricter reward shape, with many failed-trajectory nodes clipped to non-positive values.

## 5. Evaluation

Good signs:

1. Full validation, if comparable, improves broadly: `pass@16` +15.45 points and `avg@16` +14.69 points.
2. v3 is clearly better than v2 in event-graph trajectory success: 24.47% versus 10.06%.
3. The reward signal is highly aligned with outcome reward: node_reward versus global_reward correlation is 0.9977. This matches the current fusion rule, where global reward dominates and failed trajectories are penalized.
4. Redundancy penalty is low, so the reward is mostly driven by outcome, process, and counterfactual terms rather than duplicate-output artifacts.

Concerns:

1. The in-training pass@1 validation is not consistently improving: Math500-first50 drops from 0.748 to 0.641, and aggregate pass@1 drops from 0.400 to 0.373.
2. v3 late-stage rollout/event-graph success is weak: steps 151-200 average only 10.13% success.
3. Counterfactual credit is mostly negative or zero: median is -0.5, positive credit ratio is only 8.67%. This may be useful for pruning bad nodes, but it provides sparse positive learning signal.
4. Because node_reward is almost perfectly correlated with global_reward, the current result does not yet prove strong independent credit assignment beyond outcome reward shaping. It proves the graph reward layer is active and stricter than v2.

Bottom line: this is a promising engineering result and a better reward-shaping run than v2, but not yet a clean research win. The strongest claim is "GraphCredit v3 produces a stricter, outcome-aligned node reward and improves v2 trajectory success." The stronger paper claim, "fine-grained counterfactual credit improves multi-agent RL beyond Dr. MAS/global reward," still needs ablations.

## 6. Suggested Next Experiments

1. Run matched full validation for the exact v3 checkpoint whose JSONL files were analyzed, with the same evaluator and seed as the pasted step100 metrics.
2. Add ablations:
   - global only
   - global + process
   - global + counterfactual
   - full v3
   - no failure penalty
3. Reduce late-stage punitive collapse:
   - soften `failure_penalty`
   - raise `negative_credit_clip` from -0.5 to -0.25
   - lower `gamma_counterfactual` until positive credit density improves
4. Report both validation quality and credit quality:
   - pass@16 / avg@16
   - trajectory success
   - positive counterfactual ratio
   - harmful node ratio
   - node_reward variance by role
   - reward-outcome correlation

