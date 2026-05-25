# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Any, List, Tuple, Optional
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.utils import general_projection
from graphcredit_offline.core.verify_tags import has_single_final_verifier_tag, last_verifier_verdict
import numpy as np


VERIFIER_PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs
{team_context}

# Your Role
You are a "Verifier Agent". Review only the latest Solver Agent solution.
Be concise: use at most 3 short bullets, and mention only material mathematical errors or missing steps.
Do not rewrite the whole solution. Do not speculate about minor style issues.
Your final line MUST be exactly one verdict tag:
<verify>approve</verify> if the reasoning and final answer are correct.
<verify>reject</verify> if there is any material mathematical error or unsupported final answer.
Use exactly one <verify>...</verify> tag, only on the final line.
"""

@AgentRegistry.register("Verifier Agent")
class VerifierAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Verifier Agent", VERIFIER_PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
        self.start_tag = "<verify>"
        self.end_tag = "</verify>"

    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, List[str]]:
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)
        text_repsonses, valids = general_projection(
            text_repsonses,
            start_tag=self.start_tag,
            end_tag=self.end_tag,
            check_think_tag=False,
            return_whole_response=True,
        )
        strict_valids = np.array([has_single_final_verifier_tag(text) for text in text_repsonses], dtype=bool)
        valids = np.logical_and(valids, strict_valids)
        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_repsonses), dtype=object)
        return batch, text_repsonses

    def update_approved_vector(self, text_repsonses: List[str], approved_vector: np.ndarray, agent_active_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if agent_active_mask is None:
            agent_active_mask = np.ones(len(text_repsonses), dtype=bool)

        new_approved_vector: List[bool] = []
        for i in range(len(text_repsonses)):
            if agent_active_mask[i]:
                verdict = last_verifier_verdict(text_repsonses[i])
                if verdict == "approve":
                    new_approved_vector.append(True)
                elif verdict == "reject":
                    new_approved_vector.append(False)
                else:
                    new_approved_vector.append(True)
            else:
                new_approved_vector.append(True)

        new_approved_vector = np.array(new_approved_vector, dtype=bool)
        updated_vector = np.logical_or(approved_vector, new_approved_vector).astype(bool)
        return updated_vector
