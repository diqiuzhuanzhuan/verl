# Copyright 2025 Individual Contributor: Mert Unsal
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


from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


@register("batch")
class BatchRewardManager(RewardManagerBase):
    """
    A reward manager that calls compute_score with batch interface (data_sources/solution_strs/...).

    Implements run_batch to score the entire batch in one call to compute_score, and run_single
    as a fallback for the single-item path.

    Args:
        config: YAML config.
        tokenizer: Tokenizer for decoding responses.
        compute_score: Batch reward function with signature
            (data_sources, solution_strs, ground_truths, extra_infos, ...) -> list[float | dict] | Tensor.
        reward_fn_key (str): Key for data source in non_tensor_batch.
        reward_kwargs (dict): Extra keyword arguments forwarded to compute_score.
    """

    def __init__(self, config, tokenizer, compute_score, reward_fn_key="data_source", **reward_kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def _extract_batch_inputs(self, data: DataProto):
        """Extract parallel lists needed by the batch compute_score interface."""
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        solution_strs = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_ids = response_ids[i][:valid_len]
            solution_strs.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True))

        data_sources = list(data.non_tensor_batch[self.reward_fn_key])
        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]

        extra_infos = []
        for i in range(len(data)):
            item = data[i]
            extra_info = dict(item.non_tensor_batch.get("extra_info", {}))
            tool_extra_fields = item.non_tensor_batch.get("tool_extra_fields", None)
            if tool_extra_fields is not None:
                extra_info.update(tool_extra_fields.items())
            extra_info["num_turns"] = item.non_tensor_batch.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = item.non_tensor_batch.get("reward_scores", {})
            extra_infos.append(extra_info)

        return data_sources, solution_strs, ground_truths, extra_infos

    async def run_batch(self, data: DataProto) -> list[dict]:
        """Score the full batch in a single call to compute_score.

        Called by reward_loop.compute_score_batch when custom_reward_function.path is set.
        """
        data_sources, solution_strs, ground_truths, extra_infos = await self.loop.run_in_executor(
            None, lambda: self._extract_batch_inputs(data)
        )

        # compute_score_batch is sync (decorated with @run_async_in_new_loop); run in thread pool.
        scores = await self.loop.run_in_executor(
            None,
            lambda: self.compute_score(
                data_sources=data_sources,
                solution_strs=solution_strs,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
                **self.reward_kwargs,
            ),
        )

        results = []
        for i in range(len(data)):
            r = scores[i]
            result = r.item() if hasattr(r, "item") else float(r)
            reward_extra_info = {"acc": result}
            results.append({"reward_score": result, "reward_extra_info": reward_extra_info})
        return results

    async def run_single(self, data: DataProto) -> dict:
        """Score a single item by wrapping it in a one-element batch call."""
        assert len(data) == 1, "Only support single data item"
        results = await self.run_batch(data)
        return results[0]
