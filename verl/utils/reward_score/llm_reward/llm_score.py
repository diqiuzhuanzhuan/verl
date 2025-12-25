# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


"""
RULER (Relative Universal LLM-Elicited Rewards) - A general-purpose reward function for RL agents.

RULER uses an LLM-as-judge to rank multiple agent trajectories relative to each other,
requiring no labeled data or hand-crafted reward functions. It leverages the insight
that relative scoring is easier than absolute scoring, and GRPO only needs relative
scores within each group.

For detailed documentation and examples, see: https://art.openpipe.ai/fundamentals/ruler
"""

import asyncio
import json
import os
import re
from pathlib import Path
from textwrap import dedent

import torch
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from rich import print

from verl.utils.py_functional import run_async_in_new_loop


class TrajectoryScore(BaseModel):
    """Individual score for a single trajectory."""

    trajectory_id: str = Field(description="The id of the trajectory being scored.")
    explanation: str = Field(description="A short description of the trajectory's performance.")
    score: float = Field(description="A score between 0 and 1.", default=0.0)


class Response(BaseModel):
    """Response format expected from the LLM judge."""

    scores: list[TrajectoryScore] = Field(description="The scores for each trajectory.")


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> spans so they do not pollute outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


with open(Path(__file__).parent / "template" / Path("function_call_default_rubric.md")) as f:
    __BASE_DEFAULT_RUBRIC = f.read()

DEFAULT_RUBRIC = __BASE_DEFAULT_RUBRIC


async def aruler(
    message_lists: list[list[ChatCompletionMessageParam]],
    judge_model: str = "gpt-5-mini",
    extra_params: dict | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    debug: bool = False,
) -> list[TrajectoryScore]:
    """Core RULER implementation that scores a list of message trajectories.

    This is the low-level API that works with raw message lists. For integration
    with ART's training loop, use `ruler_score_group` instead.

    RULER works by:
    1. Extracting common prefixes from trajectories to save tokens
    2. Passing all trajectories to an LLM judge for relative scoring
    3. Returning scores that can be used directly as rewards in GRPO

    The key insight is that relative scores within a group are all that matters
    for GRPO, which normalizes them anyway.

    Args:
        message_lists: A list where each item is a list of ChatCompletionMessageParam
            dicts representing a single trajectory.
        judge_model: The model to use for judging. Common options:
            - "gpt-5-mini" - Fast and capabble (default)
            - "gpt-5" - Most capable but expensive
        extra_params: Additional parameters to pass to LiteLLM completion.
            Can include temperature, max_tokens, etc.
        rubric: The grading rubric. The default rubric works well for most tasks.
        debug: If True, pretty-print the judge's reasoning to help understand scores.

    Returns:
        A list of TrajectoryScore objects with scores and explanations.

    Example:
        >>> message_lists = [
        ...     [{"role": "system", "content": "You are helpful."},
        ...      {"role": "user", "content": "What is 2+2?"},
        ...      {"role": "assistant", "content": "4"}],
        ...     [{"role": "system", "content": "You are helpful."},
        ...      {"role": "user", "content": "What is 2+2?"},
        ...      {"role": "assistant", "content": "I don't know"}]
        ... ]
        >>> scores = await ruler(message_lists, debug=True)
        >>> print(scores[0].score)  # Higher score for correct answer
        0.9
    """

    # Short-circuit for the trivial case
    if not message_lists:
        return []

    # Determine the length of the longest common prefix shared by all trajectories.
    # This optimization reduces token usage when all trajectories share the same
    # system prompt or initial messages.
    message_lists = message_lists
    common_prefix_len = 0
    for idx, msg in enumerate(message_lists[0]):
        if all(len(msg_list) > idx and msg_list[idx] == msg for msg_list in message_lists):
            common_prefix_len += 1
        else:
            break

    # If there is a non-empty common prefix, serialize it once to save tokens.
    user_text = ""
    if common_prefix_len > 0:
        common_prefix_messages = message_lists[0][:common_prefix_len]
        user_text += "<context>\n" + json.dumps(common_prefix_messages, ensure_ascii=False) + "\n</context>\n\n"

    # Serialize each trajectory (minus the common prefix) for the judge.
    serialized_trajectories: list[str] = []
    for idx, full_messages in enumerate(message_lists, start=1):
        trimmed_messages = full_messages[common_prefix_len:]
        serialized_trajectories.append(
            f'<trajectory id="{idx}">\n' + json.dumps(trimmed_messages, ensure_ascii=False) + "\n</trajectory>"
        )

    user_text += "Trajectories:\n\n" + "\n\n".join(serialized_trajectories)

    judge_prompt = dedent(
        f"""
        All of the trajectories below have been given the same goal. 
        Your job is to consider each of them and give them a score between 0 and 1. 
        Take into consideration your best judgement of the agent's goal.

        Grading standards:
        {rubric}
        """
    )

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_text},
    ]

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    try:
        response = await client.chat.completions.parse(
            model=judge_model,
            messages=messages,
            max_completion_tokens=8192,
            response_format=Response,
            **extra_params if extra_params else {},
        )
    except Exception as e:
        print(f"Error in RULER: {e}")
        if debug:
            print(f"Messages: {messages}")

    assert isinstance(response, ChatCompletion)

    if len(response.choices) == 0:
        raise ValueError(f"No choices in response: {response}")
    first_choice = response.choices[0]

    if debug:
        raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
        try:
            print("\n[RULER] Pretty-printed LLM choice JSON:")
            print(json.loads(raw_content))
        except json.JSONDecodeError as e:
            print(f"[RULER] Could not parse choice content as JSON: {e}")
            print(f"[RULER] Raw choice content: {raw_content}")

    content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
    parsed = Response.model_validate_json(content)
    assert len(parsed.scores) == len(message_lists)
    return parsed.scores


def ruler(
    message_lists: list[list[ChatCompletionMessageParam]],
    judge_model: str = "gpt-5-mini",
    extra_params: dict | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    debug: bool = False,
) -> list[TrajectoryScore]:
    """Core RULER implementation that scores a list of message trajectories.

    This is the low-level API that works with raw message lists. For integration
    with ART's training loop, use `ruler_score_group` instead.

    RULER works by:
    1. Extracting common prefixes from trajectories to save tokens
    2. Passing all trajectories to an LLM judge for relative scoring
    3. Returning scores that can be used directly as rewards in GRPO

    The key insight is that relative scores within a group are all that matters
    for GRPO, which normalizes them anyway.

    Args:
        message_lists: A list where each item is a list of ChatCompletionMessageParam
            dicts representing a single trajectory.
        judge_model: The model to use for judging. Common options:
            - "openai/gpt-4o-mini" - Fast and cost-effective
            - "openai/o3" - Most capable but expensive (default)
            - "anthropic/claude-3-opus-20240229" - Alternative judge
        extra_params: Additional parameters to pass to openai completion.
            Can include temperature, max_tokens, etc.
        rubric: The grading rubric. The default rubric works well for most tasks.
        debug: If True, pretty-print the judge's reasoning to help understand scores.

    Returns:
        A list of TrajectoryScore objects with scores and explanations.

    Example:
        >>> message_lists = [
        ...     [{"role": "system", "content": "You are helpful."},
        ...      {"role": "user", "content": "What is 2+2?"},
        ...      {"role": "assistant", "content": "4"}],
        ...     [{"role": "system", "content": "You are helpful."},
        ...      {"role": "user", "content": "What is 2+2?"},
        ...      {"role": "assistant", "content": "I don't know"}]
        ... ]
        >>> scores = await ruler(message_lists, debug=True)
        >>> print(scores[0].score)  # Higher score for correct answer
        0.9
    """

    # Short-circuit for the trivial case
    if not message_lists:
        return []

    # Determine the length of the longest common prefix shared by all trajectories.
    # This optimization reduces token usage when all trajectories share the same
    # system prompt or initial messages.
    message_lists = message_lists
    common_prefix_len = 0
    for idx, msg in enumerate(message_lists[0]):
        if all(len(msg_list) > idx and msg_list[idx] == msg for msg_list in message_lists):
            common_prefix_len += 1
        else:
            break

    # If there is a non-empty common prefix, serialize it once to save tokens.
    user_text = ""
    if common_prefix_len > 0:
        common_prefix_messages = message_lists[0][:common_prefix_len]
        user_text += "<context>\n" + json.dumps(common_prefix_messages) + "\n</context>\n\n"

    # Serialize each trajectory (minus the common prefix) for the judge.
    serialized_trajectories: list[str] = []
    for idx, full_messages in enumerate(message_lists, start=1):
        trimmed_messages = full_messages[common_prefix_len:]
        serialized_trajectories.append(f'<trajectory id="{idx}">\n' + json.dumps(trimmed_messages) + "\n</trajectory>")

    user_text += "Trajectories:\n\n" + "\n\n".join(serialized_trajectories)

    judge_prompt = dedent(
        f"""
        All of the trajectories below have been given the same goal. 
        Your job is to consider each of them and give them a score between 0 and 1. 
        Take into consideration your best judgement of the agent's goal.

        Grading standards:
        {rubric}
        """
    )

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_text},
    ]

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    try:
        response = client.chat.completions.parse(
            model=judge_model,
            messages=messages,
            max_completion_tokens=4096,
            response_format=Response,
            **extra_params if extra_params else {},
        )
    except Exception as e:
        print(f"Error in RULER: {e}")
        if debug:
            print(f"Messages: {messages}")

    assert isinstance(response, ChatCompletion)

    if len(response.choices) == 0:
        raise ValueError(f"No choices in response: {response}")
    first_choice = response.choices[0]

    if debug:
        raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
        try:
            print("\n[RULER] Pretty-printed LLM choice JSON:")
            print(json.loads(raw_content))
        except json.JSONDecodeError as e:
            print(f"[RULER] Could not parse choice content as JSON: {e}")
            print(f"[RULER] Raw choice content: {raw_content}")

    content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
    parsed = Response.model_validate_json(content)
    assert len(parsed.scores) == len(message_lists)

    return parsed.scores


def render_rubric(template: str, context_prompt: str) -> str:
    """
    Safely insert context_prompt into rubric template without invoking str.format,
    which would try to interpret other {...} in the template (e.g. JSON blocks).
    """
    return template.replace("{context_prompt}", context_prompt)


async def compute_score(message_lists, rubric, judge_model="gpt-5-mini"):
    try:
        trajectory_scores = await aruler(
            message_lists, rubric=rubric, judge_model=judge_model, debug=False, extra_params={"timeout": 120}
        )
    except Exception as e:
        print(f"Error in function aruler: {e}")
        trajectory_scores = [
            TrajectoryScore(
                trajectory_id="traj_{:03d}".format(idx), explanation="The trajectory score error.", score=0.0
            )
            for idx in range(len(message_lists))
        ]
    return torch.tensor([trajectory_score.score for trajectory_score in trajectory_scores])


async def compute_score_mini_batch(message_lists, rubric, judge_model="gpt-5-mini"):
    """Score message_lists in smaller sub-batches to avoid hitting model token/size limits.

    Args:
        message_lists: list of trajectories (each a list of message dicts).
        rubric: rendered rubric string.
        judge_model: model name to use for judging.

    Returns:
        A 1-D torch.Tensor of scores in the same order as message_lists.
    """
    # Heuristic max sub-batch size; can be tuned based on model/token limits.
    max_sub_batch = int(os.getenv("RULER_MAX_SUB_BATCH", "8"))

    if not message_lists:
        return torch.tensor([], dtype=torch.float32)

    parts = []
    for i in range(0, len(message_lists), max_sub_batch):
        chunk = message_lists[i : i + max_sub_batch]
        try:
            scores = await compute_score(chunk, rubric, judge_model)
        except Exception as e:
            print(f"Error scoring mini-batch starting at {i}: {e}")
            # fallback: zeros for this chunk
            scores = torch.zeros(len(chunk), dtype=torch.float32)
        parts.append(scores)

    if not parts:
        return torch.tensor([], dtype=torch.float32)

    return torch.cat(parts)


@run_async_in_new_loop
async def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    rubric = ""
    judge_model = None
    message_lists = []
    for _data_source, solution_str, _ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):
        tools = extra_info["tools"]
        if not rubric:
            rubric = render_rubric(
                DEFAULT_RUBRIC, f"The available tools are listed here: {json.dumps(tools, ensure_ascii=False)}"
            )
        if not judge_model:
            judge_model = "gpt-5-mini" if not extra_info.get("judge_model", None) else extra_info["judge_model"]

        message_lists.append(
            [
                {"role": "user", "content": extra_info["question"]},
                {"role": "assistant", "content": remove_think_tags(solution_str)},
            ]
        )

    results = await asyncio.gather(compute_score_mini_batch(message_lists, rubric, judge_model))
    flat = []
    for tensor in results:
        flat.extend(tensor.tolist())
    return torch.tensor(flat)


if __name__ == "__main__":
    data_sources = [""] * 3
    questions = ["旧金山今天的天气怎么样？", "播放下成龙的电影", "帮我查下天气？"]
    solution_strs = [
        """
        <tool_call>
        {"name": "get_current_weather", "arguments": {"location": "San Francisco"}}
        </tool_call>
        """,
        """
        <tool_call>
        {"name": "video_play", "arguments": {"title": "下成龙"}}
        </tool_call>""",
        """
        好的，请问您想要查询哪个城市的天气呢？
        """,
    ]
    ground_truths = [None] * 3

    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_album",
                "description": "Create a new photo album, optionally using search results from the photo library.",
                "parameters": {
                    "properties": {
                        "album_name": {
                            "description": "The name of the album to be created.",
                            "title": "Album Name",
                            "type": "string",
                        },
                        "search_query": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "description": """
                            search keyword or filter used to find photos
                            (e.g., "beach", "family", "2024 vacation"). The album will include the
                            photos that match this query.
                            """,
                            "examples": ["beach", "family", "2024 vacation"],
                            "title": "Search Query",
                        },
                        "album_type": {
                            "default": "normal",
                            "description": """
                            The type of album to create. Valid options: normal, 
                            baby, face, condition, object.""",
                            "enum": ["normal", "baby", "face", "condition", "object"],
                            "examples": ["normal", "baby", "face", "condition", "object"],
                            "title": "Album Type",
                            "type": "string",
                        },
                    },
                    "required": ["album_name"],
                    "type": "object",
                },
            },
        }
    ]

    extra_infos = [{"question": question, "tools": tools} for question in questions]
    print(compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos))
