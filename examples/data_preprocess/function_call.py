# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Preprocess the query dataset to parquet format
"""

import argparse
import asyncio
import os
from typing import Any

import datasets
from fastmcp import Client
from fastmcp.tools import Tool
from rich import pretty

from verl.utils.hdfs_io import copy, makedirs


async def get_mcp_tools(mcp_cfg: dict) -> list[dict]:
    """Get tools from MCP server."""
    mcp_cfg = mcp_cfg
    client = Client(**mcp_cfg)
    async with client:
        tools = await client.list_tools()
    return tools


def convert_to_openai_tools(tools: list[Tool]) -> dict[str, list[dict[str, Any]]]:
    functions = []
    for tool in tools:
        function = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": tool.inputSchema or {},
            },
        }
        functions.append(function)
    return {"tools": functions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--mcp_server_url", default="http://127.0.0.1:8000/mcp", help="The URL of the MCP server.")
    parser.add_argument("--judge_model", default=None, help="The model to use for judging the responses.")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/ugreen_function_call",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    mcp_config = {"transport": args.mcp_server_url}
    loop = asyncio.get_event_loop()
    mcp_tools = loop.run_until_complete(get_mcp_tools(mcp_cfg=mcp_config))
    openai_format_tools = convert_to_openai_tools(mcp_tools)
    if args.debug:
        print("#### MCP tools ####")
        pretty.pprint(openai_format_tools)

    local_dataset_path = args.local_dataset_path

    data_source = "ugreen/tool_query"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(data_source, "default")

    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("query")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": ("You are a helpful assistant. "),
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "tool",
                "reward_model": {"style": "rule"},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                    "tools": openai_format_tools,
                    "judge_model": args.judge_model,
                    "need_tools_kwargs": False,
                    "interaction_kwargs": {
                        "query": question,
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
