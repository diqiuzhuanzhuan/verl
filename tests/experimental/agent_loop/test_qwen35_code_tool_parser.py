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
import pytest

from verl.experimental.agent_loop.tool_parser import Qwen35CodeToolParser, ToolParser


class DummyTokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, _ids: list[int]) -> str:
        return self.text


@pytest.mark.asyncio
async def test_qwen35_code_tool_parser_xml():
    text = (
        "hello\n"
        "<tool_call>\n"
        "<function=search_photos>\n"
        "<parameter=keyword>\n"
        "cat\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "world"
    )
    parser = Qwen35CodeToolParser(DummyTokenizer(text))

    content, function_calls = await parser.extract_tool_calls([1, 2, 3])
    assert len(function_calls) == 1
    assert function_calls[0].name == "search_photos"
    assert function_calls[0].arguments == '{"keyword": "cat"}'
    assert content == "hello\n"


@pytest.mark.asyncio
async def test_qwen35_code_tool_parser_fallback_without_tool_call_tag():
    text = "prefix\n<function=search_photos>\n<parameter=keyword>\ndog\n</parameter>\n</function>\nsuffix"
    parser = Qwen35CodeToolParser(DummyTokenizer(text))

    content, function_calls = await parser.extract_tool_calls([1])
    assert len(function_calls) == 1
    assert function_calls[0].name == "search_photos"
    assert function_calls[0].arguments == '{"keyword": "dog"}'
    assert content == "prefix\n"


@pytest.mark.asyncio
async def test_qwen35_code_tool_parser_function_parameter_style():
    text = (
        "prefix\n"
        "<tool_call>\n"
        "<function=docker_search_image>\n"
        "<parameter=image_query>\n"
        "ubuntu\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "suffix"
    )
    parser = Qwen35CodeToolParser(DummyTokenizer(text))

    content, function_calls = await parser.extract_tool_calls([1])
    assert len(function_calls) == 1
    assert function_calls[0].name == "docker_search_image"
    assert function_calls[0].arguments == '{"image_query": "ubuntu"}'
    assert content == "prefix\n"


@pytest.mark.asyncio
async def test_qwen35_code_tool_parser_type_conversion():
    text = (
        "<tool_call>\n"
        "<function=set_config>\n"
        "<parameter=retries>\n"
        "3\n"
        "</parameter>\n"
        "<parameter=enabled>\n"
        "true\n"
        "</parameter>\n"
        "<parameter=threshold>\n"
        "0.5\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    parser = Qwen35CodeToolParser(DummyTokenizer(text))

    _, function_calls = await parser.extract_tool_calls([1])
    assert len(function_calls) == 1
    assert function_calls[0].name == "set_config"
    assert function_calls[0].arguments == '{"retries": 3, "enabled": true, "threshold": 0.5}'


def test_qwen35_code_tool_parser_registered():
    parser = ToolParser.get_tool_parser("qwen35_code", DummyTokenizer(""))
    assert isinstance(parser, Qwen35CodeToolParser)
