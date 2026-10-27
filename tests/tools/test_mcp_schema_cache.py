"""Schema discovery cache tests without a live MCP server or FastMCP install."""

import asyncio
import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


@pytest.fixture
def manager():
    fastmcp = ModuleType("fastmcp")
    fastmcp.Client = object
    transports = ModuleType("fastmcp.client.transports")
    transports.SSETransport = object
    utils = ModuleType("verl.tools.utils.mcp_clients.utils")
    utils.TokenBucket = object
    utils.mcp2openai = lambda tool: tool
    path = Path(__file__).parents[2] / "verl/tools/utils/mcp_clients/McpClientManager.py"
    spec = importlib.util.spec_from_file_location("schema_cache_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "fastmcp": fastmcp,
        "fastmcp.client.transports": transports,
        "verl.tools.utils.mcp_clients.utils": utils,
    }):
        spec.loader.exec_module(module)
    return module.MCPClientManager()


def schemas():
    return [{"function": {"name": name, "parameters": {}}} for name in ("one", "two")]


def test_cache_filters_without_refetching_and_returns_copies(manager):
    calls = []

    async def discover():
        calls.append(1)
        return schemas(), {"one": "client", "two": "client"}

    manager._discover_tool_schemas = discover
    first = asyncio.run(manager.fetch_tool_schemas(["one"]))
    first[0]["function"]["parameters"]["changed"] = True
    all_tools = asyncio.run(manager.fetch_tool_schemas(None))
    assert all_tools == schemas()
    assert len(calls) == 1
    assert manager.get_client_with_tool_name("two") == "client"


def test_discovery_is_shared_across_threads_and_event_loops(manager):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    async def discover():
        calls.append(1)
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.001)
        return schemas(), {}

    manager._discover_tool_schemas = discover

    async def wait_for_owner():
        # Release only after a second loop has attached to the pending future.
        task = asyncio.create_task(manager.fetch_tool_schemas(None))
        await asyncio.sleep(0)
        release.set()
        return await task

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(asyncio.run, manager.fetch_tool_schemas(None))
        assert entered.wait(5)
        waiter = pool.submit(asyncio.run, wait_for_owner())
        assert owner.result(timeout=5) == schemas()
        assert waiter.result(timeout=5) == schemas()
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_failed_or_empty_discovery_can_be_retried(manager, failure):
    calls = []

    async def discover():
        calls.append(1)
        if len(calls) == 1:
            if failure == "exception":
                raise TimeoutError("discovery timed out")
            return [], {}
        return schemas(), {}

    manager._discover_tool_schemas = discover
    if failure == "exception":
        with pytest.raises(TimeoutError):
            asyncio.run(manager.fetch_tool_schemas(None))
    else:
        assert asyncio.run(manager.fetch_tool_schemas(None)) == []
    assert asyncio.run(manager.fetch_tool_schemas(None)) == schemas()
    assert len(calls) == 2
