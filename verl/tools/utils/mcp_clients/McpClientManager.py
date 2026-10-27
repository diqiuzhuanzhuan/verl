# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import json
import logging
import os
import socket
import threading
import time
import weakref
from concurrent.futures import Future
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit

from fastmcp import Client
from fastmcp.client.transports import SSETransport

from verl.tools.utils.mcp_clients.utils import TokenBucket, mcp2openai

logger = logging.getLogger(__name__)


def _safe_address(value):
    if not value:
        return value
    parsed = urlsplit(value)
    if not parsed.hostname:
        return "<set>"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}{parsed.path}"


def _mcp_log(message, **fields):
    # Warning level ensures diagnostic events are visible with default verl logging.
    logger.warning(
        "[MCP diagnostic] host=%s pid=%s thread=%s %s %s",
        socket.gethostname(), os.getpid(), threading.get_ident(), message, fields,
    )


class MCPClientPool:
    """A fixed-size pool of connected FastMCP clients for one event loop."""

    def __init__(self, client: Client, pool_size: int):
        self.client = client
        self.pool_size = max(pool_size, 1)
        self._idle_clients: asyncio.LifoQueue[Client] = asyncio.LifoQueue(maxsize=self.pool_size)
        self._semaphore = asyncio.Semaphore(self.pool_size)

    @asynccontextmanager
    async def acquire(self):
        await self._semaphore.acquire()
        client = None
        reuse_client = True
        try:
            try:
                client = self._idle_clients.get_nowait()
            except asyncio.QueueEmpty:
                client = self.client.new()
                await client.__aenter__()

            yield client
        except BaseException:
            reuse_client = False
            if client is not None:
                with suppress(Exception):
                    await client.close()
            raise
        finally:
            if reuse_client and client is not None:
                self._idle_clients.put_nowait(client)
            self._semaphore.release()


class MCPClientManager:
    rootServerName = "mcpServers"

    def __init__(self):
        self.initialized = False
        self.clients = []
        self.tool_client_mapping = {}
        self.rate_limiter = None
        self.pool_size = 32
        self.client_pools = weakref.WeakKeyDictionary()
        # Registry discovery runs in temporary loops on different threads. An
        # asyncio.Lock/Task cannot be shared across those loops.
        self._schema_lock = threading.Lock()
        self._schema_cache = None
        self._schema_fetch = None

    async def initialize(self, config_path, rate_limit: float = 10.0, pool_size: int = 32):
        """Initialize the MCP Client Manager and start all clients"""
        if self.initialized:
            return
        self.pool_size = pool_size
        result = self._load_config(config_path)
        servers = result[self.rootServerName]
        env = {}
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy",
                    "no_proxy", "NO_PROXY"):
            value = os.environ.get(key)
            env[key] = value if key.lower() == "no_proxy" else _safe_address(value)
        _mcp_log("initialize", config_path=config_path, cwd=os.getcwd(), proxy_env=env,
                 pool_size=pool_size, rate_limit=rate_limit)
        for name, server in servers.items():
            _mcp_log("server", name=name, url=_safe_address(server.get("url")),
                     transport="SSETransport (auth_token)" if "auth_token" in server
                     else server.get("transport", "FastMCP inferred"))
        exclude_sse_servers = {self.rootServerName: {}}
        for server_name in servers.keys():
            server = servers[server_name]
            if "auth_token" in server:
                transport = SSETransport(url=server["url"], headers={"Authorization": f"Bearer {server['auth_token']}"})
                client = Client(transport)
                self.clients.append(client)
            else:
                exclude_sse_servers[self.rootServerName][server_name] = server

        if exclude_sse_servers[self.rootServerName]:
            self.clients.append(Client(exclude_sse_servers))

        # Initialize rate limiter
        self.rate_limiter = TokenBucket(rate_limit)
        self.initialized = True

    async def call_tool(self, tool_name, parameters, timeout):
        # Apply rate limiting
        while not self.rate_limiter.acquire():
            await asyncio.sleep(0.1)

        client = self.get_client_with_tool_name(tool_name)
        pool = self._get_client_pool(client)
        async with pool.acquire() as pooled_client:
            return await pooled_client.call_tool_mcp(tool_name, parameters, timeout=timeout)

    async def fetch_tool_schemas(self, tool_selected_list: list[str] | None) -> list[dict]:
        with self._schema_lock:
            cached = self._schema_cache
            owner = cached is None and self._schema_fetch is None
            if owner:
                self._schema_fetch = Future()
            pending = self._schema_fetch

        if cached is None:
            if owner:
                try:
                    schemas, mapping = await self._discover_tool_schemas()
                    with self._schema_lock:
                        # Empty discovery remains retryable by tool_registry.
                        if schemas:
                            self._schema_cache = schemas
                            self.tool_client_mapping.update(mapping)
                        self._schema_fetch = None
                    pending.set_result(schemas)
                    cached = schemas
                except BaseException as exc:
                    with self._schema_lock:
                        self._schema_fetch = None
                    pending.set_exception(exc)
                    raise
            else:
                # A cancelled waiter must not cancel discovery for other loops.
                cached = await asyncio.shield(asyncio.wrap_future(pending))

        selected = set(tool_selected_list) if tool_selected_list else None
        return deepcopy([
            schema for schema in cached
            if selected is None or schema["function"]["name"] in selected
        ])

    async def _discover_tool_schemas(self):
        tool_schemas = []
        mapping = {}
        for index, client in enumerate(self.clients):
            started = time.monotonic()
            phase = "create_client"
            try:
                fresh_client = client.new()
                phase = "connect_initialize"
                _mcp_log("begin", client_index=index, phase=phase,
                         loop_id=id(asyncio.get_running_loop()))
                async with fresh_client:
                    _mcp_log("connected", client_index=index, elapsed=time.monotonic() - started)
                    phase = "list_tools"
                    _mcp_log("begin", client_index=index, phase=phase)
                    tools = await fresh_client.list_tools_mcp()
                    _mcp_log("tools_received", client_index=index, count=len(tools.tools),
                             elapsed=time.monotonic() - started)
                    phase = "map_schemas"
                    for tool in tools.tools:
                        mapping[tool.name] = client
                        tool_schemas.append(mcp2openai(tool))
                    phase = "close_session"
                _mcp_log("complete", client_index=index, elapsed=time.monotonic() - started)
            except BaseException:
                _mcp_log("failed", client_index=index, phase=phase,
                         elapsed=time.monotonic() - started)
                logger.exception("[MCP diagnostic] schema initialization failed")
                raise

        return tool_schemas, mapping

    def get_client_with_tool_name(self, tool_name: str):
        return self.tool_client_mapping[tool_name]

    def _get_client_pool(self, client: Client) -> MCPClientPool:
        loop = asyncio.get_running_loop()
        if loop not in self.client_pools:
            self.client_pools[loop] = {}

        loop_pools = self.client_pools[loop]
        client_id = id(client)
        if client_id not in loop_pools:
            loop_pools[client_id] = MCPClientPool(client, self.pool_size)
        return loop_pools[client_id]

    def _load_config(self, file: str) -> dict[str, Any]:
        try:
            with open(file) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f'the "{file}" file was not found')
        except Exception:
            logger.error(f'there was an error reading the "{file}" file')

        return {}


ClientManager = MCPClientManager()
