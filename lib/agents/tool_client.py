"""MCP Tool Client for connecting to external tool servers.

This module provides async clients for Model Context Protocol (MCP) servers,
enabling the agent system to call external tools like Blender, etc.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lib.utils._path import path_to_cmd

DEFAULT_MCP_STARTUP_TIMEOUT = float(os.getenv("GRASE_MCP_STARTUP_TIMEOUT", "60"))
DEFAULT_MCP_DISCOVERY_TIMEOUT = float(os.getenv("GRASE_MCP_DISCOVERY_TIMEOUT", "60"))
DEFAULT_MCP_SHUTDOWN_TIMEOUT = float(os.getenv("GRASE_MCP_SHUTDOWN_TIMEOUT", "15"))
_FAILURE_MESSAGE_LIMIT = 1000
_TRACEBACK_EXCEPTION_LINE = re.compile(
    r"^(?:[\w.]+(?:Error|Exception)|Exception|StopIteration|StopAsyncIteration|"
    r"SystemExit|KeyboardInterrupt|GeneratorExit)(?::.*)?$"
)


def _clip_failure_text(value: str, limit: int = _FAILURE_MESSAGE_LIMIT) -> str:
    """Bound diagnostics without dropping either the context or terminal cause."""
    if len(value) <= limit:
        return value
    marker = "\n... [truncated] ...\n"
    head_size = (limit - len(marker)) // 3
    tail_size = limit - len(marker) - head_size
    return value[:head_size] + marker + value[-tail_size:]


def _summarize_failure_message(message: str) -> str:
    """Prioritize exception causes and submitted-code lines over traceback paths.

    Full subprocess diagnostics remain in the transaction artifacts. The tool's
    short feedback must not consume its budget before the actionable exception,
    which is commonly thousands of characters after the first stack frame.
    Unrecognized/non-Python errors retain their original head and tail.
    """
    if len(message) <= _FAILURE_MESSAGE_LIMIT:
        return message
    if "Traceback (most recent call last):" not in message:
        return _clip_failure_text(message)
    lines = message.splitlines()
    exceptions = list(
        dict.fromkeys(
            line for line in lines if _TRACEBACK_EXCEPTION_LINE.fullmatch(line)
        )
    )
    if not exceptions:
        return _clip_failure_text(message)

    # Keep both the terminal exception and its preceding cause. Python chained
    # tracebacks can repeat the same cause, so deduplicate before allocating space.
    causes = list(reversed(exceptions[-2:]))
    cause_budget = 650 // len(causes)
    summary = [_clip_failure_text("Cause: " + cause, cause_budget) for cause in causes]
    authored_frames = list(
        dict.fromkeys(
            line.strip()
            for line in lines
            if re.match(r'\s*File "<(?:authored-code|procedural-object|string)>"', line)
        )
    )
    if authored_frames:
        summary.append(_clip_failure_text("\n".join(authored_frames[-2:]), 150))
    context = message.partition("Traceback (most recent call last):")[0].strip()
    if context:
        summary.append(_clip_failure_text("Context: " + context.splitlines()[0], 150))
    return _clip_failure_text("\n".join(summary))


class ServerHandle:
    """Async wrapper for a single MCP server connection via stdio.

    Manages the lifecycle of an MCP server process, including startup,
    tool discovery, tool execution, and graceful shutdown.

    Attributes:
        path: Path to the MCP server script.
        session: Active MCP client session once connected.
        ready: Event signaling when the server is ready for requests.
    """

    def __init__(
        self,
        path: str,
        *,
        startup_timeout: float = DEFAULT_MCP_STARTUP_TIMEOUT,
        shutdown_timeout: float = DEFAULT_MCP_SHUTDOWN_TIMEOUT,
    ) -> None:
        self.path = path
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.ready = asyncio.Event()
        self.session: ClientSession | None = None
        self.stack: AsyncExitStack | None = None

    async def start(self) -> None:
        """Start the MCP server and wait for it to be ready.

        Races the ready event against the runner task: any exception BEFORE
        ``ready.set()`` (unknown path in path_to_cmd, missing venv python at spawn,
        session.initialize() failure) previously sat unretrieved in the task while
        this method awaited the event forever — an infinite SILENT pipeline hang.
        Now it surfaces immediately as a RuntimeError naming the server."""
        self._task = asyncio.create_task(self._runner())
        ready = asyncio.create_task(self.ready.wait())
        try:
            done, _ = await asyncio.wait(
                {self._task, ready},
                timeout=self.startup_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError(
                    f"MCP server {self.path} did not initialize within "
                    f"{self.startup_timeout:g}s"
                )
            if not self.ready.is_set():
                raise RuntimeError(
                    f"MCP server {self.path} failed to start"
                ) from self._task.exception()
        except BaseException:
            await self.stop()
            raise
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)

    async def _runner(self) -> None:
        """Internal runner that maintains the server connection."""
        self.stack = AsyncExitStack()
        async with self.stack:
            env = os.environ.copy()
            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
            params = StdioServerParameters(
                command=path_to_cmd[self.path], args=[self.path], env=env
            )
            stdio, write = await self.stack.enter_async_context(stdio_client(params))
            session = await self.stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            self.session = session
            self.ready.set()
            await self._stop.wait()

    async def stop(self) -> None:
        """Stop the server and clean up resources."""
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._task), timeout=self.shutdown_timeout
                )
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(self._task, return_exceptions=True),
                        timeout=self.shutdown_timeout,
                    )
                except asyncio.TimeoutError:
                    # A broken async context may ignore cancellation. The process
                    # transport has been asked to stop; do not let cleanup hang the
                    # entire root pipeline indefinitely.
                    pass
            except (asyncio.CancelledError, Exception):
                # Cleanup must be safe for partially-started and already-failed
                # handles. The startup/discovery caller reports the original error.
                pass

    async def list_tools(self) -> list[str]:
        """List all available tools from this server.

        Returns:
            List of tool names available on this server.

        Raises:
            RuntimeError: If the server is not started.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.path} is not started.")
        tools = await self.session.list_tools()
        return [t.name for t in tools.tools]

    async def call_tool(
        self, tool_name: str, args: Optional[dict[str, Any]] = None, timeout: int = 3600
    ) -> Any:
        """Call a tool on this server.

        Args:
            tool_name: Name of the tool to call.
            args: Arguments to pass to the tool.
            timeout: Maximum time to wait for the tool to complete (seconds).

        Returns:
            The tool's response.

        Raises:
            RuntimeError: If the server is not started.
            asyncio.TimeoutError: If the tool call exceeds the timeout.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.path} is not started.")
        return await asyncio.wait_for(
            self.session.call_tool(tool_name, args), timeout=timeout
        )


class ExternalToolClient:
    """Client for connecting to multiple external MCP tool servers.

    Orchestrates connections to multiple MCP servers (e.g., Blender, image processing)
    and provides a unified interface for tool discovery and execution.

    Attributes:
        tool_to_server: Mapping from tool name to server path.
        tool_configs: Tool configurations indexed by server path.
        handles: Active ServerHandle instances indexed by server path.
        tool_servers: List of server script paths to connect to.
    """

    def __init__(
        self, tool_servers: str, args: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the external tool client.

        Args:
            tool_servers: Comma-separated list of MCP server script paths.
            args: Configuration arguments to pass to each server during initialization.
        """
        self.tool_to_server: dict[str, str] = {}
        self.tool_configs: dict[str, list[dict[str, Any]]] = {}
        self.tool_effects: dict[str, dict[str, Any]] = {}
        self.handles: dict[str, ServerHandle] = {}
        if not isinstance(tool_servers, str):
            raise ValueError("tool_servers must be a comma-separated string")
        self.tool_servers = [path.strip() for path in tool_servers.split(",")]
        if not self.tool_servers or any(not path for path in self.tool_servers):
            raise ValueError("tool_servers contains an empty server path")
        if len(self.tool_servers) != len(set(self.tool_servers)):
            raise ValueError("tool_servers contains a duplicate server path")
        self.args = args or {}
        self.startup_timeout = float(
            self.args.get("mcp_startup_timeout") or DEFAULT_MCP_STARTUP_TIMEOUT
        )
        self.discovery_timeout = float(
            self.args.get("mcp_discovery_timeout") or DEFAULT_MCP_DISCOVERY_TIMEOUT
        )

    async def connect_servers(self) -> None:
        """Connect transactionally, rejecting ambiguous tool routing.

        Startup, discovery, and the server-specific ``initialize`` call are all
        bounded. Any error unwinds every handle that reached a partial or complete
        startup before exposing the original exception.
        """
        if self.handles:
            raise RuntimeError("MCP servers are already connected")
        handles = {
            path: ServerHandle(path, startup_timeout=self.startup_timeout)
            for path in self.tool_servers
        }
        tool_to_server: dict[str, str] = {}
        tool_configs: dict[str, list[dict[str, Any]]] = {}
        tool_effects: dict[str, dict[str, Any]] = {}
        try:
            await asyncio.gather(*(handle.start() for handle in handles.values()))
            for path, handle in handles.items():
                tool_names = await asyncio.wait_for(
                    handle.list_tools(), timeout=self.discovery_timeout
                )
                result = await handle.call_tool(
                    "initialize", {"args": self.args}, timeout=self.discovery_timeout
                )
                if not getattr(result, "content", None):
                    raise RuntimeError(
                        f"MCP server {path} returned an empty initialize response"
                    )
                payload = json.loads(result.content[0].text)
                if payload.get("status") != "success":
                    raise RuntimeError(
                        f"MCP server {path} initialize failed: {payload!r}"
                    )
                output = payload.get("output")
                if not isinstance(output, dict) or not isinstance(
                    output.get("tool_configs"), list
                ):
                    raise RuntimeError(
                        f"MCP server {path} returned an invalid initialize payload"
                    )
                declared = {
                    cfg.get("function", {}).get("name")
                    for cfg in output["tool_configs"]
                    if isinstance(cfg, dict)
                }
                declared.discard(None)
                unknown = sorted(declared - set(tool_names))
                if unknown:
                    raise RuntimeError(
                        f"MCP server {path} configured tools it does not expose: {unknown}"
                    )
                collisions = sorted(declared & set(tool_to_server))
                if collisions:
                    owners = {name: tool_to_server[name] for name in collisions}
                    raise RuntimeError(
                        f"MCP tool-name collision from {path}: {collisions}; "
                        f"already provided by {owners}"
                    )
                effects = output.get("tool_effects") or {}
                if not isinstance(effects, dict):
                    raise RuntimeError(
                        f"MCP server {path} returned invalid tool_effects metadata"
                    )
                for tool_name in declared:
                    tool_to_server[tool_name] = path
                    if tool_name in effects:
                        effect = effects[tool_name]
                        if not isinstance(effect, dict):
                            raise RuntimeError(
                                f"MCP server {path} returned invalid effect metadata "
                                f"for {tool_name}"
                            )
                        tool_effects[tool_name] = effect
                tool_configs[path] = output["tool_configs"]
                print(f"MCP Server {path} connected. Tools: {tool_names}")
        except BaseException:
            await asyncio.gather(
                *(handle.stop() for handle in handles.values()),
                return_exceptions=True,
            )
            raise
        self.handles = handles
        self.tool_to_server = tool_to_server
        self.tool_configs = tool_configs
        self.tool_effects = tool_effects

    async def call_tool(
        self, tool_name: str, tool_args: Optional[dict[str, Any]] = None
    ) -> Any:
        """Call a tool by name, routing to the appropriate server.

        Args:
            tool_name: Name of the tool to call.
            tool_args: Arguments to pass to the tool.

        Returns:
            The tool's output response.

        Raises:
            RuntimeError: If the tool is not found in any connected server.
        """
        server_path = self.tool_to_server.get(tool_name)
        if not server_path:
            raise RuntimeError(f"Tool {tool_name} not found in any server.")
        handle = self.handles[server_path]
        result = await handle.call_tool(tool_name, tool_args)
        raw_text = ""
        if getattr(result, "content", None):
            raw_text = getattr(result.content[0], "text", "") or ""

        if not raw_text.strip():
            return self._failure(
                tool_name,
                server_path,
                "empty_response",
                "returned an empty MCP response",
                retryable=True,
            )

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return self._failure(
                tool_name,
                server_path,
                "non_json",
                "returned a non-JSON MCP response",
                raw=raw_text,
                retryable=True,
            )

        if not isinstance(parsed, dict) or "output" not in parsed:
            return self._failure(
                tool_name,
                server_path,
                "invalid_payload",
                "returned an unexpected MCP payload",
                raw=json.dumps(parsed, ensure_ascii=False, default=repr),
                retryable=True,
            )

        out = parsed["output"]
        if parsed.get("status") == "error":
            preserved: dict[str, Any] = {}
            if isinstance(out, dict):
                text = out.get("text")
                if isinstance(text, list):
                    message = " ".join(str(part) for part in text)
                else:
                    message = str(text or out.get("error") or "tool reported an error")
                retryable = bool(out.get("retryable", False))
                # A rule-evaluation transport can fail after collecting useful,
                # revision-bound diagnostics.  Preserve only the bounded public
                # evidence contract; dropping it makes the generator/verifier treat a
                # known backend failure as if no gate was run at all.
                for key in (
                    "rule_evidence",
                    "yaw_evidence",
                    "relationship_evidence",
                    "constraint_results",
                    "yaw_note",
                    "advisory_requirement",
                    "yaw_resolution",
                    "completion_ready",
                    # A composition flip may commit before response formatting
                    # fails.  Preserve its transaction identity/commit verdict so
                    # GeneratorAgent can enforce the mandatory re-investigation and
                    # stale any pre-flip rules pass even on an error response.
                    "required_followup",
                    "resolved_followup",
                    "scene_mutation",
                    # Explicit execute argument classification survives generic MCP
                    # error normalization so GeneratorAgent can grant only the
                    # narrowly scoped, non-mutating format retry allowance.
                    "error_code",
                    "expected_tool",
                ):
                    if key in out:
                        preserved[key] = out[key]
            else:
                message = str(out)
                retryable = False
            return self._failure(
                tool_name,
                server_path,
                "tool_error",
                message,
                raw=json.dumps(parsed, ensure_ascii=False, default=repr),
                retryable=retryable,
                preserved=preserved,
            )
        return out

    @staticmethod
    def _failure(
        tool: str,
        server: str,
        kind: str,
        message: str,
        *,
        raw: Optional[str] = None,
        retryable: bool = False,
        preserved: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Return one bounded, machine-readable failure shape for all protocols."""
        bounded_message = _summarize_failure_message(str(message))
        response: dict[str, Any] = {
            "_tool_error": True,
            "error_kind": kind,
            "tool": tool,
            "server": server,
            "retryable": retryable,
            "text": [
                f"Tool {tool} on {server} failed ({kind}): {bounded_message}. "
                "Use the evidence already available and retry only when appropriate."
            ],
        }
        if raw is not None:
            response["raw"] = _clip_failure_text(str(raw))
        if preserved:
            response.update(preserved)
        return response

    async def cleanup(self) -> None:
        """Clean up connections by stopping all MCP servers."""
        handles, self.handles = self.handles, {}
        await asyncio.gather(
            *(h.stop() for h in handles.values()), return_exceptions=True
        )
        self.tool_to_server.clear()
        self.tool_configs.clear()
        self.tool_effects.clear()
        if handles:
            print("All MCP servers stopped.")
