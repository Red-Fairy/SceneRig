"""Shared stage-entry ``get_scene_info`` preseed fetch (CT5).

Used by the generator (since 07-29) and the verifier (since 08-07): fetch the scene
inventory ONCE server-side at session entry instead of letting the model spend its
first LLM round asking for it. Each agent formats/appends its own seed message; only
the fetch + failure-shape validation live here.

Contract (verified against the live MCP stack, scratchpad/preseed_probe):
``ExternalToolClient.call_tool`` returns ``parsed["output"]`` — for get_scene_info a
``{"text": [str(scene_info)]}`` dict. Server errors carry ``"_tool_error": True``;
client-level failures come back as ``{"text": ["Tool <name> on <path> ..."]}`` prose.
Do not seed either failure — an error string in the head would be worse than letting
the model fetch for itself. (Two prior parses each missed this shape and silently
skipped every seed: v1 looked for an ``{"output": ...}`` envelope the client had
already stripped; v2 mistook the success shape for the client-failure shape.)
"""

import os
from typing import Any, Optional


async def fetch_scene_info(tool_client: Any) -> Optional[str]:
    """Return the scene-info text to seed, or None (missing tool / failure / flag off)."""
    if os.environ.get("GRASE_STAGE_PRESEED", "1") == "0":
        return None
    if "get_scene_info" not in tool_client.tool_to_server:
        return None
    try:
        resp = await tool_client.call_tool("get_scene_info", {})
    except Exception as exc:  # noqa: BLE001
        print(f"[preseed] get_scene_info failed ({exc}); model can still ask")
        return None
    if not isinstance(resp, dict) or resp.get("_tool_error"):
        return None
    texts = resp.get("text")
    if not isinstance(texts, list):
        return None
    info = "\n".join(t for t in texts if isinstance(t, str)).strip()
    if not info or info.startswith("Tool get_scene_info on "):
        return None
    return info
