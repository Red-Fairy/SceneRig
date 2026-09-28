"""Verifier Agent for analyzing generated scenes in the GRASE system.

The Verifier Agent analyzes rendered scenes from multiple viewpoints,
comparing them against target images and providing structured feedback
for the Generator Agent to refine its output.
"""

import json
import os
import traceback
from typing import Any

from lib.agents.prompt_builder import PromptBuilder
from lib.agents.scene_preseed import fetch_scene_info
from lib.agents.tool_client import ExternalToolClient
from lib.utils.common import (
    AGENT_VLM_EDGE,
    build_client,
    display_path,
    get_image_base64,
    get_model_response,
)


class VerifierAgent:
    """Agent responsible for analyzing and verifying generated scenes.

    The Verifier Agent examines rendered outputs from the Generator,
    comparing them against targets using camera manipulation tools
    and providing actionable feedback for refinement.

    Attributes:
        config: Configuration dictionary containing model settings and paths.
        memory: Working conversation memory (reset at the start of run() when
            clear_memory is set).
        saved_memory: Persistent conversation memory for logging.
        tool_client: Client for calling external MCP tools.
    """

    def __init__(self, args: dict[str, Any]) -> None:
        """Initialize the Verifier Agent.

        Args:
            args: Configuration dictionary with keys like 'model', 'api_key',
                  'verifier_tools', 'max_rounds', 'clear_memory', etc.
        """
        self.config = args
        self.agent_name = self.config.get(
            "verifier_agent_name", self.__class__.__name__
        )
        self.memory: list[dict[str, Any]] = []
        self.saved_memory: list[dict[str, Any]] = []

        # Provider-specific request shaping (OpenAI's parallel_tool_calls, reasoning
        # effort, max_completion_tokens) lives in common.get_model_response.

        # Initialize tool client
        self.tool_client = ExternalToolClient(
            self.config.get("verifier_tools"), self.config
        )

        # Initialize LLM client (build_client carries per-provider settings, e.g.
        # the fireworks 3600s timeout; config creds still win when set).
        self.client = build_client(
            self.config["model"],
            api_key=self.config.get("api_key"),
            base_url=self.config.get("api_base_url"),
        )

        # Initialize system prompt
        self.prompt_builder = PromptBuilder(self.client, self.config)
        self.system_prompt = self.prompt_builder.build_prompt("verifier", "system")
        self.memory.extend(self.system_prompt)
        self.saved_memory.extend(self.system_prompt)
        self.protected_head = len(self.memory)

    async def _preseed_scene_info(self) -> None:
        """Attach the delivered scene's bounded entry snapshot when available."""
        info = await fetch_scene_info(self.tool_client)
        if not info:
            return
        seed = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "CURRENT SCENE STATE (auto-fetched at review start; bounded "
                        "snapshot of the delivered scene). Do not call get_scene_info "
                        "to re-fetch it during this read-only review, and do not settle "
                        "visual questions from it alone:\n" + info
                    ),
                }
            ],
        }
        self.memory.append(seed)
        self.saved_memory.append(seed)

    async def run(self, user_message: dict[str, Any]) -> dict[str, Any]:
        """Verify the generated scene and provide feedback.

        Analyzes the rendered output using chain-of-thought reasoning and
        camera manipulation tools to compare against the target.

        Args:
            user_message: Dictionary containing 'argument' (sanitized generator
                          results) and 'execution' (tool execution results).

        Returns:
            Dictionary with 'text' containing feedback for the generator.
        """
        try:
            return await self._run_inner(user_message)
        except Exception as e:
            error_text = traceback.format_exc()
            self._save_error(error_text)
            print(f"Verifier failed, returning fallback feedback: {e}")
            result = {
                "approved": False,
                "text": [
                    "Verifier failed before completing this round. Continue using the latest rendered image and your own visual comparison. "
                    f"Verifier error: {type(e).__name__}: {e}"
                ],
            }
            self.saved_memory.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": result["text"][0]}],
                }
            )
            self._save_memory()
            return result

    async def _run_inner(self, user_message: dict[str, Any]) -> dict[str, Any]:
        """Internal verifier loop. Exceptions are handled by run()."""
        print(f"\n=== Running {self.agent_name} ===\n")

        # Reload scene if needed
        if "reload_scene" in self.tool_client.tool_to_server:
            print("Reload scene...")
            await self.tool_client.call_tool("reload_scene", {})
        print("Build user message...")
        self._last_ref_render = (user_message.get("argument") or {}).get(
            "current_scene_render"
        )
        user_message = self.prompt_builder.build_prompt(
            "verifier", "user", user_message
        )
        if self.config.get("clear_memory"):
            print("Clear memory...")
            self.memory = self.system_prompt + user_message
        else:
            print("Extend memory...")
            self.memory.extend(user_message)
        self.saved_memory.extend(user_message)
        await self._preseed_scene_info()
        self.protected_head = len(self.memory)
        print("Save memory...")
        self._save_memory()
        result = None

        for i in range(self.config.get("max_rounds")):
            print(f"=== Round {i} ===\n")

            # Prepare chat args
            print("Prepare chat args...")
            memory = self.prompt_builder.build_memory_append_only(
                self.memory, self.protected_head
            )
            tool_configs = self.tool_client.tool_configs
            tool_configs = [x for v in tool_configs.values() for x in v]
            chat_args = {
                "model": self.config.get("model"),
                "messages": memory,
                "tools": tool_configs,
                "tool_choice": "auto",
            }

            # Generate response
            print("Generate response...")
            response = get_model_response(
                self.client,
                chat_args,
                effort=os.environ.get("GRASE_VERIFIER_EFFORT", "medium"),
            )
            message = response.choices[0].message

            # Handle tool call
            print("Handle tool call...")
            if not message.tool_calls:
                _raw = getattr(message, "raw_content", None)
                self.memory.append(
                    {
                        **({"_raw_blocks": _raw} if _raw else {}),
                        "role": "assistant",
                        "content": message.content,
                    }
                )
                self.memory.append(
                    {
                        "role": "user",
                        "content": "Each return message must contain a tool call. Your previous message did not contain a tool call. Please reconsider.",
                    }
                )
                self.saved_memory.append(
                    {"role": "assistant", "content": message.content}
                )
                self.saved_memory.append(
                    {
                        "role": "user",
                        "content": "Each return message must contain a tool call. Your previous message did not contain a tool call. Please reconsider.",
                    }
                )
                self._save_memory()
                continue
            else:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                print(f"Call tool {tool_name}...")
                try:
                    tool_arguments = json.loads(tool_call.function.arguments)
                except (TypeError, json.JSONDecodeError) as exc:
                    print(f"Tool {tool_name}: arguments are not valid JSON ({exc})")
                    note = (
                        f"Your previous {tool_name} call carried arguments that were not "
                        "valid JSON — most likely the reply was CUT OFF at the output "
                        "token limit. Nothing was executed. Call the tool again with "
                        "COMPLETE, compact arguments."
                    )
                    text = (
                        message.content
                        or f"[{tool_name} call with unparseable arguments]"
                    )
                    self.memory.append({"role": "assistant", "content": text})
                    self.memory.append({"role": "user", "content": note})
                    self.saved_memory.append({"role": "assistant", "content": text})
                    self.saved_memory.append({"role": "user", "content": note})
                    self._save_memory()
                    continue
                tool_response = await self.tool_client.call_tool(
                    tool_name, tool_arguments
                )
                if tool_name == "render_reference_view" and (
                    (tool_response or {}).get("image") or []
                ):
                    self._last_ref_render = tool_response["image"][0]

            # Update and save memory
            print("Update and save memory...")
            self._update_memory({"assistant": message, "user": tool_response})
            self._save_memory()

            if tool_name == "end":
                result = tool_response
                break

        if result is None:
            result = await self._force_end_after_max_rounds()

        print(f"\n=== Finish {self.agent_name} process ===\n")

        if result:
            return result
        else:
            return {
                "text": [
                    "No valid response, please observe the output image and adjust your code accordingly."
                ]
            }

    def _build_tool_configs_for_names(self, names: set[str]) -> list[dict[str, Any]]:
        """Build verifier tool configs filtered to specific tool names."""
        return [
            tool
            for tools in self.tool_client.tool_configs.values()
            for tool in tools
            if tool.get("function", {}).get("name") in names
        ]

    async def _force_end_after_max_rounds(self) -> dict[str, Any]:
        """Ask the verifier for one final end call when max rounds are exhausted."""
        if "end" not in self.tool_client.tool_to_server:
            return self._fallback_max_round_result()
        prompt = (
            "Max verifier rounds have been reached. You already have enough scene information and rendered views. "
            "Do not call more investigation tools. Call the end tool now. "
            "Use the prior approval_checklist/verifier_history if present, mark items done or pending, include regression_check, "
            "include problem_images that best show blocking issues, include suggested_fixes for pending checklist items, "
            "and decide approved based on whether the current attempt satisfies the checklist without regressions."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        ref = getattr(self, "_last_ref_render", None)
        if ref and os.path.exists(ref):
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Newest REFERENCE-VIEW render of the current scene, re-attached "
                        "for this final review (the earlier copy may have scrolled out "
                        "of your visible window). Judge reference-view checklist items "
                        "from THIS image — it is available this round."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": get_image_base64(ref, max_edge=AGENT_VLM_EDGE)
                    },
                }
            )
        self.memory.append({"role": "user", "content": content})
        self.saved_memory.append({"role": "user", "content": content})
        self._save_memory()

        memory = self.prompt_builder.build_memory_append_only(
            self.memory, self.protected_head
        )
        chat_args = {
            "model": self.config.get("model"),
            "messages": memory,
            "tools": self._build_tool_configs_for_names({"end"}),
            "tool_choice": "auto",
        }
        response = get_model_response(
            self.client,
            chat_args,
            effort=os.environ.get("GRASE_VERIFIER_EFFORT", "medium"),
        )
        message = response.choices[0].message
        if message.tool_calls and message.tool_calls[0].function.name == "end":
            try:
                tool_response = await self.tool_client.call_tool(
                    "end", json.loads(message.tool_calls[0].function.arguments or "{}")
                )
                self._update_memory({"assistant": message, "user": tool_response})
                self._save_memory()
                return tool_response
            except Exception as e:
                self.memory.append(
                    {
                        "role": "assistant",
                        "content": message.content or "Verifier end tool call failed.",
                    }
                )
                self.memory.append(
                    {"role": "user", "content": f"Forced end failed: {e}"}
                )
                self.saved_memory.append(
                    {
                        "role": "assistant",
                        "content": message.content or "Verifier end tool call failed.",
                    }
                )
                self.saved_memory.append(
                    {"role": "user", "content": f"Forced end failed: {e}"}
                )
                self._save_memory()
                return self._fallback_max_round_result()

        self.memory.append(
            {
                "role": "assistant",
                "content": message.content or "No end tool call returned.",
            }
        )
        self.saved_memory.append(
            {
                "role": "assistant",
                "content": message.content or "No end tool call returned.",
            }
        )
        self._save_memory()
        return self._fallback_max_round_result()

    def _fallback_max_round_result(self) -> dict[str, Any]:
        """Return a structured rejection if the verifier still cannot call end."""
        checklist = self.config.get("stage_approval_checklist", [])
        text = (
            "Verifier reached max rounds without a valid end call. Treating this attempt as not approved. "
            "Use the current render, scene info, and prior checklist to continue."
        )
        return {
            "approved": False,
            "approval_checklist": checklist,
            "problem_images": [],
            "suggested_fixes": [],
            "regression_check": "No verifier end decision was produced before max rounds.",
            "text": [text],
        }

    def _update_memory(self, message: dict[str, Any]) -> None:
        """Update both working and saved memory with assistant and tool messages.

        Args:
            message: Dictionary containing 'assistant' (the model response) and
                     'user' (the tool response with text and optional images).
        """
        # Add tool calling
        assistant_content = message["assistant"].content
        assistant_tool_calls = message["assistant"].tool_calls[0].model_dump()
        _raw = getattr(message["assistant"], "raw_content", None)
        _extra = {"_raw_blocks": _raw} if _raw else {}
        self.memory.append(
            {
                **_extra,
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [assistant_tool_calls],
            }
        )
        self.saved_memory.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [assistant_tool_calls],
            }
        )

        # Add tool response
        tool_call_id = message["assistant"].tool_calls[0].id
        tool_call_name = message["assistant"].tool_calls[0].function.name
        tool_response = []
        user_response = []
        if "image" in message["user"]:
            tool_response.append(
                {
                    "type": "text",
                    "text": "The next user message contains the image result of the tool call.",
                }
            )
            response_texts = message["user"].get("text", [])
            response_images = message["user"]["image"]
            for idx, image in enumerate(response_images):
                text = (
                    response_texts[idx]
                    if idx < len(response_texts)
                    else f"Render image {idx}"
                )
                user_response.append({"type": "text", "text": text})
                user_response.append(
                    {
                        "type": "image_url",
                        # Tool-returned round image: agent-loop cap (768).
                        "image_url": {
                            "url": get_image_base64(image, max_edge=AGENT_VLM_EDGE)
                        },
                    }
                )
                user_response.append(
                    {
                        "type": "text",
                        "text": (
                            "Image loaded from local path: "
                            f"{display_path(image, self.config.get('scene_root'))}"
                        ),
                    }
                )
            # Preserve any text entries that have no matching image.
            for text in response_texts[len(response_images) :]:
                tool_response.append({"type": "text", "text": text})
        else:
            for text in message["user"]["text"]:
                tool_response.append({"type": "text", "text": text})

        self.memory.append(
            {
                "role": "tool",
                "content": tool_response,
                "name": tool_call_name,
                "tool_call_id": tool_call_id,
            }
        )
        self.saved_memory.append(
            {
                "role": "tool",
                "content": tool_response,
                "name": tool_call_name,
                "tool_call_id": tool_call_id,
            }
        )
        if user_response:
            self.memory.append({"role": "user", "content": user_response})
            self.saved_memory.append({"role": "user", "content": user_response})

    def _save_memory(self) -> None:
        """Save the persistent memory to a JSON file in the output directory."""
        output_file = (
            self.config.get("output_dir")
            + "/"
            + self.config.get("verifier_memory_filename", "verifier_memory.json")
        )
        with open(output_file, "w") as f:
            json.dump(self.saved_memory, f, indent=4, ensure_ascii=False)

    def _save_error(self, error_text: str) -> None:
        """Append verifier exceptions to a per-task error log."""
        output_dir = self.config.get("output_dir")
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "verifier_errors.log"), "a") as f:
            f.write(error_text)
            f.write("\n" + "=" * 80 + "\n")

    async def cleanup(self) -> None:
        """Clean up external connections."""
        await self.tool_client.cleanup()
