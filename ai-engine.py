#!/usr/bin/env python3
"""
ai-engine.py — LLM Middleware Engine

Stateless middleware that unifies LLM providers via LiteLLM and outputs
structured events (thinking, tool_call, assistant, usage, done) as NDJSON
or raw streaming text.

Usage:
    python3 ai-engine.py --prompt-file prompt.txt -f data.mermaid
    python3 ai-engine.py --prompt-text "You are an expert" --text "analyze this"
    python3 ai-engine.py --provider openai --model gpt-4o --endpoint https://api.openai.com/v1
    python3 ai-engine.py --output-format events --no-stream
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import litellm
# litellm types are dynamic; use Any for stream/batch response objects

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════
# Provider Registry
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ProviderInfo:
    """A supported LLM provider and its LiteLLM mapping."""

    key: str  # CLI-facing key
    litellm_name: str  # LiteLLM provider prefix
    default_endpoint: str  # Default API base URL
    description: str  # Human-readable description


PROVIDER_REGISTRY: dict[str, ProviderInfo] = {
    "ollama_native": ProviderInfo(
        key="ollama_native",
        litellm_name="ollama_chat",
        default_endpoint="http://192.168.10.39:11434",
        description="Ollama Chat API  (/api/chat)",
    ),
    "ollama": ProviderInfo(
        key="ollama",
        litellm_name="ollama",
        default_endpoint="http://192.168.10.39:11434",
        description="Ollama Generate API  (/api/generate)",
    ),
    "openai": ProviderInfo(
        key="openai",
        litellm_name="openai",
        default_endpoint="https://api.openai.com/v1",
        description="OpenAI API",
    ),
    "anthropic": ProviderInfo(
        key="anthropic",
        litellm_name="anthropic",
        default_endpoint="https://api.anthropic.com",
        description="Anthropic Claude API",
    ),
    "gemini": ProviderInfo(
        key="gemini",
        litellm_name="gemini",
        default_endpoint="https://generativelanguage.googleapis.com/v1beta",
        description="Google Gemini API",
    ),
    "deepseek": ProviderInfo(
        key="deepseek",
        litellm_name="deepseek",
        default_endpoint="https://api.deepseek.com/v1",
        description="DeepSeek API",
    ),
    "groq": ProviderInfo(
        key="groq",
        litellm_name="groq",
        default_endpoint="https://api.groq.com/openai/v1",
        description="Groq Cloud API  (OpenAI-compatible)",
    ),
    "together": ProviderInfo(
        key="together",
        litellm_name="together",
        default_endpoint="https://api.together.xyz/v1",
        description="Together AI API  (OpenAI-compatible)",
    ),
    "mistral": ProviderInfo(
        key="mistral",
        litellm_name="mistral",
        default_endpoint="https://api.mistral.ai/v1",
        description="Mistral AI API",
    ),
    "custom_openai": ProviderInfo(
        key="custom_openai",
        litellm_name="openai",
        default_endpoint="http://192.168.10.39:11434/v1",
        description="Custom OpenAI-compatible API  (v1 path)",
    ),
}

DEFAULT_PROVIDER = "ollama_native"


def build_provider_table() -> str:
    """Build a formatted provider-reference table for --help epilog."""
    lines = [
        "Provider              LiteLLM Prefix       Default Endpoint",
        "────────────────────────────────────────────────────────────────────────────",
    ]
    for key in PROVIDER_REGISTRY:
        info = PROVIDER_REGISTRY[key]
        marker = " (default)" if key == DEFAULT_PROVIDER else ""
        lines.append(f"{info.key:<20} {info.litellm_name:<20} {info.default_endpoint}")
    return "\n".join(lines)


def build_provider_json() -> list[dict[str, str]]:
    """Build a JSON-serializable list of providers with example endpoints."""
    providers = []
    for info in PROVIDER_REGISTRY.values():
        providers.append({
            "provider": info.key,
            "litellm_name": info.litellm_name,
            "default_endpoint": info.default_endpoint,
            "description": info.description,
            "example": f"python3 ai-engine.py --provider {info.key} --model MODEL --endpoint {info.default_endpoint}",
        })
    return providers


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def read_file(path: str) -> str:
    """Read a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def emit_event(event: dict[str, Any]) -> None:
    """Print one NDJSON event line to stdout (flush immediately)."""
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def print_raw(text: str, end: str = "") -> None:
    """Print raw text to stdout (no newline by default)."""
    sys.stdout.write(text)
    sys.stdout.write(end)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════
# Argument Parsing
# ═══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Engine — LLM Middleware (LiteLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Provider reference (--provider):
{build_provider_table()}

Environment variables:
  LLM_ENDPOINT      Override --endpoint
  LLM_MODEL         Override --model
  LLM_PROVIDER      Override --provider
  LLM_API_KEY       Override --api-key

Examples:
  python3 ai-engine.py --prompt-file prompt.txt -f data.mermaid
  python3 ai-engine.py --prompt-text "You are an expert" --text "analyze this"
  python3 ai-engine.py --provider openai --model gpt-4o --endpoint https://api.openai.com/v1
  python3 ai-engine.py --output-format events --no-stream
        """,
    )

    # ── Provider / Model / Endpoint ──────────────────────────────────
    provider_choices = list(PROVIDER_REGISTRY.keys())
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=provider_choices,
        help=f"LLM provider (default: {DEFAULT_PROVIDER}). See table below.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="qwen3.5:27b",
        help="Model name (default: %(default)s, env: LLM_MODEL)",
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        default="http://192.168.10.39:11434",
        help="API endpoint URL (default: %(default)s, env: LLM_ENDPOINT)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication (env: LLM_API_KEY). "
        "LiteLLM also respects standard env vars like OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.",
    )

    # ── Prompt Input ─────────────────────────────────────────────────
    prompt_group = parser.add_argument_group("Prompt Input (system prompt)")
    prompt_group.add_argument(
        "--prompt-file",
        default=None,
        help="Path to prompt file (default: prompt-fine.txt in script dir)",
    )
    prompt_group.add_argument(
        "--prompt-text",
        default=None,
        help="Inline prompt text (alternative to --prompt-file)",
    )

    # ── Content Input (user message) ──────────────────────────────────
    content_group = parser.add_argument_group("Content Input (user message)")
    content_group.add_argument(
        "-f",
        "--file",
        default=None,
        help="Path to request-content file (replaces --mermaid from v1)",
    )
    content_group.add_argument(
        "--text",
        default=None,
        help="Inline request-content text (alternative to -f/--file)",
    )

    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming — wait for the full response before emitting",
    )
    parser.add_argument(
        "--output-format",
        choices=["raw", "events"],
        default="raw",
        help="Output style (default: %(default)s). "
        "'raw' — plain streaming text (backward compatible). "
        "'events' — NDJSON event stream (programmatic).",
    )
    parser.add_argument(
        "--get-provider",
        action="store_true",
        help="Print supported providers as JSON and exit",
    )

    # Parse
    args = parser.parse_args()

    # Environment overrides
    if os.environ.get("LLM_ENDPOINT"):
        args.endpoint = os.environ["LLM_ENDPOINT"]
    if os.environ.get("LLM_MODEL"):
        args.model = os.environ["LLM_MODEL"]
    if os.environ.get("LLM_PROVIDER"):
        args.provider = os.environ["LLM_PROVIDER"]
    if os.environ.get("LLM_API_KEY"):
        args.api_key = os.environ["LLM_API_KEY"]

    # Fill defaults for optional paths
    if args.prompt_file is None and args.prompt_text is None:
        default_prompt = os.path.join(SCRIPT_DIR, "prompt-fine.txt")
        if os.path.isfile(default_prompt):
            args.prompt_file = default_prompt
    if args.file is None and args.text is None:
        default_content = os.path.join(SCRIPT_DIR, "saved.mermaid")
        if os.path.isfile(default_content):
            args.file = default_content

    return args


# ═══════════════════════════════════════════════════════════════════════
# Message Builder
# ═══════════════════════════════════════════════════════════════════════


def build_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    """Assemble the messages list from prompt + content args."""

    # Prompt
    prompt = ""
    if args.prompt_text:
        prompt = args.prompt_text
    elif args.prompt_file:
        try:
            prompt = read_file(args.prompt_file)
        except FileNotFoundError as e:
            print(f"Prompt file not found: {e}", file=sys.stderr)
            sys.exit(1)

    # Content (the actual request / data)
    content = ""
    if args.text:
        content = args.text
    elif args.file:
        try:
            content = read_file(args.file)
        except FileNotFoundError as e:
            print(f"Content file not found: {e}", file=sys.stderr)
            sys.exit(1)

    # Build message
    if prompt and content:
        return [{"role": "user", "content": prompt + "\n\n" + content}]
    elif prompt:
        return [{"role": "user", "content": prompt}]
    elif content:
        return [{"role": "user", "content": content}]
    else:
        return [{"role": "user", "content": "Hello!"}]


# ═══════════════════════════════════════════════════════════════════════
# Streaming Handler
# ═══════════════════════════════════════════════════════════════════════


def handle_stream(
    response: Any,  # litellm CustomStreamWrapper (dynamically typed)
    is_events: bool,
) -> None:
    """
    Process a streaming LLM response.

    Manages a small state machine:
        idle → thinking → assistant → done
        idle → assistant → done
        idle → tool_call → [assistant | done]
    """
    # State
    phase: str = "idle"  # idle | thinking | assistant | tool_call
    current_tool: dict[str, Any] | None = None
    assistant_parts: list[str] = []
    thinking_parts: list[str] = []
    finish_reason: str | None = None

    def close_phase(next_phase: str) -> None:
        """Emit end events for the current phase before transitioning."""
        nonlocal phase
        if phase == "thinking":
            if is_events:
                emit_event({"type": "thinking_end"})
        elif phase == "assistant":
            if is_events and assistant_parts:
                emit_event({"type": "assistant_end"})
        elif phase == "tool_call" and current_tool:
            if is_events:
                emit_event({
                    "type": "tool_call_end",
                    "id": current_tool["id"],
                })
        phase = next_phase

    for chunk in response:
        # ── Usage embedded in a no-choice chunk (some providers) ──────
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                _emit_usage(chunk.usage, is_events)
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        # ── Finish reason on final chunk ──────────────────────────────
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
            # The last chunk may still carry content – handle it below
            # but we break right after processing it.
            # (fall through to content/tool processing then break)

        # ── Thinking / Reasoning content ──────────────────────────────
        reasoning = _get_reasoning(delta)
        if reasoning:
            if phase not in ("thinking", "idle"):
                close_phase("thinking")
            elif phase == "idle":
                phase = "thinking"

            if phase == "thinking":
                thinking_parts.append(reasoning)
                if is_events:
                    emit_event({"type": "thinking_delta", "delta": reasoning})
                else:
                    print_raw(reasoning)

            if finish_reason is not None:
                break
            continue

        # ── Tool Calls ────────────────────────────────────────────────
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            if phase != "tool_call":
                close_phase("tool_call")

            for tc in tool_calls:
                tc_id = getattr(tc, "id", None)
                func = getattr(tc, "function", None)

                if tc_id:
                    # Beginning of a brand-new tool call
                    current_tool = {"id": tc_id, "name": "", "arguments": ""}
                    name = getattr(func, "name", "") if func else ""
                    current_tool["name"] = name
                    phase = "tool_call"
                    if is_events:
                        emit_event({"type": "tool_call_begin", "id": tc_id, "name": name})

                tc_args = getattr(func, "arguments", "") if func else ""
                if tc_args and current_tool:
                    current_tool["arguments"] += tc_args
                    if is_events:
                        emit_event({
                            "type": "tool_call_delta",
                            "id": current_tool["id"],
                            "delta": tc_args,
                        })

            if finish_reason is not None:
                break
            continue

        # ── Assistant content ─────────────────────────────────────────
        content = getattr(delta, "content", None) or ""
        if content:
            if phase not in ("assistant", "idle"):
                close_phase("assistant")
            elif phase == "idle":
                phase = "assistant"

            if phase == "assistant":
                assistant_parts.append(content)
                if is_events:
                    emit_event({"type": "assistant_delta", "delta": content})
                else:
                    print_raw(content)

        if finish_reason is not None:
            break

    # ── Close all open phases ─────────────────────────────────────────
    if phase == "thinking":
        if is_events:
            emit_event({"type": "thinking_end"})
    elif phase == "assistant":
        if is_events and assistant_parts:
            emit_event({"type": "assistant_end"})
    elif phase == "tool_call" and current_tool:
        if is_events:
            emit_event({
                "type": "tool_call_end",
                "id": current_tool["id"],
            })

    # ── Final events ──────────────────────────────────────────────────
    if is_events:
        emit_event({"type": "done", "finish_reason": finish_reason or "stop"})
    else:
        print_raw("\n")


def _get_reasoning(obj: Any) -> str:
    """Extract reasoning/thinking content from a delta or message object."""
    for attr in ("reasoning_content", "reasoning", "thinking"):
        val = getattr(obj, attr, None)
        if val:
            return val
    return ""


def _emit_usage(usage: Any, is_events: bool) -> None:
    """Emit usage info if available."""
    if not is_events:
        return
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    rt = getattr(usage, "reasoning_tokens", 0) or 0
    emit_event({
        "type": "usage",
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": rt,
    })


# ═══════════════════════════════════════════════════════════════════════
# Non-Streaming Handler
# ═══════════════════════════════════════════════════════════════════════


def handle_non_stream(response: Any, is_events: bool) -> None:  # litellm ModelResponse
    """Process a non-streaming (batched) LLM response."""
    choice = response.choices[0] if response.choices else None
    if choice is None:
        if is_events:
            emit_event({"type": "done", "finish_reason": "error"})
        else:
            print_raw("[no response]\n")
        return

    message = choice.message
    finish_reason = choice.finish_reason or "stop"

    # ── Thinking ──────────────────────────────────────────────────────
    reasoning = _get_reasoning(message)
    if reasoning:
        if is_events:
            emit_event({"type": "thinking", "content": reasoning})
            emit_event({"type": "thinking_end"})

    # ── Tool Calls ────────────────────────────────────────────────────
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            tc_id = tc.id
            func = tc.function
            if is_events:
                emit_event({"type": "tool_call_begin", "id": tc_id, "name": func.name})
                try:
                    parsed_args = json.loads(func.arguments)
                except (json.JSONDecodeError, TypeError):
                    parsed_args = func.arguments
                emit_event({"type": "tool_call_end", "id": tc_id, "arguments": parsed_args})
            else:
                print_raw(f"[tool_call: {func.name}({func.arguments})]\n")

    # ── Assistant content ─────────────────────────────────────────────
    content = getattr(message, "content", None) or ""
    if content:
        if is_events:
            emit_event({"type": "assistant", "content": content})
            emit_event({"type": "assistant_end"})
        else:
            print_raw(content, end="\n")

    # ── Usage ─────────────────────────────────────────────────────────
    if hasattr(response, "usage") and response.usage is not None:
        _emit_usage(response.usage, is_events)

    # ── Done ──────────────────────────────────────────────────────────
    if is_events:
        emit_event({"type": "done", "finish_reason": finish_reason})


# ═══════════════════════════════════════════════════════════════════════
# Engine Entry Point
# ═══════════════════════════════════════════════════════════════════════


def run_engine(args: argparse.Namespace) -> None:
    """Build messages, call LiteLLM, route to streaming or batch handler."""
    provider_info = PROVIDER_REGISTRY[args.provider]
    litellm_model = f"{provider_info.litellm_name}/{args.model}"
    messages = build_messages(args)
    stream = not args.no_stream
    is_events = args.output_format == "events"

    # LiteLLM global config
    litellm.drop_params = True  # silently drop unsupported params

    kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "stream": stream,
        "api_base": args.endpoint,
    }
    if args.api_key:
        kwargs["api_key"] = args.api_key

    try:
        response = litellm.completion(**kwargs)
    except Exception as e:
        msg = str(e)
        if is_events:
            emit_event({"type": "done", "finish_reason": "error"})
            print_raw(f"\nError: {msg}\n")
        else:
            print_raw(f"\nError: {msg}\n")
        sys.exit(1)

    if stream:
        handle_stream(response, is_events)
    else:
        handle_non_stream(response, is_events)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    if args.get_provider:
        print(json.dumps(build_provider_json(), indent=2, ensure_ascii=False))
        sys.exit(0)
    run_engine(args)


if __name__ == "__main__":
    main()
