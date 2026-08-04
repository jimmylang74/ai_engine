#!/usr/bin/env python3
"""
ai_engine.py — LLM Middleware Engine

Stateless middleware that unifies LLM providers via LiteLLM and outputs
structured events (thinking, tool_call, assistant, usage, done) as NDJSON
or raw streaming text.

Usage:
    python3 ai_engine.py --system-prompt-file prompt.txt --user-prompt-file data.mermaid
    python3 ai_engine.py --system-prompt "You are an expert" --user-prompt "analyze this"
    python3 ai_engine.py --provider openai --model gpt-4o --endpoint https://api.openai.com/v1
    python3 ai_engine.py --output-format events --no-stream
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

# litellm is imported lazily inside run_engine() to keep --help fast

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
            "example": f"python3 ai_engine.py --provider {info.key} --model MODEL --endpoint {info.default_endpoint}",
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
# Verbose Logger
# ═══════════════════════════════════════════════════════════════════════

_verbose_out: Any = None  # Optional[TextIO] — stderr or an open log file


def _init_verbose(args: argparse.Namespace) -> None:
    """Configure the verbose-log destination based on CLI args."""
    global _verbose_out
    if args.log:
        _verbose_out = open(args.log, "a", encoding="utf-8")
    elif args.verbose:
        _verbose_out = sys.stderr
    else:
        _verbose_out = None


def _close_verbose() -> None:
    """Close the verbose-log file if it was opened (not stderr)."""
    global _verbose_out
    if _verbose_out is not None and _verbose_out is not sys.stderr:
        _verbose_out.close()
        _verbose_out = None


def vlog(msg: str) -> None:
    """Write a line to the verbose log (no-op unless --verbose/--log is set)."""
    if _verbose_out is not None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        _verbose_out.write(f"[{ts}] {msg}\n")
        _verbose_out.flush()


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
  python3 ai_engine.py --system-prompt-file prompt.txt --user-prompt-file data.mermaid
  python3 ai_engine.py --system-prompt "You are an expert" --user-prompt "analyze this"
  python3 ai_engine.py --provider openai --model gpt-4o --endpoint https://api.openai.com/v1
  python3 ai_engine.py --output-format events --no-stream
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

    # ── Sampling Parameters (optional) ────────────────────────────────
    sampling_group = parser.add_argument_group("Sampling Parameters (optional)")
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: %(default)s). "
        "Only applied when the provider supports it; unsupported params are silently dropped.",
    )
    sampling_group.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p (default: %(default)s). "
        "Only applied when the provider supports it; unsupported params are silently dropped.",
    )

    # ── Prompt Input ─────────────────────────────────────────────────
    prompt_group = parser.add_argument_group("Prompt Input (system prompt)")
    prompt_group.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to system prompt file (default: prompt-fine.txt in script dir)",
    )
    prompt_group.add_argument(
        "--system-prompt",
        default=None,
        help="Inline system prompt text (alternative to --system-prompt-file)",
    )

    # ── Content Input (user message) ──────────────────────────────────
    content_group = parser.add_argument_group("Content Input (user message)")
    content_group.add_argument(
        "--user-prompt-file",
        default=None,
        help="Path to request-content file (replaces --mermaid from v1)",
    )
    content_group.add_argument(
        "--user-prompt",
        default=None,
        help="Inline request-content text (alternative to --user-prompt-file)",
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
        "--verbose",
        action="store_true",
        help="Print detailed logs: messages sent to/received from LLM, stdin input, errors, etc.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="File path to write verbose logs (implies --verbose). All verbose output goes here instead of stderr.",
    )
    parser.add_argument(
        "--get-provider",
        action="store_true",
        help="Print supported providers as JSON and exit",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON requests from stdin (one per line), run_engine for each. Used for long-running subprocess mode.",
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
    if args.system_prompt_file is None and args.system_prompt is None:
        default_prompt = os.path.join(SCRIPT_DIR, "prompt-fine.txt")
        if os.path.isfile(default_prompt):
            args.system_prompt_file = default_prompt
    if args.user_prompt_file is None and args.user_prompt is None:
        default_content = os.path.join(SCRIPT_DIR, "saved.mermaid")
        if os.path.isfile(default_content):
            args.user_prompt_file = default_content

    return args


# ═══════════════════════════════════════════════════════════════════════
# Message Builder
# ═══════════════════════════════════════════════════════════════════════


def build_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    """Assemble the messages list from prompt + content args."""

    # Prompt
    prompt = ""
    if args.system_prompt:
        prompt = args.system_prompt
    elif args.system_prompt_file:
        try:
            prompt = read_file(args.system_prompt_file)
        except FileNotFoundError as e:
            print(f"Prompt file not found: {e}", file=sys.stderr)
            sys.exit(1)

    # Content (the actual request / data)
    content = ""
    if args.user_prompt:
        content = args.user_prompt
    elif args.user_prompt_file:
        try:
            content = read_file(args.user_prompt_file)
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
                vlog(f"[RECV] thinking_delta: {reasoning}")
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
                    vlog(f"[RECV] tool_call_begin: id={tc_id} name={name}")
                    if is_events:
                        emit_event({"type": "tool_call_begin", "id": tc_id, "name": name})

                tc_args = getattr(func, "arguments", "") if func else ""
                if tc_args and current_tool:
                    current_tool["arguments"] += tc_args
                    vlog(f"[RECV] tool_call_delta: id={current_tool['id']} delta={tc_args}")
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
                vlog(f"[RECV] assistant_delta: {content}")
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

    finish_reason = finish_reason or "stop"
    vlog(f"[RECV] finish_reason={finish_reason}")
    # ── Final events ──────────────────────────────────────────────────
    if is_events:
        emit_event({"type": "done", "finish_reason": finish_reason})
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
        vlog(f"[RECV] thinking: {reasoning}")
        if is_events:
            emit_event({"type": "thinking", "content": reasoning})
            emit_event({"type": "thinking_end"})

    # ── Tool Calls ────────────────────────────────────────────────────
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            tc_id = tc.id
            func = tc.function
            vlog(f"[RECV] tool_call: id={tc_id} name={func.name} arguments={func.arguments}")
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
        vlog(f"[RECV] assistant: {content}")
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
    import litellm

    provider_info = PROVIDER_REGISTRY[args.provider]
    litellm_model = f"{provider_info.litellm_name}/{args.model}"
    messages = build_messages(args)
    stream = not args.no_stream
    is_events = args.output_format == "events"

    # ── Verbose: log request ─────────────────────────────────────
    # Optional sampling params — getattr() keeps backward compat with
    # callers that build a Namespace without these fields (defaults apply).
    temperature = getattr(args, "temperature", 0.2)
    top_p = getattr(args, "top_p", 0.9)

    vlog(f"[SEND] model={litellm_model} temperature={temperature} top_p={top_p}")
    vlog(f"[SEND] api_base={args.endpoint}")
    vlog(f"[SEND] messages={json.dumps(messages, ensure_ascii=False, indent=2)}")

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

    kwargs["temperature"] = temperature
    kwargs["top_p"] = top_p

    vlog(f"[SEND] kwargs={json.dumps({k: v for k, v in kwargs.items() if k != 'messages'}, ensure_ascii=False, indent=2)}")

    try:
        response = litellm.completion(**kwargs)
    except Exception as e:
        msg = str(e)
        vlog(f"[ERROR] {msg}")
        if is_events:
            emit_event({"type": "done", "finish_reason": "error"})
            print_raw(f"\nError: {msg}\n")
        else:
            print_raw(f"\nError: {msg}\n")
        sys.exit(1)

    vlog("[RECV] response received, starting output")
    if stream:
        handle_stream(response, is_events)
    else:
        handle_non_stream(response, is_events)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    _init_verbose(args)
    if args.get_provider:
        print(json.dumps(build_provider_json(), indent=2, ensure_ascii=False))
        _close_verbose()
        sys.exit(0)

    if args.stdin:
        import litellm

        litellm.drop_params = True
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            vlog(f"[STDIN] {line}")
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                vlog(f"[ERROR] invalid JSON from stdin: {e}")
                print_raw(f'{{"type":"error","message":"invalid JSON"}}\n')
                continue

            req_args = argparse.Namespace(
                provider=req.get("provider", args.provider),
                model=req.get("model", args.model),
                endpoint=req.get("endpoint", args.endpoint),
                api_key=req.get("api_key", args.api_key),
                temperature=req.get("temperature", getattr(args, "temperature", 0.2)),
                top_p=req.get("top_p", getattr(args, "top_p", 0.9)),
                system_prompt_file=req.get("system_prompt_file", args.system_prompt_file),
                system_prompt=req.get("system_prompt", args.system_prompt),
                user_prompt_file=req.get("user_prompt_file", args.user_prompt_file),
                user_prompt=req.get("user_prompt", ""),
                no_stream=req.get("no_stream", args.no_stream),
                output_format=req.get("output_format", args.output_format),
                get_provider=req.get("get_provider", False),
                stdin=False,
                verbose=args.verbose,
                log=args.log,
            )
            if req_args.get_provider:
                print_raw(json.dumps(build_provider_json(), ensure_ascii=False) + "\n")
            else:
                run_engine(req_args)
    else:
        run_engine(args)

    _close_verbose()


if __name__ == "__main__":
    main()
