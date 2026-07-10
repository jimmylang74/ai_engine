# ai_engine.py — LLM Middleware Engine

Stateless CLI middleware that unifies LLM providers via [LiteLLM](https://github.com/BerriAI/litellm). Supports streaming and batch responses, outputting structured NDJSON events or raw text.

## Quick Start

```bash
pip install -r requirements.txt

# Streaming raw text (default)
python3 ai_engine.py --prompt-text "You are an expert" --text "analyze this"

# NDJSON event stream (programmatic consumption)
python3 ai_engine.py --output-format events --text "hello"

# Non-streaming (wait for full response)
python3 ai_engine.py --no-stream --text "summarize this"
```

## Supported Providers

| Provider     | LiteLLM Prefix     | Default Endpoint                          |
|-------------|--------------------|------------------------------------------|
| ollama_native | `ollama_chat`     | `http://192.168.10.39:11434`            |
| ollama       | `ollama`           | `http://192.168.10.39:11434`            |
| openai       | `openai`           | `https://api.openai.com/v1`              |
| anthropic    | `anthropic`        | `https://api.anthropic.com`              |
| gemini       | `gemini`           | `https://generativelanguage.googleapis.com/v1beta` |
| deepseek     | `deepseek`         | `https://api.deepseek.com/v1`            |
| groq         | `groq`             | `https://api.groq.com/openai/v1`         |
| together     | `together`         | `https://api.together.xyz/v1`            |
| mistral      | `mistral`          | `https://api.mistral.ai/v1`              |
| custom_openai | `openai`          | `http://192.168.10.39:11434/v1`          |

Default provider: **ollama_native**

## Usage

### Provider / Model Selection

```bash
python3 ai_engine.py --provider openai --model gpt-4o --endpoint https://api.openai.com/v1
python3 ai_engine.py --provider anthropic --model claude-sonnet-4-20250514 --api-key sk-...
python3 ai_engine.py --provider ollama_native --model qwen3.5:27b
```

### Input: System Prompt

```bash
python3 ai_engine.py --prompt-file prompt.txt --text "user message"
python3 ai_engine.py --prompt-text "You are a helpful assistant" --text "hello"
```

If neither `--prompt-file` nor `--prompt-text` is given, the engine falls back to `prompt-fine.txt` in the script directory (if it exists).

### Input: User Content

```bash
python3 ai_engine.py -f data.mermaid          # read content from file
python3 ai_engine.py --text "inline text"      # inline content
python3 ai_engine.py                           # falls back to saved.mermaid (if it exists)
```

### Output Formats

**Raw** (default) — plain streaming text, backward compatible:

```bash
python3 ai_engine.py --output-format raw --text "hello"
```

**Events** — NDJSON event stream, one JSON object per line:

```bash
python3 ai_engine.py --output-format events --text "hello"
```

### Debugging / Verbose Logging

```bash
# Print detailed logs to stderr (messages sent to/received from LLM, stdin input, errors)
python3 ai_engine.py --verbose --text "hello"

# Redirect all verbose logs to a file
python3 ai_engine.py --log /tmp/ai-engine.log --text "hello"

# Combine with --stdin mode to trace all requests
python3 ai_engine.py --stdin --output-format events --log /tmp/ai-engine.log
```

Verbose output includes:
- `[SEND]` — model, API endpoint, and the full messages payload sent to the LLM
- `[RECV]` — each chunk (thinking, assistant content, tool calls) received from the LLM
- `[STDIN]` — each JSON request line read from stdin (in `--stdin` mode)
- `[ERROR]` — exception messages and invalid JSON errors

### Provider Discovery

```bash
# List all supported providers as JSON
python3 ai_engine.py --get-provider | jq .
```

#### Event Types

| Event              | Fields                                  | When                     |
|-------------------|-----------------------------------------|--------------------------|
| `thinking_delta`  | `delta`                                 | Reasoning chunk arrives   |
| `thinking_end`    | —                                       | Reasoning phase finished  |
| `assistant_delta` | `delta`                                 | Content chunk arrives     |
| `assistant_end`   | —                                       | Content phase finished    |
| `tool_call_begin` | `id`, `name`                            | Tool call starts          |
| `tool_call_delta` | `id`, `delta`                           | Tool call argument chunk  |
| `tool_call_end`   | `id`, `arguments`                       | Tool call finished        |
| `usage`           | `prompt_tokens`, `completion_tokens`, `reasoning_tokens` | Token usage |
| `done`            | `finish_reason`                         | Response complete         |

### Non-Streaming

```bash
python3 ai_engine.py --no-stream --text "hello"
```

### Query Providers (JSON)

```bash
python3 ai_engine.py --get-provider
```

Outputs a JSON array of all supported providers:

```json
[
  {
    "provider": "ollama_native",
    "litellm_name": "ollama_chat",
    "default_endpoint": "http://192.168.10.39:11434",
    "description": "Ollama Chat API  (/api/chat)",
    "example": "python3 ai_engine.py --provider ollama_native --model MODEL --endpoint http://192.168.10.39:11434"
  },
  ...
]
```

## Environment Variables

| Variable       | Overrides          | Default              |
|---------------|--------------------|-----------------------|
| `LLM_ENDPOINT` | `--endpoint`      | `http://192.168.10.39:11434` |
| `LLM_MODEL`    | `--model`         | `qwen3.5:27b`        |
| `LLM_PROVIDER` | `--provider`      | `ollama_native`      |
| `LLM_API_KEY`  | `--api-key`       | —                     |

Standard LiteLLM env vars are also supported (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Python Import API

You can import `ai_engine` as a module in other Python programs. LiteLLM is loaded lazily — only when `run_engine()` is first called.

```python
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from ai_engine import run_engine
from argparse import Namespace

messages = [
    "What is 1+1?",
    "What color is the sky?",
    "Say hello in Chinese.",
]

for i, text in enumerate(messages, 1):
    print(f"\n[{i}/{len(messages)}] Sending: {text}\n")

    args = Namespace(
        provider="ollama_native",
        model="qwen3.5:27b",
        endpoint="http://192.168.10.39:11434",
        api_key=None,
        prompt_file=None,
        prompt_text=None,
        file=None,
        text=text,
        no_stream=True,
        output_format="raw",
        get_provider=False,
        verbose=False,
        log=None,
    )

    start = time.perf_counter()
    run_engine(args)
    elapsed = time.perf_counter() - start
    print(f"\n[elapsed: {elapsed:.2f}s]")
```

First call includes LiteLLM import overhead; subsequent calls skip it because Python caches modules in `sys.modules`.

Full example: [example_import.py](example_import.py)

## Subprocess Long-Running Mode (`--stdin`)

Start a persistent `ai_engine.py` process that reads JSON requests from stdin, avoiding repeated LiteLLM import overhead:

```bash
# Start the engine in stdin mode
python3 ai_engine.py --stdin --output-format events

# Send requests via stdin (one JSON per line)
echo '{"text":"hello","no_stream":true,"output_format":"events"}' | python3 ai_engine.py --stdin --output-format events
```

Client example:

```python
import json
import subprocess
import sys

proc = subprocess.Popen(
    [sys.executable, "-u", "ai_engine.py", "--stdin", "--output-format", "events"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
)

request = {"text": "hello", "no_stream": True, "output_format": "events"}
proc.stdin.write(json.dumps(request) + "\n")
proc.stdin.flush()

for line in proc.stdout:
    event = json.loads(line)
    print(event)
    if event["type"] == "done":
        break
```

Full example: [example_subprocess.py](example_subprocess.py)

## Examples

```bash
# Analyze a Mermaid diagram with a local Ollama model
python3 ai_engine.py --prompt-file prompt.txt -f diagram.mermaid

# Use OpenAI with NDJSON events, piped to jq
python3 ai_engine.py \
  --provider openai --model gpt-4o \
  --output-format events --no-stream \
  --text "explain this code" | jq .

# Set credentials via environment
export LLM_API_KEY=sk-...
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
python3 ai_engine.py --text "hello"
```

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt)

## Architecture

```
                    ┌─────────────────────┐
  prompt ──┐        │                     │
           ├───────►│   ai_engine.py      │────► raw text (stdout)
  content ─┘        │                     │────► NDJSON events (stdout)
                    │  ┌───────────────┐  │
                    │  │  LiteLLM      │  │
                    │  │  Provider     │  │
                    │  │  Registry     │  │
                    │  └───────┬───────┘  │
                    └──────────┼──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   LLM Provider API  │
                    │  (local / cloud)    │
                    └─────────────────────┘
```

The engine is stateless — each invocation is a single request/response cycle with no persistent connections or session state.
