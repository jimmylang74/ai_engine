# ai-engine.py — LLM Middleware Engine

Stateless CLI middleware that unifies LLM providers via [LiteLLM](https://github.com/BerriAI/litellm). Supports streaming and batch responses, outputting structured NDJSON events or raw text.

## Quick Start

```bash
pip install -r requirements.txt

# Streaming raw text (default)
python3 ai-engine.py --prompt-text "You are an expert" --text "analyze this"

# NDJSON event stream (programmatic consumption)
python3 ai-engine.py --output-format events --text "hello"

# Non-streaming (wait for full response)
python3 ai-engine.py --no-stream --text "summarize this"
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
python3 ai-engine.py --provider openai --model gpt-4o --endpoint https://api.openai.com/v1
python3 ai-engine.py --provider anthropic --model claude-sonnet-4-20250514 --api-key sk-...
python3 ai-engine.py --provider ollama_native --model qwen3.5:27b
```

### Input: System Prompt

```bash
python3 ai-engine.py --prompt-file prompt.txt --text "user message"
python3 ai-engine.py --prompt-text "You are a helpful assistant" --text "hello"
```

If neither `--prompt-file` nor `--prompt-text` is given, the engine falls back to `prompt-fine.txt` in the script directory (if it exists).

### Input: User Content

```bash
python3 ai-engine.py -f data.mermaid          # read content from file
python3 ai-engine.py --text "inline text"      # inline content
python3 ai-engine.py                           # falls back to saved.mermaid (if it exists)
```

### Output Formats

**Raw** (default) — plain streaming text, backward compatible:

```bash
python3 ai-engine.py --output-format raw --text "hello"
```

**Events** — NDJSON event stream, one JSON object per line:

```bash
python3 ai-engine.py --output-format events --text "hello"
```

### Provider Discovery

```bash
# List all supported providers as JSON
python3 ai-engine.py --get-provider | jq .
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
python3 ai-engine.py --no-stream --text "hello"
```

### Query Providers (JSON)

```bash
python3 ai-engine.py --get-provider
```

Outputs a JSON array of all supported providers:

```json
[
  {
    "provider": "ollama_native",
    "litellm_name": "ollama_chat",
    "default_endpoint": "http://192.168.10.39:11434",
    "description": "Ollama Chat API  (/api/chat)",
    "example": "python3 ai-engine.py --provider ollama_native --model MODEL --endpoint http://192.168.10.39:11434"
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

## Examples

```bash
# Analyze a Mermaid diagram with a local Ollama model
python3 ai-engine.py --prompt-file prompt.txt -f diagram.mermaid

# Use OpenAI with NDJSON events, piped to jq
python3 ai-engine.py \
  --provider openai --model gpt-4o \
  --output-format events --no-stream \
  --text "explain this code" | jq .

# Set credentials via environment
export LLM_API_KEY=sk-...
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
python3 ai-engine.py --text "hello"
```

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt)

## Architecture

```
                    ┌─────────────────────┐
  prompt ──┐        │                     │
           ├───────►│   ai-engine.py      │────► raw text (stdout)
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
