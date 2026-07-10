# AI Engine — API 文档

## 概述

`ai_engine.py` 是 LLM 中间层引擎，上游 Agent/App 通过 CLI 参数指定 provider、model、endpoint、prompt 和请求内容，引擎调用 LiteLLM 适配不同 LLM 后端，并将响应以 **raw 文本**或 **NDJSON 事件流**两种格式输出。

### 架构角色

```
Agent / App
    │
    │  CLI args (provider, model, endpoint, prompt, content, …)
    ▼
┌─────────────────┐
│   ai_engine.py   │  ← 无状态中间层，不维护对话历史
│   (LiteLLM)      │
└────────┬────────┘
         │
         ▼
    LLM Backend (Ollama / OpenAI / Anthropic / Gemini / …)
         │
         ▼
    stdout (raw text 或 NDJSON events)
```

### 设计原则

- **无状态**: `ai_engine.py` 不维护 LLM state（不管理 conversation history）。每次调用只发送单轮 messages，streaming 输出完成后即退出。
- **单一输出通道**: 所有输出走 stdout。调用者通过 `--output-format` 切换 raw / events 格式。
- **诊断输出**: 诊断信息、警告默认走 stderr；通过 `--verbose` 可以输出详细的请求/响应日志，`--log <file>` 则重定向到文件。
- **错误输出**: 错误信息打印到 stderr；调用者可以 `2>/dev/null` 屏蔽。

---

## CLI 参数

### Provider / Model / Endpoint

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--provider` | str | `ollama_native` | LLM provider 名称。详见下方 Provider 表格。 |
| `-m` / `--model` | str | `qwen3.5:27b` | 模型名称。可通过 `LLM_MODEL` 环境变量覆盖。 |
| `-e` / `--endpoint` | str | `http://192.168.10.39:11434` | API 地址。可通过 `LLM_ENDPOINT` 环境变量覆盖。 |
| `--api-key` | str | — | API 密钥。也可用 `LLM_API_KEY` 环境变量或 LiteLLM 标准的 `OPENAI_API_KEY` 等。 |

### Prompt 输入（system prompt）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--prompt-file` | str | `prompt-fine.txt` | Prompt 文件路径。与 `--prompt-text` 二选一。 |
| `--prompt-text` | str | — | 内联 Prompt 文本。与 `--prompt-file` 二选一。 |

### Content 输入（user message）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-f` / `--file` | str | `saved.mermaid` | 请求内容文件路径（替代 v1 的 `--mermaid`）。与 `--text` 二选一。 |
| `--text` | str | — | 内联请求内容文本。与 `-f` / `--file` 二选一。 |

### 输出控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--no-stream` | bool | `False` | 关闭 streaming，等完整响应再输出。 |
| `--output-format` | str | `raw` | `raw` — 纯文本流（向后兼容）；`events` — NDJSON 事件流（程序化调用）。 |

### 调试日志

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--verbose` | bool | `False` | 打印详细日志到 stderr：发送给 LLM 的消息、LLM 返回的消息、stdin 输入、错误信息等。 |
| `--log` | str | — | 将所有 `--verbose` 信息写入指定文件（而非 stderr）。指定此参数相当于同时启用 `--verbose`。 |

### 其他

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--get-provider` | bool | `False` | 以 JSON 格式输出支持的 provider 列表后退出。不调用 LLM。 |
| `--stdin` | bool | `False` | 从 stdin 循环读取 JSON 行请求（每行一个 JSON），逐个调用 `run_engine()`。用于长驻子进程模式，避免每次请求重新加载 LiteLLM。 |

---

## Provider 参考

| CLI 名称 | LiteLLM 前缀 | 默认 Endpoint | 说明 |
|----------|-------------|---------------|------|
| `ollama_native` | `ollama_chat` | `http://192.168.10.39:11434` | Ollama Chat API (`/api/chat`)。**默认** |
| `ollama` | `ollama` | `http://192.168.10.39:11434` | Ollama Generate API (`/api/generate`) |
| `openai` | `openai` | `https://api.openai.com/v1` | OpenAI API |
| `anthropic` | `anthropic` | `https://api.anthropic.com` | Anthropic Claude API |
| `gemini` | `gemini` | `https://generativelanguage.googleapis.com/v1beta` | Google Gemini API |
| `deepseek` | `deepseek` | `https://api.deepseek.com/v1` | DeepSeek API |
| `groq` | `groq` | `https://api.groq.com/openai/v1` | Groq Cloud API（OpenAI 兼容） |
| `together` | `together` | `https://api.together.xyz/v1` | Together AI API（OpenAI 兼容） |
| `mistral` | `mistral` | `https://api.mistral.ai/v1` | Mistral AI API |
| `custom_openai` | `openai` | `http://192.168.10.39:11434/v1` | 自定义 OpenAI 兼容 API（v1 路径） |

LiteLLM 最终构造的模型全名为 `{litellm_prefix}/{model}`，例如 `ollama_chat/qwen3.5:27b`、`openai/gpt-4o`。

---

## 环境变量

| 变量 | 覆盖参数 | 说明 |
|------|---------|------|
| `LLM_ENDPOINT` | `--endpoint` | API 地址 |
| `LLM_MODEL` | `--model` | 模型名称 |
| `LLM_PROVIDER` | `--provider` | Provider 名称 |
| `LLM_API_KEY` | `--api-key` | API 密钥 |

LiteLLM 也识别标准环境变量如 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 等。

---

## 输出格式

### `raw` 模式（默认）

Streaming 时：直接将 LLM 返回的 thinking 和 assistant 文本输出到 stdout，无 JSON 包装。

```
# thinking 内容直接输出（如有）
我先分析一下这个网络拓扑…

# assistant 内容接着输出
根据您的网络拓扑…
```

非 Streaming 时：一个请求一个响应，输出完整文本后换行。

### `events` 模式

每一行都是一个独立的 JSON 对象（NDJSON），包含事件类型和相关数据。调用方逐行读取并根据 `type` 字段分发处理。

```
{"type":"thinking_delta","delta":"我"}
{"type":"thinking_delta","delta":"先"}
{"type":"thinking_delta","delta":"分析"}
{"type":"thinking_end"}
{"type":"assistant_delta","delta":"今天"}
{"type":"assistant_delta","delta":"天气"}
{"type":"assistant_delta","delta":"不错。"}
{"type":"assistant_end"}
{"type":"usage","prompt_tokens":120,"completion_tokens":500,"reasoning_tokens":300}
{"type":"done","finish_reason":"stop"}
```

---

## 事件规范

### 事件时序

#### `--no-stream`（非 Streaming）

所有事件在完整响应到达后一次性输出。`thinking` 和 `assistant` 都一次性携带完整内容。

```
thinking? (with full content)
    ↓
thinking_end
    ↓
(tool_call_begin → tool_call_end)?   ← per tool call; tool_call_end 含 arguments
    ↓
assistant (with full content)
    ↓
assistant_end
    ↓
usage?
    ↓
done
```

`?` 表示该事件可选（某些模型/场景不会产生）。

#### Streaming（默认）

事件随着 LLM 的流式响应逐块输出。

```
(thinking_delta* → thinking_end)?    ← reasoning_content fragments
    ↓
(tool_call_begin → tool_call_delta* → tool_call_end)?  ← tool_calls
    ↓
(assistant_delta* → assistant_end)?  ← content fragments
    ↓
usage?
    ↓
done
```

`?` 表示该事件可选。调用者自行累积 `thinking_delta`、`assistant_delta` 的 `delta` 字段得到完整内容。

### 事件类型

#### `thinking` (Non-streaming 专用)

LLM 的完整推理/思考内容。

```json
{
  "type": "thinking",
  "content": "我先分析一下这个网络拓扑的架构..."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `thinking` |
| `content` | string | 完整推理内容 |

---

#### `thinking_delta` (Streaming 专用)

推理内容的流式片段。

```json
{
  "type": "thinking_delta",
  "delta": "我"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `thinking_delta` |
| `delta` | string | 本次流式片段 |

---

#### `thinking_end`

推理内容结束标记。

```json
{
  "type": "thinking_end"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `thinking_end` |

---

#### `tool_call_begin`

LLM 发起一个工具调用。

```json
{
  "type": "tool_call_begin",
  "id": "call_xxx",
  "name": "search"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `tool_call_begin` |
| `id` | string | 工具调用唯一 ID |
| `name` | string | 工具名称 |

---

#### `tool_call_delta` (Streaming 专用)

工具调用参数（JSON 字符串）的流式片段。

```json
{
  "type": "tool_call_delta",
  "id": "call_xxx",
  "delta": "{\"query\":\""
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `tool_call_delta` |
| `id` | string | 工具调用 ID |
| `delta` | string | 参数 JSON 文本的片段 |
| `index` | number | Streaming 中同批 tool call 的顺序编号（可选） |

---

#### `tool_call_end`

工具调用结束标记。Streaming 模式下调用者自行累积所有 `tool_call_delta` 的 `delta` 得到完整参数；`--no-stream` 模式下 `arguments` 包含完整解析后的参数。

```json
// Streaming — 纯标记
{
  "type": "tool_call_end",
  "id": "call_xxx"
}

// --no-stream — 携带完整参数
{
  "type": "tool_call_end",
  "id": "call_xxx",
  "arguments": { "query": "天气" }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `tool_call_end` |
| `id` | string | 工具调用 ID |
| `arguments` | object \| string (可选) | `--no-stream` 模式下存在，为完整参数对象 |

---

#### `assistant` (Non-streaming 专用)

LLM 的完整回复文本。

```json
{
  "type": "assistant",
  "content": "今天天气不错。"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `assistant` |
| `content` | string | 完整回复文本 |

---

#### `assistant_delta` (Streaming 专用)

LLM 回复文本的流式片段。

```json
{
  "type": "assistant_delta",
  "delta": "今天"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `assistant_delta` |
| `delta` | string | 本次流式片段 |

---

#### `assistant_end`

LLM 回复结束标记。调用者自行累积所有 `assistant_delta` 的 `delta` 字段得到完整文本。

```json
{
  "type": "assistant_end"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `assistant_end` |

---

#### `usage`

Token 用量信息（部分模型不提供，此时不会产生该事件）。

```json
{
  "type": "usage",
  "prompt_tokens": 120,
  "completion_tokens": 500,
  "reasoning_tokens": 300
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `usage` |
| `prompt_tokens` | int | 输入 token 数 |
| `completion_tokens` | int | 输出 token 数 |
| `reasoning_tokens` | int | 推理 token 数（仅部分模型支持，否则为 0） |

---

#### `done`

响应结束。**所有场景下都会产生**，表示 HTTP Response / SSE 流 / WebSocket 已关闭。

```json
{
  "type": "done",
  "finish_reason": "stop"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `done` |
| `finish_reason` | string | 结束原因。可能值：`stop`、`length`、`tool_calls`、`error` 等。 |

若请求过程中发生异常，`finish_reason` 为 `"error"`，且错误信息会打印到 stderr（`events` 模式下 `done` 事件仍会输出）：

```json
{"type":"done","finish_reason":"error"}
```

---

## 示例

### 1. 基础用法（raw 模式，默认）

```bash
python3 ai_engine.py
```

自动读取 `prompt-fine.txt` 作为 prompt、`saved.mermaid` 作为请求内容，调用默认 Ollama 的 `qwen3.5:27b`。

### 2. 指定文件

```bash
python3 ai_engine.py --prompt-file my-prompt.txt -f my-data.txt
```

### 3. 内联输入

```bash
python3 ai_engine.py --prompt-text "你是一名网络专家" --text "分析这段日志..."
```

### 4. 切换 provider

```bash
python3 ai_engine.py \
    --provider openai \
    --model gpt-4o \
    --endpoint https://api.openai.com/v1 \
    --api-key sk-xxx \
    --prompt-text "Translate to French" \
    --text "Hello world"
```

```bash
python3 ai_engine.py \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --endpoint https://api.anthropic.com \
    --api-key sk-ant-xxx \
    --text "Summarize this article"
```

### 5. 非 Streaming + Events 输出（程序化调用）

```bash
python3 ai_engine.py \
    --no-stream \
    --output-format events \
    --prompt-text "You are a helpful assistant" \
    --text "What is 2+2?"
```

期望输出（一次响应，所有事件在一个批次中输出）：

```json
{"type":"assistant","content":"2 + 2 = 4"}
{"type":"assistant_end"}
{"type":"usage","prompt_tokens":20,"completion_tokens":5,"reasoning_tokens":0}
{"type":"done","finish_reason":"stop"}
```

### 6. Streaming + Events 输出

```bash
python3 ai_engine.py \
    --output-format events \
    --text "Explain quantum computing" \
    --provider ollama_native \
    --model qwen3.5:27b
```

期望输出（流式，每行独立输出）：

```json
{"type":"thinking_delta","delta":"量子"}
{"type":"thinking_delta","delta":"计算"}
{"type":"thinking_end"}
{"type":"assistant_delta","delta":"量子"}
{"type":"assistant_delta","delta":"计算是一种"}
{"type":"assistant_end"}
{"type":"usage","prompt_tokens":40,"completion_tokens":100,"reasoning_tokens":30}
{"type":"done","finish_reason":"stop"}
```

### 7. Python Import API

将 `ai_engine` 作为模块直接 import，无需 subprocess。LiteLLM 延迟加载——仅在首次调用 `run_engine()` 时导入。

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

首次调用包含 LiteLLM import 开销；后续调用跳过加载（Python 在 `sys.modules` 中缓存模块）。

完整示例：[example_import.py](example_import.py)

### 8. 进程式消费（Python subprocess）

```python
import json
import subprocess
import sys

proc = subprocess.Popen(
    [
        sys.executable, "ai_engine.py",
        "--output-format", "events",
        "--no-stream",
        "--prompt-text", "You are a poet",
        "--text", "Write a haiku about Python",
    ],
    stdout=subprocess.PIPE,
    text=True,
)

final_content = ""
for line in proc.stdout:
    event = json.loads(line)
    if event["type"] == "assistant":
        final_content = event["content"]
    elif event["type"] == "usage":
        print(f"Tokens: {event}", file=sys.stderr)
    elif event["type"] == "done":
        break

print(f"Final result: {final_content}")
```

### 9. Subprocess 长驻内存模式（`--stdin`）

启动常驻子进程，通过 stdin 逐行接收 JSON 请求，避免每次调用重新加载 LiteLLM：

```bash
echo '{"text":"hello","no_stream":true,"output_format":"events"}' \
  | python3 ai_engine.py --stdin --output-format events
```

Python 客户端示例：

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

完整示例：[example_subprocess.py](example_subprocess.py)

### 10. bash 逐行处理 events

```bash
python3 ai_engine.py --output-format events --no-stream \
    --text "hello" | while read -r line; do
    type=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['type'])")
    case "$type" in
        assistant_delta)
            echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('delta',''), end='')"
            ;;
        done)
            echo ""
            echo "--- DONE: $(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('finish_reason',''))") ---"
            ;;
    esac
done
```

### 11. 使用环境变量

```bash
export LLM_ENDPOINT=https://api.openai.com/v1
export LLM_MODEL=gpt-4o
export LLM_API_KEY=sk-xxx
python3 ai_engine.py --text "hello"
```

```bash
# 混合：参数优先级高于环境变量
LLM_ENDPOINT=http://localhost:11434 python3 ai_engine.py --model llama3.2 --text "hi"
```

### 12. 自定义 OpenAI 兼容 API

```bash
python3 ai_engine.py \
    --provider custom_openai \
    --endpoint https://my-proxy.example.com/v1 \
    --api-key sk-my-key \
    --model my-custom-model \
    --text "test"
```

### 13. 调试日志

```bash
# 查看发送给 LLM 的请求和返回的响应（输出到 stderr）
python3 ai_engine.py --verbose --text "hello"

# 将日志保存到文件（便于排查问题）
python3 ai_engine.py --log /tmp/ai-engine.log --text "hello"

# 在 --stdin 长驻模式下追踪每条请求
python3 ai_engine.py --stdin --output-format events --log /tmp/ai-engine.log
```

Verbose 日志包含以下标记：
- `[SEND]` — 发送给 LLM 的模型名、API 地址和完整 messages 负载
- `[RECV]` — 从 LLM 收到的每个流式块（thinking、assistant 内容、tool_call）
- `[STDIN]` — 从 stdin 读到的每行 JSON 请求（仅 `--stdin` 模式）
- `[ERROR]` — 异常消息和 JSON 解析错误

### 14. 查询支持的 Provider（JSON 输出）

```bash
python3 ai_engine.py --get-provider
```

输出示例：

```json
[
  {
    "provider": "ollama_native",
    "litellm_name": "ollama_chat",
    "default_endpoint": "http://192.168.10.39:11434",
    "description": "Ollama Chat API  (/api/chat)",
    "example": "python3 ai_engine.py --provider ollama_native --model MODEL --endpoint http://192.168.10.39:11434"
  },
  {
    "provider": "openai",
    "litellm_name": "openai",
    "default_endpoint": "https://api.openai.com/v1",
    "description": "OpenAI API",
    "example": "python3 ai_engine.py --provider openai --model MODEL --endpoint https://api.openai.com/v1"
  },
  ...
]
```

该命令不调用 LLM，仅返回静态 provider 列表后退出（exit code 0）。上层 Agent 可通过解析此 JSON 自动发现可用 provider 和 endpoint。

```bash
# 程序化获取 provider 列表
providers=$(python3 ai_engine.py --get-provider)
default_endpoint=$(echo "$providers" | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['default_endpoint'])")
```

---

## 与 v1 (`verify-ai-analysis.py`) 的差异

| 项目 | v1 (`verify-ai-analysis.py`) | v2 (`ai_engine.py`) |
|------|------------------------------|---------------------|
| LLM 适配 | 硬编码 Ollama HTTP API | LiteLLM 适配任意 provider |
| `--endpoint` | 默认 `http://192.168.10.39:11434/api/chat` | 默认 `http://192.168.10.39:11434`（不含路径） |
| `-f` 参数 | `--mermaid` 读取 Mermaid 文件 | `--file` 读取任意请求内容文件 |
| `--text` | 不支持 | 新增，内联请求内容 |
| `-p` / `--prompt` | 单个参数 | 拆分为 `--prompt-file` + `--prompt-text` |
| 输出格式 | 固定 raw 文本 + THINKING 动画 | `raw`（默认）或 `events`（NDJSON 事件流） |
| 返回路径 | HTTP Streaming 直接输出 | LiteLLM Streaming / Non-streaming 统一输出 |
| Smart 状态 | ✓ 保留流式拼接 | ✓ 保留，另外增加 events 格式 |
| 无状态 | ✗ 隐含单次调用状态 | ✓ 明确无状态 |

---

## 常见问题

### Q: 为什么 events 模式下最后一行的 `done` 可能在 `error` 后？

当请求失败时，引擎先输出 `done` 事件（`finish_reason: "error"`），再将错误信息打印到 stderr，然后以 exit code 1 退出。所以调用方永远会收到 `done` 事件。

### Q: 如何判断响应对应哪个请求？

`ai_engine.py` 是单请求-单响应的。每个进程实例处理一个请求并退出。如果需要并发，上层启动多个子进程即可。

### Q: 部分模型的 `reasoning_tokens` 始终为 0？

只有提供 reasoning token 计数的模型（如 DeepSeek-R1、某些 Ollama 模型）才会返回非零值。OpenAI `o1`/`o3` 系列也会在 `usage` 中提供 `completion_tokens_details.reasoning_tokens`，LiteLLM 会映射到 `usage.reasoning_tokens`。

### Q: 为什么 no-stream 和 streaming 的 assistant 事件不同？

为了与 `thinking`/`thinking_delta` 命名对称。`--no-stream` 用 `assistant`（完整 content），streaming 用 `assistant_delta`（片段），调用者按各自场景处理即可。
