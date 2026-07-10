#!/usr/bin/env python3
"""
Example: subprocess 长驻内存方式调用 ai_engine.py

启动一个常驻 Python 进程，通过 stdin/stdout 通信，
避免每次请求都重新加载 LiteLLM。
"""

import json
import subprocess
import sys
import os

ENGINE = os.path.join(os.path.dirname(__file__), "ai_engine.py")


class AIEngineClient:
    """常驻子进程客户端，通过 JSON 行协议通信。"""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-u", ENGINE, "--stdin", "--output-format", "events"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def ask(self, text: str, *, model: str = "qwen3.5:27b",
            prompt_text: str | None = None) -> dict:
        request = {
            "text": text,
            "model": model,
            "no_stream": True,
            "output_format": "events",
        }
        if prompt_text:
            request["prompt_text"] = prompt_text

        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()

        result = {"content": "", "usage": None, "finish_reason": None}
        for line in self.proc.stdout:
            event = json.loads(line)
            if event["type"] == "assistant":
                result["content"] = event["content"]
            elif event["type"] == "usage":
                result["usage"] = event
            elif event["type"] == "done":
                result["finish_reason"] = event.get("finish_reason")
                break
        return result

    def get_providers(self) -> list:
        self.proc.stdin.write(json.dumps({"get_provider": True}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


if __name__ == "__main__":
    client = AIEngineClient()

    print("=== Supported Providers ===")
    providers = client.get_providers()
    for p in providers:
        print(f"  {p['provider']:20s}  {p['description']}")

    questions = [
        "1+1=?",
        "天空是什么颜色？",
        "用中文说你好",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Q: {q}")
        result = client.ask(q)
        print(f"A: {result['content']}")
        if result["usage"]:
            print(f"Tokens: {result['usage']}")

    client.close()
