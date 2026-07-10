#!/usr/bin/env python3
"""Example: importing ai-engine as a module, calling run_engine() multiple times."""

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
    print(f"\n{'='*50}")
    print(f"[{i}/{len(messages)}] Sending: {text}")
    print(f"{'='*50}\n")

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
    )

    start = time.perf_counter()
    run_engine(args)
    elapsed = time.perf_counter() - start
    print(f"\n[elapsed: {elapsed:.2f}s]")
