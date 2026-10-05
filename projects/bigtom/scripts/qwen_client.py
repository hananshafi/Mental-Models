"""
OpenAI-compatible async client for the NRP Qwen endpoint.

Used by all annotation / generation scripts in this pipeline.

Usage:
    from qwen_client import QwenClient, parallel_chat

    client = QwenClient()
    answer = await client.chat_async([{"role": "user", "content": "hi"}])

    results = parallel_chat(
        client,
        prompts=[[{"role": "user", "content": p}] for p in my_prompts],
        max_workers=32,
    )
"""
import os
import asyncio
import time
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI, OpenAI


NRP_BASE_URL = "https://ellm.nrp-nautilus.io/v1"
NRP_API_KEY = os.environ.get("QWEN_API_KEY", "x8JuCE7Jy3IBzT6vmOOCSj4QR1Zr5shs")
DEFAULT_MODEL = "qwen3"  # the 397B-A17B model in the NRP registry


class QwenClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = NRP_BASE_URL,
                 api_key: str = NRP_API_KEY, timeout: float = 300.0,
                 max_retries: int = 5, enable_thinking: bool = False):
        self.model = model
        self.sync = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.async_ = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        # Qwen3 is a thinking model by default — we disable reasoning tokens so
        # the response budget is spent on real content, not on chain-of-thought.
        self.default_extra_body = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    def _merge_extra(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        extra = dict(self.default_extra_body)
        if "extra_body" in kwargs and kwargs["extra_body"]:
            extra.update(kwargs["extra_body"])
        kwargs["extra_body"] = extra
        return kwargs

    # ── sync ───────────────────────────────────────────────────────────────
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        kwargs = self._merge_extra(kwargs)
        for attempt in range(self.max_retries):
            try:
                resp = self.sync.chat.completions.create(
                    model=self.model, messages=messages, **kwargs,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = min(2 ** attempt, 30)
                print(f"  [retry {attempt+1}/{self.max_retries}] {type(e).__name__}: {e} - sleeping {wait}s")
                time.sleep(wait)

    # ── async ──────────────────────────────────────────────────────────────
    async def chat_async(self, messages: List[Dict[str, str]], **kwargs) -> str:
        kwargs = self._merge_extra(kwargs)
        for attempt in range(self.max_retries):
            try:
                resp = await self.async_.chat.completions.create(
                    model=self.model, messages=messages, **kwargs,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return ""  # don't crash whole batch on one bad row
                wait = min(2 ** attempt, 30)
                await asyncio.sleep(wait)
        return ""


async def _gather_with_sem(sem, client, messages, **kwargs):
    async with sem:
        return await client.chat_async(messages, **kwargs)


def parallel_chat(client: QwenClient, prompts: List[List[Dict[str, str]]],
                  max_workers: int = 32, **kwargs) -> List[str]:
    """Fire many chat requests concurrently and return results in order."""
    async def _run():
        sem = asyncio.Semaphore(max_workers)
        tasks = [_gather_with_sem(sem, client, p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)

    return asyncio.run(_run())


if __name__ == "__main__":
    # Smoke test
    client = QwenClient()
    out = client.chat([{"role": "user", "content": "Say exactly: OK"}],
                      temperature=0.0, max_tokens=8)
    print(f"Sync test: {out!r}")

    outs = parallel_chat(
        client,
        prompts=[[{"role": "user", "content": f"Say the number {i}"}] for i in range(5)],
        max_workers=5, temperature=0.0, max_tokens=16,
    )
    for i, o in enumerate(outs):
        print(f"  Async [{i}]: {o!r}")
