"""LLM client for OpenEvolve runs.

make_client is injected through OpenEvolve's init_client hook and returns an
OpenAILLM subclass with two changes. The output limit is sent under the field
named by $EVOLVE_MAX_TOKENS_FIELD: OpenEvolve always sends max_tokens, but
the field that caps a whole response including reasoning differs by API
(DeepSeek: max_tokens; DashScope: max_completion_tokens). Each API response's
usage is recorded (raw, plus normalized to mycode's canonical fields) before
the response text enters diff parsing, so requests whose diff later fails to
parse are still counted. Requests that die without a response are logged with
usage null; usage the upstream billed but never returned cannot be recovered.
Records go to a per-process usage-<pid>.jsonl in $EVOLVE_USAGE_LOG_DIR.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import openai
from openevolve.llm.openai import OpenAILLM


def make_client(model_cfg):
    return Client(model_cfg)


class Client(OpenAILLM):
    async def _call_api(self, params):
        # OpenEvolve reuses params across retries, so rename on a copy.
        params = params.copy()
        params[os.environ["EVOLVE_MAX_TOKENS_FIELD"]] = params.pop("max_tokens")
        # Async client: when OpenEvolve's outer timeout cancels the wait, the
        # request is closed instead of living on in a thread. Retries are left
        # to OpenEvolve's own loop, so the client must not retry as well. It is
        # created per call because OpenEvolve runs every iteration in a fresh
        # asyncio.run() loop and a pooled connection cannot outlive its loop.
        client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.api_base, timeout=self.timeout, max_retries=0
        )
        try:
            async with client:
                response = await client.chat.completions.create(**params)
        # BaseException: cancellation by the outer timeout is not an Exception.
        # Usage the upstream billed for a cancelled request cannot be recovered.
        except BaseException as exc:
            _log(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "model": params.get("model"),
                    "error": repr(exc),
                    "usage": None,
                }
            )
            raise
        raw = response.usage.model_dump() if response.usage else None
        _log(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "model": params.get("model"),
                "response_id": getattr(response, "id", None),
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
                "raw_usage": raw,
                "usage": _normalize(raw),
            }
        )
        return response.choices[0].message.content


def _normalize(raw: dict | None) -> dict | None:
    """Mycode's canonical usage fields; missing values stay missing."""
    if not raw:
        return None
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cache_read = prompt_details.get("cached_tokens")
    if cache_read is None:
        # DeepSeek reports cache hits as a top-level extension field.
        cache_read = raw.get("prompt_cache_hit_tokens")
    usage = {
        "total_tokens": raw.get("total_tokens"),
        "input_tokens": raw.get("prompt_tokens"),
        "output_tokens": raw.get("completion_tokens"),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": prompt_details.get("cache_write_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
    }
    return {key: value for key, value in usage.items() if value is not None}


def _log(record: dict) -> None:
    log_dir = os.environ.get("EVOLVE_USAGE_LOG_DIR")
    if log_dir:
        with (Path(log_dir) / f"usage-{os.getpid()}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
