"""Usage records for the baselines' OpenAI-compatible model clients (LLM-SR,
SR-Scientist), one JSON line per request in usage.jsonl.

A record has the timestamp and model, then either error, or response_id,
finish_reason, raw_usage as the API returned it and usage in mycode's field
names (total, input, output, cache_read, cache_write and reasoning tokens);
fields the API did not return stay missing.
"""

import json
from datetime import UTC, datetime
from pathlib import Path


def record(path: Path, model: str, response=None, error: str | None = None) -> None:
    entry = {"timestamp": datetime.now(UTC).isoformat(), "model": model}
    if response is None:
        entry |= {"error": error, "usage": None}
    else:
        choice = response.choices[0] if response.choices else None
        raw = response.usage.model_dump() if response.usage else None
        entry |= {
            "response_id": response.id,
            "finish_reason": choice.finish_reason if choice else None,
            "raw_usage": raw,
            "usage": _usage(raw) if raw else None,
        }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _usage(raw: dict) -> dict:
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cache_read = prompt_details.get("cached_tokens")
    if cache_read is None:
        cache_read = raw.get("prompt_cache_hit_tokens")  # DeepSeek's extension field
    usage = {
        "total_tokens": raw.get("total_tokens"),
        "input_tokens": raw.get("prompt_tokens"),
        "output_tokens": raw.get("completion_tokens"),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": prompt_details.get("cache_write_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
    }
    return {key: value for key, value in usage.items() if value is not None}
