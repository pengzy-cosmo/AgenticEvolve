#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "mycode-sdk>=0.16.0", "pyyaml"]
# ///
"""Anytime curves for one or more runs.

Reads run directories of both systems: ours (archive.jsonl + sessions/) and
OpenEvolve comparison runs (evals/ + usage/ as written by
examples/openevolve/run.py). Two panels: best-so-far
fitness against the number of formal evaluator calls (the initial program is
evaluation 1), and against cumulative estimated LLM cost. Failed evaluations
advance the evaluation axis without changing best-so-far.

Runs record token usage only. Cost is estimated at plot time from mycode's
model catalog (its pricing and estimate_cost), identically for both systems,
over completed requests: requests that died without a response have no usage
and are excluded (their count is printed). A model without catalog pricing
has no cost curve.

Usage: uv run scripts/curves.py [LABEL=]RUN_DIR [[LABEL=]RUN_DIR ...] [-o curves.png]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from mycode.models import estimate_cost, infer_provider_from_model, lookup_model_metadata


@dataclass
class RunData:
    name: str
    fitness_key: str
    initial: dict  # metrics/timestamp of the run-start evaluation
    evaluations: list[dict]  # rows with status/metrics/timestamp, evaluation order
    requests: list[tuple[datetime, dict | None, str]]  # (timestamp, usage, model)
    provider: str | None
    behavior: bool  # rows carry our per-commit behavior fields


def load_run(run_dir: Path, label: str | None) -> RunData:
    name = label or run_dir.name
    if (run_dir / "evals").is_dir():
        return _load_openevolve_run(run_dir, name)
    return _load_agentic_run(run_dir, name)


def _load_agentic_run(run_dir: Path, name: str) -> RunData:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    initial = json.loads((run_dir / "initial.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "archive.jsonl").read_text().splitlines()]

    model = config["llm"]["model"]
    requests = []
    for messages in sorted(run_dir.glob("sessions/*/messages.jsonl")):
        for line in messages.read_text().splitlines():
            meta = json.loads(line).get("meta") or {}
            if meta.get("usage"):
                requests.append((datetime.fromisoformat(meta["timestamp"]), meta["usage"], model))
    requests.sort(key=lambda r: r[0])

    provider = config["llm"].get("provider") or infer_provider_from_model(model)
    return RunData(name, config["map_elites"]["fitness"], initial, rows, requests, provider, True)


def _load_openevolve_run(run_dir: Path, name: str) -> RunData:
    run = json.loads((run_dir / "run.json").read_text())

    evals: list[dict] = []
    for path in sorted((run_dir / "evals").glob("evals-*.jsonl")):
        evals.extend(json.loads(line) for line in path.read_text().splitlines())
    evals.sort(key=lambda r: r["timestamp"])
    if not evals:
        raise SystemExit(f"{run_dir}: no evaluation records")

    requests = []
    for path in sorted((run_dir / "usage").glob("usage-*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            # usage stays None for requests that died without a response.
            requests.append(
                (datetime.fromisoformat(record["timestamp"]), record.get("usage"), record["model"])
            )
    requests.sort(key=lambda r: r[0])

    # The controller evaluates the initial program before any iteration runs.
    provider = infer_provider_from_model(run["llm"]["model"])
    return RunData(name, run["fitness"], evals[0], evals[1:], requests, provider, False)


def best_so_far(run: RunData) -> list[float]:
    """Series over evaluator calls; the first entry is the initial program."""
    best = float(run.initial["metrics"][run.fitness_key])
    series = [best]
    for row in run.evaluations:
        if row["status"] == "ok":
            best = max(best, float(row["metrics"][run.fitness_key]))
        series.append(best)
    return series


def request_costs(run: RunData) -> list[tuple[datetime, float]] | None:
    """(timestamp, cost) of every completed request, or None when the model
    has no catalog pricing."""
    pricing: dict[str, dict | None] = {}
    costs = []
    for timestamp, usage, model in run.requests:
        if usage is None:
            continue
        if model not in pricing:
            meta = lookup_model_metadata(provider_type=run.provider, model=model)
            pricing[model] = meta.pricing if meta else None
        cost = estimate_cost(usage, pricing[model])
        if cost is None:
            return None
        costs.append((timestamp, cost["total"]))
    return costs


def summarize(run: RunData, costs: list[tuple[datetime, float]] | None) -> None:
    ok = [r for r in run.evaluations if r["status"] == "ok"]
    failed = [r for r in run.evaluations if r["status"] != "ok"]
    tokens: dict[str, int] = {}
    for _, usage, _ in run.requests:
        for key, value in (usage or {}).items():
            if isinstance(value, int):
                tokens[key] = tokens.get(key, 0) + value
    initial = float(run.initial["metrics"][run.fitness_key])
    no_usage = sum(usage is None for _, usage, _ in run.requests)
    cost = "no catalog pricing" if costs is None else f"est_cost=${sum(c for _, c in costs):.4f}"
    print(
        f"{run.name}: evaluations={1 + len(run.evaluations)} "
        f"(initial + {len(run.evaluations)}: ok={len(ok)} failed={len(failed)})"
    )
    print(f"  initial {run.fitness_key}={initial} -> best {best_so_far(run)[-1]}")
    print(f"  requests={len(run.requests)} (no response: {no_usage}) {cost}")
    print("  tokens: " + " ".join(f"{k}={v}" for k, v in sorted(tokens.items())))
    if run.behavior:
        print(f"  per-commit model_calls={[r['behavior']['model_calls'] for r in run.evaluations]}")
        print(
            f"  per-commit analyses={[r['behavior']['run_analysis_count'] for r in run.evaluations]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directory, optionally as LABEL=path")
    parser.add_argument("-o", "--output", type=Path, default=Path("curves.png"))
    args = parser.parse_args()

    fig, (by_eval, by_cost) = plt.subplots(1, 2, figsize=(12, 5))
    for arg in args.runs:
        label, sep, path = arg.rpartition("=")
        run = load_run(Path(path), label if sep else None)
        costs = request_costs(run)
        summarize(run, costs)
        series = best_so_far(run)
        by_eval.step(range(1, len(series) + 1), series, where="post", label=run.name)

        if costs:
            stamps = [datetime.fromisoformat(run.initial["timestamp"])] + [
                datetime.fromisoformat(r["timestamp"]) for r in run.evaluations
            ]
            x_cost = [sum(c for t, c in costs if t <= s) for s in stamps]
            by_cost.step(x_cost, series, where="post", label=run.name)

    by_eval.set_xlabel("formal evaluations")
    by_cost.set_xlabel("cumulative estimated LLM cost of completed requests (USD)")
    for ax in (by_eval, by_cost):
        ax.set_ylabel("best-so-far fitness")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
