"""Benchmark runner: execute an experiment matrix of agentic_evolve and
OpenEvolve runs, interleaved and in parallel, recording provenance.

Usage:
    uv run scripts/bench.py MATRIX.yaml --llm LLM.yaml [--runs-dir DIR] [--parallel 4]
        [--only ENTRY ...] [--retry-failed] [--allow-dirty] [--dry-run]

The matrix is the protocol and lives in the repository; the llm file is the
deployment (model, endpoint, key variable), lives outside it, and is recorded
with every attempt. Runs land in --runs-dir, default
runs/<matrix stem>-<llm stem>.

Matrix file:

    repeats: 5
    entries:
      obp-full:
        system: agentic_evolve
        config: examples/online_bin_packing/config.yaml
        override: {evolution: {num_islands: 3, max_commits: 30}}
      obp-openevolve:
        system: openevolve
        task: examples/online_bin_packing
        iterations: 180

LLM file, one block per system (OpenEvolve has only an OpenAI-compatible
client):

    agentic_evolve: {model: deepseek-flash}   # our llm config block
    openevolve:
      model: deepseek-flash
      api_base: https://api.deepseek.com
      api_key_env: DEEPSEEK_API_KEY
      max_tokens: 384000
      max_tokens_field: max_tokens

The agentic_evolve block is merged into every override; the openevolve block
is passed to the OpenEvolve driver as --llm. Every (entry, repeat) pair gets the directory <runs_dir>/<entry>/r<repeat>/
holding the override or llm file, one child run directory per attempt (created by the launched program), attempt-<n>.log with the child's
stdout and stderr, and attempts.jsonl with one record per attempt: command,
git commit / tag / source tree hash, timestamps, exit code, run directory.
OpenEvolve's seed is the repeat number.

Pairs are launched in interleaved order (repeat 1 of every entry, then repeat
2, ...) through a pool of --parallel slots. An attempt is recorded when it
starts and updated when it ends (ok, failed, or interrupted; a record still
marked running belongs to a runner that died). Pairs with a successful attempt
are skipped; pairs whose attempts all ended otherwise are skipped unless
--retry-failed is given, so infrastructure failures are rerun deliberately
and every attempt stays on record. Ctrl-C terminates the children. The working
tree must be clean unless --allow-dirty is given.
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml


def git(*args: str) -> str | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def command(entry: dict, pair_dir: Path, repeat: int) -> list[str]:
    if entry["system"] == "agentic_evolve":
        cmd = ["uv", "run", "agentic-evolve", entry["config"]]
        cmd += ["--override", str(pair_dir / "override.yaml"), "--runs-dir", str(pair_dir)]
        return cmd
    if entry["system"] == "openevolve":
        cmd = ["uv", "run", "--group", "openevolve", "examples/openevolve/run.py"]
        cmd += ["--task", entry["task"], "--llm", str(pair_dir / "llm.yaml")]
        cmd += ["--iterations", str(entry["iterations"]), "--seed", str(repeat)]
        cmd += ["--runs-dir", str(pair_dir)]
        return cmd
    raise ValueError(f"unknown system {entry['system']!r}")


def write_inputs(entry: dict, pair_dir: Path, llm: dict) -> None:
    if entry["system"] == "agentic_evolve":
        override = {**entry.get("override", {}), "llm": llm["agentic_evolve"]}
        (pair_dir / "override.yaml").write_text(yaml.safe_dump(override, sort_keys=False))
    else:
        (pair_dir / "llm.yaml").write_text(yaml.safe_dump(llm["openevolve"], sort_keys=False))


def attempts(pair_dir: Path) -> list[dict]:
    path = pair_dir / "attempts.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_attempt(pair_dir: Path, record: dict) -> None:
    """Append the record, or replace the record with the same attempt number.
    Written through a temporary file so an interruption cannot truncate the history."""
    records = [r for r in attempts(pair_dir) if r["attempt"] != record["attempt"]] + [record]
    tmp = pair_dir / "attempts.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    tmp.replace(pair_dir / "attempts.jsonl")


async def launch(
    name: str, entry: dict, pair_dir: Path, repeat: int, llm: dict, provenance: dict
) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    write_inputs(entry, pair_dir, llm)
    number = len(attempts(pair_dir)) + 1
    cmd = command(entry, pair_dir, repeat)
    before = {p for p in pair_dir.iterdir() if p.is_dir()}
    started = datetime.now(UTC)
    # Recorded as running before the child starts: an interrupted runner
    # leaves the attempt on record and the next one takes a fresh number.
    record = {
        "attempt": number,
        "entry": name,
        "repeat": repeat,
        "command": cmd,
        **provenance,
        "started": started.isoformat(),
        "status": "running",
    }
    save_attempt(pair_dir, record)
    print(f"[{started:%H:%M:%S}] start {name} r{repeat} attempt {number}", flush=True)
    with (pair_dir / f"attempt-{number}.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(*cmd, stdout=log, stderr=log)
        try:
            code = await process.wait()
        except asyncio.CancelledError:
            process.terminate()
            code = await process.wait()
            record["status"] = "interrupted"
            raise
        finally:
            new_dirs = sorted(p for p in pair_dir.iterdir() if p.is_dir() and p not in before)
            if record["status"] == "running":
                record["status"] = "ok" if code == 0 else "failed"
            record |= {
                "finished": datetime.now(UTC).isoformat(),
                "exit_code": code,
                "run_dir": str(new_dirs[-1]) if new_dirs else None,
            }
            save_attempt(pair_dir, record)
            print(f"[{datetime.now(UTC):%H:%M:%S}] {record['status']} {name} r{repeat}", flush=True)


async def run_all(jobs: list, parallel: int) -> None:
    slots = asyncio.Semaphore(parallel)

    async def guarded(job):
        async with slots:
            await launch(*job)

    await asyncio.gather(*(guarded(job) for job in jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, help="default runs/<matrix stem>-<llm stem>")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--only", nargs="*", help="entry names to run; default all")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="list the pairs that would launch")
    args = parser.parse_args()

    dirty = git("status", "--porcelain")
    if dirty and not args.allow_dirty:
        sys.exit("working tree is not clean; commit first or pass --allow-dirty")
    llm = yaml.safe_load(args.llm.read_text())
    provenance = {
        "matrix": str(args.matrix),
        "llm": llm,
        "git_commit": git("rev-parse", "HEAD"),
        "git_tag": git("describe", "--tags", "--exact-match", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(dirty),
    }

    matrix = yaml.safe_load(args.matrix.read_text())
    runs_dir = args.runs_dir or Path("runs") / f"{args.matrix.stem}-{args.llm.stem}"
    entries = matrix["entries"]
    if args.only:
        entries = {name: entries[name] for name in args.only}

    jobs = []
    for repeat in range(1, matrix["repeats"] + 1):
        for name, entry in entries.items():
            pair_dir = runs_dir / name / f"r{repeat}"
            done = attempts(pair_dir)
            if any(a["status"] == "ok" for a in done):
                continue
            if done and not args.retry_failed:
                statuses = ", ".join(a["status"] for a in done)
                print(f"skip {name} r{repeat}: attempts {statuses}; use --retry-failed")
                continue
            jobs.append((name, entry, pair_dir, repeat, llm, provenance))

    print(
        f"{len(jobs)} run(s) to launch, {args.parallel} in parallel; tree {provenance['git_tree']}"
    )
    if args.dry_run:
        for name, entry, pair_dir, repeat, _, _ in jobs:
            print(f"  {name} r{repeat}: {' '.join(command(entry, pair_dir, repeat))}")
        return
    try:
        asyncio.run(run_all(jobs, args.parallel))
    except KeyboardInterrupt:
        sys.exit("interrupted; running attempts were terminated and recorded")


if __name__ == "__main__":
    main()
