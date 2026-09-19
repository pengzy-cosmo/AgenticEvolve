"""Benchmark runner: execute an experiment matrix of agentic_evolve, OpenEvolve
and symbolic-regression baseline runs, interleaved and in parallel, recording
provenance.

Usage:
    uv run scripts/bench.py MATRIX.yaml --llm LLM.yaml [--runs-dir DIR] [--parallel 4]
        [--llmsr-repo DIR] [--srscientist-repo DIR]
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

A matrix with a `problems` list is expanded before anything else: every entry
becomes one entry per problem, named <entry>-<problem>, with {problem}
substituted in its string values and the problem recorded; the expanded
entries are ordered by problem, so a launch interleaves the methods. The
systems for the LLM-SRBench problems (examples/llm_srbench) are llmsr (with
`samples`), srscientist and pysr; the first two need the repositories their
runners use, given by --llmsr-repo and --srscientist-repo:

    repeats: 1
    problems: [CRK21, BPG18]
    entries:
      full:
        system: agentic_evolve
        config: examples/llm_srbench/tasks/{problem}/config.yaml
      llmsr: {system: llmsr, samples: 1000}
      srscientist: {system: srscientist}
      pysr: {system: pysr}

LLM file, one block per system in the matrix that takes a model (pysr takes
none). The agentic_evolve block is our llm config block, merged into every
override; the others are OpenAI-compatible client settings passed to the
system's runner as --llm:

    agentic_evolve: {model: deepseek-flash}
    openevolve: &client
      model: deepseek-flash
      api_base: https://api.deepseek.com
      api_key_env: DEEPSEEK_API_KEY
      max_tokens: 384000
      max_tokens_field: max_tokens
    llmsr: *client
    srscientist: *client

Every (entry, repeat) pair gets the directory <runs_dir>/<entry>/r<repeat>/
holding the override or llm file, one child run directory per attempt
(created by the launched program), attempt-<n>.log with the child's stdout
and stderr, and attempts.jsonl with one record per attempt: command, git
commit / tag / source tree hash, timestamps, exit code, run directory. The
seed of every system but ours is the repeat number.

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

SRBENCH = "examples/llm_srbench"


def git(*args: str) -> str | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def expand(matrix: dict) -> dict:
    """The entries, one per (entry, problem) when the matrix lists problems."""
    if "problems" not in matrix:
        return matrix["entries"]
    return {
        f"{name}-{problem}": {
            key: value.format(problem=problem) if isinstance(value, str) else value
            for key, value in entry.items()
        }
        | {"problem": problem}
        for problem in matrix["problems"]
        for name, entry in matrix["entries"].items()
    }


def command(entry: dict, pair_dir: Path, repeat: int, repos: dict) -> list[str]:
    system = entry["system"]
    if system == "agentic_evolve":
        cmd = ["uv", "run", "agentic-evolve", entry["config"]]
        cmd += ["--override", str(pair_dir / "override.yaml"), "--runs-dir", str(pair_dir)]
        return cmd
    if system == "openevolve":
        cmd = ["uv", "run", "--group", "openevolve", "examples/openevolve/run.py"]
        cmd += ["--task", entry["task"], "--llm", str(pair_dir / "llm.yaml")]
        cmd += ["--iterations", str(entry["iterations"]), "--seed", str(repeat)]
        cmd += ["--runs-dir", str(pair_dir)]
        return cmd
    if system in ("llmsr", "srscientist", "pysr"):
        cmd = ["uv", "run", f"{SRBENCH}/run_{system}.py", "--problem", entry["problem"]]
        cmd += ["--seed", str(repeat), "--runs-dir", str(pair_dir)]
        if system != "pysr":
            cmd += ["--llm", str(pair_dir / "llm.yaml"), f"--{system}-repo", str(repos[system])]
        if system == "llmsr":
            cmd += ["--samples", str(entry["samples"])]
        return cmd
    raise ValueError(f"unknown system {system!r}")


def write_inputs(entry: dict, pair_dir: Path, llm: dict) -> None:
    system = entry["system"]
    if system == "agentic_evolve":
        override = {**entry.get("override", {}), "llm": llm["agentic_evolve"]}
        (pair_dir / "override.yaml").write_text(yaml.safe_dump(override, sort_keys=False))
    elif system != "pysr":
        (pair_dir / "llm.yaml").write_text(yaml.safe_dump(llm[system], sort_keys=False))


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
    name: str, entry: dict, pair_dir: Path, repeat: int, llm: dict, repos: dict, provenance: dict
) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    write_inputs(entry, pair_dir, llm)
    number = len(attempts(pair_dir)) + 1
    cmd = command(entry, pair_dir, repeat, repos)
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
    parser.add_argument("--llmsr-repo", type=Path, help="LLM-SRBench clone, for llmsr entries")
    parser.add_argument(
        "--srscientist-repo", type=Path, help="SR-Scientist clone, for srscientist entries"
    )
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
    entries = expand(matrix)
    if args.only:
        entries = {name: entries[name] for name in args.only}
    systems = {e["system"] for e in entries.values()}
    if missing := systems - {"pysr"} - llm.keys():
        sys.exit(f"llm file lacks the block(s) {', '.join(sorted(missing))}")
    repos = {"llmsr": args.llmsr_repo, "srscientist": args.srscientist_repo}
    for system, repo in repos.items():
        if repo is None and system in systems:
            sys.exit(f"{system} entries need --{system}-repo")

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
            jobs.append((name, entry, pair_dir, repeat, llm, repos, provenance))

    print(
        f"{len(jobs)} run(s) to launch, {args.parallel} in parallel; tree {provenance['git_tree']}"
    )
    if args.dry_run:
        for name, entry, pair_dir, repeat, _, _, _ in jobs:
            print(f"  {name} r{repeat}: {' '.join(command(entry, pair_dir, repeat, repos))}")
        return
    try:
        asyncio.run(run_all(jobs, args.parallel))
    except KeyboardInterrupt:
        sys.exit("interrupted; running attempts were terminated and recorded")


if __name__ == "__main__":
    main()
