#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "huggingface_hub", "numpy", "openai", "pandas", "pyarrow", "scipy", "sympy", "tqdm"]
# ///
"""Run the SR-Scientist baseline on one LSR-Synth problem, with our LLM deployment.

Uses the inference code of the SR-Scientist repository (Xia et al., ICLR
2026), pinned to commit 00a44a1, cloned to the directory given by
--srscientist-repo:

    git clone https://github.com/GAIR-NLP/SR-Scientist REPO
    git -C REPO checkout 00a44a1

Its search is kept: 40 turns, each a fresh conversation of at most 25
assistant messages (1000 model calls at most) with an `equation_evaluator`
and a `data_analyzer` tool, a MAPE target that tightens turn by turn, the
best three candidates so far carried into the next turn's prompt, completion
below MAPE 1e-6. Four things are replaced through the module attributes the
searcher calls; its repository is not modified:
- the model client: our deployment, the output limit under the API's field,
  retries, every request recorded (client.py). A request that fails every
  retry ends its conversation, as the searcher's own failures do; the run then
  still finishes but exits non-zero, with the number of such requests in
  run.json, so scripts/bench.py marks the attempt failed. The searcher fixes
  max_completion_tokens 8192 and records nothing.
- the scoring of candidates: the shared evaluator (trace, size check,
  least-squares fit) in a 30 s subprocess, both in the tool and in the
  end-of-turn re-evaluation that drives the MAPE target, the carried
  candidates and the completion (the searcher's own BFGS fit in both places).
  The system prompt and the tool schema describe the shared evaluator instead
  of the BFGS wrapper.
- the code execution of `data_analyzer`: a 30 s local subprocess with the
  training rows on stdin as JSON, in place of the SandboxFusion service.
- the end-of-turn evaluation runs in threads, not processes, so it sees the
  replaced attributes and the candidate cache.

The searcher sees training data only; score.py scores its candidates. It has
no random source: the seed is recorded only.

Usage:
    uv run experiments/llm_srbench/run_srscientist.py --problem CRK21 --llm llm.yaml --seed 1
        --srscientist-repo REPO [--turns 40] [--assistant-turns 25] [--runs-dir runs/srscientist]

The run directory receives run.json (written at the start, completed at the
end), usage.jsonl (one record per request), transcript.jsonl (every message of
every turn's conversation, appended as the run goes), candidates.jsonl (one
record per distinct candidate at its first evaluation: code, size, train
mse/nmse/mape and fitted params, or the error) and srscientist.json, the
searcher's own output.
"""

# ruff: noqa: E402
import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import types
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import openai
import pandas as pd
import yaml

HERE = Path(__file__).parent
# The evaluation protocol (evaluator.py, expression.py) is the symbolic regression example's.
PROTOCOL = HERE.parents[1] / "examples/symbolic_regression"
sys.path[:0] = [str(HERE), str(PROTOCOL)]
from client import record
from dataset import problems
from expression import MAX_SIZE, RULE

RETRIES = 8
TIMEOUT = 30  # seconds per code execution and per candidate evaluation
SEARCH = {"num_turns": 40, "max_assistant_turns": 25, "mape_threshold": 0.001, "top_k": 3}
EVALUATOR_DESCRIPTION = (
    "The `equation_evaluator` tool fits the 10 entries of `params` to the data by least squares "
    "(scipy least_squares from all ones) and reports the mean squared error (MSE), the normalized "
    f"MSE (NMSE), the mean absolute percentage error (MAPE) and whether MAPE meets the target. {RULE}\n"
)
EVALUATE = f"""
import json, sys
sys.path.insert(0, {str(PROTOCOL)!r})
import numpy as np
from evaluator import fit
from expression import check, compile, load, size, trace

task = json.loads(sys.stdin.read())
rows = np.asarray(task["rows"], dtype=float)
y, x = rows[:, 0], rows[:, 1:]
equation = load("import numpy as np\n" + task["code"])  # the searcher's candidates assume np
expr = check(trace(equation, task["inputs"]))
equation = compile(expr, task["inputs"])
params, mse = fit(equation, x, y)
if not np.isfinite(mse):
    raise ValueError(f"fit produced a non-finite mse: {{mse}}")
pred = equation(*x.T, params=params)
nonzero = ~np.isclose(y, 0)
mape = float(np.mean(np.abs((pred[nonzero] - y[nonzero]) / y[nonzero]))) if nonzero.any() else 0.0
open("result.json", "w").write(json.dumps({{"size": size(expr), "mse": mse, "nmse": mse / float(np.var(y)),
                                            "mape": mape, "params": params.tolist()}}))
"""


def import_srscientist(repo: Path):
    """The repository's inference modules."""
    sys.path.insert(0, str(repo / "inference"))
    sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=None)  # imported, unused
    from infer import get_results, inference, tool

    return inference, tool, get_results


def run(script: str, stdin: str) -> tuple[int | None, str, str, str | None]:
    """Run a Python script in a fresh temp directory and its own process group, killed whole
    on timeout. Returns (returncode, None on timeout; stdout; stderr; result.json if written)."""
    with tempfile.TemporaryDirectory() as cwd:
        env = os.environ | {key: cwd for key in ("HOME", "TMPDIR", "TMP", "TEMP")}
        env |= {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(stdin, timeout=TIMEOUT)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            returncode = None
        result = Path(cwd, "result.json")
        return returncode, stdout, stderr, result.read_text() if result.exists() else None


class Candidates:
    """Shared evaluation of the searcher's candidates: each distinct code once, recorded on first sight."""

    def __init__(self, inputs: list[str], rows: np.ndarray, log: Path, clean):
        self.inputs = inputs
        self.rows = rows.tolist()
        self.log = log
        self.clean = clean  # the searcher's code-fence stripping, so both paths share a key
        self.cache: dict[str, dict] = {}
        self.lock = threading.Lock()

    def evaluate(self, code: str, source: str) -> dict:
        code = self.clean(code)
        with self.lock:
            if code not in self.cache:
                self.cache[code] = self._evaluate(code)
                entry = {"timestamp": datetime.now(UTC).isoformat(), "source": source, "code": code}
                with self.log.open("a") as f:
                    f.write(json.dumps(entry | self.cache[code]) + "\n")
            return self.cache[code]

    def _evaluate(self, code: str) -> dict:
        stdin = json.dumps({"code": code, "inputs": self.inputs, "rows": self.rows})
        returncode, _, stderr, result = run(EVALUATE, stdin)
        if returncode is None:
            return {"error": f"evaluation timed out after {TIMEOUT}s"}
        if result is None:
            last = stderr.strip().splitlines()
            return {"error": last[-1] if last else f"evaluation exited with {returncode}"}
        return json.loads(result)


def tool_functions(candidates: Candidates, rows_json: str):
    """Replacements for the searcher's two tools, in the signatures it calls them with."""

    async def equation_evaluator(equation, stdin_data, sandbox_fusion_url, mape_threshold):
        result = await asyncio.to_thread(candidates.evaluate, equation, "tool")
        if "error" in result:
            text = f"Evaluation failed: {result['error']}"
        else:
            met = result["mape"] < mape_threshold
            text = (
                f"MSE loss: {result['mse']:.6e}; NMSE loss: {result['nmse']:.6e}; "
                f"Mean absolute percentage error: {result['mape']:.4%}; "
                f"expression size: {result['size']} nodes (at most {MAX_SIZE}).\n"
                f"{'Success' if met else 'Failure'}: The mean absolute percentage error is "
                f"{'smaller' if met else 'larger'} than {mape_threshold:.4%}."
            )
        return text, {"tool_name": "equation_evaluator", **result}

    async def execute_code(code, stdin_data, sandbox_fusion_url):
        returncode, stdout, stderr, _ = await asyncio.to_thread(
            run, candidates.clean(code), rows_json
        )
        if returncode is None:
            text = f"Execution Failed: timed out after {TIMEOUT}s\n\n---STDOUT---\n{stdout}\n\n---STDERR---\n{stderr}"
        elif stderr:
            text = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        else:
            text = f"STDOUT:\n{stdout}"
        return text, {"tool_name": "data_analyzer", "returncode": returncode}

    def get_function_by_name(name):
        return {"equation_evaluator": equation_evaluator, "data_analyzer": execute_code}[name]

    return get_function_by_name


def branch_worker(candidates: Candidates, parse_equation):
    """The searcher's end-of-turn evaluation of the conversation's candidates, in the tables
    it reads; the test-set table has the layout without values (it sees training data only)."""

    def evaluate_branch_worker(branch, example_data):
        equations = list(dict.fromkeys(parse_equation(branch.messages[2:])))
        results = [candidates.evaluate(code, "turn_end") for code in equations]
        train = {
            "equations": equations,
            "mse": [r.get("mse") for r in results],
            "nmse": [r.get("nmse") for r in results],
            "mape": [r.get("mape") for r in results],
            "optimized_params": [r.get("params") for r in results],
            "success": [int("mape" in r and r["mape"] < branch.mape_threshold) for r in results],
        }
        # The searcher aggregates both tables using the training table's keys.
        train |= {key: [None] * len(results) for key in ("acc_0.1", "acc_0.01", "acc_0.001")}
        metrics = ("mse", "nmse", "mape", "acc_0.1", "acc_0.01", "acc_0.001", "optimized_params")
        test = {"equations": equations} | {key: [None] * len(results) for key in metrics}
        return branch.branch_id, train, test

    return evaluate_branch_worker


def client_class(llm: dict, run_dir: Path):
    """AsyncOpenAI stand-in: our deployment, the output limit under the API's field, retries,
    every request recorded in usage.jsonl and the conversation appended to transcript.jsonl."""

    class Client:
        exhausted = 0  # requests that failed RETRIES times

        def __init__(self, api_key=None, base_url=None):
            self.client = openai.AsyncOpenAI(
                api_key=os.environ[llm["api_key_env"]], base_url=llm["api_base"], max_retries=0
            )
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
            self.seen = 0  # messages of the current conversation already written

        async def create(self, **request):
            del request["max_completion_tokens"]
            request[llm["max_tokens_field"]] = llm["max_tokens"]
            if llm.get("reasoning_effort"):
                request["reasoning_effort"] = llm["reasoning_effort"]
            self.transcribe(request["messages"])  # written before the request: survives a failure
            for attempt in range(RETRIES):
                try:
                    response = await self.client.chat.completions.create(**request)
                except Exception as exc:
                    record(run_dir / "usage.jsonl", request["model"], error=repr(exc))
                    print(f"request failed ({attempt + 1}/{RETRIES}): {exc!r}", file=sys.stderr)
                    await asyncio.sleep(2**attempt)
                    continue
                record(run_dir / "usage.jsonl", request["model"], response)
                self.transcribe([*request["messages"], response.choices[0].message.model_dump()])
                return response
            type(self).exhausted += 1
            raise RuntimeError(f"request failed {RETRIES} times")

        def transcribe(self, messages: list) -> None:
            """Append the conversation's messages not written yet."""
            if len(messages) < self.seen:
                self.seen = 0  # a new turn's conversation
            with (run_dir / "transcript.jsonl").open("a") as f:
                for message in messages[self.seen :]:
                    f.write(json.dumps(message) + "\n")
            self.seen = len(messages)

    return Client


def system_prompt(original: str) -> str:
    """The searcher's system prompt, the BFGS wrapper code replaced by the shared evaluator's description."""
    start = original.index("For the `equation_evaluator`")
    end = original.index("In utilizing the tool")
    return original[:start] + EVALUATOR_DESCRIPTION + original[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--srscientist-repo", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=SEARCH["num_turns"])
    parser.add_argument("--assistant-turns", type=int, default=SEARCH["max_assistant_turns"])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/srscientist"))
    args = parser.parse_args()

    inference, tool, get_results = import_srscientist(args.srscientist_repo.resolve())
    logging.getLogger().setLevel(logging.WARNING)  # the searcher logs every step at INFO
    problem = next(p for p in problems() if p.name == args.problem)
    inputs = problem.symbols[1:]
    llm = yaml.safe_load(args.llm.read_text())
    search = SEARCH | {"num_turns": args.turns, "max_assistant_turns": args.assistant_turns}

    run_dir = args.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{problem.name}-seed{args.seed}"
    run_dir.mkdir(parents=True)
    started = datetime.now(UTC)

    candidates = Candidates(
        inputs, problem.train, run_dir / "candidates.jsonl", get_results._clean_code_block
    )
    inference.AsyncOpenAI = Client = client_class(llm, run_dir)
    inference.get_function_by_name = tool_functions(candidates, json.dumps(problem.train.tolist()))
    inference.evaluate_branch_worker = branch_worker(candidates, get_results.parse_equation)
    inference.SYSTEM_PROMPT = system_prompt(inference.SYSTEM_PROMPT)
    tool.tools[0]["function"]["description"] = (
        "Accepts a mathematical equation as a Python function string, fits its parameters to the "
        "dataset by least squares, and returns performance metrics (MSE, NMSE, MAPE) and the "
        "expression size to evaluate its goodness of fit."
    )
    concurrent.futures.ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor

    parquet = run_dir / "problem.parquet"
    pd.DataFrame(
        [
            {
                "dataset_identifier": f"lsr_synth/{problem.subset}",
                "symbols": problem.symbols,
                "symbol_descs": problem.descriptions,
                "symbol_properties": ["O"] + ["V"] * len(inputs),
                "expression": "",  # the truth stays out of the run
                "samples": {"train": problem.train.tolist()},
            }
        ]
    ).to_parquet(parquet)
    info = {
        "problem": problem.name,
        "subset": problem.subset,
        "seed": args.seed,
        "llm": llm,
        "search": search | {"timeout_seconds": TIMEOUT, "max_size": MAX_SIZE},
        "started": started.isoformat(),
    }
    (run_dir / "run.json").write_text(json.dumps(info, indent=2))
    asyncio.run(
        inference.run_inference_batch(
            model_name=llm["model"],
            model_url=llm["api_base"],
            api_key="",
            sandbox_urls=[""],
            parquet_file_path=str(parquet),
            source=None,
            output_json_path=str(run_dir / "srscientist.json"),
            **search,
        )
    )

    info |= {
        "candidates": len(candidates.cache),
        "failed_requests": Client.exhausted,
        "finished": datetime.now(UTC).isoformat(),
    }
    (run_dir / "run.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {run_dir}")
    if Client.exhausted:
        sys.exit(f"{Client.exhausted} request(s) failed {RETRIES} times")


if __name__ == "__main__":
    main()
