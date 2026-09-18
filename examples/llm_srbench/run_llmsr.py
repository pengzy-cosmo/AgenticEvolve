#!/usr/bin/env -S uv run
# /// script
# requires-python = "==3.11.*"  # the LLM-SR code uses ast.Str, removed in 3.12
# dependencies = ["h5py", "huggingface_hub", "numpy", "openai", "pyarrow", "pyyaml", "requests", "scipy", "sympy", "absl-py"]
# ///
"""Run the LLM-SR baseline on one LSR-Synth problem, with our LLM deployment.

Uses the LLM-SR implementation of the LLM-SRBench repository, pinned to
commit 6fdbe409, cloned to the directory given by --llmsr-repo:

    git clone https://github.com/deep-symbolic-mathematics/llm-srbench REPO
    git -C REPO checkout 6fdbe409b75ea749c0a93c69b6861637d804dec4

The benchmark's search is kept (10 islands, 2 examples per prompt, 4 samples
per prompt, 30 s evaluation time limit, the best training score decides the
final equation). Three things are ours. The sample budget is given on the
command line. Samples are scored by the shared evaluator (trace, size check,
least-squares fit; see evaluator.py) in place of the program's own BFGS
evaluate(), and the prompt states the evaluator's rule. The model is called
through our client, which sends the output limit under the API's field,
records every response's usage, and hands the search an empty sample after
the retries are exhausted (an empty sample fails evaluation and is counted);
the repository's own client caps every response at 1024 tokens, retries
forever, and records nothing.

The model file has the OpenEvolve layout (model, api_base, api_key_env,
max_tokens, max_tokens_field, optional reasoning_effort).

Usage:
    uv run examples/llm_srbench/run_llmsr.py --problem CRK0 --llm llm.yaml --samples 1000
        --seed 0 --llmsr-repo REPO [--runs-dir runs/llmsr]

The run directory receives run.json, usage.jsonl (one record per request),
best.py (the final equation, importable), and llmsr/ with the searcher's own
sample logs (samples/samples_<n>.json: function, score, sample order).
"""

# ruff: noqa: E402
import argparse
import json
import logging
import multiprocessing
import os
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

# The sandbox forks after numpy is loaded; a multithreaded BLAS crashes in the
# forked child on macOS, so the evaluation runs single-threaded.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import openai
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import evaluator as shared
from dataset import problems

RETRIES = 8


class _NullWriter:
    """Stands in for torch's SummaryWriter, which the searcher's profiler
    imports; the tensorboard log is not used."""

    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass


def _import_llmsr(repo: Path):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "methods"))
    tensorboard = types.ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = _NullWriter
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules["torch.utils"] = types.ModuleType("torch.utils")
    sys.modules["torch.utils.tensorboard"] = tensorboard
    from bench.dataclasses import SEDTask
    from llmsr import config, evaluator, sampler
    from methods.llmsr.searcher import LLMSRSearcher

    return SEDTask, config, sampler, evaluator, LLMSRSearcher


def _normalize(raw):
    """Mycode's canonical usage fields; missing values stay missing."""
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


def make_sandbox_class(evaluator_module, inputs: list[str]):
    """The searcher's sandbox, scoring samples with the shared evaluator.

    The final program's "optimize" and "execute" runs keep the sandbox's own path.
    """

    class Sandbox(evaluator_module.LocalSandbox):
        def _compile_and_run_function(
            self,
            program,
            function_to_run,
            function_to_evolve,
            dataset,
            numba_accelerate,
            result_queue,
            shared_array_np=None,
        ):
            if function_to_run != "evaluate":
                return super()._compile_and_run_function(
                    program,
                    function_to_run,
                    function_to_evolve,
                    dataset,
                    numba_accelerate,
                    result_queue,
                    shared_array_np,
                )
            try:
                namespace = {}
                exec(program, namespace)
                expr = shared.check(shared.trace(namespace["equation"], inputs))
                x, y = np.column_stack(dataset["inputs"]), np.asarray(dataset["outputs"])
                _, mse = shared.fit(shared.compile(expr, inputs), x, y)
                result_queue.put((-mse, True) if np.isfinite(mse) else (None, False))
            except Exception as exc:
                print(f"Execution Error: {exc}")
                result_queue.put((None, False))

    return Sandbox


def make_llm_class(llm: dict, usage_log: Path, sampler_module):
    """The searcher's LLM: one chat completion per sample through our deployment."""

    class LLM(sampler_module.LLM):
        instruction = (
            "You are a helpful assistant tasked with discovering mathematical function "
            "structures for scientific systems. Complete the 'equation' function below, "
            f"considering the physical meaning and relationships of inputs. {shared.RULE}\n\n"
        )

        def draw_samples(self, prompt, config):
            raise NotImplementedError("the searcher samples through async_draw_single_sample")

        async def async_draw_single_sample(self, prompt: str, config) -> str:
            params = {
                "model": llm["model"],
                "messages": [{"role": "user", "content": self.instruction + prompt}],
                "temperature": 1.0,
                llm["max_tokens_field"]: llm["max_tokens"],
            }
            if llm.get("reasoning_effort"):
                params["reasoning_effort"] = llm["reasoning_effort"]
            # The searcher runs each prompt in a new event loop; a client's
            # connection pool must not outlive the loop it was used in.
            async with openai.AsyncOpenAI(
                api_key=os.environ[llm["api_key_env"]],
                base_url=llm["api_base"],
                timeout=1200,
                max_retries=0,
            ) as client:
                for attempt in range(RETRIES):
                    record = {"timestamp": datetime.now(UTC).isoformat(), "model": llm["model"]}
                    try:
                        response = await client.chat.completions.create(**params)
                    except Exception as exc:
                        record |= {"error": repr(exc), "usage": None}
                        _log(usage_log, record)
                        print(f"request failed ({attempt + 1}/{RETRIES}): {exc!r}", file=sys.stderr)
                        continue
                    raw = response.usage.model_dump() if response.usage else None
                    record |= {
                        "response_id": response.id,
                        "finish_reason": response.choices[0].finish_reason
                        if response.choices
                        else None,
                        "raw_usage": raw,
                        "usage": _normalize(raw) if raw else None,
                    }
                    _log(usage_log, record)
                    content = response.choices[0].message.content if response.choices else None
                    return sampler_module._extract_body(content or "", config)
            return ""

    return LLM


def _log(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--llmsr-repo", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/llmsr"))
    args = parser.parse_args()

    # The searcher's sandbox evaluates in a child process whose target lives
    # in the modules imported below; a spawned child cannot import them.
    multiprocessing.set_start_method("fork")
    for name in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    SEDTask, config, sampler, llmsr_evaluator, LLMSRSearcher = _import_llmsr(
        args.llmsr_repo.resolve()
    )
    problem = next(p for p in problems() if p.name == args.problem)
    llm = yaml.safe_load(args.llm.read_text())

    run_dir = args.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{problem.name}-seed{args.seed}"
    run_dir.mkdir(parents=True)
    usage_log = run_dir / "usage.jsonl"
    started = datetime.now(UTC)
    np.random.seed(args.seed)  # island and example selection use numpy's global state

    cfg = config.Config(
        experience_buffer=config.ExperienceBufferConfig(num_islands=10),
        use_api=True,
        api_model=llm["model"],
        samples_per_prompt=4,
    )
    searcher = LLMSRSearcher(
        "llmsr",
        cfg,
        make_llm_class(llm, usage_log, sampler),
        global_max_sample_num=args.samples,
        log_path=run_dir / "llmsr",
    )
    searcher.class_config = config.ClassConfig(
        llm_class=searcher.class_config.llm_class,
        sandbox_class=make_sandbox_class(llmsr_evaluator, problem.symbols[1:]),
    )
    task = SEDTask(
        name=problem.name,
        symbols=problem.symbols,
        symbol_descs=problem.descriptions,
        symbol_properties=["O"] + ["V"] * (len(problem.symbols) - 1),
        samples=problem.train,
        desc=None,
    )
    result = searcher.discover(task)[0]

    (run_dir / "best.py").write_text("import numpy as np\n\n\n" + result.equation.program_format)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "problem": problem.name,
                "subset": problem.subset,
                "seed": args.seed,
                "samples": args.samples,
                "llm": llm,
                "llmsr_config": {
                    "num_islands": 10,
                    "functions_per_prompt": 2,
                    "samples_per_prompt": 4,
                    "evaluate_timeout_seconds": cfg.evaluate_timeout_seconds,
                    "temperature": 1.0,
                    "max_size": shared.MAX_SIZE,
                },
                "best_program_score": result.aux["best_program_score"],
                "best_program_sample_order": result.aux["best_program_sample_order"],
                "started": started.isoformat(),
                "finished": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
