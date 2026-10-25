# AgenticEvolve

Agentic LLM evolution: each island is one persistent LLM session that evolves a Python program through six controlled tools (`read`, `write`, `edit`, `run_analysis`, `commit`, `read_commit`). The framework owns evaluation, the archive, per-island MAP-Elites, migration, and context compaction; the LLM decides what to change and when to commit.

## Install

Python 3.12+ and `uv`.

```bash
uv sync
```

## Run

```bash
export DEEPSEEK_API_KEY=...
uv run agentic-evolve examples/online_bin_packing/config.yaml
```

Override any setting with a second YAML file. Dictionaries merge, other values replace.

```bash
printf 'evolution: {num_islands: 1, max_commits: 3}\n' > smoke.yaml
uv run agentic-evolve examples/online_bin_packing/config.yaml --override smoke.yaml --runs-dir runs/smoke
```

Settings and defaults: [CONFIG.md](CONFIG.md).

## Task directory

```
config.yaml         description, paths, settings
initial_program.py  starting program.py
evaluator.py        evaluate(program_path) -> dict with the fitness field and
                    the feature fields, each feature in [0, 1]
resources/          optional public data, solver code, and notes
openevolve.yaml     optional OpenEvolve settings for the comparison driver
```

The description defines the task: state the function interface, execution rules, evaluation protocol, and meaning of each metric. Use `task.resources` for public files. They are listed in the system prompt, available through `read`, and copied into each analysis and evaluation directory. Python modules in that directory can be imported by the candidate program. The original resources are read-only through the tools; changes to execution copies are discarded after each run.

Keep private evaluator inputs, such as hidden instances and reference values, beside the evaluator and outside the resources directory.

## Examples

| Example | Task |
|---|---|
| [circle_packing](examples/circle_packing) | 26 circles in the unit square, maximizing the sum of radii. The minimal task directory. |
| [online_bin_packing](examples/online_bin_packing) | A priority rule for online bin packing, in FunSearch's setting. |
| [dynamic_job_shop](examples/dynamic_job_shop) | A dispatching rule for a dynamic job shop, against a published rule. |
| [symbolic_regression](examples/symbolic_regression) | The equation behind an oscillator's data, under the LLM-SRBench protocol. |

## Run directory

```
runs/<timestamp>/
  config.yaml    effective config
  task/          snapshot of the task directory
  initial.json   evaluation of the initial program
  archive.jsonl  every commit, code inlined
  best.py        global best program
  best.json      its id, metrics, description
  sessions/      model conversations, one directory per island
  run.log
```

Runs store token counts. Cost is estimated at plot time from the mycode model catalog.

## Notes

- OS sandboxing uses the built-in `sandbox-exec` on macOS and requires `bubblewrap` with unprivileged user namespaces on Linux. Windows runs without an OS sandbox. See [CONFIG.md](CONFIG.md) for sandbox settings.
- For comparison runs against OpenEvolve and experiment matrices, see [experiments/README.md](experiments/README.md).

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```
