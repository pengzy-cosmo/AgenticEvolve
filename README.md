# agentic-evolve

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
data/               optional raw inputs, copied into every analysis execution
openevolve.yaml     optional OpenEvolve settings for the comparison driver
```

The description is the model's only source of task knowledge. State the harness semantics, the full function interface, the evaluation protocol, and the meaning of every metric. Evaluator inputs (hidden instances, reference values) stay beside the evaluator and never enter the analysis environment. See [examples/](examples/) for task examples.

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

- Generated code runs in local subprocesses with a timeout and process-group kill, without a sandbox. `executor.py` is the isolation boundary.
- Comparison runs against OpenEvolve and the experiment matrix: [experiments/README.md](experiments/README.md).

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```
