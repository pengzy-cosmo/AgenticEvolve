# Experiments

Tooling for the paper's runs. The matrix files here are the protocol; the model is given at launch. Every attempt records the git commit and source tree it ran from.

## Matrix

`heuristics.yaml`: entries are either an agentic_evolve config with an optional override, or an OpenEvolve task with an iteration count; each entry is repeated `repeats` times.

```yaml
repeats: 5
entries:
  obp-full:
    system: agentic_evolve
    config: examples/online_bin_packing/config.yaml
  obp-no-analysis:
    system: agentic_evolve
    config: examples/online_bin_packing/config.yaml
    override:
      evolution: {max_analyses_per_commit: 0}
  obp-openevolve:
    system: openevolve
    task: examples/online_bin_packing
    iterations: 180
```

`symbolic_regression.yaml`: the LLM-SRBench comparison (`experiments/llm_srbench`). A matrix with a `problems` list is expanded first: every entry becomes one entry per problem, named `<entry>-<problem>`, with `{problem}` substituted in its string values; the expanded entries are ordered by problem, so the methods interleave. The systems are agentic_evolve on the generated task directories, `llmsr` with a sample count, `srscientist` and `pysr`; their seed is the repeat number. The LLM-SR and SR-Scientist runners exit non-zero when a request failed all its retries, so a run hit by an outage is recorded as failed and relaunched with `--retry-failed`; its results stay on disk.

```yaml
repeats: 1
problems: [CRK21, CRK35, BPG18]
entries:
  full:
    system: agentic_evolve
    config: experiments/llm_srbench/tasks/{problem}/config.yaml
  llmsr: {system: llmsr, samples: 1000}
  srscientist: {system: srscientist}
  pysr: {system: pysr}
```

## Model file

Kept outside the repository. One block per system that takes a model (`pysr` takes none). The `agentic_evolve` block is our llm block; the others are OpenAI-compatible client settings, one layout for OpenEvolve, LLM-SR and SR-Scientist, passed to the system's runner as `--llm`.

```yaml
agentic_evolve:          # the llm block of CONFIG.md
  model: deepseek-flash
  context_window: 500000        # session context cap
openevolve: &client
  model: deepseek-flash
  api_base: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  max_tokens: 384000            # the model's output limit, same value as our side
  max_tokens_field: max_tokens  # the API's field for a whole response incl. reasoning
llmsr: *client
srscientist: *client
```

`max_tokens_field` names the request field that caps a whole response, reasoning included (DeepSeek: `max_tokens`, DashScope: `max_completion_tokens`); mycode chooses it per provider, the baselines' clients send what the file says. `reasoning_effort` is optional in every block and must be the same value in all; omitted means the provider default. For a model missing from the mycode catalog, set `max_tokens` in the `agentic_evolve` block too. Before a formal launch, check the effective values: the first lines of a run's `run.log` (ours) and `run.json` (the baselines).

## Run

```bash
uv run scripts/bench.py experiments/heuristics.yaml --llm ~/deepseek.yaml --parallel 3
uv run scripts/bench.py experiments/symbolic_regression.yaml --llm ~/deepseek.yaml --parallel 3 \
    --llmsr-repo ~/llm-srbench --srscientist-repo ~/SR-Scientist
```

The symbolic regression matrix needs the task directories (`uv run experiments/llm_srbench/make_tasks.py`) and the two baseline repositories at the commits named in `run_llmsr.py` and `run_srscientist.py`; PySR installs its Julia dependencies on first use.

Results go to `runs/heuristics-deepseek/` (`runs/<matrix stem>-<llm stem>`, or `--runs-dir`). Pairs launch interleaved: repeat 1 of every entry, then repeat 2, and so on. Each pair has its own directory:

```
runs/heuristics-deepseek/<entry>/r<n>/
  override.yaml or llm.yaml   inputs written for this pair
  <run>/                      the run directory created by the launched program
  attempt-<k>.log             stdout and stderr of attempt k
  attempts.jsonl              one record per attempt: command, matrix, llm, git commit,
                              tag, source tree hash, timestamps, exit code, run directory
```

An attempt is recorded when it starts and updated when it ends: `ok`, `failed`, or `interrupted` (Ctrl-C terminates the children); a record left at `running` belongs to a runner that died. Pairs with a successful attempt are skipped on relaunch. Pairs whose attempts all ended otherwise are skipped unless `--retry-failed` is given. The working tree must be clean unless `--allow-dirty` is given. `--dry-run` lists the commands; `--only ENTRY ...` restricts the entries.

## Curves

```bash
uv run scripts/curves.py Full=runs/heuristics-deepseek/obp-full/r1/<run> OpenEvolve=runs/heuristics-deepseek/obp-openevolve/r1/<run> -o curves.png
```

Reads run directories of both systems and plots best-so-far fitness against formal evaluations and against estimated cost. Runs store token counts; prices come from the mycode model catalog at plot time.

## OpenEvolve driver

`experiments/openevolve/run.py` runs OpenEvolve 0.3.2 on a task directory: the task description becomes the system message, evaluations go through the same subprocess executor, requests are logged with their usage. The task's `openevolve.yaml` holds the model-independent settings. For direct use, `--llm` takes a YAML file containing only the fields inside the `openevolve` block above, without the block wrapper. `bench.py` writes this file automatically.

```bash
uv run --group openevolve experiments/openevolve/run.py --task examples/online_bin_packing --llm llm.yaml --iterations 180 --seed 0
```
