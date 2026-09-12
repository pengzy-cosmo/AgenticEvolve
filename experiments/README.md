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

## Model file

Kept outside the repository. One block per system; OpenEvolve has only an OpenAI-compatible client.

```yaml
agentic_evolve:          # the llm block of CONFIG.md
  model: deepseek-flash
  context_window: 500000        # session context cap
openevolve:
  model: deepseek-flash
  api_base: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  max_tokens: 384000            # the model's output limit, same value as our side
  max_tokens_field: max_tokens  # the API's field for a whole response incl. reasoning
```

`max_tokens_field` names the request field that caps a whole response, reasoning included (DeepSeek: `max_tokens`, DashScope: `max_completion_tokens`); mycode chooses it per provider, our OpenEvolve client sends what the file says. `reasoning_effort` is optional in both blocks and must be the same value in both; omitted means the provider default. For a model missing from the mycode catalog, set `max_tokens` in the `agentic_evolve` block too. Before a formal launch, check the effective values: the first lines of a run's `run.log` (ours) and `run.json` (OpenEvolve).

## Run

```bash
uv run scripts/bench.py experiments/heuristics.yaml --llm ~/deepseek.yaml --parallel 3
```

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

`examples/openevolve/run.py` runs OpenEvolve 0.3.2 on a task directory: the task description becomes the system message, evaluations go through the same subprocess executor, requests are logged with their usage. The task's `openevolve.yaml` holds the model-independent settings. For direct use, `--llm` takes a YAML file containing only the fields inside the `openevolve` block above, without the block wrapper. `bench.py` writes this file automatically.

```bash
uv run --group openevolve examples/openevolve/run.py --task examples/online_bin_packing --llm llm.yaml --iterations 180 --seed 0
```
