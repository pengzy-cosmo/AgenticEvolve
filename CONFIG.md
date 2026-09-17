# Configuration

A run is configured by one YAML file, normally the task directory's `config.yaml`. Everything not listed as required has the default shown.

```yaml
task:                                # required
  initial_program: initial_program.py  # starting program.py
  evaluator: evaluator.py              # defines evaluate(program_path) -> dict
  description: |                       # given verbatim to the model in the system prompt
    ...
  # data: data                         # optional directory of raw task inputs, copied
                                       # into every analysis execution directory

llm:
  model: deepseek-flash                # required; the only required llm field
  # provider: deepseek                 # inferred from the model name when omitted
  # base_url: https://...              # provider default when omitted
  # api_key_env: DEEPSEEK_API_KEY      # environment variable holding the key;
                                       # the provider's usual variable when omitted
  # context_window: 1000000            # model catalog value when omitted
  # max_tokens: 384000                 # output limit; model catalog value when omitted
  # reasoning_effort: high             # provider default when omitted

evolution:
  num_islands: 3                       # independent sessions evolving in parallel
  max_commits: 30                      # formal evaluations per island; the run ends
                                       # when every island has spent them
  max_analyses_per_commit: 10          # run_analysis calls allowed between commits;
                                       # 0 removes the analysis capability entirely
  migration_interval: 3                # every N commits an island receives candidates
  migration_rate: 0.3                  # ... the top fraction of its ring neighbors' elites
  # max_total_tokens: 3000000          # optional hard stop per island on total tokens
  # max_requests: 333                  # optional hard stop per island on model requests
                                       # (every response counts); the response that
                                       # reaches it still runs its tool calls

map_elites:
  features: [complexity, bf_divergence]  # required; metric fields spanning the grid,
                                         # each normalized to [0, 1] by the evaluator
  fitness: combined_score              # metric field to maximize
  feature_bins: 10                     # bins per feature axis

executor:
  timeout: 120                         # seconds per evaluation or analysis execution
  output_limit: 10000                  # characters of stdout/stderr returned to the model

compact:
  compact_threshold: 0.7               # fraction of the context window at which the
                                       # session is compacted after the current commit
  safety_threshold: 0.9                # fraction at which the turn is cancelled and
                                       # compacted immediately
```

Paths in `task` are relative to the config file. The evaluator's dict must contain the fitness field and every feature field as finite numbers; anything else fails the commit.

## Overrides

`--override FILE` merges a second YAML file over the config: dictionaries merge recursively, other values replace. The experiment matrix uses this to set the mode and budget without editing the task:

```yaml
evolution:
  max_analyses_per_commit: 0           # No-analysis mode
llm:
  model: qwen3.8-27b
  provider: alibaba
```

## Modes

`max_analyses_per_commit` defines the two modes. With a positive value (Full) the model has `analysis.py` and `run_analysis` and may execute code between formal evaluations; with 0 (No-analysis) those are removed from the tools and the system prompt, and the model only edits, commits, and reads history. The compaction settings, budgets, and archive behave identically in both.
