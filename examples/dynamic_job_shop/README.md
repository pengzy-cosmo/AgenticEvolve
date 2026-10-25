# Dynamic job shop

A dispatching rule for a dynamic job shop: 10 machines, jobs arriving by a Poisson process, each with its own route. Whenever a machine becomes free with two or more operations waiting, the evolved `priority(queue, now)` picks the one to start. The initial program is shortest processing time first. Fitness is the inverse of the tardiness ratio: per scenario, the program's mean tardiness divided by that of rule R1 (Ferreira et al. 2021), combined by geometric mean over 5 scenarios of load and due-date tightness, each on 10 hidden arrival streams. 1.0 matches R1. One formal evaluation simulates for about 20 seconds.

```bash
uv run agentic-evolve examples/dynamic_job_shop/config.yaml
```

| File | Purpose |
|---|---|
| `simulator.py` | The shop simulator the evaluator runs. |
| `reference.json` | R1's mean tardiness per scenario on the hidden streams. After changing the simulator or the hidden seeds, regenerate it with `uv run examples/dynamic_job_shop/evaluator.py`. |
| `calibrate.py` | Compares classical and published dispatching rules on the scenarios, without an LLM. |
| `generalize.py` | Scores programs on 10 fresh arrival streams per scenario, with R1 simulated on the same streams. |
| `openevolve.yaml` | Settings for the OpenEvolve comparison in `experiments/`. |

After a run, check the best program on the fresh streams:

```bash
uv run examples/dynamic_job_shop/generalize.py runs/<timestamp>/best.py
```
