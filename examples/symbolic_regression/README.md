# Symbolic regression

Find the equation behind 1000 observations of a driven nonlinear oscillator: its acceleration as a function of position `x`, time `t` and velocity `v`. The task follows the LLM-SRBench protocol of our experiments (`experiments/llm_srbench`). The program defines `equation(x, t, v, params)`. The evaluator traces it into a single expression of at most 30 nodes, fits the 10 `params` by least squares from all ones, and uses -NMSE on the observations as fitness. The initial program is linear in the inputs.

```bash
uv run agentic-evolve examples/symbolic_regression/config.yaml
```

| File | Purpose |
|---|---|
| `data/train.csv` | The observations, given to the model as a task resource. |
| `expression.py` | Tracing, node counting and checks, used by the evaluator. |
| `make_data.py` | Generates the observations. It holds the ground truth, which the model never sees. |

The ground truth is

```
dv/dt = F0 sin(t) - c v|v| - omega0^2 x - alpha x^3,  F0 = 0.8, c = 0.3, omega0^2 = 1.5, alpha = 0.5
```

a forced oscillator with cubic stiffness and quadratic drag. In this form it has 21 nodes and fits to NMSE 2e-18. The textbook Duffing form, with linear drag, stops at 1.5e-2.
