# Online bin packing

The online bin packing setting of FunSearch (Romera-Paredes et al., Nature 2024). Items arrive one at a time and are placed irrevocably. The evolved `priority(item, bins)` scores the bins that can hold the item, and the item goes into the highest-scoring one. The initial program is Best Fit. Fitness is 1 / (1 + excess), where excess is the fraction of bins used beyond the Martello-Toth L2 lower bound, over 100 hidden sequences of 120 items.

```bash
uv run agentic-evolve examples/online_bin_packing/config.yaml
```

| File | Purpose |
|---|---|
| `sequences.txt` | The hidden sequences. Only the evaluator reads them. |
| `generalize.py` | Scores programs on sequences they were not evolved on: the 20 u120 instances of OR-Library (`or1.txt`, licence in `or1_LICENSE.txt`) and 100 fresh sequences. Best Fit and the heuristic FunSearch published for these instances are printed as references. |
| `openevolve.yaml` | Settings for the OpenEvolve comparison in `experiments/`. |

After a run, check the best program on the unseen sequences:

```bash
uv run examples/online_bin_packing/generalize.py runs/<timestamp>/best.py
```
