#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "huggingface_hub", "numpy", "pyarrow", "scipy", "sympy"]
# ///
"""Select one LSR-Synth problem per domain from the data and the ground truth.

Within each domain, in order:
1. complexity (nodes of the ground truth's expression tree) within the
   domain's interquartile range;
2. informative OOD split: OOD target variance at least 10% of the ID test
   targets', every OOD input spanning at least 5% of its ID range;
3. identifiable structure: with any one basis function of the expanded ground
   truth removed, the refit's train NMSE stays above 1e-3;
4. the most complex survivor, ties by problem index.

Usage: uv run examples/llm_srbench/select.py   (minutes: the refits)
"""

import re
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from dataset import problems

CONSTANTS = ["F0", "omega0", "beta", "mu", "alpha", "gamma"]  # the physics problems' symbols
OOD_VARIANCE = 0.1
OOD_SPREAD = 0.05
NECESSARY = 1e-3


def truth(problem):
    """Ground truth as a sympy expression; constants without a numeric value stay symbols."""
    text = re.sub(r"\b(?!sin|cos|exp|log|sqrt)([A-Za-z_]\w*)\(t\)", r"\1", problem.expression)
    text = re.sub(r"\d+\.\d+_([A-Za-z])\b", r"k_\1", text)
    names = {n: sp.Symbol(n) for n in [*CONSTANTS, *problem.symbols[1:]]}
    return sp.sympify(text, locals={**names, "Abs": sp.Abs})


def bases(expr, inputs):
    """Distinct basis functions of the expanded truth, outer coefficients dropped."""
    found = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        _, rest = term.as_independent(*inputs, as_Add=False)
        found[rest] = None
    return list(found)


def refit_nmse(basis, inputs, train):
    """Best train NMSE of sum(k_i * basis_i) with the inner constants free too."""
    params, start = [], []

    def free(node):
        if isinstance(node, sp.Float) or (isinstance(node, sp.Symbol) and node not in inputs):
            params.append(sp.Symbol(f"c{len(params)}"))
            start.append(float(node) if isinstance(node, sp.Float) else 1.0)
            return params[-1]
        return node.func(*[free(a) for a in node.args], evaluate=False) if node.args else node

    model = sp.Add(*[free(sp.Symbol(f"k{i}") * b) for i, b in enumerate(basis)], evaluate=False)
    f = sp.lambdify([*inputs, *params], model, modules="numpy")
    x, y = train[:, 1:].T, train[:, 0]

    def residual(c):
        with np.errstate(all="ignore"):
            r = np.asarray(f(*x, *c), dtype=complex) - y
        return np.where(np.isfinite(r) & (abs(r.imag) < 1e-12), r.real, 1e3)

    rng = np.random.default_rng(0)
    best = np.inf
    for attempt in range(6):
        x0 = np.array(start) * (1 if attempt == 0 else rng.uniform(0.3, 3, len(start)))
        result = least_squares(residual, x0, max_nfev=20000)
        best = min(best, np.mean(result.fun**2) / np.var(y))
        if best < 1e-9:
            break
    return best


def main():
    by_domain = {}
    for p in problems():
        by_domain.setdefault(p.subset, []).append(p)
    for domain, group in by_domain.items():
        nodes = {p.name: len(list(sp.preorder_traversal(truth(p)))) for p in group}
        low, high = np.percentile(list(nodes.values()), [25, 75])
        survivors = []
        for p in sorted(group, key=lambda p: (-nodes[p.name], int(re.sub(r"\D", "", p.name)))):
            if not low <= nodes[p.name] <= high:
                continue
            variance = np.var(p.ood_test[:, 0]) / np.var(p.test[:, 0])
            spread = min(
                np.ptp(p.ood_test[:, i]) / np.ptp(p.test[:, i]) for i in range(1, p.train.shape[1])
            )
            if variance < OOD_VARIANCE or spread < OOD_SPREAD:
                continue
            inputs = [sp.Symbol(s) for s in p.symbols[1:]]
            basis = bases(truth(p), inputs)
            weakest = min(
                refit_nmse(basis[:i] + basis[i + 1 :], inputs, p.train) for i in range(len(basis))
            )
            print(
                f"{domain} {p.name}: nodes {nodes[p.name]}, ood variance {variance:.2f}, "
                f"bases {len(basis)}, weakest drop-one nmse {weakest:.1e}"
                + ("" if weakest > NECESSARY else " (not identifiable)")
            )
            if weakest > NECESSARY:
                survivors.append(p)
        chosen = survivors[0]
        print(f"==> {domain}: {chosen.name}   {sp.sstr(truth(chosen))}\n")


if __name__ == "__main__":
    main()
