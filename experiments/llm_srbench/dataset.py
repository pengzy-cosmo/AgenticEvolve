"""The LSR-Synth problems of LLM-SRBench (Shojaee et al., ICML 2025).

Read from the HuggingFace dataset nnheui/llm-srbench at a fixed revision. The
dataset is gated: run `hf auth login` once. Every problem has 4000 training
points, 500 in-domain test points and 500 out-of-domain test points; the first
column of each matrix is the target variable.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO_ID = "nnheui/llm-srbench"
REVISION = "4f22d48100d94125d3af1e7bd40897fcc7f9597d"
SUBSETS = ("chem_react", "bio_pop_growth", "phys_osc", "matsci")


@dataclass
class Problem:
    name: str
    subset: str
    symbols: list[str]  # target first, then the inputs; the matrix columns
    descriptions: list[str]
    expression: str  # ground truth, never shown to a search method
    train: np.ndarray
    test: np.ndarray
    ood_test: np.ndarray


def problems() -> list[Problem]:
    root = Path(snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION))
    found = []
    with h5py.File(root / "lsr_bench_data.hdf5") as h5:
        for subset in SUBSETS:
            table = pq.read_table(root / f"data/lsr_synth_{subset}-00000-of-00001.parquet")
            for row in table.to_pylist():
                if row["symbol_properties"] != ["O"] + ["V"] * (len(row["symbols"]) - 1):
                    raise ValueError(f"{row['name']}: unexpected variable roles")
                group = h5[f"lsr_synth/{subset}/{row['name']}"]
                found.append(
                    Problem(
                        row["name"],
                        subset,
                        row["symbols"],
                        row["symbol_descs"],
                        row["expression"],
                        *(group[k][...].astype(np.float64) for k in ("train", "test", "ood_test")),
                    )
                )
    return found
