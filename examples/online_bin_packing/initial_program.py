"""Online bin packing: score each feasible bin for the arriving item."""

import numpy as np


def priority(item: int, bins: np.ndarray) -> np.ndarray:
    """Returns the priority with which to add item to each feasible bin.

    Args:
        item: size of the arriving item.
        bins: remaining capacities of the feasible bins (each fits the item).

    Returns:
        Array of the same size as bins with the priority score of each bin.
    """
    # Best fit: prefer the bin that the item fills most tightly.
    return -(bins - item)
