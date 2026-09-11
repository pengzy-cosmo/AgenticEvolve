"""Offline bin packing: assign every item to a bin, one bin id per item."""


def pack(items, capacity):
    """Next-fit: fill the current bin, open a new one when the item does not fit."""
    assignment = []
    current, load = 0, 0
    for item in items:
        if load + item > capacity:
            current += 1
            load = 0
        assignment.append(current)
        load += item
    return assignment
