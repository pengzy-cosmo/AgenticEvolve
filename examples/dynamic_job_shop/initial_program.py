"""Dynamic job shop: score the operations waiting at a machine that just became free."""

import numpy as np


def priority(queue, now: float) -> np.ndarray:
    """Returns the priority of each waiting operation; the highest starts next.

    Args:
        queue: the waiting operations as equal-length numpy float arrays:
            pt (this operation's processing time), next_pt (processing time
            of the job's next operation, 0 if last), remaining_work (this and
            later operations), remaining_ops, total_work, arrival (job arrival
            time), due (job due date), queued_since (when the operation joined
            this queue), winq (work waiting at the job's next machine plus the
            remainder of the operation in progress there); plus the scalar
            utilization of the current scenario.
        now: current simulation time.

    Returns:
        Array of the same length as queue.pt with one score per operation.
    """
    # Shortest processing time first.
    return -queue.pt
