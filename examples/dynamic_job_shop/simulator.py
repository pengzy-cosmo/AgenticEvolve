"""Dynamic job shop simulator for evolving dispatching rules.

Jobs arrive over time (Poisson process), each with its own machine route and
integer processing times; due dates follow the TWK rule. Scheduling is
non-delay: whenever a machine is free and has waiting operations it starts one
immediately. With one waiting operation there is no choice; with two or more
the dispatching rule priority(queue, now) scores them and the highest score
starts (ties: earliest arrival at this queue). The rule never sees jobs that
have not arrived.

The 500 warm-up jobs are not scored; arrivals continue until jobs 501..2500
(by arrival index) have all completed, so the shop never empties out during
the scored period. The metric is the mean tardiness of the scored jobs.

Generative parameters are given by Scenario; everything is reproducible from
the seed (numpy default_rng, pinned by uv.lock).
"""

from dataclasses import dataclass
from heapq import heappop, heappush
from types import SimpleNamespace

import numpy as np

MACHINES = 10
OPS_MIN, OPS_MAX = 2, 14  # operations per job, uniform
PT_MIN, PT_MAX = 1, 99  # integer processing time, uniform
WARMUP = 500
RECORDED = 2000
CHUNK = 500  # jobs generated per RNG draw
MAX_JOBS = 20 * (WARMUP + RECORDED)  # guard against rules that never finish

# Columns of the per-machine queue arrays.
PT, NEXT_PT, REMAINING_WORK, REMAINING_OPS, TOTAL_WORK, ARRIVAL, DUE, QUEUED_SINCE, NEXT_MACHINE = (
    range(9)
)
COLUMNS = 9


@dataclass(frozen=True)
class Scenario:
    utilization: float
    due_factor: float  # TWK tightness: due = arrival + due_factor * total processing time

    @property
    def mean_interarrival(self) -> float:
        mean_pt = (PT_MIN + PT_MAX) / 2
        mean_ops = (OPS_MIN + OPS_MAX) / 2
        return mean_pt * mean_ops / (MACHINES * self.utilization)


class _Jobs:
    """Lazily generated arrival stream: arrival time, route, processing times, due date."""

    def __init__(self, scenario: Scenario, seed: int):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.clock = 0.0
        self.arrival: list[float] = []
        self.route: list[np.ndarray] = []
        self.pt: list[np.ndarray] = []
        self.due: list[float] = []

    def ensure(self, index: int) -> None:
        while index >= len(self.arrival):
            self._generate(CHUNK)

    def _generate(self, count: int) -> None:
        rng = self.rng
        gaps = rng.exponential(self.scenario.mean_interarrival, count)
        n_ops = rng.integers(OPS_MIN, OPS_MAX + 1, count)
        for gap, n in zip(gaps, n_ops, strict=True):
            self.clock += gap
            # Revisits allowed, but consecutive operations use different machines.
            route = np.empty(n, dtype=np.int64)
            route[0] = rng.integers(MACHINES)
            for i in range(1, n):
                route[i] = (route[i - 1] + rng.integers(1, MACHINES)) % MACHINES
            pt = rng.integers(PT_MIN, PT_MAX + 1, n)
            self.arrival.append(self.clock)
            self.route.append(route)
            self.pt.append(pt)
            self.due.append(self.clock + self.scenario.due_factor * float(pt.sum()))


class Shop:
    def __init__(self):
        cap = 64
        self.queue = [np.zeros((cap, COLUMNS)) for _ in range(MACHINES)]
        self.queue_job = [np.zeros(cap, dtype=np.int64) for _ in range(MACHINES)]
        self.count = [0] * MACHINES
        self.waiting_work = np.zeros(MACHINES)  # processing time waiting at each machine
        self.busy = [False] * MACHINES
        self.busy_until = np.zeros(MACHINES)

    def push(self, m: int, row: np.ndarray, job: int) -> None:
        n = self.count[m]
        if n == len(self.queue_job[m]):
            self.queue[m] = np.concatenate([self.queue[m], np.zeros_like(self.queue[m])])
            self.queue_job[m] = np.concatenate(
                [self.queue_job[m], np.zeros_like(self.queue_job[m])]
            )
        self.queue[m][n] = row
        self.queue_job[m][n] = job
        self.count[m] = n + 1
        self.waiting_work[m] += row[PT]

    def pop(self, m: int, i: int) -> int:
        n = self.count[m] - 1
        job = int(self.queue_job[m][i])
        self.waiting_work[m] -= self.queue[m][i, PT]
        self.queue[m][i] = self.queue[m][n]
        self.queue_job[m][i] = self.queue_job[m][n]
        self.count[m] = n
        return job


def simulate(
    priority, scenario: Scenario, seed: int, trace: list | None = None
) -> tuple[float, int, int]:
    """Runs one replication.

    Returns (mean tardiness of scored jobs, number of rule decisions, number of
    decisions that did not pick a shortest waiting operation). If trace is a
    list, every started operation is appended as (job, op, machine, start, end).
    """
    jobs = _Jobs(scenario, seed)
    shop = Shop()
    op_index: list[int] = []
    events: list[tuple[float, int, int, int]] = []  # (time, seq, kind, payload)
    seq = 0
    ARRIVE, FINISH = 0, 1

    jobs.ensure(0)
    heappush(events, (jobs.arrival[0], seq, ARRIVE, 0))
    next_job = 1
    scored_done = 0
    tardiness_sum = 0.0
    decisions = divergent = 0

    def row_for(job: int, i: int, now: float) -> np.ndarray:
        route, pt = jobs.route[job], jobs.pt[job]
        last = i == len(route) - 1
        return np.array(
            [
                pt[i],
                0.0 if last else pt[i + 1],
                pt[i:].sum(),
                len(route) - i,
                pt.sum(),
                jobs.arrival[job],
                jobs.due[job],
                now,
                -1.0 if last else route[i + 1],
            ]
        )

    def start(m: int, job: int, now: float) -> None:
        nonlocal seq
        shop.busy[m] = True
        seq += 1
        end = now + jobs.pt[job][op_index[job]]
        shop.busy_until[m] = end
        heappush(events, (end, seq, FINISH, job))
        if trace is not None:
            trace.append((job, op_index[job], m, now, end))

    def dispatch(m: int, now: float) -> None:
        nonlocal decisions, divergent
        n = shop.count[m]
        if n == 1:
            i = 0
        else:
            rows = shop.queue[m][:n]
            # WINQ: work waiting at the next machine plus the remainder of its current operation.
            nxt = np.maximum(rows[:, NEXT_MACHINE].astype(np.int64), 0)
            load = shop.waiting_work + np.maximum(shop.busy_until - now, 0.0)
            winq = np.where(rows[:, NEXT_MACHINE] >= 0, load[nxt], 0.0)
            # The rule gets a copy; ties and statistics below read the shop's own rows.
            view = rows.copy()
            queue = SimpleNamespace(
                pt=view[:, PT],
                next_pt=view[:, NEXT_PT],
                remaining_work=view[:, REMAINING_WORK],
                remaining_ops=view[:, REMAINING_OPS],
                total_work=view[:, TOTAL_WORK],
                arrival=view[:, ARRIVAL],
                due=view[:, DUE],
                queued_since=view[:, QUEUED_SINCE],
                winq=winq,
                utilization=scenario.utilization,
            )
            scores = np.asarray(priority(queue, now), dtype=float)
            if scores.shape != (n,):
                raise ValueError("priority must return one score per waiting operation")
            if np.isnan(scores).any():
                raise ValueError("priority returned NaN")
            i = int(np.lexsort((rows[:, QUEUED_SINCE], -scores))[0])
            decisions += 1
            divergent += int(rows[i, PT] > rows[:, PT].min())
        start(m, shop.pop(m, i), now)

    # Every event at the current time is applied before any machine dispatches,
    # so simultaneous completions and arrivals are all visible to the rule.
    while scored_done < RECORDED:
        now = events[0][0]
        while events and events[0][0] == now:
            _, _, kind, job = heappop(events)
            if kind == ARRIVE:
                op_index.append(0)
                if next_job >= MAX_JOBS:
                    raise RuntimeError("simulation did not terminate")
                jobs.ensure(next_job)
                seq += 1
                heappush(events, (jobs.arrival[next_job], seq, ARRIVE, next_job))
                next_job += 1
            else:
                shop.busy[int(jobs.route[job][op_index[job]])] = False
                op_index[job] += 1
                if op_index[job] == len(jobs.route[job]):
                    if WARMUP <= job < WARMUP + RECORDED:
                        scored_done += 1
                        tardiness_sum += max(0.0, now - jobs.due[job])
                    continue
            shop.push(int(jobs.route[job][op_index[job]]), row_for(job, op_index[job], now), job)
        for m in range(MACHINES):
            if not shop.busy[m] and shop.count[m]:
                dispatch(m, now)

    return tardiness_sum / RECORDED, decisions, divergent
