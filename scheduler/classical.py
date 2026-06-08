"""Classical CPU scheduling baselines."""

from __future__ import annotations

from scheduler.simulation import Process, RuntimeProcess, clone_processes, metrics


def _arrive(
    processes: list[RuntimeProcess],
    ready: list[RuntimeProcess],
    index: int,
    now: float,
) -> int:
    while index < len(processes) and processes[index].arrival <= now:
        ready.append(processes[index])
        index += 1
    return index


def _finish(proc: RuntimeProcess, now: float) -> dict[str, float]:
    return {
        "pid": float(proc.pid),
        "arrival": proc.arrival,
        "burst": proc.burst,
        "finish": now,
    }


def run_fcfs(processes: list[Process]) -> dict[str, float]:
    jobs = clone_processes(processes)
    ready: list[RuntimeProcess] = []
    completed: list[dict[str, float]] = []
    now = 0.0
    idx = 0

    while len(completed) < len(jobs):
        idx = _arrive(jobs, ready, idx, now)
        if not ready:
            now = jobs[idx].arrival
            continue
        proc = ready.pop(0)
        now += proc.remaining
        proc.remaining = 0.0
        completed.append(_finish(proc, now))

    return metrics(completed, now)


def run_sjf(processes: list[Process]) -> dict[str, float]:
    jobs = clone_processes(processes)
    ready: list[RuntimeProcess] = []
    completed: list[dict[str, float]] = []
    now = 0.0
    idx = 0

    while len(completed) < len(jobs):
        idx = _arrive(jobs, ready, idx, now)
        if not ready:
            now = jobs[idx].arrival
            continue
        choice = min(range(len(ready)), key=lambda i: ready[i].remaining)
        proc = ready.pop(choice)
        now += proc.remaining
        proc.remaining = 0.0
        completed.append(_finish(proc, now))

    return metrics(completed, now)


def run_rr(processes: list[Process], quantum: float = 20.0) -> dict[str, float]:
    jobs = clone_processes(processes)
    ready: list[RuntimeProcess] = []
    completed: list[dict[str, float]] = []
    now = 0.0
    idx = 0

    while len(completed) < len(jobs):
        idx = _arrive(jobs, ready, idx, now)
        if not ready:
            now = jobs[idx].arrival
            continue

        proc = ready.pop(0)
        run_for = min(quantum, proc.remaining)
        now += run_for
        proc.remaining -= run_for
        idx = _arrive(jobs, ready, idx, now)

        if proc.remaining <= 1e-8:
            completed.append(_finish(proc, now))
        else:
            ready.append(proc)

    return metrics(completed, now)
