"""Validate CPUSchedEnv with Gymnasium's environment checker."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gymnasium.utils.env_checker import check_env

from env.cpu_sched_env import CPUSchedEnv


def main() -> int:
    check_env(CPUSchedEnv(seed=123), skip_render_check=True)
    print("CPUSchedEnv passed Gymnasium validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
