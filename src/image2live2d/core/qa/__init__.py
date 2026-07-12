"""Stage 7' — QA harness. "Does it look good in motion?"

Implementation lives in :mod:`.harness`; this package re-exports its public API.
"""

from __future__ import annotations

from .harness import (
    batch,
    BatchReport,
    check,
    deform_at,
    evaluate,
    MAX_DISPLACEMENT,
    ParamSweep,
    plan_sweeps,
    RigReport,
    sweep_report,
    SweepReport,
)

__all__ = [
    "batch",
    "BatchReport",
    "check",
    "deform_at",
    "evaluate",
    "MAX_DISPLACEMENT",
    "ParamSweep",
    "plan_sweeps",
    "RigReport",
    "sweep_report",
    "SweepReport",
]
