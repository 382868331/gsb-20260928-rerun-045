"""数值计算实验框架：问题、求解器、迭代记录三层边界。"""

from .problems import LinearSystem, PreconditionError
from .criteria import (StoppingCriterion, RelativeResidual, CombinedCriterion,
                       StagnationGuard, BudgetExhausted, Stagnated)
from .records import IterationRecord, SolveResult, TerminationReason
from .base import IterativeSolver, Checkpoint
from .cg import ConjugateGradient
from .stationary import Richardson
from .reuse import ReuseReport, assess_reuse
from .series import SeriesStep, SeriesResult, solve_problem_series
from .multistart import (StartPath, PathRecord, MultiStartResult,
                         MultiStartSession)
from .continuation import (ContinuationNode, AttemptRecord,
                           ContinuationPath, ContinuationResult,
                           ContinuationSession)

__all__ = [
    "LinearSystem", "PreconditionError",
    "StoppingCriterion", "RelativeResidual", "CombinedCriterion",
    "StagnationGuard", "BudgetExhausted", "Stagnated",
    "IterationRecord", "SolveResult", "TerminationReason",
    "IterativeSolver", "Checkpoint",
    "ConjugateGradient", "Richardson",
    "ReuseReport", "assess_reuse",
    "SeriesStep", "SeriesResult", "solve_problem_series",
    "StartPath", "PathRecord", "MultiStartResult", "MultiStartSession",
    "ContinuationNode", "AttemptRecord", "ContinuationPath",
    "ContinuationResult", "ContinuationSession",
]
