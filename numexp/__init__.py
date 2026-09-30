"""数值计算实验框架：问题、求解器、迭代记录三层边界。"""

from .problems import LinearSystem, PreconditionError
from .criteria import (StoppingCriterion, RelativeResidual, CombinedCriterion,
                       BudgetExhausted)
from .records import IterationRecord, SolveResult, TerminationReason
from .base import IterativeSolver, Checkpoint
from .cg import ConjugateGradient
from .stationary import Richardson
from .sequence import CarryoverNote, SequenceStep, describe_carryover, solve_sequence
from .multistart import (MultiStartRunner, MultiStartResult, PathOutcome,
                         StartPoint)

__all__ = [
    "LinearSystem", "PreconditionError",
    "StoppingCriterion", "RelativeResidual", "CombinedCriterion",
    "BudgetExhausted",
    "IterationRecord", "SolveResult", "TerminationReason",
    "IterativeSolver", "Checkpoint",
    "ConjugateGradient", "Richardson",
    "CarryoverNote", "SequenceStep", "describe_carryover", "solve_sequence",
    "MultiStartRunner", "MultiStartResult", "PathOutcome", "StartPoint",
]
