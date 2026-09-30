"""数值计算实验框架：问题、求解器、迭代记录三层边界。"""

from .problems import LinearSystem, PreconditionError
from .criteria import (StoppingCriterion, RelativeResidual, CombinedCriterion,
                       BudgetExhausted)
from .records import IterationRecord, SolveResult, TerminationReason
from .base import IterativeSolver, Checkpoint
from .checkpoint import CheckpointMismatchError
from .problems import matrix_fingerprint, vector_fingerprint
from .cg import ConjugateGradient
from .stationary import Richardson

__all__ = [
    "LinearSystem", "PreconditionError",
    "StoppingCriterion", "RelativeResidual", "CombinedCriterion",
    "BudgetExhausted",
    "IterationRecord", "SolveResult", "TerminationReason",
    "IterativeSolver", "Checkpoint", "CheckpointMismatchError",
    "matrix_fingerprint", "vector_fingerprint",
    "ConjugateGradient", "Richardson",
]
