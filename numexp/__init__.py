"""数值计算实验框架。

三层边界：
- Problem（问题）：被求解的数学对象与已知解等元数据；
- Solver（求解器）：只负责推进一个迭代步，不决定何时停止；
- Record/Result（记录与结果）：每步真实残差、停止原因、可独立复核的报告。

停止条件（numexp.stopping）与结果检查（numexp.verification）对所有求解器通用，
后续新增求解方法时直接复用。
"""
from numexp.problems import LinearSystem, ProblemError
from numexp.records import IterationRecord, SolveResult, StopReason
from numexp.stopping import (
    StoppingSuite,
    ResidualTolerance,
    MaxIterations,
    NumericalAnomaly,
)
from numexp.solvers import IterativeSolver, JacobiSolver, ConjugateGradientSolver
from numexp.solvers.base import MethodPreconditionError, IterationBreakdown
from numexp.checkpoint import Checkpointer
from numexp.verification import verify_result, VerificationReport
from numexp import reference, cases

__all__ = [
    "LinearSystem",
    "ProblemError",
    "IterationRecord",
    "SolveResult",
    "StopReason",
    "StoppingSuite",
    "ResidualTolerance",
    "MaxIterations",
    "NumericalAnomaly",
    "IterativeSolver",
    "JacobiSolver",
    "ConjugateGradientSolver",
    "MethodPreconditionError",
    "IterationBreakdown",
    "Checkpointer",
    "verify_result",
    "VerificationReport",
    "reference",
    "cases",
]
