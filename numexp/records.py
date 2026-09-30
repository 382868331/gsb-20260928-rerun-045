"""迭代记录与求解结果层。"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class TerminationReason(enum.Enum):
    CONVERGED = "converged"          # 达到精度
    BUDGET_EXHAUSTED = "budget_exhausted"  # 预算用尽，不能叫成功
    NUMERICAL_BREAKDOWN = "numerical_breakdown"  # NaN/Inf/除零等异常
    PRECONDITION_FAILED = "precondition_failed"  # 方法前提被破坏

    @property
    def success(self) -> bool:
        return self is TerminationReason.CONVERGED


@dataclass
class IterationRecord:
    """单步迭代记录：迭代号、真实残差范数、当前近似解快照（可选）。"""

    iteration: int
    residual_norm: float
    x: np.ndarray | None = None


@dataclass
class SolveResult:
    """一次求解的完整结果。

    history 中每一步保存的残差都是用原矩阵重新计算的真实残差。
    """

    x: np.ndarray
    reason: TerminationReason
    message: str
    history: list[IterationRecord] = field(default_factory=list)
    problem_identity: dict = field(default_factory=dict)
    solver_name: str = ""
    continuation: str = "fresh"

    @property
    def success(self) -> bool:
        return self.reason.success

    @property
    def iterations(self) -> int:
        return self.history[-1].iteration if self.history else 0

    @property
    def residual_history(self) -> list[float]:
        return [h.residual_norm for h in self.history]

    def verify(self, problem, tol: float = 1e-6) -> bool:
        """结果检查：独立计算真实残差验证，而不是相信求解器自报值。"""
        true_norm = problem.residual_norm(self.x)
        b_norm = float(np.linalg.norm(problem.b))
        scale = b_norm if b_norm > 0.0 else 1.0
        return np.isfinite(true_norm) and true_norm <= tol * scale
