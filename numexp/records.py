"""记录层：逐步迭代记录与带明确停止原因的结果。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class StopReason(str, Enum):
    """停止原因。只有 CONVERGED 表示成功，其余都必须如实报告。"""

    CONVERGED = "converged"                      # 达到精度
    BUDGET_EXHAUSTED = "budget_exhausted"        # 迭代预算用尽
    NUMERICAL_BREAKDOWN = "numerical_breakdown"  # NaN/Inf、除零、负曲率等异常

    @property
    def success(self) -> bool:
        return self is StopReason.CONVERGED


@dataclass
class IterationRecord:
    """单步记录。residual_norm 来自重新计算的真实残差，而非递推估计。"""

    iteration: int
    residual_norm: float
    relative_residual: float
    metrics: dict = field(default_factory=dict)
    x: np.ndarray | None = None  # 默认不保存向量，避免大型问题爆内存


@dataclass
class SolveResult:
    solver_name: str
    problem_name: str
    reason: StopReason
    message: str
    x: np.ndarray
    records: list[IterationRecord]
    residual_history: list[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.reason.success

    @property
    def iterations(self) -> int:
        return len(self.residual_history) - 1

    @property
    def final_residual(self) -> float:
        return self.residual_history[-1] if self.residual_history else float("nan")

    def summary(self) -> str:
        return (
            f"{self.solver_name} on {self.problem_name}: {self.reason.value} "
            f"after {self.iterations} iters, ||r||={self.final_residual:.3e} ({self.message})"
        )
