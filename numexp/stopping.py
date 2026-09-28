"""停止条件层：与具体求解器解耦，任何迭代求解器都可复用。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from numexp.records import IterationRecord, StopReason


@dataclass
class StoppingDecision:
    reason: StopReason
    message: str


class Criterion:
    def check(
        self, record: IterationRecord, state: dict, iteration: int
    ) -> StoppingDecision | None:
        raise NotImplementedError


@dataclass
class ResidualTolerance(Criterion):
    """真实相对残差（辅以绝对残差）达到精度。"""

    rtol: float = 1e-8
    atol: float = 0.0

    def check(self, record, state, iteration):
        if record.residual_norm <= self.atol + self.rtol * state["b_norm"]:
            return StoppingDecision(
                StopReason.CONVERGED,
                f"||r||={record.residual_norm:.3e} 满足 rtol={self.rtol:g}, atol={self.atol:g}",
            )
        return None


@dataclass
class MaxIterations(Criterion):
    """迭代预算用尽（不是成功）。"""

    max_iter: int = 1000

    def check(self, record, state, iteration):
        if iteration >= self.max_iter:
            return StoppingDecision(
                StopReason.BUDGET_EXHAUSTED,
                f"达到最大迭代步数 {self.max_iter}，残差仍为 {record.residual_norm:.3e}",
            )
        return None


class NumericalAnomaly(Criterion):
    """NaN/Inf 等数值异常（不是成功）。"""

    def check(self, record, state, iteration):
        x = state["x"]
        if not np.all(np.isfinite(x)):
            return StoppingDecision(
                StopReason.NUMERICAL_BREAKDOWN,
                f"第 {iteration} 步迭代向量出现 NaN/Inf",
            )
        if not np.isfinite(record.residual_norm):
            return StoppingDecision(
                StopReason.NUMERICAL_BREAKDOWN,
                f"第 {iteration} 步真实残差为 NaN/Inf",
            )
        return None


class StoppingSuite:
    """组合多个停止条件，按顺序裁决；默认包含精度、预算与数值异常。"""

    def __init__(self, *criteria: Criterion):
        self.criteria = list(criteria) or [
            ResidualTolerance(),
            MaxIterations(),
            NumericalAnomaly(),
        ]

    def check(
        self, record: IterationRecord, state: dict, iteration: int
    ) -> StoppingDecision | None:
        for criterion in self.criteria:
            decision = criterion.check(record, state, iteration)
            if decision is not None:
                return decision
        return None
