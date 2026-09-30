"""停止条件层：与具体求解方法解耦，可被任意迭代求解器复用。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problems import LinearSystem


class NumericalBreakdown(RuntimeError):
    """迭代中出现数值异常（NaN/Inf、除零、非正曲率等）。"""


class BudgetExhausted(Exception):
    """迭代预算用尽。"""


@dataclass
class StoppingCriterion:
    """停止条件基类。

    子类实现 ``check``，返回 True 表示达到精度应停止；
    数值异常与预算由基类统一处理，新增求解方法无需重复实现。
    """

    max_iter: int = 1000

    def signature(self) -> dict:
        """数学停止设定；迭代预算 max_iter 不属于问题/进度身份。"""
        return {"type": type(self).__name__}

    def check(self, iteration: int, x: np.ndarray, problem: LinearSystem,
              residual_norm: float) -> bool:
        raise NotImplementedError

    def enforce_budget(self, iteration: int) -> None:
        if iteration >= self.max_iter:
            raise BudgetExhausted(
                f"迭代预算用尽：{iteration} >= max_iter={self.max_iter}")

    @staticmethod
    def check_finite(x: np.ndarray, residual_norm: float, iteration: int) -> None:
        if not np.isfinite(residual_norm) or not np.all(np.isfinite(x)):
            raise NumericalBreakdown(
                f"第 {iteration} 步出现 NaN/Inf（残差范数 {residual_norm:.3e}）")


@dataclass
class RelativeResidual(StoppingCriterion):
    """相对残差 ||b - Ax|| / ||b|| <= tol。"""

    tol: float = 1e-8

    def signature(self) -> dict:
        return {"type": type(self).__name__, "tol": self.tol}

    def check(self, iteration, x, problem, residual_norm) -> bool:
        b_norm = float(np.linalg.norm(problem.b))
        scale = b_norm if b_norm > 0.0 else 1.0
        return residual_norm <= self.tol * scale


@dataclass
class CombinedCriterion(StoppingCriterion):
    """组合多个精度条件，全部满足才停止（预算与异常仍由基类统一处理）。"""

    conditions: tuple = ()

    def signature(self) -> dict:
        return {"type": type(self).__name__,
                "conditions": [c.signature() for c in self.conditions]}

    def check(self, iteration, x, problem, residual_norm) -> bool:
        return all(c.check(iteration, x, problem, residual_norm)
                   for c in self.conditions)
