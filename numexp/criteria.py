"""停止条件层：与具体求解方法解耦，可被任意迭代求解器复用。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problems import LinearSystem


class NumericalBreakdown(RuntimeError):
    """迭代中出现数值异常（NaN/Inf、除零、非正曲率等）。"""


class BudgetExhausted(Exception):
    """迭代预算用尽。"""


class Stagnated(Exception):
    """迭代停滞：真实残差在一个窗口内没有实质下降。

    与预算用尽区分：停滞说明当前方法/起点组合暂时没有进展，
    多起点场景下应把剩余预算让给其他路径。
    """


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


@dataclass
class StagnationGuard(StoppingCriterion):
    """包装精度条件：窗口内真实残差下降不足则抛 Stagnated。

    预算与数值异常仍委托给被包装的条件处理；停滞判定只看
    基类每步重算的真实残差范数。
    """

    inner: StoppingCriterion | None = None
    window: int = 30
    min_decay: float = 0.9

    def __post_init__(self):
        if self.inner is None:
            self.inner = RelativeResidual()
        self._residuals: list[float] = []

    def signature(self) -> dict:
        return {"type": type(self).__name__,
                "inner": self.inner.signature(),
                "window": self.window,
                "min_decay": self.min_decay}

    def enforce_budget(self, iteration: int) -> None:
        self.inner.enforce_budget(iteration)

    def check(self, iteration, x, problem, residual_norm) -> bool:
        history = self._residuals
        history.append(residual_norm)
        if len(history) > self.window:
            reference = history[-1 - self.window]
            if residual_norm > self.min_decay * reference:
                raise Stagnated(
                    f"第 {iteration} 步判定停滞：最近 {self.window} 步真实残差 "
                    f"从 {reference:.3e} 仅变为 {residual_norm:.3e}")
        return self.inner.check(iteration, x, problem, residual_norm)
