"""共轭梯度法（手写实现，非现成完整求解器的包装）。

每步残差记录由基类用 r = b - Ax 重新计算真实值；
迭代内部同时维护递推残差用于方向构造。
"""

from __future__ import annotations

import numpy as np

from .base import IterativeSolver
from .criteria import StoppingCriterion
from .checkpoint import Checkpoint
from .problems import LinearSystem


class ConjugateGradient(IterativeSolver):
    name = "cg"

    def __init__(self, criterion: StoppingCriterion | None = None,
                 record_solutions: bool = False,
                 checkpoint: str | Checkpoint | None = None,
                 spd_tol: float | None = None):
        super().__init__(criterion, record_solutions, checkpoint)
        self.spd_tol = spd_tol

    def validate(self, problem: LinearSystem) -> None:
        # CG 前提：A 必须对称正定；不满足则直接拒绝。
        problem.check_spd(self.spd_tol)

    def initial_state(self, problem, x0):
        r = problem.true_residual(x0)
        return {"x": x0.copy(), "r": r, "p": r.copy(),
                "r_dot": float(r @ r)}

    def step(self, problem, state):
        x, r, p, r_dot = state["x"], state["r"], state["p"], state["r_dot"]
        Ap = problem.matvec(p)
        pAp = float(p @ Ap)
        if not np.isfinite(pAp):
            from .criteria import NumericalBreakdown
            raise NumericalBreakdown("矩阵-向量积出现 NaN/Inf")
        if pAp <= 0.0:
            # 非正曲率：说明矩阵在该方向上非正定（大矩阵 SPD 检查未做分解时由此捕获）
            raise FloatingPointError(
                f"p^T A p = {pAp:.3e} <= 0，矩阵非正定，CG 无法继续")
        alpha = r_dot / pAp
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        r_dot_new = float(r_new @ r_new)
        beta = r_dot_new / r_dot
        p_new = r_new + beta * p
        return {"x": x_new, "r": r_new, "p": p_new, "r_dot": r_dot_new}
