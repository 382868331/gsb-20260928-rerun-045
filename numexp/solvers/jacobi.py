"""Jacobi 迭代：x^{k+1} = D^{-1}(b - (L+U) x^k)。

定常迭代不保证收敛；不收敛时由预算/异常条件如实终止，不伪装成成功。
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from numexp.problems import LinearSystem
from numexp.solvers.base import IterativeSolver, MethodPreconditionError, IterationBreakdown


class JacobiSolver(IterativeSolver):
    name = "jacobi"

    def validate(self, problem: LinearSystem) -> None:
        diag = problem.A.diagonal() if sparse.issparse(problem.A) else np.diag(problem.A)
        if not np.all(np.isfinite(diag)):
            raise MethodPreconditionError("Jacobi 要求对角元均为有限实数")
        if np.any(diag == 0.0):
            raise MethodPreconditionError("Jacobi 要求对角元全部非零，发现零对角元")
        self._diag = diag

    def init_state(self, problem: LinearSystem, x0: np.ndarray) -> dict:
        return {"x": x0.copy()}

    def step(self, problem: LinearSystem, state: dict):
        x = state["x"]
        Ax = problem.matvec(x)
        x_new = x + (problem.b - Ax) / self._diag
        if not np.all(np.isfinite(x_new)):
            raise IterationBreakdown("Jacobi 更新出现 NaN/Inf（迭代发散）")
        state["x"] = x_new
        return x_new, {}
