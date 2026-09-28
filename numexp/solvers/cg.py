"""共轭梯度法（CG）：手工实现，面向大型稀疏对称正定系统。

前提：A 为实对称正定矩阵。非对称直接拒绝；对小规模矩阵额外用 Cholesky
尝试确认正定性，非正定同样拒绝。大规模稀疏矩阵只检查对称性，正定与否
在迭代中以负曲率/零分母形式暴露并作为数值异常终止。
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from numexp.problems import LinearSystem
from numexp.solvers.base import IterativeSolver, MethodPreconditionError, IterationBreakdown

_SYM_TOL = 1e-10
_MAX_EXACT_CHOLESKY_N = 500


class ConjugateGradientSolver(IterativeSolver):
    name = "cg"

    def __init__(self, *args, sym_tol: float = _SYM_TOL, **kwargs):
        super().__init__(*args, **kwargs)
        self.sym_tol = sym_tol

    def validate(self, problem: LinearSystem) -> None:
        A = problem.A
        if sparse.issparse(A):
            diff = (A - A.T).tocoo()
            scale = max(1.0, abs(A).max())
            asym = abs(diff.data).max() if diff.nnz else 0.0
        else:
            scale = max(1.0, np.abs(A).max())
            asym = np.abs(A - A.T).max()
        if asym > self.sym_tol * scale:
            raise MethodPreconditionError(
                f"CG 要求对称矩阵，非对称度 {asym:.3e} 超过容差 {self.sym_tol:g}"
            )
        if not sparse.issparse(A) and problem.n <= _MAX_EXACT_CHOLESKY_N:
            try:
                np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                raise MethodPreconditionError("CG 要求正定矩阵，Cholesky 分解失败") from None

    def init_state(self, problem: LinearSystem, x0: np.ndarray) -> dict:
        r = problem.residual(x0)
        return {"x": x0.copy(), "r": r, "p": r.copy(), "rr": float(r @ r)}

    def step(self, problem: LinearSystem, state: dict):
        r, p, rr = state["r"], state["p"], state["rr"]
        Ap = problem.matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0.0 or not np.isfinite(pAp):
            raise IterationBreakdown(
                f"CG 遇到非正曲率 p^T A p = {pAp:.3e}（矩阵可能非正定或迭代已发散）"
            )
        alpha = rr / pAp
        x_new = state["x"] + alpha * p
        r_new = r - alpha * Ap
        rr_new = float(r_new @ r_new)
        if not np.isfinite(rr_new):
            raise IterationBreakdown("CG 更新出现 NaN/Inf")
        beta = rr_new / rr
        p_new = r_new + beta * p
        state.update(x=x_new, r=r_new, p=p_new, rr=rr_new)
        return x_new, {"alpha": alpha, "pAp": pAp, "r_dot_r": rr_new}
