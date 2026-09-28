"""结果检查：与求解器内部状态无关地复核结果，任何求解器都可复用。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from numexp.problems import LinearSystem
from numexp.records import SolveResult, StopReason


@dataclass
class VerificationReport:
    solver_name: str
    problem_name: str
    claimed_reason: StopReason
    recomputed_residual: float
    matches_history: bool
    true_error: float | None
    passed: bool

    def __str__(self) -> str:
        err = "n/a" if self.true_error is None else f"{self.true_error:.3e}"
        return (
            f"[{self.solver_name}/{self.problem_name}] 声明={self.claimed_reason.value}, "
            f"复核残差={self.recomputed_residual:.3e}, 与记录一致={self.matches_history}, "
            f"真解误差={err}, 通过={self.passed}"
        )


def verify_result(
    problem: LinearSystem,
    result: SolveResult,
    *,
    rtol: float = 1e-8,
    atol: float = 0.0,
    history_tol: float = 1e-10,
) -> VerificationReport:
    """独立重算真实残差并检查：声明与实际是否相符。

    - 声称收敛，则复核残差必须确实低于阈值；
    - 声称预算用尽/数值异常，则不应误判为成功（残差明显高于阈值）；
    - 每步历史记录的最后一个残差必须与重算值一致。
    """
    x = result.x
    r_norm = float(np.linalg.norm(problem.residual(x)))
    b_norm = float(np.linalg.norm(problem.b))
    threshold = atol + rtol * b_norm
    matches = abs(r_norm - result.final_residual) <= history_tol * max(
        1.0, result.final_residual
    )
    true_error = (
        float(np.linalg.norm(x - problem.x_true))
        if problem.x_true is not None
        else None
    )
    if result.reason is StopReason.CONVERGED:
        passed = matches and r_norm <= threshold
    else:
        passed = matches and r_norm > threshold
    return VerificationReport(
        solver_name=result.solver_name,
        problem_name=problem.name,
        claimed_reason=result.reason,
        recomputed_residual=r_norm,
        matches_history=matches,
        true_error=true_error,
        passed=passed,
    )
