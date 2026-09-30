"""定常迭代法：Richardson 迭代 x_{k+1} = x_k + omega (b - A x_k)。

用于演示可复用的停止/检查框架，并提供一个不收敛/发散案例：
当 omega 不在 A 谱的收敛区间内时，残差会增长直到预算用尽或数值爆炸。
"""

from __future__ import annotations

import numpy as np

from .base import IterativeSolver
from .criteria import StoppingCriterion
from .checkpoint import Checkpoint
from .problems import LinearSystem


class Richardson(IterativeSolver):
    name = "richardson"

    def __init__(self, omega: float = 1.0,
                 criterion: StoppingCriterion | None = None,
                 record_solutions: bool = False,
                 checkpoint: str | Checkpoint | None = None):
        super().__init__(criterion, record_solutions, checkpoint)
        self.omega = float(omega)

    def validate(self, problem: LinearSystem) -> None:
        # Richardson 只要求问题本身是合法方阵方程组；收敛性由实验案例体现。
        if not np.isfinite(self.omega) or self.omega == 0.0:
            from .problems import PreconditionError
            raise PreconditionError("omega 必须是非零有限值")

    def solver_settings(self) -> dict:
        return {"omega": self.omega}

    def initial_state(self, problem, x0):
        return {"x": x0.copy()}

    def step(self, problem, state):
        x = state["x"]
        r = problem.true_residual(x)
        return {"x": x + self.omega * r}
