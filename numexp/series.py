"""相关问题序列的求解：热启动与冷启动对照。

用户把上一问题的近似解作为下一问题的起点（热启动），同时每个问题
也从零向量独立求解（冷启动）作为对照。每次求解都是独立的新运行：
从迭代 0 重新记录，目标问题身份与收敛判断各自独立，绝不把前一问题
的剩余迭代状态当作兼容检查点。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problems import LinearSystem
from .records import SolveResult
from .reuse import ReuseReport, assess_reuse


@dataclass
class SeriesStep:
    """序列中一个问题的求解记录。"""

    index: int
    problem_identity: dict
    cold: SolveResult
    warm: SolveResult | None          # 第一问或维度变化时没有热启动
    reuse: ReuseReport | None         # 相对上一问题的可沿用性说明

    @property
    def best(self) -> SolveResult:
        """按当前问题的真实残差选取本问较优结果。"""
        candidates = [r for r in (self.warm, self.cold) if r is not None]
        return min(candidates, key=lambda r: r.history[-1].residual_norm)


@dataclass
class SeriesResult:
    steps: list[SeriesStep] = field(default_factory=list)

    @property
    def final_x(self) -> np.ndarray | None:
        return self.steps[-1].best.x if self.steps else None


def solve_problem_series(problems: list[LinearSystem],
                         make_solver) -> SeriesResult:
    """依次求解相关问题序列。

    Parameters
    ----------
    problems : 按顺序求解的问题列表，彼此相关但不要求相同。
    make_solver : 无参 callable，每次调用返回一个全新的迭代求解器实例
        （保证热启动与冷启动互不影响，也不复用任何内部状态）。

    每个问题求解两次：冷启动（零向量）与热启动（上一问题较优近似解，
    仅当维度一致时）。热启动的初值残差按当前问题重新计算。
    """
    result = SeriesResult()
    prev_problem: LinearSystem | None = None
    prev_x: np.ndarray | None = None
    for index, problem in enumerate(problems):
        reuse = (assess_reuse(prev_problem, problem)
                 if prev_problem is not None else None)
        cold = make_solver().solve(problem)
        warm = None
        if prev_x is not None and prev_x.shape[0] == problem.n:
            warm = make_solver().solve(problem, x0=prev_x)
        step = SeriesStep(index=index,
                          problem_identity=problem.identity(),
                          cold=cold, warm=warm, reuse=reuse)
        result.steps.append(step)
        prev_problem = problem
        prev_x = step.best.x
    return result
