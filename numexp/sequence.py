"""问题序列求解：相关但不同的问题逐个求解，热启动与冷启动对照。

每个问题独立求解、独立记录目标问题身份与收敛判断；上一问题
能沿用的只有近似解 x（作为新初值），求解器内部状态（搜索方向、
递推残差、迭代计数）一律不沿用，旧检查点也不会被当作兼容检查点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problems import LinearSystem
from .criteria import StoppingCriterion
from .records import SolveResult


@dataclass
class CarryoverNote:
    """相邻两次求解之间"什么变了、什么能沿用"的明确说明。"""

    matrix_changed: bool
    rhs_changed: bool
    dimension_changed: bool
    criterion_changed: bool | None  # 未提供停止条件对照时为 None
    reusable: str
    text: str

    @property
    def problem_changed(self) -> bool:
        return (self.matrix_changed or self.rhs_changed
                or self.dimension_changed)


def describe_carryover(old_problem: LinearSystem, new_problem: LinearSystem,
                       old_criterion: StoppingCriterion | None = None,
                       new_criterion: StoppingCriterion | None = None
                       ) -> CarryoverNote:
    """比较前后两个问题（及停止条件），明确说明可沿用的内容。"""
    old_id = old_problem.identity()
    new_id = new_problem.identity()
    matrix_changed = old_id["A_sha256"] != new_id["A_sha256"]
    rhs_changed = old_id["b_sha256"] != new_id["b_sha256"]
    dimension_changed = old_id["n"] != new_id["n"]
    if old_criterion is not None and new_criterion is not None:
        criterion_changed = (old_criterion.signature()
                             != new_criterion.signature())
    else:
        criterion_changed = None

    changes = []
    if dimension_changed:
        changes.append("维度变化（%d -> %d）" % (old_id["n"], new_id["n"]))
    if matrix_changed:
        changes.append("矩阵 A 已变化")
    if rhs_changed:
        changes.append("右端项 b 已变化")
    if criterion_changed:
        changes.append("停止条件（容差等数学设定）已变化")

    if not changes:
        reusable = ("问题与停止条件完全一致：检查点可按原进度恢复，"
                    "求解器内部状态可沿用。")
        text = "未检测到变化：" + reusable
    else:
        if dimension_changed:
            reusable = ("无可沿用内容：维度已变化，旧近似解不能作为新初值，"
                        "必须重新提供初值并从迭代 0 开始。")
        else:
            reusable = ("可沿用的只有上一问题的近似解 x，且仅作为新初值；"
                        "求解器内部状态（搜索方向、递推残差、迭代计数）不可沿用，"
                        "旧检查点不能按原进度恢复，只能以 continuation='restart' "
                        "受控重新开始，新运行从迭代 0 重新记录。")
        text = "、".join(changes) + "。" + reusable

    return CarryoverNote(matrix_changed=matrix_changed,
                         rhs_changed=rhs_changed,
                         dimension_changed=dimension_changed,
                         criterion_changed=criterion_changed,
                         reusable=reusable, text=text)


@dataclass
class SequenceStep:
    """序列中一个问题的求解记录：热启动结果 + 冷启动对照。"""

    index: int
    problem_identity: dict
    carryover: CarryoverNote | None  # 相对上一问题的变化说明；首步为 None
    warm: SolveResult                # 以上一问题近似解为起点（首步同冷启动）
    cold: SolveResult | None         # 从零向量起点独立求解的对照


def solve_sequence(problems, solver_factory,
                   compare_cold_start: bool = True) -> list[SequenceStep]:
    """依次求解一系列相关问题。

    Parameters
    ----------
    problems : LinearSystem 序列，相关问题但不必相同。
    solver_factory : 无参工厂，每次调用返回一个全新的 IterativeSolver
        （不得携带检查点路径——序列中每次求解都是独立的新运行，
        上一问题的剩余迭代状态不允许被当作兼容检查点）。
    compare_cold_start : 是否同时对每个问题从零向量冷启动求解作对照。

    热启动只传递上一问题的近似解 x 作为 x0，求解器内部状态由
    当前问题重新构造；每个 SolveResult 各自记录目标问题身份与收敛判断。
    """
    steps: list[SequenceStep] = []
    prev_problem = None
    prev_criterion = None
    prev_x = None
    for index, problem in enumerate(problems):
        solver = solver_factory()
        carryover = None
        x0 = None
        if prev_problem is not None:
            carryover = describe_carryover(prev_problem, problem,
                                           prev_criterion, solver.criterion)
            if not carryover.dimension_changed and prev_x is not None:
                x0 = prev_x
        warm = solver.solve(problem, x0=x0)
        if x0 is not None:
            warm.continuation = "warm_started_from_previous_solution"
        cold = None
        if compare_cold_start:
            cold = solver_factory().solve(problem)
        steps.append(SequenceStep(index=index,
                                  problem_identity=problem.identity(),
                                  carryover=carryover,
                                  warm=warm, cold=cold))
        prev_problem = problem
        prev_criterion = solver.criterion
        prev_x = warm.x
    return steps
