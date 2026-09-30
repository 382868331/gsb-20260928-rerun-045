"""迭代求解器基类：统一驱动停止判定、真实残差记录与检查点续算。"""

from __future__ import annotations

import numpy as np

from .criteria import (StoppingCriterion, RelativeResidual, BudgetExhausted,
                       NumericalBreakdown, Stagnated)
from .checkpoint import Checkpoint
from .problems import LinearSystem
from .records import IterationRecord, SolveResult, TerminationReason


class IterativeSolver:
    """迭代求解器抽象基类。

    子类只需实现：
      * ``name``：求解器名称；
      * ``validate(problem)``：方法前提检查，不满足抛 PreconditionError；
      * ``initial_state(problem, x0)``：构造内部状态 dict；
      * ``step(problem, state)``：推进一个迭代并返回新状态。

    基类统一负责：每步计算真实残差、停止条件、异常分类、
    预算判定、检查点存取。后续新增 Krylov/分裂迭代方法可直接复用。
    """

    name = "iterative"

    def __init__(self, criterion: StoppingCriterion | None = None,
                 record_solutions: bool = False,
                 checkpoint: str | Checkpoint | None = None):
        self.criterion = criterion or RelativeResidual()
        self.record_solutions = record_solutions
        if isinstance(checkpoint, Checkpoint) or checkpoint is None:
            self.checkpoint = checkpoint
        else:
            self.checkpoint = Checkpoint(checkpoint)

    # ---- 子类接口 -----------------------------------------------------

    def validate(self, problem: LinearSystem) -> None:
        raise NotImplementedError

    def initial_state(self, problem: LinearSystem,
                      x0: np.ndarray) -> dict:
        raise NotImplementedError

    def step(self, problem: LinearSystem, state: dict) -> dict:
        raise NotImplementedError

    def _state_arrays(self, state: dict) -> dict[str, np.ndarray]:
        """默认检查点序列化：保存状态中的全部 ndarray 与数值标量。"""
        arrays = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                arrays[k] = v
            elif isinstance(v, (int, float, bool, np.generic)):
                arrays[k] = np.asarray(v)
        return arrays

    def _scalar_keys(self, state: dict) -> list[str]:
        return [k for k, v in state.items()
                if not isinstance(v, np.ndarray)
                and isinstance(v, (int, float, bool, np.generic))]

    def solver_settings(self) -> dict:
        """会改变迭代数学过程的求解器设定；由子类补充。"""
        return {}

    # ---- 统一驱动 -----------------------------------------------------

    def solve(self, problem: LinearSystem, x0: np.ndarray | None = None,
              history: list[IterationRecord] | None = None,
              continuation: str = "resume",
              restart_from: str | Checkpoint | None = None) -> SolveResult:
        if continuation not in ("resume", "restart"):
            raise ValueError("continuation 必须是 'resume' 或 'restart'")
        if continuation == "resume" and restart_from is not None:
            raise ValueError("restart_from 只能与 continuation='restart' 一起使用")
        self.validate(problem)
        if x0 is None:
            x0 = np.zeros(problem.n)
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (problem.n,):
            from .problems import PreconditionError
            raise PreconditionError("x0 形状与问题维度不一致")

        start_iter = 0
        checkpoint_read_only = False
        continuation_used = "fresh"
        source_checkpoint = None
        if restart_from is not None:
            source_checkpoint = (restart_from if isinstance(restart_from, Checkpoint)
                                 else Checkpoint(restart_from))
            if not source_checkpoint.exists():
                raise FileNotFoundError(f"接续源检查点不存在：{source_checkpoint.path}")
            if (self.checkpoint is not None and self.checkpoint.exists()
                    and (self.checkpoint.resolved_path()
                         != source_checkpoint.resolved_path())):
                raise ValueError(
                    "目标检查点已存在；受控重新开始应使用新的目标路径，"
                    "不能覆盖其他运行记录")
            if (self.checkpoint is not None
                    and self.checkpoint.resolved_path()
                    == source_checkpoint.resolved_path()):
                checkpoint_read_only = True
        elif continuation == "restart" and self.checkpoint is not None and self.checkpoint.exists():
            source_checkpoint = self.checkpoint
            checkpoint_read_only = True

        if source_checkpoint is not None and source_checkpoint.exists():
            old_iteration, old_x, _ = source_checkpoint.approximation(problem)
            state = self.initial_state(problem, old_x)
            start_iter = 0
            continuation_used = f"restarted_from_iteration_{old_iteration}"
        elif continuation == "resume" and self.checkpoint is not None and self.checkpoint.exists():
            self.checkpoint.assert_resumable(
                self.name, problem, self.criterion, self.solver_settings())
            start_iter, arrays, meta = self.checkpoint.load(self.name)
            state = self.initial_state(problem, x0)
            for key in meta.get("scalar_keys", []):
                if key in arrays:
                    state[key] = arrays.pop(key).item()
            state.update(arrays)
            continuation_used = "resumed"
        else:
            state = self.initial_state(problem, x0)

        recs: list[IterationRecord] = history if history is not None else []

        def log(iteration: int, x: np.ndarray) -> float:
            r_norm = problem.residual_norm(x)
            self.criterion.check_finite(x, r_norm, iteration)
            recs.append(IterationRecord(
                iteration=iteration,
                residual_norm=r_norm,
                x=x.copy() if self.record_solutions else None,
            ))
            return r_norm

        reason = TerminationReason.CONVERGED
        message = "达到精度要求"
        try:
            iteration = start_iter
            r_norm = log(iteration, state["x"])
            if self.criterion.check(iteration, state["x"], problem, r_norm):
                pass
            else:
                while True:
                    state = self.step(problem, state)
                    iteration += 1
                    r_norm = log(iteration, state["x"])
                    if self.criterion.check(iteration, state["x"], problem, r_norm):
                        break
                    if self.checkpoint is not None and not checkpoint_read_only:
                        self.checkpoint.save(
                            self.name, iteration, self._state_arrays(state),
                            problem_identity=problem.identity(),
                            criterion_signature=self.criterion.signature(),
                            solver_settings=self.solver_settings(),
                            scalar_keys=self._scalar_keys(state))
                    self.criterion.enforce_budget(iteration)
        except NumericalBreakdown as exc:
            reason = TerminationReason.NUMERICAL_BREAKDOWN
            message = str(exc)
        except BudgetExhausted as exc:
            reason = TerminationReason.BUDGET_EXHAUSTED
            message = str(exc)
        except Stagnated as exc:
            reason = TerminationReason.STAGNATED
            message = str(exc)
        except ZeroDivisionError as exc:
            reason = TerminationReason.NUMERICAL_BREAKDOWN
            message = f"第 {iteration} 步除零（迭代 breakdown）：{exc}"
        except FloatingPointError as exc:
            reason = TerminationReason.NUMERICAL_BREAKDOWN
            message = f"第 {iteration} 步浮点异常：{exc}"

        if (reason is TerminationReason.CONVERGED and self.checkpoint is not None
                and not checkpoint_read_only):
            self.checkpoint.remove()

        return SolveResult(x=state["x"].copy(), reason=reason,
                           message=message, history=recs,
                           problem_identity=problem.identity(),
                           solver_name=self.name,
                           continuation=continuation_used)
