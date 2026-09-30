"""迭代求解器基类：统一驱动停止判定、真实残差记录与检查点续算。"""

from __future__ import annotations

import numpy as np

from .criteria import (StoppingCriterion, RelativeResidual, BudgetExhausted,
                       NumericalBreakdown)
from .checkpoint import Checkpoint, CheckpointMismatchError
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
        """影响迭代内部状态含义的求解器参数；恢复原进度时必须一致。

        预算（criterion.max_iter）不在其中：允许放宽预算后续算。
        子类（如带步长/松弛参数的方法）应覆盖并加入对应参数。
        """
        return {"name": self.name}

    def _settings_key(self) -> dict:
        return {"solver": self.solver_settings(),
                "criterion": self.criterion.settings_key()}

    # ---- 统一驱动 -----------------------------------------------------

    def solve(self, problem: LinearSystem, x0: np.ndarray | None = None,
              history: list[IterationRecord] | None = None,
              continuation: str = "resume") -> SolveResult:
        """求解 A x = b。

        continuation 控制已有检查点的用法：

        * ``"resume"``（默认）：恢复原问题的原进度，迭代号/内部状态
          （如 CG 的递推残差与搜索方向）原样延续。恢复前严格校验问题
          身份（A、b）与关键求解设定，不一致立即抛
          ``CheckpointMismatchError`` 明确拒绝，而不是盲目迭代。
          ``max_iter`` 属于预算，允许调整；精度设定改变则拒绝。
        * ``"restart"``：受控接续，只把检查点里的旧近似解 x 当作新问题
          的初值，内部状态全部重新初始化，迭代从 0 重新计数。
          此时检查点只读，绝不覆盖旧记录。
        """
        if continuation not in ("resume", "restart"):
            raise ValueError(
                "continuation 必须是 'resume' 或 'restart'")
        self.validate(problem)

        start_iter = 0
        resumed = False
        ckpt = self.checkpoint
        checkpoint_exists = ckpt is not None and ckpt.exists()
        if checkpoint_exists and continuation == "resume":
            # 恢复前先核对问题身份与关键设定，不一致立即明确拒绝。
            ckpt.verify_resume(self.name, problem, self._settings_key())
            if x0 is None:
                x0 = np.zeros(problem.n)
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (problem.n,):
                from .problems import PreconditionError
                raise PreconditionError("x0 形状与问题维度不一致")
            start_iter, arrays, meta = ckpt.load(self.name)
            state = self.initial_state(problem, x0)
            for key in meta.get("scalar_keys", []):
                if key in arrays:
                    state[key] = arrays.pop(key).item()
            state.update(arrays)
            resumed = True
        else:
            if checkpoint_exists and continuation == "restart":
                # 受控接续：只借用旧近似解，旧记录保持只读、绝不覆盖。
                _, old_x, _old_solver = ckpt.approximate_solution()
                if old_x.shape != (problem.n,):
                    raise CheckpointMismatchError(
                        f"旧近似解维度 {old_x.shape[0]} 与新问题维度 "
                        f"{problem.n} 不一致，不能作为初值重启")
                if x0 is None:
                    x0 = old_x
                ckpt = None  # 本次重启不写该检查点，保留旧问题的原记录
            if x0 is None:
                x0 = np.zeros(problem.n)
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (problem.n,):
                from .problems import PreconditionError
                raise PreconditionError("x0 形状与问题维度不一致")
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
                    if ckpt is not None:
                        ckpt.save(
                            self.name, iteration, self._state_arrays(state),
                            problem_id=problem.problem_id,
                            problem_identity=problem.identity,
                            settings_key=self._settings_key(),
                            scalar_keys=self._scalar_keys(state))
                    self.criterion.enforce_budget(iteration)
        except NumericalBreakdown as exc:
            reason = TerminationReason.NUMERICAL_BREAKDOWN
            message = str(exc)
        except BudgetExhausted as exc:
            reason = TerminationReason.BUDGET_EXHAUSTED
            message = str(exc)
        except ZeroDivisionError as exc:
            reason = TerminationReason.NUMERICAL_BREAKDOWN
            message = f"第 {iteration} 步除零（迭代 breakdown）：{exc}"
        except FloatingPointError as exc:
            reason = TerminationReason.NUMERICAL_BREAKDOWN
            message = f"第 {iteration} 步浮点异常：{exc}"

        if reason is TerminationReason.CONVERGED and ckpt is not None:
            ckpt.remove()

        return SolveResult(x=state["x"].copy(), reason=reason,
                           message=message, history=recs,
                           problem_id=problem.problem_id,
                           resumed=resumed)
