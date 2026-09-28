"""迭代求解器基类：统一停止裁决、真实残差记录与检查点续算。

子类只需实现：
- validate(problem)：检查方法前提，不满足则抛出 MethodPreconditionError；
- init_state(problem, x0)：初始化迭代状态；
- step(problem, state)：推进一个迭代步，返回 (x, metrics)。

核心循环不调用任何现成完整求解器，停止逻辑全部来自可复用的 StoppingSuite。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from numexp.problems import LinearSystem
from numexp.records import IterationRecord, SolveResult, StopReason
from numexp.stopping import StoppingSuite
from numexp.checkpoint import Checkpointer


class MethodPreconditionError(ValueError):
    """输入不满足该数值方法的前提条件（如 CG 遇到非对称矩阵、Jacobi 遇到零对角元）。"""


class IterationBreakdown(RuntimeError):
    """迭代内部数值崩溃（除零、负曲率等）。"""


class IterativeSolver(ABC):
    name = "iterative"

    def __init__(
        self,
        stopping: StoppingSuite | None = None,
        *,
        record_vectors: bool = False,
        checkpointer: Checkpointer | None = None,
    ):
        self.stopping = stopping or StoppingSuite()
        self.record_vectors = record_vectors
        self.checkpointer = checkpointer

    # ---- 子类实现的三个钩子 ----
    @abstractmethod
    def validate(self, problem: LinearSystem) -> None: ...

    @abstractmethod
    def init_state(self, problem: LinearSystem, x0: np.ndarray) -> dict: ...

    @abstractmethod
    def step(self, problem: LinearSystem, state: dict) -> tuple[np.ndarray, dict]: ...

    # ---- 通用运行循环 ----
    def solve(
        self,
        problem: LinearSystem,
        x0: np.ndarray | None = None,
        *,
        max_iter: int | None = None,
    ) -> SolveResult:
        b_norm = float(np.linalg.norm(problem.b))
        state = self.init_state(
            problem,
            np.zeros(problem.n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1),
        )
        records, history, iteration, decision, resumed = self._restore(
            problem, state, b_norm
        )
        # 续算时检查点已确认过前提；全新求解必须先校验方法前提
        if not resumed:
            self.validate(problem)
        suite = self._suite_with_budget(max_iter)

        while decision is None:
            try:
                x, metrics = self.step(problem, state)
            except IterationBreakdown as exc:
                x = np.asarray(state.get("x", np.full(problem.n, np.nan)), dtype=float)
                record = self._make_record(iteration + 1, problem, x, b_norm, {})
                records.append(record)
                history.append(record.residual_norm)
                decision = _decision(StopReason.NUMERICAL_BREAKDOWN, str(exc))
                break
            iteration += 1
            record = self._make_record(iteration, problem, x, b_norm, metrics)
            records.append(record)
            history.append(record.residual_norm)
            state["x"] = x
            decision = suite.check(record, {"x": x, "b_norm": b_norm}, iteration)
            if decision is None and self.checkpointer is not None:
                self.checkpointer.save(self, problem, state, records, history, iteration)

        if self.checkpointer is not None and decision is not None:
            if decision.reason.success:
                self.checkpointer.clear(problem)
        return SolveResult(
            solver_name=self.name,
            problem_name=problem.name,
            reason=decision.reason,
            message=decision.message,
            x=records[-1].x if self.record_vectors else np.asarray(state["x"]),
            records=records,
            residual_history=history,
        )

    # ---- 内部工具 ----
    def _suite_with_budget(self, max_iter):
        if max_iter is None:
            return self.stopping
        from numexp.stopping import Criterion, MaxIterations

        criteria = [
            MaxIterations(max_iter) if isinstance(c, MaxIterations) else c
            for c in self.stopping.criteria
        ]
        return StoppingSuite(*criteria)

    def _make_record(self, iteration, problem, x, b_norm, metrics):
        r = problem.residual(x)
        r_norm = float(np.linalg.norm(r))
        rel = r_norm / b_norm if b_norm > 0 else r_norm
        return IterationRecord(
            iteration=iteration,
            residual_norm=r_norm,
            relative_residual=float(rel),
            metrics=dict(metrics),
            x=x.copy() if self.record_vectors else None,
        )

    def _restore(self, problem, state, b_norm):
        if self.checkpointer is None:
            return (*self._initial_records(problem, state, b_norm), False)
        data = self.checkpointer.load(problem)
        if data is None:
            return (*self._initial_records(problem, state, b_norm), False)
        if data["solver_name"] != self.name:
            raise ValueError(
                f"检查点属于求解器 {data['solver_name']}，不能用 {self.name} 续算"
            )
        state.update(data["state"])
        records = [_record_from_dict(d) for d in data["records"]]
        history = list(data["history"])
        return records, history, int(data["iteration"]), None, True

    def _initial_records(self, problem, state, b_norm):
        record = self._make_record(0, problem, state["x"], b_norm, {})
        return [record], [record.residual_norm], 0, None


def _decision(reason, message):
    from numexp.stopping import StoppingDecision

    return StoppingDecision(reason, message)


def _record_from_dict(d):
    return IterationRecord(
        iteration=d["iteration"],
        residual_norm=d["residual_norm"],
        relative_residual=d["relative_residual"],
        metrics=d.get("metrics", {}),
        x=None,
    )
