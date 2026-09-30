"""多起点求解：统一预算管理、路径隔离、按真实残差选择。

设计要点：

* 统一预算：所有路径共享一个迭代预算池，每条路径实际消耗的迭代步数
  从池中扣减；池用尽时会话暂停，可以追加预算后继续。
* 路径隔离：某条路径失败（前提拒绝、数值异常）或停滞只影响该路径，
  其余路径继续使用剩余预算。
* 选择依据：最终选择以当前问题的真实残差 b - Ax 重新计算为准，
  而不是各路径自报的历史值。
* 暂停/重开：会话状态（各路径预算消耗、起点来源、结果）持久化到
  会话目录；每条路径的求解器检查点独立存放，热启动来源的旧检查点
  只读、绝不覆盖。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .checkpoint import Checkpoint
from .criteria import RelativeResidual, StagnationGuard, StoppingCriterion
from .problems import LinearSystem, PreconditionError
from .records import SolveResult, TerminationReason

SESSION_VERSION = 1


@dataclass
class StartPath:
    """一条求解路径的起点描述。"""

    label: str
    x0: np.ndarray | None = None        # None 表示零向量
    origin: str = "fresh"               # 起点来源说明（fresh / warm:* / checkpoint:*）
    restart_from: str | None = None     # 可选：只读的旧检查点，受控重新开始


@dataclass
class PathRecord:
    """一条路径的持久化记录：来源、预算消耗与结果。"""

    label: str
    origin: str
    status: str = "pending"             # pending / paused / TerminationReason 的 value
    iterations: int = 0                 # 该路径累计消耗的迭代步数
    best_residual: float | None = None  # 结束时按当前问题计算的真实残差
    message: str = ""
    x: np.ndarray | None = None         # 结束时的近似解

    @property
    def finished(self) -> bool:
        # paused：统一预算池用尽被迫暂停，追加预算后可从路径检查点续跑
        return self.status not in ("pending", "paused")

    @property
    def converged(self) -> bool:
        return self.status == TerminationReason.CONVERGED.value


@dataclass
class MultiStartResult:
    """多起点求解结果；best 由当前问题的真实残差决定。"""

    problem_identity: dict
    budget_total: int
    budget_used: int
    records: list[PathRecord] = field(default_factory=list)

    @property
    def pending(self) -> list[PathRecord]:
        return [r for r in self.records if not r.finished]

    def best(self, problem: LinearSystem) -> PathRecord | None:
        """按当前问题重新计算真实残差，选出最优路径。"""
        best_record = None
        best_norm = np.inf
        for record in self.records:
            if record.x is None:
                continue
            norm = problem.residual_norm(record.x)
            if norm < best_norm:
                best_norm = norm
                best_record = record
        return best_record


class MultiStartSession:
    """一次多起点求解会话，支持暂停与恢复。

    Parameters
    ----------
    problem : 当前目标问题；所有路径与最终选择都针对它。
    make_solver : callable(criterion, checkpoint, path) -> IterativeSolver，
        每次调用返回全新求解器实例；criterion 由会话按剩余预算配置。
    paths : 起点列表，每条路径独立消耗统一预算。
    total_budget : 所有路径共享的迭代预算池。
    criterion : 精度停止条件模板；其 max_iter 由会话按剩余预算覆盖。
    stagnation : 可选 (window, min_decay)，为每条路径包装 StagnationGuard。
    session_dir : 会话目录；给定后每次运行结束持久化，可暂停重开。
    """

    def __init__(self, problem: LinearSystem, make_solver,
                 paths: list[StartPath], total_budget: int,
                 criterion: StoppingCriterion | None = None,
                 stagnation: tuple[int, float] | None = None,
                 session_dir: str | Path | None = None):
        if total_budget <= 0:
            raise ValueError("统一预算必须为正整数")
        self.problem = problem
        self.make_solver = make_solver
        self.criterion = criterion or RelativeResidual()
        self.stagnation = stagnation
        self.total_budget = int(total_budget)
        self.budget_used = 0
        self.session_dir = Path(session_dir) if session_dir else None
        self._paths = list(paths)
        self.records = [PathRecord(label=p.label, origin=p.origin)
                        for p in self._paths]

    # ---- 预算与停止条件 ----------------------------------------------

    @property
    def remaining_budget(self) -> int:
        return self.total_budget - self.budget_used

    def _path_checkpoint(self, label: str) -> str | None:
        if self.session_dir is None:
            return None
        return str(self.session_dir / "paths" / label)

    def _make_criterion(self, path_iterations: int) -> StoppingCriterion:
        criterion = copy.copy(self.criterion)
        # 续跑的路径从已消耗步数继续计数，预算池只追加剩余量
        criterion.max_iter = path_iterations + self.remaining_budget
        if self.stagnation is not None:
            window, min_decay = self.stagnation
            criterion = StagnationGuard(inner=criterion, window=window,
                                        min_decay=min_decay)
        return criterion

    # ---- 运行 ---------------------------------------------------------

    def run(self, extra_budget: int = 0) -> MultiStartResult:
        """运行所有未完成路径；可选先向预算池追加预算。"""
        if extra_budget:
            if extra_budget < 0:
                raise ValueError("追加预算不能为负")
            self.total_budget += int(extra_budget)
        for path, record in zip(self._paths, self.records):
            if record.finished or self.remaining_budget <= 0:
                continue
            self._run_path(path, record)
        if self.session_dir is not None:
            self.save()
        return self.result()

    def _run_path(self, path: StartPath, record: PathRecord) -> None:
        checkpoint = self._path_checkpoint(path.label)
        criterion = self._make_criterion(record.iterations)
        try:
            solver = self.make_solver(criterion, checkpoint, path)
        except PreconditionError as exc:
            record.status = TerminationReason.PRECONDITION_FAILED.value
            record.message = str(exc)
            return
        try:
            if checkpoint and Checkpoint(checkpoint).exists():
                # 本会话上次暂停时留下的路径检查点：严格身份校验后原进度恢复
                result = solver.solve(self.problem, continuation="resume")
            elif path.restart_from is not None:
                # 旧检查点只读：受控重新开始，绝不覆盖来源
                result = solver.solve(self.problem, x0=path.x0,
                                      continuation="restart",
                                      restart_from=path.restart_from)
            else:
                result = solver.solve(self.problem, x0=path.x0)
        except PreconditionError as exc:
            record.status = TerminationReason.PRECONDITION_FAILED.value
            record.message = str(exc)
            return
        steps = max(len(result.history) - 1, 0)
        record.iterations += steps
        self.budget_used += steps
        if result.reason is TerminationReason.BUDGET_EXHAUSTED:
            # 路径预算即预算池剩余量，耗尽说明池已空：标记暂停而非终结
            record.status = "paused"
        else:
            record.status = result.reason.value
        record.message = result.message
        record.x = result.x
        record.best_residual = self.problem.residual_norm(result.x)

    def result(self) -> MultiStartResult:
        return MultiStartResult(problem_identity=self.problem.identity(),
                                budget_total=self.total_budget,
                                budget_used=self.budget_used,
                                records=list(self.records))

    # ---- 暂停 / 重开 ---------------------------------------------------

    def save(self) -> None:
        """把各路径的预算消耗、来源与结果持久化到会话目录。"""
        if self.session_dir is None:
            raise ValueError("未设置 session_dir，无法保存会话")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for index, (path, record) in enumerate(zip(self._paths, self.records)):
            if path.x0 is not None:
                arrays[f"x0_{index}"] = np.asarray(path.x0, dtype=float)
            if record.x is not None:
                arrays[f"x_{index}"] = np.asarray(record.x, dtype=float)
        payload = {
            "version": SESSION_VERSION,
            "problem": self.problem.identity(),
            "total_budget": self.total_budget,
            "budget_used": self.budget_used,
            "stagnation": self.stagnation,
            "paths": [
                {"label": p.label, "origin": p.origin,
                 "restart_from": p.restart_from,
                 "has_x0": p.x0 is not None}
                for p in self._paths
            ],
            "records": [
                {"label": r.label, "origin": r.origin, "status": r.status,
                 "iterations": r.iterations,
                 "best_residual": r.best_residual, "message": r.message,
                 "has_x": r.x is not None}
                for r in self.records
            ],
        }
        tmp_npz = self.session_dir / ".session.writing.npz"
        tmp_json = self.session_dir / "session.json.tmp"
        np.savez(tmp_npz, **arrays)
        tmp_json.write_text(json.dumps(payload), encoding="utf-8")
        tmp_npz.replace(self.session_dir / "session.npz")
        tmp_json.replace(self.session_dir / "session.json")

    @classmethod
    def load(cls, session_dir: str | Path, problem: LinearSystem,
             make_solver, criterion: StoppingCriterion | None = None
             ) -> "MultiStartSession":
        """从会话目录恢复；问题身份必须一致，各路径记录原样保留。"""
        session_dir = Path(session_dir)
        meta_path = session_dir / "session.json"
        data_path = session_dir / "session.npz"
        if not meta_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"会话记录不存在：{session_dir}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("version") != SESSION_VERSION:
            raise ValueError("会话记录格式不受支持")
        if payload.get("problem") != problem.identity():
            raise ValueError("会话记录的目标问题与当前问题不一致，不能恢复；"
                             "请针对新问题新建会话（可从旧近似解热启动）")
        with np.load(data_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        paths = []
        for index, item in enumerate(payload["paths"]):
            x0 = arrays.get(f"x0_{index}") if item["has_x0"] else None
            paths.append(StartPath(label=item["label"], x0=x0,
                                   origin=item["origin"],
                                   restart_from=item["restart_from"]))
        stagnation = payload.get("stagnation")
        session = cls(problem, make_solver, paths,
                      total_budget=payload["total_budget"],
                      criterion=criterion,
                      stagnation=tuple(stagnation) if stagnation else None,
                      session_dir=session_dir)
        session.budget_used = payload["budget_used"]
        for index, item in enumerate(payload["records"]):
            record = session.records[index]
            record.status = item["status"]
            record.iterations = item["iterations"]
            record.best_residual = item["best_residual"]
            record.message = item["message"]
            if item["has_x"]:
                record.x = arrays[f"x_{index}"]
        return session
