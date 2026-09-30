"""多起点求解：统一预算管理、路径隔离、按当前问题真实残差选择。

- 统一预算：所有路径共享一个总迭代预算，按轮分片下发；
  某条路径失败（数值异常/前提拒绝）或停滞（残差不再改善）即被隔离，
  不再消耗预算，但不影响其他路径继续。
- 结果选择：最终以各路径近似解在当前问题上的真实残差 ||b - Ax||
  为依据，而不是任何迭代内部递推值。
- 暂停/重开：工作目录中的 manifest 记录每条路径的预算、来源与结果，
  每条路径使用各自独立的检查点文件；作为起点的旧检查点只读，
  绝不覆盖。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .problems import LinearSystem, PreconditionError
from .records import SolveResult, TerminationReason


@dataclass
class StartPoint:
    """一条求解路径的起点来源。"""

    label: str
    x0: np.ndarray | None = None          # 直接给定初值
    restart_from: str | None = None       # 或从旧检查点受控重新开始（只读）

    def origin(self) -> str:
        if self.restart_from is not None:
            return f"restart_from={self.restart_from}"
        if self.x0 is not None:
            return "x0=用户给定初值"
        return "x0=零向量"


@dataclass
class PathOutcome:
    """一条路径的最终状态：预算消耗、来源、结果与当前问题真实残差。"""

    label: str
    origin: str
    status: str                # converged / budget_exhausted / failed / stagnant
    iterations: int            # 该路径实际消耗的迭代预算
    residual_norm: float       # 当前问题上的真实残差 ||b - Ax||
    x: np.ndarray | None
    message: str = ""


@dataclass
class MultiStartResult:
    problem_identity: dict
    total_budget: int
    used_budget: int
    paths: list[PathOutcome]
    best_label: str
    best_x: np.ndarray
    best_residual: float
    converged: bool
    selection_note: str = "按当前问题的真实残差 ||b-Ax|| 最小选择"


class MultiStartRunner:
    """多起点求解驱动器。

    Parameters
    ----------
    problem : 当前目标问题；所有路径的残差都以它计算。
    solver_factory : (checkpoint_path, criterion) -> IterativeSolver，
        每次调用返回全新求解器；checkpoint_path 由运行器按路径分配。
    criterion_factory : (max_iter) -> StoppingCriterion，
        每轮为路径构造该轮预算切片对应的停止条件。
    starts : StartPoint 列表，各路径起点。
    total_budget : 所有路径共享的总迭代预算。
    workdir : 工作目录，保存 manifest 与各路径独立检查点，支持暂停重开。
    slice_iter : 每轮每条路径最多获得的迭代数。
    stagnation_rounds / stagnation_rtol : 连续若干轮残差相对改善不足
        即判定停滞并隔离该路径。
    """

    MANIFEST = "multistart.json"
    FINALS = "finals.npz"

    def __init__(self, problem: LinearSystem, solver_factory,
                 criterion_factory, starts: list[StartPoint],
                 total_budget: int, workdir: str | Path,
                 slice_iter: int = 10, stagnation_rounds: int = 3,
                 stagnation_rtol: float = 1e-8):
        if total_budget <= 0:
            raise ValueError("total_budget 必须为正")
        self.problem = problem
        self.solver_factory = solver_factory
        self.criterion_factory = criterion_factory
        self.starts = starts
        self.total_budget = int(total_budget)
        self.workdir = Path(workdir)
        self.slice_iter = max(1, int(slice_iter))
        self.stagnation_rounds = max(1, int(stagnation_rounds))
        self.stagnation_rtol = float(stagnation_rtol)

    # ---- 暂停/重开：manifest 与结果快照 -------------------------------

    @property
    def _manifest_path(self) -> Path:
        return self.workdir / self.MANIFEST

    def _checkpoint_path(self, index: int) -> Path:
        return self.workdir / f"path-{index}"

    def _save_state(self, paths: list[dict]) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "problem": self.problem.identity(),
            "total_budget": self.total_budget,
            "used_budget": sum(p["used"] for p in paths),
            "paths": [{k: v for k, v in p.items() if k != "x"} for p in paths],
        }
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._manifest_path)
        arrays = {f"x{i}": p["x"] for i, p in enumerate(paths)
                  if p["x"] is not None}
        if arrays:
            np.savez(self.workdir / self.FINALS, **arrays)

    def _load_state(self) -> list[dict] | None:
        if not self._manifest_path.exists():
            return None
        meta = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if meta.get("version") != 1:
            raise ValueError("多起点运行记录格式不受支持")
        if meta.get("problem") != self.problem.identity():
            raise ValueError(
                "工作目录中的多起点记录属于另一个问题，不能在此问题上重开；"
                "请换用新的 workdir，或从旧近似解受控重新开始")
        if int(meta.get("total_budget", 0)) > self.total_budget:
            raise ValueError("新总预算小于已记录预算，无法重开")
        finals = {}
        finals_path = self.workdir / self.FINALS
        if finals_path.exists():
            with np.load(finals_path) as npz:
                finals = {k: npz[k] for k in npz.files}
        paths = []
        for index, saved in enumerate(meta["paths"]):
            path = dict(saved)
            path["x"] = finals.get(f"x{index}")
            if path["status"] == "budget_exhausted" and                     self.total_budget > int(meta.get("total_budget", 0)):
                path["status"] = "active"  # 预算调大后允许该路径继续
            paths.append(path)
        return paths

    # ---- 主循环 -------------------------------------------------------

    def run(self) -> MultiStartResult:
        paths = self._load_state()
        if paths is None:
            paths = [{
                "label": s.label,
                "origin": s.origin(),
                "status": "active",
                "used": 0,
                "iterations_done": 0,
                "residual_norm": None,
                "best_residual": None,
                "stall": 0,
                "message": "",
                "x": None,
            } for s in self.starts]

        while True:
            used_total = sum(p["used"] for p in paths)
            remaining = self.total_budget - used_total
            active = [i for i, p in enumerate(paths) if p["status"] == "active"]
            if remaining <= 0 or not active:
                break
            for i in active:
                if self.total_budget - sum(p["used"] for p in paths) <= 0:
                    break
                self._run_slice(paths, i)
                self._save_state(paths)
        if sum(p["used"] for p in paths) >= self.total_budget:
            for p in paths:
                if p["status"] == "active":
                    p["status"] = "budget_exhausted"
            self._save_state(paths)

        outcomes = []
        for p in paths:
            residual = (self.problem.residual_norm(p["x"])
                        if p["x"] is not None else None)
            outcomes.append(PathOutcome(
                label=p["label"], origin=p["origin"], status=p["status"],
                iterations=p["used"], residual_norm=residual, x=p["x"],
                message=p["message"]))
        candidates = [o for o in outcomes if o.x is not None
                      and np.isfinite(o.residual_norm)]
        if not candidates:
            raise RuntimeError("所有路径都失败，没有任何可用近似解")
        best = min(candidates, key=lambda o: o.residual_norm)
        return MultiStartResult(
            problem_identity=self.problem.identity(),
            total_budget=self.total_budget,
            used_budget=sum(p["used"] for p in paths),
            paths=outcomes, best_label=best.label, best_x=best.x,
            best_residual=best.residual_norm,
            converged=best.status == "converged")

    def _run_slice(self, paths: list[dict], index: int) -> None:
        path = paths[index]
        remaining = self.total_budget - sum(p["used"] for p in paths)
        budget = min(self.slice_iter, remaining)
        if budget <= 0:
            return
        # 迭代号跨轮累计，切片预算要换算成累计 max_iter
        criterion = self.criterion_factory(path["iterations_done"] + budget)
        checkpoint = str(self._checkpoint_path(index))
        solver = self.solver_factory(checkpoint, criterion)
        start = self.starts[index]
        try:
            if path["iterations_done"] == 0 and start.restart_from is not None:
                result = solver.solve(self.problem, continuation="restart",
                                      restart_from=start.restart_from)
            elif path["iterations_done"] == 0:
                result = solver.solve(self.problem, x0=start.x0)
            else:
                result = solver.solve(self.problem)  # 从本路径检查点原进度恢复
        except PreconditionError as exc:
            path["status"] = "failed"
            path["message"] = f"前提检查拒绝：{exc}"
            return
        except Exception as exc:  # 路径隔离：任何失败都不影响其他路径
            path["status"] = "failed"
            path["message"] = f"路径异常：{type(exc).__name__}: {exc}"
            return

        new_iterations = result.iterations - path["iterations_done"]
        path["iterations_done"] = result.iterations
        path["used"] += new_iterations
        path["x"] = result.x.copy()
        residual = self.problem.residual_norm(result.x)  # 当前问题真实残差
        path["residual_norm"] = residual
        path["message"] = result.message

        if result.reason is TerminationReason.CONVERGED:
            path["status"] = "converged"
            return
        if result.reason is TerminationReason.NUMERICAL_BREAKDOWN:
            path["status"] = "failed"
            return
        # 预算切片用尽：检查停滞
        best = path["best_residual"]
        if best is not None and residual >= best * (1.0 - self.stagnation_rtol):
            path["stall"] += 1
        else:
            path["stall"] = 0
        if best is None or residual < best:
            path["best_residual"] = residual
        if path["stall"] >= self.stagnation_rounds:
            path["status"] = "stagnant"
            path["message"] = (f"连续 {path['stall']} 轮残差相对改善不足 "
                               f"{self.stagnation_rtol:.1e}，判定停滞并隔离")
