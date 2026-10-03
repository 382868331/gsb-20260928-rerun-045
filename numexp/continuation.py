"""参数连续跟踪：逐步改变参数观察解的变化，失败回退缩小步长。

设计要点：

* 参数路径：用户指定参数起点与终点，会话自适应地把参数从当前
  已确认节点向前推进，每个已确认节点都保存实际参数、近似解、
  起点来源与该步实际消耗。
* 失败回退：某一步未收敛（预算内未达到精度、前提拒绝、数值异常）
  时，回到上一个已确认节点，把推进幅度减半重试；失败尝试只记录
  消耗，绝不作为已收敛结果进入节点序列。
* 统一预算：总预算覆盖成功步与失败尝试；池用尽时会话暂停，
  追加预算后沿原路径继续。
* 多方案比较：可从任一已确认节点另开方案（branch），原节点与
  预算记录不被覆盖；多条路径到达同一参数时按真实残差比较。
  未到达终点的路径明确标记为未完成，不以最近节点冒充终点结果。
* 暂停/重开：会话状态（各路径节点、尝试记录、预算消耗）持久化
  到会话目录，重开后继续原路径。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .criteria import RelativeResidual, StoppingCriterion
from .problems import PreconditionError
from .records import TerminationReason

SESSION_VERSION = 1


@dataclass
class ContinuationNode:
    """参数路径上一个已确认（收敛）的节点。"""

    param: float
    x: np.ndarray
    residual_norm: float      # 按该参数对应问题重算的真实残差
    source: str               # 起点来源（initial / node:param / branch:...）
    cost: int                 # 到达该节点的成功求解消耗的迭代步数


@dataclass
class AttemptRecord:
    """一次推进尝试的记录，无论成败都计入预算。"""

    param: float              # 实际尝试的目标参数
    origin: str               # 本次起点来源
    step: float               # 本次推进幅度（带符号）
    cost: int                 # 实际消耗的迭代步数
    status: str               # converged / budget_exhausted / ...
    accepted: bool            # 只有收敛才为 True；失败残差不进入节点


@dataclass
class ContinuationPath:
    """一条参数推进方案：节点序列、尝试记录与自适应步长。"""

    label: str
    target: float
    step: float                        # 当前推进幅度（绝对值）
    status: str = "active"             # active / paused / stalled / completed
    nodes: list[ContinuationNode] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    pending_x0: np.ndarray | None = None   # 尚无节点时的初始起点
    pending_origin: str = "initial"

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def current(self) -> ContinuationNode | None:
        return self.nodes[-1] if self.nodes else None

    @property
    def final_node(self) -> ContinuationNode | None:
        """终点节点；未到达终点时返回 None，不以最近节点代替。"""
        return self.nodes[-1] if self.completed else None

    @property
    def total_cost(self) -> int:
        return sum(a.cost for a in self.attempts)


@dataclass
class ContinuationResult:
    """一次运行的快照：预算与各路径状态。"""

    param_start: float
    param_end: float
    budget_total: int
    budget_used: int
    paths: list[ContinuationPath]

    @property
    def completed_paths(self) -> list[ContinuationPath]:
        return [p for p in self.paths if p.completed]


class ContinuationSession:
    """参数连续跟踪会话。

    Parameters
    ----------
    make_problem : callable(param) -> LinearSystem，按参数构造问题。
    make_solver : callable(criterion) -> 全新迭代求解器实例。
    param_start, param_end : 参数起点与终点。
    total_budget : 全部路径共享的迭代预算池（成功步与失败尝试都扣减）。
    criterion : 精度停止条件模板；max_iter 由会话按单次尝试预算覆盖。
    attempt_budget : 单次推进尝试的迭代上限，避免一次失败耗尽预算池。
    initial_step : 初始推进幅度；默认直接尝试一步到达终点。
    min_step : 最小推进幅度；小于它仍不收敛则路径标记 stalled。
    growth : 成功后步长放大倍数。
    x0 : 参数起点处的初始猜测（默认零向量）。
    session_dir : 会话目录；给定后每次运行结束持久化，可暂停重开。
    """

    def __init__(self, make_problem, make_solver,
                 param_start: float, param_end: float,
                 total_budget: int,
                 criterion: StoppingCriterion | None = None,
                 attempt_budget: int = 200,
                 initial_step: float | None = None,
                 min_step: float | None = None,
                 growth: float = 2.0,
                 x0: np.ndarray | None = None,
                 session_dir: str | Path | None = None):
        if total_budget <= 0:
            raise ValueError("统一预算必须为正整数")
        if attempt_budget <= 0:
            raise ValueError("单次尝试预算必须为正整数")
        if param_start == param_end:
            raise ValueError("参数起点与终点不能相同")
        self.make_problem = make_problem
        self.make_solver = make_solver
        self.param_start = float(param_start)
        self.param_end = float(param_end)
        self.total_budget = int(total_budget)
        self.budget_used = 0
        self.criterion = criterion or RelativeResidual()
        self.attempt_budget = int(attempt_budget)
        interval = abs(self.param_end - self.param_start)
        self.initial_step = float(initial_step) if initial_step else interval
        self.min_step = (float(min_step) if min_step
                         else interval * 1e-8)
        self.growth = float(growth)
        self.session_dir = Path(session_dir) if session_dir else None
        self.paths: list[ContinuationPath] = []
        main = ContinuationPath(label="main", target=self.param_end,
                                step=self.initial_step)
        main.pending_x0 = (None if x0 is None
                           else np.asarray(x0, dtype=float))
        self.paths.append(main)

    # ---- 预算 ---------------------------------------------------------

    @property
    def remaining_budget(self) -> int:
        return self.total_budget - self.budget_used

    def _make_criterion(self) -> StoppingCriterion:
        criterion = copy.copy(self.criterion)
        criterion.max_iter = min(self.attempt_budget, self.remaining_budget)
        return criterion

    # ---- 推进 ---------------------------------------------------------

    def run(self, extra_budget: int = 0) -> ContinuationResult:
        """推进所有活动路径；可选先向预算池追加预算。"""
        if extra_budget:
            if extra_budget < 0:
                raise ValueError("追加预算不能为负")
            self.total_budget += int(extra_budget)
            for path in self.paths:
                if path.status == "paused":
                    path.status = "active"
        while self.remaining_budget > 0:
            active = [p for p in self.paths if p.status == "active"]
            if not active:
                break
            for path in active:
                if self.remaining_budget <= 0:
                    break
                self._advance(path)
        if self.session_dir is not None:
            self.save()
        return self.result()

    def _advance(self, path: ContinuationPath) -> None:
        node = path.current
        if node is None:
            base_param, base_x, origin = (self.param_start,
                                          path.pending_x0,
                                          path.pending_origin)
            target_param = self.param_start
        else:
            base_param, base_x = node.param, node.x
            origin = f"node:{base_param!r}"
            direction = 1.0 if path.target > base_param else -1.0
            target_param = base_param + direction * path.step
            # 不越过终点
            if (direction > 0) == (path.target > base_param):
                overshoot = ((target_param - path.target) * direction > 0)
                if overshoot:
                    target_param = path.target
        step_signed = target_param - base_param
        problem = self.make_problem(target_param)
        criterion = self._make_criterion()
        try:
            solver = self.make_solver(criterion)
            result = solver.solve(problem, x0=base_x)
            cost = max(len(result.history) - 1, 0)
            status = result.reason.value
            x_new = result.x
        except PreconditionError as exc:
            cost, status, x_new = 0, TerminationReason.PRECONDITION_FAILED.value, None
            result = None
        self.budget_used += cost
        accepted = status == TerminationReason.CONVERGED.value
        path.attempts.append(AttemptRecord(
            param=target_param, origin=origin, step=step_signed,
            cost=cost, status=status, accepted=accepted))
        if accepted:
            residual = problem.residual_norm(x_new)
            path.nodes.append(ContinuationNode(
                param=target_param, x=x_new, residual_norm=residual,
                source=origin, cost=cost))
            if target_param == path.target:
                path.status = "completed"
            else:
                path.step = min(path.step * self.growth,
                                abs(path.target - target_param))
            return
        # 失败：结果不进入节点序列
        if self.remaining_budget <= 0:
            path.status = "paused"      # 预算池用尽：暂停，可追加预算续跑
            return
        path.step = abs(step_signed) / 2.0
        if path.step < self.min_step:
            path.status = "stalled"     # 步长已小于下限仍不收敛

    # ---- 分支与比较 ----------------------------------------------------

    def branch(self, source_label: str, node_index: int, label: str,
               initial_step: float | None = None,
               target: float | None = None) -> ContinuationPath:
        """从已有路径的某个已确认节点另开方案；原路径记录不被修改。"""
        if any(p.label == label for p in self.paths):
            raise ValueError(f"方案名已存在：{label}")
        source = next((p for p in self.paths if p.label == source_label), None)
        if source is None:
            raise ValueError(f"找不到来源路径：{source_label}")
        node = source.nodes[node_index]
        path = ContinuationPath(
            label=label,
            target=self.param_end if target is None else float(target),
            step=initial_step or self.initial_step)
        path.nodes.append(ContinuationNode(
            param=node.param, x=np.array(node.x, copy=True),
            residual_norm=node.residual_norm,
            source=f"branch:{source_label}#{node_index}", cost=0))
        self.paths.append(path)
        return path

    def compare_at(self, param: float, tol: float = 1e-12) -> list[dict]:
        """比较各路径到达同一参数时的解；残差按该参数的问题重算。"""
        problem = self.make_problem(param)
        entries = []
        for path in self.paths:
            for node in path.nodes:
                if abs(node.param - param) <= tol:
                    entries.append({
                        "path": path.label,
                        "param": node.param,
                        "x": node.x,
                        "residual_norm": problem.residual_norm(node.x),
                        "source": node.source,
                        "cost": node.cost,
                    })
        return entries

    def result(self) -> ContinuationResult:
        return ContinuationResult(param_start=self.param_start,
                                  param_end=self.param_end,
                                  budget_total=self.total_budget,
                                  budget_used=self.budget_used,
                                  paths=list(self.paths))

    # ---- 暂停 / 重开 ---------------------------------------------------

    def save(self) -> None:
        if self.session_dir is None:
            raise ValueError("未设置 session_dir，无法保存会话")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        arrays = {}
        paths_payload = []
        for pi, path in enumerate(self.paths):
            if path.pending_x0 is not None:
                arrays[f"pending_{pi}"] = np.asarray(path.pending_x0, float)
            nodes = []
            for ni, node in enumerate(path.nodes):
                arrays[f"node_{pi}_{ni}"] = np.asarray(node.x, float)
                nodes.append({"param": node.param,
                              "residual_norm": node.residual_norm,
                              "source": node.source, "cost": node.cost})
            attempts = [{"param": a.param, "origin": a.origin,
                         "step": a.step, "cost": a.cost,
                         "status": a.status, "accepted": a.accepted}
                        for a in path.attempts]
            paths_payload.append({
                "label": path.label, "target": path.target,
                "step": path.step, "status": path.status,
                "pending_origin": path.pending_origin,
                "has_pending_x0": path.pending_x0 is not None,
                "nodes": nodes, "attempts": attempts})
        payload = {
            "version": SESSION_VERSION,
            "param_start": self.param_start, "param_end": self.param_end,
            "total_budget": self.total_budget,
            "budget_used": self.budget_used,
            "attempt_budget": self.attempt_budget,
            "initial_step": self.initial_step,
            "min_step": self.min_step, "growth": self.growth,
            "paths": paths_payload,
        }
        tmp_npz = self.session_dir / ".session.writing.npz"
        tmp_json = self.session_dir / "session.json.tmp"
        np.savez(tmp_npz, **arrays)
        tmp_json.write_text(json.dumps(payload), encoding="utf-8")
        tmp_npz.replace(self.session_dir / "session.npz")
        tmp_json.replace(self.session_dir / "session.json")

    @classmethod
    def load(cls, session_dir: str | Path, make_problem, make_solver,
             criterion: StoppingCriterion | None = None
             ) -> "ContinuationSession":
        """从会话目录恢复；参数区间必须一致，各路径记录原样保留。"""
        session_dir = Path(session_dir)
        meta_path = session_dir / "session.json"
        data_path = session_dir / "session.npz"
        if not meta_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"会话记录不存在：{session_dir}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("version") != SESSION_VERSION:
            raise ValueError("会话记录格式不受支持")
        with np.load(data_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        session = cls(make_problem, make_solver,
                      param_start=payload["param_start"],
                      param_end=payload["param_end"],
                      total_budget=payload["total_budget"],
                      criterion=criterion,
                      attempt_budget=payload["attempt_budget"],
                      initial_step=payload["initial_step"],
                      min_step=payload["min_step"],
                      growth=payload["growth"],
                      session_dir=session_dir)
        session.budget_used = payload["budget_used"]
        session.paths = []
        for pi, item in enumerate(payload["paths"]):
            path = ContinuationPath(label=item["label"],
                                    target=item["target"],
                                    step=item["step"],
                                    status=item["status"])
            path.pending_origin = item["pending_origin"]
            if item["has_pending_x0"]:
                path.pending_x0 = arrays[f"pending_{pi}"]
            for ni, n in enumerate(item["nodes"]):
                path.nodes.append(ContinuationNode(
                    param=n["param"], x=arrays[f"node_{pi}_{ni}"],
                    residual_norm=n["residual_norm"],
                    source=n["source"], cost=n["cost"]))
            for a in item["attempts"]:
                path.attempts.append(AttemptRecord(
                    param=a["param"], origin=a["origin"], step=a["step"],
                    cost=a["cost"], status=a["status"],
                    accepted=a["accepted"]))
            session.paths.append(path)
        return session
