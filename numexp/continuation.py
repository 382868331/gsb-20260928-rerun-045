"""参数连续跟踪：逐步改变参数、失败回退、统一预算与多路径比较。

设计要点：

* 参数跟踪：用户给出参数起点 t_start 与终点 t_end 以及问题族
  problem_at(t)；每条方案（scheme）从已确认节点出发，以上一节点
  的解作为热启动推进到下一个参数值，沿途记录解的变化。
* 失败回退：某一步未收敛（预算上限内未达精度、数值异常、前提拒绝）
  时，记录这次失败尝试及其实际消耗，然后缩小步长，从上一个已确认
  节点重新出发；失败尝试的残差与近似解只入日志，绝不作为下一步
  的起点或收敛结果。
* 统一预算：成功步与失败尝试的迭代消耗都从同一个预算池扣减；
  池用尽时方案标记为 paused，追加预算后沿原路径继续。
* 多路径：可从任一已确认节点另开方案（fork），原节点与预算记录
  不被覆盖；compare_at(t) 用 problem_at(t) 重算真实残差，
  比较不同方案在同一参数处的解。
* 未完成不替代：方案只有确认到达 t_end 才算完成；
  solution_at(t) 在没有确认节点时返回 None，不以最近结果代替。
* 起点需确认：参数起点 t_start 处的初始解不直接算作已确认节点；
  首次运行时按 problem_at(t_start) 实际求解验证，消耗计入统一预算，
  只有达到精度才成为已确认节点；未确认的起点在暂停重开后仍需验证，
  不会因保存接续而跳过验证或重复扣减。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .criteria import RelativeResidual, StoppingCriterion
from .problems import LinearSystem, PreconditionError
from .records import TerminationReason

SESSION_VERSION = 1


@dataclass
class ContinuationNode:
    """一个已确认的参数节点：解在该参数处达到精度要求。"""

    index: int
    t: float
    x: np.ndarray
    iterations: int                   # 到达该节点那一次成功推进的消耗
    origin: str                       # 起点来源（fresh / fork:* / node:*）


@dataclass
class ContinuationAttempt:
    """一次推进尝试的记录，成功与失败都保留实际参数与消耗。"""

    scheme: str
    t_from: float
    t_target: float
    step: float
    iterations: int                   # 本次尝试实际消耗的迭代步数
    status: str                       # converged / 失败原因 / paused
    residual: float                   # 尝试结束时的真实残差（失败仅记录）
    message: str = ""

    @property
    def converged(self) -> bool:
        return self.status == TerminationReason.CONVERGED.value


@dataclass
class ContinuationScheme:
    """一条参数跟踪方案：已确认节点序列与当前步长。"""

    label: str
    t_end: float
    nodes: list = field(default_factory=list)
    step: float = 0.0                 # 下一步将尝试的步长
    status: str = "active"            # active / completed / stalled / paused
    pending: dict | None = None       # 待确认起点 {"t", "x", "origin"}；None 表示已确认

    @property
    def current(self) -> ContinuationNode:
        if not self.nodes:
            raise ValueError("方案 " + self.label + " 的起点尚未确认，没有当前节点")
        return self.nodes[-1]

    @property
    def reached_end(self) -> bool:
        return bool(self.nodes) and self.nodes[-1].t == self.t_end

    def solution_at(self, t: float):
        """参数 t 处已确认的解；没有确认节点时返回 None（不以最近代替）。"""
        for node in self.nodes:
            if node.t == t:
                return node.x
        return None


@dataclass
class ContinuationResult:
    """一次运行的快照：预算、方案与全部尝试记录。"""

    budget_total: int
    budget_used: int
    schemes: list
    attempts: list

    @property
    def completed(self) -> list:
        return [s for s in self.schemes if s.reached_end]

    @property
    def incomplete(self) -> list:
        return [s for s in self.schemes if not s.reached_end]


class ContinuationSession:
    """一次参数跟踪会话，支持暂停重开与从旧节点另开方案。

    Parameters
    ----------
    problem_at : callable(t) -> LinearSystem，参数化问题族。
    t_start, t_end : 参数起点与终点（t_start < t_end）。
    make_solver : callable(criterion) -> IterativeSolver，每次返回全新
        求解器实例；criterion 由会话按剩余预算配置。
    total_budget : 所有方案共享的迭代预算池，成功步与失败尝试都扣减。
    criterion : 精度停止条件模板；其 max_iter 由会话按本次尝试的
        可用预算覆盖。
    initial_step, min_step : 初始步长与最小步长；失败回退时步长减半，
        低于 min_step 仍失败则方案标记为 stalled。
    max_step_iters : 单次推进允许的最大迭代数（None 表示不限，只受
        预算池约束）；用于给后续参数预留预算。
    x0 : t_start 处的初始解（None 表示零向量）。
    session_dir : 会话目录；给定后每次运行结束持久化，可暂停重开。
    """

    def __init__(self, problem_at, t_start: float, t_end: float,
                 make_solver, total_budget: int,
                 criterion: StoppingCriterion | None = None,
                 initial_step: float | None = None,
                 min_step: float | None = None,
                 max_step_iters: int | None = None,
                 x0: np.ndarray | None = None,
                 session_dir: str | Path | None = None):
        if not t_start < t_end:
            raise ValueError("参数起点必须小于终点")
        if total_budget <= 0:
            raise ValueError("统一预算必须为正整数")
        self.problem_at = problem_at
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.make_solver = make_solver
        self.criterion = criterion or RelativeResidual()
        self.total_budget = int(total_budget)
        self.budget_used = 0
        self.initial_step = float(initial_step) if initial_step else (self.t_end - self.t_start) / 4.0
        self.min_step = float(min_step) if min_step else self.initial_step / 64.0
        self.max_step_iters = max_step_iters
        self.session_dir = Path(session_dir) if session_dir else None
        self.attempts: list = []
        start_x = np.zeros(problem_at(self.t_start).n) if x0 is None else np.asarray(x0, dtype=float)
        # 起点只是待确认初值：首次运行时按 t_start 处的问题实际求解，
        # 达到精度才列为已确认节点；求解消耗计入统一预算。
        pending = {"t": self.t_start, "x": start_x, "origin": "fresh"}
        self.schemes = [ContinuationScheme(label="main", t_end=self.t_end,
                                           nodes=[], step=self.initial_step,
                                           pending=pending)]

    # ---- 预算 ---------------------------------------------------------

    @property
    def remaining_budget(self) -> int:
        return self.total_budget - self.budget_used

    # ---- 方案管理 -----------------------------------------------------

    def fork_scheme(self, source_label: str, node_index: int,
                    new_label: str) -> ContinuationScheme:
        """从某方案已确认节点另开方案；原节点与预算记录不被覆盖。"""
        if any(s.label == new_label for s in self.schemes):
            raise ValueError("方案名已存在：" + new_label)
        source = self._scheme(source_label)
        if not 0 <= node_index < len(source.nodes):
            raise IndexError("节点序号越界：" + str(node_index))
        anchor = source.nodes[node_index]
        root = ContinuationNode(index=0, t=anchor.t, x=np.array(anchor.x, copy=True),
                                iterations=0,
                                origin="fork:" + source.label + "#node" + str(node_index))
        scheme = ContinuationScheme(label=new_label, t_end=self.t_end,
                                    nodes=[root], step=self.initial_step)
        self.schemes.append(scheme)
        return scheme

    def _scheme(self, label: str) -> ContinuationScheme:
        for scheme in self.schemes:
            if scheme.label == label:
                return scheme
        raise KeyError("未知方案：" + label)

    # ---- 运行 ---------------------------------------------------------

    def run(self, extra_budget: int = 0) -> ContinuationResult:
        """推进所有未完成方案；可选先向预算池追加预算。"""
        if extra_budget:
            if extra_budget < 0:
                raise ValueError("追加预算不能为负")
            self.total_budget += int(extra_budget)
        for scheme in self.schemes:
            if scheme.status == "paused":
                scheme.status = "active"
            while scheme.status == "active" and self.remaining_budget > 0:
                self._advance(scheme)
        if self.session_dir is not None:
            self.save()
        return self.result()

    def _advance(self, scheme: ContinuationScheme) -> None:
        if scheme.pending is not None:
            self._confirm_start(scheme)
            return
        node = scheme.current
        if node.t == self.t_end:
            scheme.status = "completed"
            return
        step = min(scheme.step, self.t_end - node.t)
        t_target = node.t + step
        allowed = self.remaining_budget
        if self.max_step_iters is not None:
            allowed = min(allowed, self.max_step_iters)
        if allowed <= 0:
            scheme.status = "paused"
            return
        problem = self.problem_at(t_target)
        criterion = copy.copy(self.criterion)
        criterion.max_iter = allowed
        try:
            solver = self.make_solver(criterion)
            outcome = solver.solve(problem, x0=node.x)
            iterations = outcome.iterations
            status = outcome.reason.value
            residual = problem.residual_norm(outcome.x)
            message = outcome.message
            success = outcome.success
        except PreconditionError as exc:
            iterations, residual, message = 0, np.inf, str(exc)
            status = TerminationReason.PRECONDITION_FAILED.value
            success = False
        self.budget_used += iterations
        pool_empty = self.remaining_budget <= 0
        if not success and pool_empty:
            # 预算池在本次尝试中耗尽：标记暂停，追加预算后从已确认节点重试
            status = "paused"
        self.attempts.append(ContinuationAttempt(
            scheme=scheme.label, t_from=node.t, t_target=t_target, step=step,
            iterations=iterations, status=status, residual=residual,
            message=message))
        if success:
            scheme.nodes.append(ContinuationNode(
                index=len(scheme.nodes), t=t_target, x=outcome.x,
                iterations=iterations,
                origin="node:" + str(node.index) + "@" + scheme.label))
            scheme.step = min(step * 1.5, self.t_end - self.t_start)
            if t_target == self.t_end:
                scheme.status = "completed"
        elif status == "paused":
            scheme.status = "paused"
        else:
            # 失败回退：缩小步长，从上一个已确认节点重新出发；
            # 失败尝试的残差只留在 attempts 日志，不作为下一步起点
            scheme.step = step * 0.5
            if scheme.step < self.min_step:
                scheme.status = "stalled"

    def _confirm_start(self, scheme: ContinuationScheme) -> None:
        """求解 t_start 处的问题以确认起点；只有达到精度才列为已确认节点。

        消耗计入统一预算并记录尝试；预算不足标记 paused（追加预算后续验），
        其余失败标记 stalled，失败依据保留在 attempts 日志中。
        """
        pending = scheme.pending
        t = pending["t"]
        allowed = self.remaining_budget
        if self.max_step_iters is not None:
            allowed = min(allowed, self.max_step_iters)
        if allowed <= 0:
            scheme.status = "paused"
            return
        problem = self.problem_at(t)
        criterion = copy.copy(self.criterion)
        criterion.max_iter = allowed
        try:
            solver = self.make_solver(criterion)
            outcome = solver.solve(problem, x0=pending["x"])
            iterations = outcome.iterations
            status = outcome.reason.value
            residual = problem.residual_norm(outcome.x)
            message = outcome.message
            success = outcome.success
        except PreconditionError as exc:
            iterations, residual, message = 0, np.inf, str(exc)
            status = TerminationReason.PRECONDITION_FAILED.value
            success = False
        self.budget_used += iterations
        if not success and self.remaining_budget <= 0:
            status = "paused"
        self.attempts.append(ContinuationAttempt(
            scheme=scheme.label, t_from=t, t_target=t, step=0.0,
            iterations=iterations, status=status, residual=residual,
            message=message))
        if success:
            scheme.nodes.append(ContinuationNode(
                index=0, t=t, x=outcome.x, iterations=iterations,
                origin=pending["origin"]))
            scheme.pending = None
        elif status == "paused":
            scheme.status = "paused"
        else:
            scheme.status = "stalled"

    # ---- 查询 ---------------------------------------------------------

    def result(self) -> ContinuationResult:
        return ContinuationResult(budget_total=self.total_budget,
                                  budget_used=self.budget_used,
                                  schemes=list(self.schemes),
                                  attempts=list(self.attempts))

    def compare_at(self, t: float) -> list:
        """比较各方案在同一参数处的解；真实残差按 problem_at(t) 重算。

        只统计在该参数处有已确认节点的方案；未到达该参数的方案不出现在
        结果中（不可用最近节点代替）。
        """
        problem = self.problem_at(t)
        entries = []
        for scheme in self.schemes:
            x = scheme.solution_at(t)
            if x is None:
                continue
            entries.append({"scheme": scheme.label, "t": t, "x": x,
                            "residual": problem.residual_norm(x)})
        return entries

    def final_solution(self, label: str):
        """方案到达终点时的解；未到达终点返回 None，不以最近结果代替。"""
        scheme = self._scheme(label)
        return scheme.solution_at(self.t_end)

    def verify_node(self, label: str, node_index: int) -> float:
        """按节点自身参数重算真实残差，独立复核该确认节点。"""
        scheme = self._scheme(label)
        if not 0 <= node_index < len(scheme.nodes):
            raise IndexError("节点序号越界：" + str(node_index))
        node = scheme.nodes[node_index]
        return self.problem_at(node.t).residual_norm(node.x)

    # ---- 暂停 / 重开 ---------------------------------------------------

    def _identity(self) -> dict:
        return {"t_start": self.t_start, "t_end": self.t_end,
                "start_problem": self.problem_at(self.t_start).identity(),
                "end_problem": self.problem_at(self.t_end).identity()}

    def save(self) -> None:
        """持久化各方案的已确认节点、预算消耗与全部尝试记录。"""
        if self.session_dir is None:
            raise ValueError("未设置 session_dir，无法保存会话")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for s_idx, scheme in enumerate(self.schemes):
            for node in scheme.nodes:
                arrays["x_%d_%d" % (s_idx, node.index)] = np.asarray(node.x, dtype=float)
            if scheme.pending is not None:
                arrays["pending_%d" % s_idx] = np.asarray(scheme.pending["x"], dtype=float)
        payload = {
            "version": SESSION_VERSION,
            "identity": self._identity(),
            "total_budget": self.total_budget,
            "budget_used": self.budget_used,
            "initial_step": self.initial_step,
            "min_step": self.min_step,
            "max_step_iters": self.max_step_iters,
            "schemes": [
                {"label": s.label, "step": s.step, "status": s.status,
                 "pending": (None if s.pending is None else
                             {"t": s.pending["t"], "origin": s.pending["origin"]}),
                 "nodes": [{"index": n.index, "t": n.t,
                            "iterations": n.iterations, "origin": n.origin}
                           for n in s.nodes]}
                for s in self.schemes
            ],
            "attempts": [
                {"scheme": a.scheme, "t_from": a.t_from, "t_target": a.t_target,
                 "step": a.step, "iterations": a.iterations, "status": a.status,
                 "residual": a.residual, "message": a.message}
                for a in self.attempts
            ],
        }
        tmp_npz = self.session_dir / ".continuation.writing.npz"
        tmp_json = self.session_dir / "continuation.json.tmp"
        np.savez(tmp_npz, **arrays)
        tmp_json.write_text(json.dumps(payload), encoding="utf-8")
        tmp_npz.replace(self.session_dir / "continuation.npz")
        tmp_json.replace(self.session_dir / "continuation.json")

    @classmethod
    def load(cls, session_dir, problem_at, make_solver,
             criterion: StoppingCriterion | None = None) -> "ContinuationSession":
        """从会话目录恢复；参数区间与端点问题身份必须一致。

        恢复后继续原路径：已确认节点、预算消耗与尝试日志原样保留，
        旧记录不被覆盖。
        """
        session_dir = Path(session_dir)
        meta_path = session_dir / "continuation.json"
        data_path = session_dir / "continuation.npz"
        if not meta_path.exists() or not data_path.exists():
            raise FileNotFoundError("会话记录不存在：" + str(session_dir))
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("version") != SESSION_VERSION:
            raise ValueError("会话记录格式不受支持")
        identity = payload["identity"]
        session = cls(problem_at, identity["t_start"], identity["t_end"],
                      make_solver, total_budget=payload["total_budget"],
                      criterion=criterion,
                      initial_step=payload["initial_step"],
                      min_step=payload["min_step"],
                      max_step_iters=payload.get("max_step_iters"),
                      session_dir=session_dir)
        if session._identity() != identity:
            raise ValueError("会话记录的参数区间或端点问题与当前不一致，"
                             "不能恢复；请针对新问题族新建会话")
        with np.load(data_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        session.budget_used = payload["budget_used"]
        session.schemes = []
        for s_idx, item in enumerate(payload["schemes"]):
            nodes = [ContinuationNode(index=n["index"], t=n["t"],
                                      x=arrays["x_%d_%d" % (s_idx, n["index"])],
                                      iterations=n["iterations"],
                                      origin=n["origin"])
                     for n in item["nodes"]]
            pending_info = item.get("pending")
            pending = None
            if pending_info is not None:
                pending = {"t": pending_info["t"],
                           "x": arrays["pending_%d" % s_idx],
                           "origin": pending_info["origin"]}
            session.schemes.append(ContinuationScheme(
                label=item["label"], t_end=session.t_end, nodes=nodes,
                step=item["step"], status=item["status"], pending=pending))
        session.attempts = [ContinuationAttempt(**a) for a in payload["attempts"]]
        return session
