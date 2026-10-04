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
import math
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

    # ---- 节点核验报告 --------------------------------------------------

    def verification_report(self) -> dict:
        """生成可导出的节点核验报告（dict，可直接 json 序列化）。

        每个已确认节点独立按其自身参数问题重算真实残差，不信任求解时
        自报值；尚未确认成功的初始猜测与暂停/失败尝试单独成节，
        绝不混入已确认节点。预算按起点校正、后续推进、失败尝试三类汇总。
        """
        criterion_sig = self.criterion.signature()
        tolerance = _extract_tolerance(self.criterion)

        schemes_section = []
        for scheme in self.schemes:
            nodes_section = []
            for node in scheme.nodes:
                entry = _verified_node_entry(scheme, node, self.problem_at,
                                             tolerance, criterion_sig)
                nodes_section.append(entry)

            unconfirmed = None
            if scheme.pending is not None:
                t0 = scheme.pending["t"]
                x0 = scheme.pending["x"]
                problem = self.problem_at(t0)
                unconfirmed = {
                    "scheme": scheme.label,
                    "t": _json_number(t0),
                    "origin": scheme.pending["origin"],
                    "residual": _json_number(problem.residual_norm(x0)),
                    "status": scheme.status,
                }

            start_iters, advance_iters, failed_iters = _partition_budget(
                scheme, self.attempts, self.t_start)
            schemes_section.append({
                "label": scheme.label,
                "status": scheme.status,
                "stopped_at": _stopped_at(scheme, self.t_start, self.t_end),
                "confirmed_nodes": nodes_section,
                "unconfirmed_start": unconfirmed,
                "budget": {
                    "start_confirmation": start_iters,
                    "advances": advance_iters,
                    "failed_attempts": failed_iters,
                    "total": start_iters + advance_iters + failed_iters,
                },
            })

        paused_attempts = [_attempt_entry(a) for a in self.attempts
                           if a.status == "paused"]
        failed_attempts = [_attempt_entry(a) for a in self.attempts
                           if not a.converged and a.status != "paused"]

        start_budget = sum(s["budget"]["start_confirmation"]
                           for s in schemes_section)
        advance_budget = sum(s["budget"]["advances"] for s in schemes_section)
        failed_budget = sum(s["budget"]["failed_attempts"]
                            for s in schemes_section)

        return {
            "report_type": "continuation_node_verification",
            "version": SESSION_VERSION,
            "parameter_range": {
                "t_start": _json_number(self.t_start),
                "t_end": _json_number(self.t_end),
            },
            "criterion": criterion_sig,
            "tolerance": tolerance,
            "budget": {
                "total": int(self.total_budget),
                "used": int(self.budget_used),
                "remaining": int(self.remaining_budget),
                "start_confirmation": int(start_budget),
                "advances": int(advance_budget),
                "failed_attempts": int(failed_budget),
            },
            "schemes": schemes_section,
            "paused_attempts": paused_attempts,
            "failed_attempts": failed_attempts,
        }

    def export_report_json(self, path: str | Path | None = None,
                           indent: int = 2) -> str:
        """导出 JSON 核验报告；给定 path 时写入文件，返回 JSON 文本。"""
        text = json.dumps(self.verification_report(), ensure_ascii=False,
                          indent=indent, allow_nan=True, sort_keys=True)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        return text

    def export_report_text(self, path: str | Path | None = None) -> str:
        """导出简洁文本核验报告；给定 path 时写入文件，返回文本。"""
        text = _format_report_text(self.verification_report())
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        return text

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


# ---- 核验报告辅助函数 --------------------------------------------------

def _find_relative_residual(criterion) -> RelativeResidual | None:
    """递归找出停止条件中实际生效的相对残差容差（支持组合/停滞包装）。"""
    if isinstance(criterion, RelativeResidual):
        return criterion
    inner = getattr(criterion, "inner", None)
    if inner is not None:
        found = _find_relative_residual(inner)
        if found is not None:
            return found
    for cond in getattr(criterion, "conditions", ()):
        found = _find_relative_residual(cond)
        if found is not None:
            return found
    return None


def _extract_tolerance(criterion) -> dict:
    """报告使用的容差设定；非 RelativeResidual 条件时 tol 记为 None。"""
    rel = _find_relative_residual(criterion)
    if rel is not None:
        return {"kind": "relative_residual",
                "definition": "||b - A x||_2 / ||b||_2 <= tol",
                "tol": _json_number(rel.tol)}
    return {"kind": type(criterion).__name__, "tol": None}


def _json_number(value):
    """转为 JSON 安全数值；非有限值（inf/NaN）记为 None。"""
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _verified_node_entry(scheme, node, problem_at, tolerance, criterion_sig):
    """按节点自身参数独立重算残差并给出核验结论。"""
    problem = problem_at(node.t)
    residual = problem.residual_norm(node.x)
    b_norm = float(np.linalg.norm(problem.b))
    scale = b_norm if b_norm > 0.0 else 1.0
    tol = tolerance.get("tol")
    if tol is not None:
        threshold = tol * scale
        relative = residual / scale if scale else residual
        passed = bool(math.isfinite(residual) and residual <= threshold)
    else:
        threshold = None
        relative = None
        passed = bool(math.isfinite(residual))
    return {
        "index": int(node.index),
        "t": _json_number(node.t),
        "origin": node.origin,
        "iterations": int(node.iterations),
        "residual": _json_number(residual),
        "b_norm": _json_number(b_norm),
        "relative_residual": _json_number(relative) if relative is not None else None,
        "tolerance": tolerance,
        "threshold": _json_number(threshold) if threshold is not None else None,
        "criterion": criterion_sig,
        "conclusion": "confirmed_pass" if passed else "confirmed_fail",
    }


def _partition_budget(scheme, attempts, t_start):
    """把某方案的预算消耗拆为起点校正、后续推进、失败尝试三类。"""
    start_iters = advance_iters = failed_iters = 0
    for attempt in attempts:
        if attempt.scheme != scheme.label:
            continue
        is_start = attempt.t_from == attempt.t_target == t_start
        if attempt.converged:
            if is_start:
                start_iters += attempt.iterations
            else:
                advance_iters += attempt.iterations
        else:
            failed_iters += attempt.iterations
    return start_iters, advance_iters, failed_iters


def _stopped_at(scheme, t_start, t_end):
    """说明方案停在起点（未确认/已确认）、后续节点还是终点。"""
    if not scheme.nodes:
        return {"kind": "start_unconfirmed", "t": _json_number(t_start)}
    t = scheme.nodes[-1].t
    if scheme.status == "completed" or t == t_end:
        kind = "end"
    elif t == t_start:
        kind = "start"
    else:
        kind = "followup"
    return {"kind": kind, "t": _json_number(t)}


def _attempt_entry(attempt: ContinuationAttempt) -> dict:
    return {
        "scheme": attempt.scheme,
        "kind": ("start_confirmation"
                 if attempt.t_from == attempt.t_target else "advance"),
        "t_from": _json_number(attempt.t_from),
        "t_target": _json_number(attempt.t_target),
        "step": _json_number(attempt.step),
        "iterations": int(attempt.iterations),
        "status": attempt.status,
        "residual": _json_number(attempt.residual),
        "message": attempt.message,
    }


def _fmt_float(value, digits=6):
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return ("inf" if value > 0 else "-inf") if not math.isnan(value) else "nan"
    return f"{value:.{digits}g}"


def _format_report_text(report: dict) -> str:
    """生成简洁中文文本核验报告。"""
    lines = []
    pr = report["parameter_range"]
    lines.append("节点核验报告 (continuation node verification)")
    lines.append("参数区间: t_start=%s  t_end=%s" % (
        _fmt_float(pr["t_start"]), _fmt_float(pr["t_end"])))
    tol = report["tolerance"]
    lines.append("容差: %s tol=%s" % (tol["kind"], _fmt_float(tol.get("tol"))))
    bg = report["budget"]
    lines.append(
        "预算: 总额=%d 已用=%d 剩余=%d | 起点校正=%d 后续推进=%d 失败尝试=%d" % (
            bg["total"], bg["used"], bg["remaining"],
            bg["start_confirmation"], bg["advances"], bg["failed_attempts"]))
    for scheme in report["schemes"]:
        stop = scheme["stopped_at"]
        lines.append("")
        lines.append("方案 %s [状态=%s, 停于=%s t=%s]" % (
            scheme["label"], scheme["status"], stop["kind"],
            _fmt_float(stop["t"])))
        bd = scheme["budget"]
        lines.append("  预算: 起点校正=%d 后续推进=%d 失败尝试=%d 合计=%d" % (
            bd["start_confirmation"], bd["advances"],
            bd["failed_attempts"], bd["total"]))
        if scheme["confirmed_nodes"]:
            lines.append("  已确认节点:")
            for node in scheme["confirmed_nodes"]:
                lines.append(
                    "    #%d t=%s 来源=%s 消耗=%d 真实残差=%s (相对=%s, 阈值=%s) => %s" % (
                        node["index"], _fmt_float(node["t"]), node["origin"],
                        node["iterations"], _fmt_float(node["residual"]),
                        _fmt_float(node["relative_residual"]),
                        _fmt_float(node["threshold"]), node["conclusion"]))
        else:
            lines.append("  已确认节点: 无")
        pending = scheme["unconfirmed_start"]
        if pending is not None:
            lines.append("  尚未确认的初始猜测: t=%s 来源=%s 真实残差=%s 状态=%s" % (
                _fmt_float(pending["t"]), pending["origin"],
                _fmt_float(pending["residual"]), pending["status"]))

    def _dump_attempts(title, attempts):
        if not attempts:
            return
        lines.append("")
        lines.append(title + ":")
        for attempt in attempts:
            lines.append(
                "  方案=%s 类型=%s %s->%s 步长=%s 消耗=%d 状态=%s 真实残差=%s%s" % (
                    attempt["scheme"], attempt["kind"],
                    _fmt_float(attempt["t_from"]), _fmt_float(attempt["t_target"]),
                    _fmt_float(attempt["step"]), attempt["iterations"],
                    attempt["status"], _fmt_float(attempt["residual"]),
                    ("  " + attempt["message"]) if attempt["message"] else ""))

    _dump_attempts("暂停尝试（未计入确认节点）", report["paused_attempts"])
    _dump_attempts("失败尝试（未计入确认节点）", report["failed_attempts"])
    return "\n".join(lines) + "\n"
