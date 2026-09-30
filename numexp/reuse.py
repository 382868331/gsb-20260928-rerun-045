"""输入变化时的可沿用性说明。

矩阵、右端项或容差变化时，明确哪些内容可以沿用到新问题的求解，
哪些必须重建。结论分三档：

* 原进度恢复（resumable）：问题身份与停止条件数学设定完全一致，
  检查点中的求解器内部状态可以继续使用；
* 受控热启动：维度一致时，旧近似解 x 可以作为新问题的初始猜测，
  但求解器内部迭代状态（搜索方向、递推残差等）必须由当前问题重建；
* 不可沿用：维度变化时旧近似解也不能用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .problems import LinearSystem


@dataclass
class ReuseReport:
    """一次新旧输入对比的结论。"""

    matrix_changed: bool
    rhs_changed: bool
    criterion_changed: bool | None   # 未提供停止条件对比时为 None
    dimension_changed: bool
    resumable: bool
    reusable: list[str] = field(default_factory=list)
    not_reusable: list[str] = field(default_factory=list)

    @property
    def anything_changed(self) -> bool:
        return (self.matrix_changed or self.rhs_changed
                or bool(self.criterion_changed) or self.dimension_changed)

    def summary(self) -> str:
        lines = []
        changes = []
        if self.dimension_changed:
            changes.append("维度")
        if self.matrix_changed:
            changes.append("矩阵")
        if self.rhs_changed:
            changes.append("右端项")
        if self.criterion_changed:
            changes.append("容差/停止条件")
        lines.append("变化项：" + ("、".join(changes) if changes else "无"))
        lines.append("可沿用：" + ("；".join(self.reusable) or "无"))
        lines.append("不可沿用：" + ("；".join(self.not_reusable) or "无"))
        lines.append("原进度恢复：" + ("允许" if self.resumable else "不允许"))
        return "\n".join(lines)


def assess_reuse(old_problem: LinearSystem, new_problem: LinearSystem,
                 old_criterion=None, new_criterion=None) -> ReuseReport:
    """对比新旧问题（及可选的停止条件），给出可沿用内容说明。"""
    old_id = old_problem.identity()
    new_id = new_problem.identity()
    dimension_changed = old_id["n"] != new_id["n"]
    matrix_changed = old_id["A_sha256"] != new_id["A_sha256"]
    rhs_changed = old_id["b_sha256"] != new_id["b_sha256"]
    if old_criterion is None or new_criterion is None:
        criterion_changed = None
    else:
        criterion_changed = (old_criterion.signature()
                             != new_criterion.signature())

    problem_unchanged = not (dimension_changed or matrix_changed
                             or rhs_changed)
    resumable = problem_unchanged and criterion_changed is False

    reusable: list[str] = []
    not_reusable: list[str] = []
    if resumable:
        reusable.append("问题与停止条件完全一致，检查点中的求解器内部状态"
                        "可按原进度恢复（max_iter 预算允许加大）")
    if not dimension_changed:
        reusable.append("旧近似解 x 可作为当前问题的初始猜测（热启动），"
                        "残差按当前问题重新计算")
    else:
        not_reusable.append("维度变化，旧近似解不能作为新初值")
    if problem_unchanged and criterion_changed:
        not_reusable.append("停止条件变化，旧检查点不能按原进度恢复，"
                            "只能受控重新开始")
    if not problem_unchanged:
        not_reusable.append("求解器内部迭代状态（如 CG 的搜索方向与递推残差）"
                            "不可沿用，必须由当前问题重建")
        not_reusable.append("旧检查点不能作为兼容检查点恢复原进度，"
                            "只能以只读方式取出旧近似解受控重新开始")

    return ReuseReport(matrix_changed=matrix_changed,
                       rhs_changed=rhs_changed,
                       criterion_changed=criterion_changed,
                       dimension_changed=dimension_changed,
                       resumable=resumable,
                       reusable=reusable,
                       not_reusable=not_reusable)
