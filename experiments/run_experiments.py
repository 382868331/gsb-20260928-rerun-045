"""实验脚本：对照、病态、不收敛三类案例。

用法：python -m experiments.run_experiments
"""

from __future__ import annotations

import tempfile
import json
from pathlib import Path

import numpy as np

from numexp import (ConjugateGradient, Richardson, RelativeResidual,
                    PreconditionError, LinearSystem)
from numexp import (StartPath, MultiStartSession, solve_problem_series)
from numexp import ContinuationSession
from numexp.reference import (direct_solve, spd_1d_poisson,
                              spd_sparse_laplacian, nonsymmetric_system,
                              indefinite_system)


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _summarize(tag: str, result, problem, x_ref=None) -> None:
    print(f"[{tag}] 终止原因: {result.reason.value} | 成功: {result.success} "
          f"| 迭代数: {result.iterations} | {result.message}")
    hist = result.residual_history
    print(f"        真实残差: 初值 {hist[0]:.3e} -> 终值 {hist[-1]:.3e}")
    print(f"        独立结果检查 verify(1e-6): {result.verify(problem, 1e-6)}")
    if x_ref is not None:
        err = float(np.linalg.norm(result.x - x_ref)
                    / max(np.linalg.norm(x_ref), 1e-300))
        print(f"        与直接解相对误差: {err:.3e}")


def experiment_reference_check() -> None:
    _print_header("实验1：小规模 CG 与直接求解对照（n=50）")
    problem, _ = spd_1d_poisson(50)
    x_ref = direct_solve(problem)
    result = ConjugateGradient(RelativeResidual(tol=1e-10, max_iter=500)).solve(problem)
    _summarize("CG", result, problem, x_ref)


def experiment_ill_conditioned() -> None:
    _print_header("实验2：病态系统（n=100, kappa 缩放）")
    for kappa in (1.0, 1e3, 1e6):
        problem, _ = spd_1d_poisson(100, kappa=kappa)
        result = ConjugateGradient(
            RelativeResidual(tol=1e-8, max_iter=2000)).solve(problem)
        _summarize(f"kappa={kappa:g}", result, problem)


def experiment_divergent() -> None:
    _print_header("实验3：Richardson 不收敛案例（预算用尽 ≠ 成功）")
    problem, _ = spd_1d_poisson(40)
    result = Richardson(omega=1.0,
                        criterion=RelativeResidual(tol=1e-10, max_iter=30)).solve(problem)
    _summarize("Richardson omega=1", result, problem)
    print("        残差是否单调增长:",
          all(b >= a for a, b in zip(result.residual_history,
                                     result.residual_history[1:])))


def experiment_precondition_reject() -> None:
    _print_header("实验4：方法前提拒绝")
    cg = ConjugateGradient(RelativeResidual(max_iter=10))
    for tag, factory in (("非对称", nonsymmetric_system),
                         ("对称不定", indefinite_system)):
        try:
            cg.solve(factory())
            print(f"[{tag}] 未拒绝（错误！）")
        except PreconditionError as exc:
            print(f"[{tag}] 已拒绝: {exc}")


def experiment_checkpoint_resume() -> None:
    _print_header("实验5：检查点原进度恢复与受控重新开始（n=80）")
    problem, info = spd_sparse_laplacian(80)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "cg-original"
        first = ConjugateGradient(
            RelativeResidual(tol=1e-12, max_iter=5), checkpoint=str(ckpt)).solve(problem)
        print(f"[第一段] 预算5步后: {first.reason.value}, 迭代 {first.iterations}, "
              f"残差 {first.residual_history[-1]:.3e}")
        second = ConjugateGradient(
            RelativeResidual(tol=1e-12, max_iter=1000),
            checkpoint=str(ckpt)).solve(problem, history=first.history)
        print(f"[续算段] {second.reason.value}, 总迭代 {second.iterations}, "
              f"残差 {second.residual_history[-1]:.3e}")
        err = float(np.linalg.norm(second.x - info["x_exact"])
                    / np.linalg.norm(info["x_exact"]))
        print(f"        与精确解相对误差: {err:.3e}")
        print(f"        检查点收敛后已清理: {not ckpt.with_suffix('.npz').exists()}")

        restart_problem, _ = spd_1d_poisson(80, seed=80)
        source = Path(tmp) / "cg-restart-source"
        old = ConjugateGradient(
            RelativeResidual(tol=1e-12, max_iter=4),
            checkpoint=str(source)).solve(restart_problem)
        print(f"\n[旧记录] CG 迭代 {old.iterations} 后停止: {old.reason.value}")
        variants = (
            ("矩阵改变", LinearSystem(restart_problem.A + 0.001 * np.eye(80),
                                      restart_problem.b),
             RelativeResidual(tol=5e-2, max_iter=5000)),
            ("右端项改变", LinearSystem(restart_problem.A,
                                        restart_problem.b + 0.1),
             RelativeResidual(tol=5e-2, max_iter=5000)),
            ("停止条件改变", restart_problem,
             RelativeResidual(tol=5e-2, max_iter=5000)),
        )
        for index, (tag, new_problem, criterion) in enumerate(variants):
            target = Path(tmp) / f"richardson-new-{index}"
            result = Richardson(
                0.4, criterion, checkpoint=str(target)
            ).solve(new_problem, continuation="restart",
                     restart_from=str(source))
            print(f"[{tag}] {result.continuation}: 第0步真实残差 "
                  f"{result.history[0].residual_norm:.3e} -> "
                  f"{result.residual_history[-1]:.3e}, "
                  f"verify={result.verify(new_problem, criterion.tol)}")
        source_meta = source.with_suffix(".json").read_text(encoding="utf-8")
        print(f"        旧记录保留: solver={json.loads(source_meta)['solver']}, "
              f"iteration={json.loads(source_meta)['iteration']}")


def main() -> None:
    experiment_reference_check()
    experiment_ill_conditioned()
    experiment_divergent()
    experiment_precondition_reject()
    experiment_checkpoint_resume()
    experiment_problem_series()
    experiment_multistart()
    experiment_continuation()


def experiment_continuation() -> None:
    _print_header("实验8：参数连续跟踪——回退推进、分支比较与暂停重开")
    n = 30
    b = np.ones(n)

    def make_problem(t):
        # t<=0.4 时 A=I（平稳段）；之后最小特征值降到 0.05（难收敛段）
        s = max(0.0, (t - 0.4) / 0.6)
        c = np.geomspace(10.0 ** (-1.3 * s), 1.0, n)
        return LinearSystem(np.diag(c), b)

    def make_solver(criterion):
        return Richardson(omega=1.0, criterion=criterion)

    with tempfile.TemporaryDirectory() as tmp:
        session = ContinuationSession(
            make_problem, make_solver, param_start=0.0, param_end=1.0,
            total_budget=400, criterion=RelativeResidual(tol=1e-6),
            attempt_budget=200, initial_step=0.6,
            session_dir=str(Path(tmp) / "cont"))
        first = session.run()
        path = first.paths[0]
        print(f"[预算受限] 状态={path.status} 已耗={first.budget_used} "
              f"完成={path.completed}（未到达终点时 final_node 为 "
              f"{path.final_node}）")
        restored = ContinuationSession.load(
            Path(tmp) / "cont", make_problem, make_solver,
            criterion=RelativeResidual(tol=1e-6))
        second = restored.run(extra_budget=40000)
        path = second.paths[0]
        print(f"[重开追加预算] 状态={path.status} 已耗={second.budget_used}")
        for a in path.attempts:
            mark = "确认" if a.accepted else "失败"
            print(f"  尝试 t={a.param:.4f} 步长={a.step:+.4f} "
                  f"耗={a.cost:4d} {mark} ({a.status})")
        print("  已确认节点: "
              + ", ".join(f"t={nd.param:.4f}(耗{nd.cost})"
                          for nd in path.nodes))
        # 从中间节点另开方案，两条路径到达同一参数后按真实残差比较
        mid = len(path.nodes) // 2
        restored.branch("main", mid, label="alt", initial_step=0.05)
        restored.run()
        for entry in restored.compare_at(1.0):
            print(f"  比较 t=1.0: 路径[{entry['path']}] "
                  f"真实残差={entry['residual_norm']:.3e} "
                  f"来源={entry['source']}")


def experiment_problem_series() -> None:
    _print_header("实验6：相关问题序列——热启动与冷启动对照")
    base, _ = spd_1d_poisson(60, seed=6)
    problems = [
        base,
        LinearSystem(base.A + 0.02 * np.eye(base.n), base.b),
        LinearSystem(base.A + 0.02 * np.eye(base.n), base.b + 0.3),
    ]
    make_solver = lambda: ConjugateGradient(
        RelativeResidual(tol=1e-10, max_iter=1000))
    series = solve_problem_series(problems, make_solver)
    for step in series.steps:
        print(f"[问题{step.index}] 冷启动: {step.cold.iterations} 步, "
              f"残差 {step.cold.residual_history[-1]:.3e}")
        if step.warm is not None:
            print(f"         热启动: {step.warm.iterations} 步, "
                  f"初值残差 {step.warm.history[0].residual_norm:.3e} -> "
                  f"{step.warm.residual_history[-1]:.3e}")
        if step.reuse is not None:
            print("         " + step.reuse.summary().replace(chr(10), chr(10) + "         "))


def experiment_multistart() -> None:
    _print_header("实验7：多起点统一预算——失败隔离、暂停重开、按真实残差选择")
    problem, _ = spd_1d_poisson(40)

    def make(criterion, checkpoint, path):
        omega = {"慢速": 0.1, "合适": 0.4, "发散": 1.0}[path.label]
        return Richardson(omega, criterion, checkpoint=checkpoint)

    paths = [StartPath("慢速", origin="fresh"),
             StartPath("合适", origin="fresh"),
             StartPath("发散", origin="fresh")]
    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp) / "multistart"
        session = MultiStartSession(
            problem, make, paths, total_budget=200,
            criterion=RelativeResidual(tol=1e-8),
            stagnation=(50, 0.9), session_dir=session_dir)
        first = session.run()
        print(f"[第一段] 预算 {first.budget_used}/{first.budget_total}")
        for r in first.records:
            print(f"  路径[{r.label}] 来源={r.origin} 状态={r.status} "
                  f"已耗迭代={r.iterations}")
        restored = MultiStartSession.load(session_dir, problem, make,
                                          criterion=RelativeResidual(tol=1e-8))
        second = restored.run(extra_budget=30000)
        print(f"[重开追加预算后] 预算 {second.budget_used}/{second.budget_total}")
        for r in second.records:
            residual = (f"{r.best_residual:.3e}"
                        if r.best_residual is not None else "无")
            print(f"  路径[{r.label}] 状态={r.status} 累计迭代={r.iterations} "
                  f"真实残差={residual}")
        best = second.best(problem)
        print(f"  最终选择: 路径[{best.label}]，按当前问题重算残差 "
              f"{problem.residual_norm(best.x):.3e}，收敛={best.converged}")


if __name__ == "__main__":
    main()
