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


if __name__ == "__main__":
    main()
