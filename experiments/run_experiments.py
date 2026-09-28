"""第一版实验：Jacobi 与 CG 在良性/病态/不收敛案例上的表现。

运行：python -m experiments.run_experiments
"""
from __future__ import annotations

import numpy as np

from numexp import (
    StoppingSuite,
    ResidualTolerance,
    MaxIterations,
    NumericalAnomaly,
    JacobiSolver,
    ConjugateGradientSolver,
    MethodPreconditionError,
    verify_result,
    reference,
    cases,
)
from numexp.records import StopReason


def make_suite(max_iter: int, rtol: float = 1e-8) -> StoppingSuite:
    return StoppingSuite(
        ResidualTolerance(rtol=rtol),
        MaxIterations(max_iter),
        NumericalAnomaly(),
    )


def run() -> None:
    print("=" * 78)
    print("一、良性大型稀疏系统：CG 求解二维泊松方程")
    print("=" * 78)
    poisson = cases.poisson2d(k=24)  # n=576 稀疏
    cg = ConjugateGradientSolver(make_suite(max_iter=2000))
    res = cg.solve(poisson)
    print(res.summary())
    report = verify_result(poisson, res)
    print(report)
    assert res.reason is StopReason.CONVERGED
    assert report.passed

    print()
    print("=" * 78)
    print("二、小规模直接法对照（n<=200，仅作参考）")
    print("=" * 78)
    dd = cases.diagonally_dominant(n=40)
    x_ref = reference.direct_solve(dd)
    for solver in (
        JacobiSolver(make_suite(max_iter=5000)),
        ConjugateGradientSolver(make_suite(max_iter=5000)),
    ):
        r = solver.solve(dd)
        gap = float(np.linalg.norm(r.x - x_ref))
        print(f"{r.summary()}  与直接解之差 ||x-x_ref||={gap:.3e}")
        assert r.converged
        assert gap < 1e-6

    print()
    print("=" * 78)
    print("三、病态案例：Hilbert 矩阵（n=12，条件数约 1e16）")
    print("=" * 78)
    hilbert = cases.hilbert(n=12)
    cg_h = ConjugateGradientSolver(make_suite(max_iter=300, rtol=1e-10))
    r_h = cg_h.solve(hilbert)
    print(r_h.summary())
    rep_h = verify_result(hilbert, r_h, rtol=1e-10)
    print(rep_h)
    x_direct = reference.direct_solve(hilbert)
    print(f"直接法真解误差 ||x-x_true||={np.linalg.norm(x_direct - hilbert.x_true):.3e}")
    print("（病态案例即使残差很小，真解误差也可能很大——这是问题本身的病态性）")

    print()
    print("=" * 78)
    print("四、不收敛案例：Jacobi 在非对角占优系统上预算用尽/发散")
    print("=" * 78)
    diverging = cases.jacobi_divergent()
    jac = JacobiSolver(make_suite(max_iter=100))
    r_div = jac.solve(diverging)
    print(r_div.summary())
    assert r_div.reason in (
        StopReason.BUDGET_EXHAUSTED,
        StopReason.NUMERICAL_BREAKDOWN,
    )
    assert not r_div.converged
    print("=> 未收敛被如实标记，没有一律叫成功")

    print()
    print("=" * 78)
    print("五、前提拒绝：错误方法 + 不满足前提的输入")
    print("=" * 78)
    for label, problem in (
        ("非对称矩阵交给 CG", cases.nonsymmetric()),
        ("对称不定矩阵交给 CG", cases.indefinite_symmetric()),
        ("零对角元交给 Jacobi", cases.zero_diagonal()),
    ):
        solver = (
            ConjugateGradientSolver(make_suite(max_iter=10))
            if "CG" in label
            else JacobiSolver(make_suite(max_iter=10))
        )
        try:
            solver.solve(problem)
        except MethodPreconditionError as exc:
            print(f"拒绝 [{label}]：{exc}")
        else:
            raise AssertionError(f"应当拒绝：{label}")


if __name__ == "__main__":
    run()
