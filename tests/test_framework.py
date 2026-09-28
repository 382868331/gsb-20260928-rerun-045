"""框架测试：真实残差、停止原因、检查点续算、前提拒绝、直接法对照。"""
from __future__ import annotations

import numpy as np
import pytest

from numexp import (
    StoppingSuite,
    ResidualTolerance,
    MaxIterations,
    NumericalAnomaly,
    JacobiSolver,
    ConjugateGradientSolver,
    MethodPreconditionError,
    Checkpointer,
    verify_result,
    reference,
    cases,
    LinearSystem,
    ProblemError,
)
from numexp.records import StopReason


def suite(max_iter, rtol=1e-8):
    return StoppingSuite(
        ResidualTolerance(rtol=rtol),
        MaxIterations(max_iter),
        NumericalAnomaly(),
    )


def test_cg_converges_sparse_spd_and_residuals_are_real():
    problem = cases.poisson2d(k=10)
    solver = ConjugateGradientSolver(suite(2000, rtol=1e-10), record_vectors=True)
    result = solver.solve(problem)
    assert result.converged
    assert len(result.residual_history) == result.iterations + 1
    for record in result.records:
        r_norm = np.linalg.norm(problem.residual(record.x))
        assert abs(r_norm - record.residual_norm) < 1e-10 * max(1.0, r_norm)
    report = verify_result(problem, result, rtol=1e-10)
    assert report.passed
    assert report.true_error < 1e-6


def test_jacobi_matches_direct_solution_on_small_system():
    problem = cases.diagonally_dominant(n=20)
    x_ref = reference.direct_solve(problem)
    result = JacobiSolver(suite(5000)).solve(problem)
    assert result.converged
    assert np.linalg.norm(result.x - x_ref) < 1e-6


def test_budget_exhaustion_is_not_success():
    problem = cases.jacobi_divergent()
    result = JacobiSolver(suite(20)).solve(problem)
    assert result.reason in (
        StopReason.BUDGET_EXHAUSTED,
        StopReason.NUMERICAL_BREAKDOWN,
    )
    assert not result.converged
    report = verify_result(problem, result, rtol=1e-8)
    assert report.passed


def test_cg_rejects_nonsymmetric():
    with pytest.raises(MethodPreconditionError):
        ConjugateGradientSolver(suite(100)).solve(cases.nonsymmetric())


def test_cg_rejects_indefinite_small_dense():
    with pytest.raises(MethodPreconditionError):
        ConjugateGradientSolver(suite(100)).solve(cases.indefinite_symmetric(n=15))


def test_jacobi_rejects_zero_diagonal():
    with pytest.raises(MethodPreconditionError):
        JacobiSolver(suite(100)).solve(cases.zero_diagonal())


def test_malformed_problem_rejected():
    with pytest.raises(ProblemError):
        LinearSystem(np.eye(3), np.ones(2))


def test_checkpoint_resume_reproduces_trajectory(tmp_path):
    problem = cases.poisson2d(k=8)

    cp1 = Checkpointer(tmp_path / "run_a")
    solver1 = ConjugateGradientSolver(suite(5), checkpointer=cp1)
    partial = solver1.solve(problem)
    assert partial.reason is StopReason.BUDGET_EXHAUSTED
    assert partial.iterations == 5
    checkpoint_file = tmp_path / "run_a" / f"checkpoint_{problem.name}.npz"
    assert checkpoint_file.exists()

    cp2 = Checkpointer(tmp_path / "run_a")
    solver2 = ConjugateGradientSolver(suite(2000), checkpointer=cp2)
    resumed = solver2.solve(problem)
    assert resumed.converged
    assert len(resumed.residual_history) > 6
    assert resumed.iterations >= 5

    full = ConjugateGradientSolver(suite(2000)).solve(problem)
    k = resumed.iterations
    assert np.allclose(
        resumed.residual_history[: k + 1],
        full.residual_history[: k + 1],
        atol=1e-12,
    )
    assert np.allclose(resumed.x, full.x, atol=1e-10)
    assert not checkpoint_file.exists()


def test_checkpoint_rejects_changed_problem(tmp_path):
    problem = cases.poisson2d(k=6)
    cp = Checkpointer(tmp_path)
    ConjugateGradientSolver(suite(3), checkpointer=cp).solve(problem)
    other = cases.diagonally_dominant(n=36)
    other.name = problem.name
    with pytest.raises(ValueError, match="指纹"):
        ConjugateGradientSolver(suite(3), checkpointer=cp).solve(other)


def test_resume_requires_matching_solver(tmp_path):
    problem = cases.diagonally_dominant(n=16)
    cp = Checkpointer(tmp_path)
    JacobiSolver(suite(3), checkpointer=cp).solve(problem)
    with pytest.raises(ValueError, match="求解器"):
        ConjugateGradientSolver(suite(3), checkpointer=cp).solve(problem)


def test_ill_conditioned_case_runs_with_honest_report():
    problem = cases.hilbert(n=10)
    result = ConjugateGradientSolver(suite(500, rtol=1e-12)).solve(problem)
    report = verify_result(problem, result, rtol=1e-12)
    assert report.matches_history
    assert report.passed == result.converged
