"""框架测试：前提检查、真实残差、停止分类、检查点续算、对照。"""

import json

import numpy as np
import pytest

from numexp import (ConjugateGradient, Richardson, RelativeResidual,
                    PreconditionError, TerminationReason, LinearSystem)
from numexp.reference import (direct_solve, spd_1d_poisson,
                              spd_sparse_laplacian, nonsymmetric_system,
                              indefinite_system)
from scipy import sparse


def test_cg_matches_direct_solution():
    problem, info = spd_1d_poisson(30)
    result = ConjugateGradient(
        RelativeResidual(tol=1e-10, max_iter=300)).solve(problem)
    assert result.success
    x_ref = direct_solve(problem)
    assert np.allclose(result.x, x_ref, atol=1e-8)
    assert np.allclose(result.x, info["x_exact"], atol=1e-8)
    # 开启解快照后，历史记录每一步的残差都应由原矩阵重算复现
    solver = ConjugateGradient(RelativeResidual(tol=1e-10, max_iter=300),
                               record_solutions=True)
    rec = solver.solve(problem)
    for h in rec.history:
        assert h.residual_norm == pytest.approx(problem.residual_norm(h.x))


def test_sparse_matrix_supported_by_cg():
    problem, info = spd_sparse_laplacian(200)
    result = ConjugateGradient(
        RelativeResidual(tol=1e-10, max_iter=500)).solve(problem)
    assert result.success
    assert np.linalg.norm(result.x - info["x_exact"]) / np.linalg.norm(
        info["x_exact"]) < 1e-8


def test_true_residual_decreases_and_is_trusted():
    problem, _ = spd_1d_poisson(40)
    solver = ConjugateGradient(RelativeResidual(tol=1e-9, max_iter=200),
                               record_solutions=True)
    result = solver.solve(problem)
    assert result.history[-1].residual_norm < 1e-7
    # 独立验证
    assert result.verify(problem, tol=1e-6)


def test_cg_rejects_nonsymmetric():
    problem = nonsymmetric_system()
    with pytest.raises(PreconditionError, match="不对称"):
        ConjugateGradient().solve(problem)


def test_cg_rejects_indefinite_small():
    problem = indefinite_system(8)
    with pytest.raises(PreconditionError, match="正定"):
        ConjugateGradient().solve(problem)


def test_invalid_shapes_rejected():
    with pytest.raises(PreconditionError):
        from numexp import LinearSystem
        LinearSystem(np.eye(3), np.ones(2))
    with pytest.raises(PreconditionError):
        from numexp import LinearSystem
        LinearSystem(np.ones((2, 3)), np.ones(2))


def test_matrix_identity_is_content_based():
    dense = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    coo = sparse.coo_matrix(dense)
    csr = sparse.csr_matrix(dense)
    assert LinearSystem(dense, b).identity() == LinearSystem(coo, b).identity()
    assert LinearSystem(csr, b).identity() == LinearSystem(coo, b).identity()


def test_budget_exhausted_is_not_success():
    problem, _ = spd_1d_poisson(60)
    result = ConjugateGradient(
        RelativeResidual(tol=1e-14, max_iter=3)).solve(problem)
    assert not result.success
    assert result.reason is TerminationReason.BUDGET_EXHAUSTED
    assert result.iterations == 3


def test_richardson_divergence_classified():
    problem, _ = spd_1d_poisson(40)
    result = Richardson(omega=1.0,
                        criterion=RelativeResidual(tol=1e-12, max_iter=40)
                        ).solve(problem)
    assert result.reason in (TerminationReason.BUDGET_EXHAUSTED,
                             TerminationReason.NUMERICAL_BREAKDOWN)
    assert not result.success
    # 残差整体应在增长（发散）
    hist = result.residual_history
    assert hist[-1] > hist[0]


def test_richardson_converges_with_good_omega():
    problem, _ = spd_1d_poisson(30)
    result = Richardson(omega=0.4,
                        criterion=RelativeResidual(tol=1e-8, max_iter=5000)
                        ).solve(problem)
    assert result.success
    assert result.verify(problem)


def test_checkpoint_resume(tmp_path):
    problem, info = spd_sparse_laplacian(300)
    ckpt = tmp_path / "cg"
    first = ConjugateGradient(
        RelativeResidual(tol=1e-12, max_iter=4), checkpoint=str(ckpt)
    ).solve(problem)
    assert first.reason is TerminationReason.BUDGET_EXHAUSTED
    assert ckpt.with_suffix(".npz").exists()
    assert ckpt.with_suffix(".json").exists()

    second = ConjugateGradient(
        RelativeResidual(tol=1e-12, max_iter=1000),
        checkpoint=str(ckpt)).solve(problem, history=first.history)
    assert second.success
    # 续算后迭代号连续，从 4 开始而不是从 0 重新计数
    assert second.history[4].iteration == 4
    assert second.history[-1].iteration > 4
    assert np.linalg.norm(second.x - info["x_exact"]) / np.linalg.norm(
        info["x_exact"]) < 1e-9
    # 收敛后检查点被清理
    assert not ckpt.with_suffix(".npz").exists()


def test_checkpoint_solver_mismatch_rejected(tmp_path):
    problem, _ = spd_1d_poisson(20)
    ckpt = tmp_path / "run"
    ConjugateGradient(RelativeResidual(max_iter=1),
                      checkpoint=str(ckpt)).solve(problem)
    with pytest.raises(ValueError, match="检查点属于"):
        Richardson(0.5, RelativeResidual(max_iter=10),
                   checkpoint=str(ckpt)).solve(problem)


def test_checkpoint_problem_and_criterion_mismatch_rejected(tmp_path):
    problem, _ = spd_1d_poisson(20, seed=20)
    ckpt = tmp_path / "cg-run"
    first = ConjugateGradient(
        RelativeResidual(tol=1e-12, max_iter=3),
        checkpoint=str(ckpt)).solve(problem)
    assert first.reason is TerminationReason.BUDGET_EXHAUSTED

    changed_A = LinearSystem(problem.A + 0.01 * np.eye(problem.n), problem.b)
    with pytest.raises(ValueError, match="与当前问题不一致"):
        ConjugateGradient(
            RelativeResidual(tol=1e-12, max_iter=10),
            checkpoint=str(ckpt)).solve(changed_A)

    changed_b = LinearSystem(problem.A, problem.b + 0.1)
    with pytest.raises(ValueError, match="与当前问题不一致"):
        ConjugateGradient(
            RelativeResidual(tol=1e-12, max_iter=10),
            checkpoint=str(ckpt)).solve(changed_b)

    with pytest.raises(ValueError, match="停止条件不一致"):
        ConjugateGradient(
            RelativeResidual(tol=1e-10, max_iter=10),
            checkpoint=str(ckpt)).solve(problem)

    richardson_ckpt = tmp_path / "richardson-run"
    Richardson(0.4, RelativeResidual(tol=1e-12, max_iter=3),
               checkpoint=str(richardson_ckpt)).solve(problem)
    with pytest.raises(ValueError, match="与当前设定不一致"):
        Richardson(0.3, RelativeResidual(tol=1e-12, max_iter=10),
                   checkpoint=str(richardson_ckpt)).solve(problem)

    old_meta = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8"))
    assert old_meta["problem"] == problem.identity()


def test_controlled_restart_separates_new_run_from_old_progress(tmp_path):
    problem, _ = spd_1d_poisson(20, seed=21)
    source = tmp_path / "cg-old"
    first = ConjugateGradient(
        RelativeResidual(tol=1e-12, max_iter=3),
        checkpoint=str(source)).solve(problem)
    assert first.reason is TerminationReason.BUDGET_EXHAUSTED

    changed_A = LinearSystem(problem.A + 0.01 * np.eye(problem.n), problem.b)
    changed_b = LinearSystem(problem.A, problem.b + 0.1)
    variants = (
        ("same-checkpoint-matrix", changed_A,
         RelativeResidual(tol=1e-12, max_iter=3000), source),
        ("new-checkpoint-rhs", changed_b,
         RelativeResidual(tol=1e-12, max_iter=3000), tmp_path / "richardson-rhs"),
        ("new-checkpoint-criterion", problem,
         RelativeResidual(tol=1e-10, max_iter=3000), tmp_path / "richardson-crit"),
    )

    for _, new_problem, criterion, target in variants:
        result = Richardson(
            0.4, criterion, checkpoint=str(target)
        ).solve(new_problem, continuation="restart", restart_from=str(source))
        assert result.success
        assert result.verify(new_problem, tol=criterion.tol)
        assert result.history[0].iteration == 0
        assert result.continuation == "restarted_from_iteration_3"
        assert result.problem_identity == new_problem.identity()
        assert result.solver_name == "richardson"

        old_meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        assert old_meta["problem"] == problem.identity()
        assert old_meta["solver"] == "cg"
        assert old_meta["iteration"] == 3

    assert source.with_suffix(".npz").exists()
    assert not (tmp_path / "richardson-rhs.npz").exists()
    assert not (tmp_path / "richardson-crit.npz").exists()


def test_indefinite_large_captured_by_breakdown():
    # 大矩阵不做稠密 Cholesky；构造对称但对角元为负 -> 立即拒绝
    from numexp import LinearSystem
    n = 250
    A = np.eye(n)
    A[0, 0] = -1.0
    problem = LinearSystem(A, np.ones(n))
    with pytest.raises(PreconditionError):
        ConjugateGradient().solve(problem)


def test_x0_shape_rejected():
    problem, _ = spd_1d_poisson(10)
    with pytest.raises(PreconditionError):
        ConjugateGradient().solve(problem, x0=np.zeros(9))
