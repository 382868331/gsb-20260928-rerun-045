"""检查点身份校验与受控接续测试。

分别改变矩阵 A、右端向量 b、停止条件，验证：
* 原进度恢复（resume）在恢复前就明确拒绝，而非盲目迭代后报告预算耗尽；
* 受控接续（restart）只借用旧近似解、内部状态全部重建，
  真实残差按新问题重算，旧检查点原封不动。
"""

import numpy as np
import pytest

from numexp import (Checkpoint, CheckpointMismatchError, CombinedCriterion,
                    ConjugateGradient, RelativeResidual,
                    TerminationReason)
from numexp.reference import spd_sparse_laplacian


def _run_first_leg(problem, ckpt, max_iter=4, tol=1e-12):
    return ConjugateGradient(
        RelativeResidual(tol=tol, max_iter=max_iter),
        checkpoint=str(ckpt)).solve(problem)


def _changed_b(problem, seed=99):
    rng = np.random.default_rng(seed)
    return type(problem)(problem.A, rng.standard_normal(problem.n))


def _changed_A(problem, seed=98):
    A = problem.A.toarray() if hasattr(problem.A, "toarray") \
        else np.array(problem.A)
    rng = np.random.default_rng(seed)
    A2 = A.copy()
    idx = int(rng.integers(0, problem.n))
    A2[idx, idx] += 1.0
    return type(problem)(A2, problem.b)


@pytest.mark.parametrize("change", ["b", "A", "tol", "criterion_type"])
def test_resume_rejects_mismatched_problem_or_settings(tmp_path, change):
    problem, _ = spd_sparse_laplacian(120)
    ckpt = tmp_path / "run"
    first = _run_first_leg(problem, ckpt)
    assert first.reason is TerminationReason.BUDGET_EXHAUSTED

    new_problem = problem
    criterion = RelativeResidual(tol=1e-12, max_iter=1000)
    expected = ""
    if change == "b":
        new_problem = _changed_b(problem)
        expected = "右端向量"
    elif change == "A":
        new_problem = _changed_A(problem)
        expected = "矩阵"
    elif change == "tol":
        criterion = RelativeResidual(tol=1e-6, max_iter=1000)
        expected = "精度设定"
    else:
        criterion = CombinedCriterion(
            conditions=(RelativeResidual(tol=1e-12, max_iter=1000),),
            max_iter=1000)
        expected = "精度设定"

    solver = ConjugateGradient(criterion, checkpoint=str(ckpt))
    with pytest.raises(CheckpointMismatchError, match=expected):
        solver.solve(new_problem)
    meta = Checkpoint(str(ckpt)).inspect()
    assert meta["iteration"] == 4
    assert meta["problem_id"] == problem.problem_id


def test_resume_allows_budget_change_but_records_true_residual(tmp_path):
    problem, info = spd_sparse_laplacian(200)
    ckpt = tmp_path / "run"
    first = _run_first_leg(problem, ckpt, max_iter=3)
    second = ConjugateGradient(
        RelativeResidual(tol=1e-12, max_iter=1000),
        checkpoint=str(ckpt)).solve(problem, history=first.history)
    assert second.success and second.resumed
    assert second.problem_id == problem.problem_id
    assert second.history[-1].residual_norm == pytest.approx(
        problem.residual_norm(second.x), rel=1e-12)
    assert np.linalg.norm(second.x - info["x_exact"]) \
        / np.linalg.norm(info["x_exact"]) < 1e-9
    assert not ckpt.with_suffix(".npz").exists()


def test_restart_with_new_b_uses_true_residual_and_keeps_old(tmp_path):
    problem, _ = spd_sparse_laplacian(120)
    ckpt = tmp_path / "run"
    _run_first_leg(problem, ckpt)
    old_meta = Checkpoint(str(ckpt)).inspect()

    new_problem = _changed_b(problem)
    solver = ConjugateGradient(
        RelativeResidual(tol=1e-12, max_iter=1000), checkpoint=str(ckpt))
    result = solver.solve(new_problem, continuation="restart")
    assert result.success and not result.resumed
    assert result.problem_id == new_problem.problem_id
    assert result.problem_id != problem.problem_id
    assert result.verify(new_problem, tol=1e-6)
    assert new_problem.residual_norm(result.x) < 1e-8
    meta = Checkpoint(str(ckpt)).inspect()
    assert meta["problem_id"] == problem.problem_id
    assert meta["iteration"] == 4
    assert meta == old_meta


def test_restart_with_changed_A_converges_on_new_problem(tmp_path):
    problem, _ = spd_sparse_laplacian(80)
    ckpt = tmp_path / "run"
    _run_first_leg(problem, ckpt, max_iter=2)
    new_problem = _changed_A(problem)
    result = ConjugateGradient(
        RelativeResidual(tol=1e-10, max_iter=2000),
        checkpoint=str(ckpt)).solve(new_problem, continuation="restart")
    assert result.success
    assert new_problem.residual_norm(result.x) < 1e-7
    assert Checkpoint(str(ckpt)).inspect()["problem_id"] \
        == problem.problem_id


def test_restart_rejects_dimension_mismatch(tmp_path):
    problem, _ = spd_sparse_laplacian(60)
    ckpt = tmp_path / "run"
    _run_first_leg(problem, ckpt, max_iter=1)
    other, _ = spd_sparse_laplacian(30, seed=7)
    with pytest.raises(CheckpointMismatchError, match="维度"):
        ConjugateGradient(RelativeResidual(max_iter=10),
                          checkpoint=str(ckpt)).solve(
            other, continuation="restart")


def test_cross_method_restart_from_richardson_warm_start(tmp_path):
    from numexp import Richardson
    problem, info = spd_sparse_laplacian(150)
    ckpt = tmp_path / "rich"
    rich = Richardson(0.4, RelativeResidual(tol=1e-12, max_iter=20),
                      checkpoint=str(ckpt)).solve(problem)
    assert not rich.success
    warm_residual = problem.residual_norm(rich.x)
    cg = ConjugateGradient(RelativeResidual(tol=1e-12, max_iter=1000),
                           checkpoint=str(ckpt))
    with pytest.raises(CheckpointMismatchError, match="求解方法"):
        cg.solve(problem)
    result = cg.solve(problem, continuation="restart")
    assert result.success
    assert result.history[0].iteration == 0
    assert result.history[0].residual_norm == pytest.approx(
        warm_residual, rel=1e-12)
    assert np.linalg.norm(result.x - info["x_exact"]) \
        / np.linalg.norm(info["x_exact"]) < 1e-9
    assert Checkpoint(str(ckpt)).inspect()["solver"] == "richardson"
