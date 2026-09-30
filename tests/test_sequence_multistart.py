"""问题序列与多起点统一预算的测试。"""

import json

import numpy as np
import pytest

from numexp import (ConjugateGradient, Richardson, RelativeResidual,
                    LinearSystem, TerminationReason, describe_carryover,
                    solve_sequence, MultiStartRunner, StartPoint)
from numexp.reference import spd_1d_poisson, direct_solve


def _cg_factory(tol=1e-10, max_iter=500):
    return lambda: ConjugateGradient(RelativeResidual(tol=tol, max_iter=max_iter))


def test_describe_carryover_no_change():
    problem, _ = spd_1d_poisson(20)
    note = describe_carryover(problem, problem,
                              RelativeResidual(tol=1e-8),
                              RelativeResidual(tol=1e-8))
    assert not note.problem_changed
    assert note.criterion_changed is False
    assert "原进度恢复" in note.text


def test_describe_carryover_matrix_rhs_tol_dimension():
    problem, _ = spd_1d_poisson(20, seed=1)
    changed_A = LinearSystem(problem.A + 0.01 * np.eye(problem.n), problem.b)
    note = describe_carryover(problem, changed_A,
                              RelativeResidual(tol=1e-8),
                              RelativeResidual(tol=1e-10))
    assert note.matrix_changed and not note.rhs_changed
    assert note.criterion_changed is True
    assert "矩阵 A 已变化" in note.text
    assert "停止条件" in note.text
    assert "近似解 x" in note.text and "仅作为新初值" in note.text
    assert "内部状态" in note.text and "不可沿用" in note.text

    changed_b = LinearSystem(problem.A, problem.b + 0.5)
    note_b = describe_carryover(problem, changed_b)
    assert note_b.rhs_changed and not note_b.matrix_changed
    assert note_b.criterion_changed is None
    assert "右端项 b 已变化" in note_b.text

    bigger, _ = spd_1d_poisson(30, seed=1)
    note_dim = describe_carryover(problem, bigger)
    assert note_dim.dimension_changed
    assert "无可沿用内容" in note_dim.text


def test_sequence_warm_and_cold_independent_records():
    base, _ = spd_1d_poisson(40, seed=5)
    problems = [base]
    for k in (1, 2):
        problems.append(LinearSystem(base.A, base.b + 0.05 * k))
    steps = solve_sequence(problems, _cg_factory())
    assert len(steps) == 3
    assert steps[0].carryover is None
    for k, step in enumerate(steps):
        # 每次求解独立记录目标问题与收敛判断
        assert step.warm.problem_identity == problems[k].identity()
        assert step.cold.problem_identity == problems[k].identity()
        assert step.warm.success and step.cold.success
        # 新运行从迭代 0 重新记录，不沿用上一问题的迭代状态
        assert step.warm.history[0].iteration == 0
        assert step.cold.history[0].iteration == 0
        x_ref = direct_solve(problems[k])
        assert np.allclose(step.warm.x, x_ref, atol=1e-8)
        assert np.allclose(step.cold.x, x_ref, atol=1e-8)
    assert steps[1].warm.continuation == "warm_started_from_previous_solution"
    assert steps[1].cold.continuation == "fresh"
    assert steps[1].carryover.rhs_changed
    # 热启动通常不比冷启动慢
    assert steps[1].warm.iterations <= steps[1].cold.iterations
    assert steps[2].warm.iterations <= steps[2].cold.iterations


def _cg_solver_factory(checkpoint, criterion):
    return ConjugateGradient(criterion, checkpoint=checkpoint)


def test_multistart_unified_budget_and_residual_selection(tmp_path):
    problem, info = spd_1d_poisson(60, seed=7)
    rng = np.random.default_rng(0)
    starts = [StartPoint("zero"),
              StartPoint("random-a", x0=rng.standard_normal(problem.n)),
              StartPoint("random-b", x0=rng.standard_normal(problem.n))]
    runner = MultiStartRunner(
        problem, _cg_solver_factory,
        lambda budget: RelativeResidual(tol=1e-12, max_iter=budget),
        starts, total_budget=200, workdir=tmp_path / "ms", slice_iter=7)
    result = runner.run()
    # 统一预算：总消耗不超过预算
    assert result.used_budget <= 200
    assert sum(p.iterations for p in result.paths) == result.used_budget
    assert result.converged
    # 最终选择以当前问题真实残差为依据
    for p in result.paths:
        assert p.residual_norm == pytest.approx(problem.residual_norm(p.x))
    best = min(result.paths, key=lambda p: p.residual_norm)
    assert result.best_label == best.label
    assert result.best_residual == pytest.approx(problem.residual_norm(result.best_x))
    assert np.linalg.norm(result.best_x - info["x_exact"]) / np.linalg.norm(
        info["x_exact"]) < 1e-8


def test_multistart_failure_isolated_and_stagnation_detected(tmp_path):
    problem, _ = spd_1d_poisson(40, seed=8)

    def factory(checkpoint, criterion):
        # 路径1 使用发散的 Richardson，其余路径用 CG
        if checkpoint.endswith("path-1"):
            return Richardson(1.0, criterion, checkpoint=checkpoint)
        return ConjugateGradient(criterion, checkpoint=checkpoint)

    starts = [StartPoint("cg-zero"),
              StartPoint("richardson-divergent"),
              StartPoint("cg-ones", x0=np.ones(problem.n))]
    runner = MultiStartRunner(
        problem, factory,
        lambda budget: RelativeResidual(tol=1e-10, max_iter=budget),
        starts, total_budget=400, workdir=tmp_path / "ms", slice_iter=5)
    result = runner.run()
    status = {p.label: p.status for p in result.paths}
    # 发散路径被隔离（数值异常或停滞），但不影响其他路径收敛
    assert status["richardson-divergent"] in ("failed", "stagnant")
    assert status["cg-zero"] == "converged"
    assert status["cg-ones"] == "converged"
    assert result.converged and result.best_residual < 1e-8


def test_multistart_pause_resume_preserves_paths(tmp_path):
    problem, info = spd_1d_poisson(60, seed=9)
    workdir = tmp_path / "ms"
    starts = [StartPoint("zero"), StartPoint("ones", x0=np.ones(problem.n))]

    first = MultiStartRunner(
        problem, _cg_solver_factory,
        lambda budget: RelativeResidual(tol=1e-10, max_iter=budget),
        starts, total_budget=6, workdir=workdir, slice_iter=3).run()
    assert not first.converged
    assert first.used_budget <= 6

    manifest = json.loads((workdir / "multistart.json").read_text(encoding="utf-8"))
    assert [p["label"] for p in manifest["paths"]] == ["zero", "ones"]
    assert manifest["paths"][1]["origin"] == "x0=用户给定初值"
    saved_used = [p["used"] for p in manifest["paths"]]

    # 暂停后重开：同一 workdir、更大预算；各路径预算与来源保留并继续累计
    second = MultiStartRunner(
        problem, _cg_solver_factory,
        lambda budget: RelativeResidual(tol=1e-10, max_iter=budget),
        starts, total_budget=800, workdir=workdir, slice_iter=5).run()
    assert second.converged
    for p, before in zip(second.paths, saved_used):
        assert p.iterations >= before
    origins = {p.label: p.origin for p in second.paths}
    assert origins["ones"] == "x0=用户给定初值"
    assert np.linalg.norm(second.best_x - info["x_exact"]) / np.linalg.norm(
        info["x_exact"]) < 1e-8

    # 换问题后不能在同一 workdir 上重开
    changed = LinearSystem(problem.A, problem.b + 1.0)
    with pytest.raises(ValueError, match="另一个问题"):
        MultiStartRunner(
            changed, _cg_solver_factory,
            lambda budget: RelativeResidual(tol=1e-10, max_iter=budget),
            starts, total_budget=900, workdir=workdir).run()


def test_multistart_restart_source_checkpoint_read_only(tmp_path):
    old_problem, _ = spd_1d_poisson(50, seed=11)
    source = tmp_path / "old-cg"
    old = ConjugateGradient(RelativeResidual(tol=1e-12, max_iter=4),
                            checkpoint=str(source)).solve(old_problem)
    assert old.reason is TerminationReason.BUDGET_EXHAUSTED
    source_meta_before = source.with_suffix(".json").read_text(encoding="utf-8")

    # 相关问题（右端项变化）：旧近似解只作为起点，旧检查点不被覆盖
    new_problem = LinearSystem(old_problem.A, old_problem.b + 0.1)
    starts = [StartPoint("from-old", restart_from=str(source)),
              StartPoint("zero")]
    result = MultiStartRunner(
        new_problem, _cg_solver_factory,
        lambda budget: RelativeResidual(tol=1e-12, max_iter=budget),
        starts, total_budget=300, workdir=tmp_path / "ms", slice_iter=5).run()
    assert result.converged
    origins = {p.label: p.origin for p in result.paths}
    assert origins["from-old"] == f"restart_from={source}"
    # 源检查点不被覆盖
    assert source.with_suffix(".json").read_text(encoding="utf-8") == source_meta_before
    meta = json.loads(source_meta_before)
    assert meta["problem"] == old_problem.identity()
    assert meta["iteration"] == 4
