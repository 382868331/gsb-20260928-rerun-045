"""框架测试：前提检查、真实残差、停止分类、检查点续算、对照。"""

import json

import numpy as np
import pytest

from numexp import (ConjugateGradient, Richardson, RelativeResidual,
                    PreconditionError, TerminationReason, LinearSystem)
from numexp import (StagnationGuard, assess_reuse, solve_problem_series,
                    StartPath, MultiStartSession)
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


def test_stagnation_guard_classified():
    problem, _ = spd_1d_poisson(40)
    criterion = StagnationGuard(
        inner=RelativeResidual(tol=1e-12, max_iter=5000),
        window=10, min_decay=0.99)
    result = Richardson(0.01, criterion).solve(problem)
    assert result.reason is TerminationReason.STAGNATED
    assert not result.success


def test_assess_reuse_reports_changes():
    problem, _ = spd_1d_poisson(20, seed=7)
    same = assess_reuse(problem, problem,
                        RelativeResidual(tol=1e-8), RelativeResidual(tol=1e-8))
    assert same.resumable
    assert not same.anything_changed

    changed_A = LinearSystem(problem.A + 0.01 * np.eye(problem.n), problem.b)
    report = assess_reuse(problem, changed_A,
                          RelativeResidual(tol=1e-8), RelativeResidual(tol=1e-10))
    assert report.matrix_changed and not report.rhs_changed
    assert report.criterion_changed
    assert not report.resumable
    assert any("热启动" in s for s in report.reusable)
    assert any("内部迭代状态" in s for s in report.not_reusable)
    assert "矩阵" in report.summary() and "容差" in report.summary()

    bigger, _ = spd_1d_poisson(30, seed=7)
    dim_report = assess_reuse(problem, bigger)
    assert dim_report.dimension_changed
    assert not dim_report.reusable
    assert dim_report.criterion_changed is None


def test_series_warm_and_cold_are_independent_runs():
    base, _ = spd_1d_poisson(20, seed=11)
    problems = [
        base,
        LinearSystem(base.A + 0.05 * np.eye(base.n), base.b),
        LinearSystem(base.A + 0.05 * np.eye(base.n), base.b + 0.2),
    ]
    make_solver = lambda: ConjugateGradient(
        RelativeResidual(tol=1e-10, max_iter=500))
    series = solve_problem_series(problems, make_solver)
    assert len(series.steps) == 3
    assert series.steps[0].warm is None
    prev_best = series.steps[0].best
    for index, step in enumerate(series.steps):
        assert step.problem_identity == problems[index].identity()
        for result in (step.cold, step.warm):
            if result is None:
                continue
            assert result.success
            # 每次求解独立从迭代 0 记录，目标问题身份各自独立
            assert result.history[0].iteration == 0
            assert result.problem_identity == problems[index].identity()
            assert result.continuation == "fresh"
        if index > 0:
            # 热启动初值残差按当前问题重新计算
            expected = problems[index].residual_norm(prev_best.x)
            assert step.warm.history[0].residual_norm == pytest.approx(expected)
            assert step.reuse is not None and not step.reuse.resumable
        prev_best = step.best
    assert series.steps[1].reuse.matrix_changed
    assert series.steps[2].reuse.rhs_changed


def _richardson_factory(omega_by_label):
    def make(criterion, checkpoint, path):
        return Richardson(omega_by_label[path.label], criterion,
                          checkpoint=checkpoint)
    return make


def test_multistart_failure_isolated_and_best_by_true_residual(tmp_path):
    problem, _ = spd_1d_poisson(30)
    paths = [StartPath("divergent", origin="fresh"),
             StartPath("good", origin="fresh")]
    session = MultiStartSession(
        problem,
        _richardson_factory({"divergent": 1.0, "good": 0.4}),
        paths, total_budget=20000,
        criterion=RelativeResidual(tol=1e-8),
        stagnation=(20, 0.95),
        session_dir=tmp_path / "session")
    result = session.run()
    by_label = {r.label: r for r in result.records}
    # 发散路径停滞或异常结束，不影响好路径收敛
    assert by_label["divergent"].status in (
        TerminationReason.STAGNATED.value,
        TerminationReason.NUMERICAL_BREAKDOWN.value,
        TerminationReason.BUDGET_EXHAUSTED.value)
    assert by_label["good"].converged
    best = result.best(problem)
    assert best.label == "good"
    assert problem.residual_norm(best.x) == pytest.approx(best.best_residual)
    assert result.budget_used <= result.budget_total


def test_multistart_precondition_failure_isolated():
    problem = nonsymmetric_system()

    def make(criterion, checkpoint, path):
        if path.label == "cg":
            return ConjugateGradient(criterion, checkpoint=checkpoint)
        return Richardson(0.1, criterion, checkpoint=checkpoint)

    paths = [StartPath("cg"), StartPath("richardson")]
    session = MultiStartSession(
        problem, make, paths, total_budget=20000,
        criterion=RelativeResidual(tol=1e-8))
    result = session.run()
    by_label = {r.label: r for r in result.records}
    assert by_label["cg"].status == TerminationReason.PRECONDITION_FAILED.value
    assert by_label["richardson"].finished


def test_multistart_pause_resume_preserves_paths(tmp_path):
    problem, _ = spd_1d_poisson(30)
    session_dir = tmp_path / "session"
    make = _richardson_factory({"a": 0.4, "b": 0.4})
    paths = [StartPath("a", origin="fresh"), StartPath("b", origin="fresh")]
    session = MultiStartSession(
        problem, make, paths, total_budget=50,
        criterion=RelativeResidual(tol=1e-8), session_dir=session_dir)
    first = session.run()
    assert first.pending, "预算应不足以完成全部路径"
    assert (session_dir / "session.json").exists()
    path_ckpt = session_dir / "paths" / "a.npz"
    assert path_ckpt.exists()

    restored = MultiStartSession.load(session_dir, problem, make,
                                      criterion=RelativeResidual(tol=1e-8))
    assert restored.budget_used == first.budget_used
    assert [r.origin for r in restored.records] == ["fresh", "fresh"]
    second = restored.run(extra_budget=20000)
    assert not second.pending
    assert all(r.converged for r in second.records)
    # 续跑路径的累计预算包含暂停前消耗的步数
    assert all(r.iterations > 0 for r in second.records)
    best = second.best(problem)
    assert best is not None and problem.residual_norm(best.x) < 1e-6

    with pytest.raises(ValueError, match="目标问题与当前问题不一致"):
        other, _ = spd_1d_poisson(30, seed=99)
        MultiStartSession.load(session_dir, other, make)


def test_multistart_restart_source_checkpoint_not_overwritten(tmp_path):
    problem, _ = spd_1d_poisson(20, seed=5)
    source = tmp_path / "old-cg"
    old = ConjugateGradient(RelativeResidual(tol=1e-12, max_iter=3),
                            checkpoint=str(source)).solve(problem)
    assert old.reason is TerminationReason.BUDGET_EXHAUSTED
    paths = [StartPath("warm", origin="checkpoint:old-cg",
                       restart_from=str(source)),
             StartPath("cold", origin="fresh")]
    make = _richardson_factory({"warm": 0.4, "cold": 0.4})
    session = MultiStartSession(
        problem, make, paths, total_budget=20000,
        criterion=RelativeResidual(tol=1e-8),
        session_dir=tmp_path / "session")
    result = session.run()
    assert all(r.converged for r in result.records)
    old_meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    assert old_meta["solver"] == "cg"
    assert old_meta["iteration"] == 3
    assert old_meta["problem"] == problem.identity()


# ---- 参数连续跟踪 -----------------------------------------------------

def _continuation_family(t):
    """小问题族：t 小接近单位阵（平稳段），t 大条件数上升（难收敛段）。"""
    n = 6
    diag = 1.0 + t * np.arange(1, n + 1)
    A = np.diag(diag) + 0.05 * t * (np.ones((n, n)) - np.eye(n))
    return LinearSystem(A, np.ones(n))


def _make_continuation(tmp_path=None, budget=3000, **kwargs):
    from numexp import ContinuationSession
    x0 = np.linalg.solve(_continuation_family(0.0).A, np.ones(6))
    options = dict(criterion=RelativeResidual(tol=1e-4),
                   initial_step=0.5, min_step=1e-3, max_step_iters=5, x0=x0)
    options.update(kwargs)
    return ContinuationSession(_continuation_family, 0.0, 1.0,
                               lambda c: ConjugateGradient(c),
                               total_budget=budget, **options)


def test_continuation_backtracks_and_reaches_end():
    session = _make_continuation()
    result = session.run()
    scheme = result.schemes[0]
    assert scheme.reached_end and scheme.status == "completed"
    # 失败尝试存在且消耗计入统一预算
    failed = [a for a in result.attempts if not a.converged]
    assert failed, "难收敛段应触发失败回退"
    assert result.budget_used == sum(a.iterations for a in result.attempts)
    assert result.budget_used == sum(n.iterations for n in scheme.nodes) + \
        sum(a.iterations for a in failed)
    # 每次失败后以更小步长从同一已确认节点重新出发
    for attempt in failed:
        retry = next(a for a in result.attempts
                     if a.t_from == attempt.t_from and a.step < attempt.step)
        assert retry.t_target < attempt.t_target
    # 失败尝试的参数不会变成已确认节点（除非后来真正收敛）
    node_ts = [n.t for n in scheme.nodes]
    for attempt in failed:
        later = [a for a in result.attempts
                 if a.t_target == attempt.t_target and a.converged]
        if not later:
            assert attempt.t_target not in node_ts
    # 每个已确认节点按当前参数重算真实残差都满足精度
    for node in scheme.nodes[1:]:
        problem = _continuation_family(node.t)
        assert problem.residual_norm(node.x) <= 1e-4 * np.linalg.norm(problem.b) * 1.05
    # 终点解可独立验证
    final = session.final_solution("main")
    assert final is not None
    assert _continuation_family(1.0).residual_norm(final) < 1e-3


def test_continuation_incomplete_path_not_substituted(tmp_path):
    session = _make_continuation(tmp_path, budget=12,
                                 session_dir=str(tmp_path / "sess"))
    result = session.run()
    scheme = result.schemes[0]
    assert not scheme.reached_end
    assert result.incomplete and not result.completed
    # 未到达终点：final_solution 与 compare_at 都不以最近结果代替
    assert session.final_solution("main") is None
    assert session.compare_at(1.0) == []
    # 暂停重开：预算与尝试记录保留，追加预算后沿原路径继续
    used_before = result.budget_used
    attempts_before = len(result.attempts)
    from numexp import ContinuationSession
    resumed = ContinuationSession.load(tmp_path / "sess", _continuation_family,
                                       lambda c: ConjugateGradient(c),
                                       criterion=RelativeResidual(tol=1e-4))
    assert resumed.budget_used == used_before
    assert len(resumed.attempts) == attempts_before
    result2 = resumed.run(extra_budget=3000)
    assert result2.schemes[0].reached_end
    assert result2.budget_used > used_before
    assert len(result2.attempts) > attempts_before
    assert resumed.final_solution("main") is not None


def test_continuation_load_rejects_changed_family(tmp_path):
    session = _make_continuation(tmp_path, budget=12,
                                 session_dir=str(tmp_path / "sess"))
    session.run()
    from numexp import ContinuationSession
    with pytest.raises(ValueError, match="不一致"):
        ContinuationSession.load(tmp_path / "sess",
                                 lambda t: _continuation_family(t + 1.0),
                                 lambda c: ConjugateGradient(c))


def test_continuation_fork_and_compare_at_same_parameter(tmp_path):
    session = _make_continuation(tmp_path, budget=3000,
                                 session_dir=str(tmp_path / "sess"))
    result = session.run()
    main = result.schemes[0]
    assert main.reached_end
    anchor = main.nodes[1]
    nodes_before = [(n.t, n.origin) for n in main.nodes]
    budget_before = result.budget_used
    # 从旧节点另开方案：原节点与预算记录不被覆盖
    fork = session.fork_scheme("main", 1, "forked")
    assert fork.nodes[0].origin == "fork:main#node1"
    assert fork.nodes[0].t == anchor.t
    result2 = session.run()
    assert [(n.t, n.origin) for n in main.nodes] == nodes_before
    assert result2.schemes[1].reached_end
    assert result2.budget_used > budget_before
    # 两条路径在同一参数处的解可比较，残差按该参数问题重算
    entries = session.compare_at(anchor.t)
    assert {e["scheme"] for e in entries} == {"main", "forked"}
    problem = _continuation_family(anchor.t)
    for entry in entries:
        assert entry["residual"] == pytest.approx(
            problem.residual_norm(entry["x"]))
    # 两条路径都真正到达终点
    finals = session.compare_at(1.0)
    assert {e["scheme"] for e in finals} == {"main", "forked"}
    for entry in finals:
        assert entry["residual"] < 1e-3


def test_continuation_start_requires_actual_solve():
    """起点不直接算已确认：首次运行按 t_start 处问题求解验证。"""
    from numexp import ContinuationSession
    # 独立七维线性族：零初值在 t=0 处真实残差 ||b|| = sqrt(140) ≈ 11.83
    def family(t):
        n = 7
        b = 1.0 + np.arange(n)
        diag = 2.0 + np.arange(n) + t
        A = np.diag(diag) + 0.1 * t * (np.ones((n, n)) - np.eye(n))
        return LinearSystem(A, b)
    # 无效初值：零向量残差大，单步预算内无法收敛
    session = ContinuationSession(family, 0.0, 1.0,
                                  lambda c: ConjugateGradient(c),
                                  total_budget=3000,
                                  criterion=RelativeResidual(tol=1e-4),
                                  initial_step=0.5, min_step=1e-3,
                                  max_step_iters=2)
    assert session.schemes[0].nodes == []  # 构造时不确认
    result = session.run()
    scheme = result.schemes[0]
    start_attempts = [a for a in result.attempts
                      if a.t_from == 0.0 and a.t_target == 0.0]
    assert start_attempts, "起点确认必须实际求解并记录尝试"
    first = start_attempts[0]
    assert not first.converged
    assert first.residual > 1e-4 * np.linalg.norm(1.0 + np.arange(7))
    # 起点未确认：不列节点、不算到达、final_solution 为 None
    assert all(n.t != 0.0 for n in scheme.nodes)
    assert session.final_solution("main") is None
    # 确认消耗计入统一预算
    assert result.budget_used >= first.iterations > 0


def test_continuation_start_valid_x0_confirmed_at_start_problem():
    """有效初值：满足 t_start 方程与精度即确认，残差可按自身参数重算。"""
    session = _make_continuation()
    result = session.run()
    scheme = result.schemes[0]
    root = scheme.nodes[0]
    assert root.t == 0.0
    problem0 = _continuation_family(0.0)
    assert problem0.residual_norm(root.x) <= 1e-4 * np.linalg.norm(problem0.b) * 1.05
    # verify_node 按节点自身参数重算残差
    for idx in range(len(scheme.nodes)):
        node = scheme.nodes[idx]
        expected = _continuation_family(node.t).residual_norm(node.x)
        assert session.verify_node("main", idx) == pytest.approx(expected)


def test_continuation_start_degenerate_exact_x0_zero_cost():
    """退化情形：x0 已是精确解，确认零消耗且仍记录一次尝试。"""
    session = _make_continuation()  # x0 为 t=0 处精确解
    result = session.run()
    start_attempts = [a for a in result.attempts
                      if a.t_from == 0.0 and a.t_target == 0.0]
    assert len(start_attempts) == 1
    assert start_attempts[0].converged
    assert start_attempts[0].iterations == 0
    assert result.schemes[0].nodes[0].iterations == 0


def test_continuation_start_confirmation_budget_pause_resume(tmp_path):
    """预算不足：起点确认标记待续，重开后不跳验证也不重复扣减。"""
    from numexp import ContinuationSession
    def family(t):
        n = 7
        b = 1.0 + np.arange(n)
        diag = 2.0 + np.arange(n) + t
        A = np.diag(diag) + 0.1 * t * (np.ones((n, n)) - np.eye(n))
        return LinearSystem(A, b)
    kwargs = dict(criterion=RelativeResidual(tol=1e-12),
                  initial_step=0.5, min_step=1e-3, max_step_iters=10,
                  session_dir=str(tmp_path / "sess"))
    session = ContinuationSession(family, 0.0, 1.0,
                                  lambda c: ConjugateGradient(c),
                                  total_budget=3, **kwargs)
    result = session.run()
    scheme = result.schemes[0]
    assert scheme.status == "paused"
    assert not scheme.nodes
    used_before = result.budget_used
    assert used_before == 3
    # 暂停重开：待确认起点保留，追加预算后续验，不重复扣减
    resumed = ContinuationSession.load(tmp_path / "sess", family,
                                       lambda c: ConjugateGradient(c),
                                       criterion=RelativeResidual(tol=1e-12))
    assert resumed.schemes[0].pending is not None
    assert resumed.budget_used == used_before
    result2 = resumed.run(extra_budget=3000)
    scheme2 = result2.schemes[0]
    assert scheme2.nodes and scheme2.nodes[0].t == 0.0
    assert scheme2.pending is None
    # 预算账目一致：总消耗等于全部尝试之和，且不重扣首次确认
    assert result2.budget_used == sum(a.iterations for a in result2.attempts)
    start_attempts = [a for a in result2.attempts
                      if a.t_from == 0.0 and a.t_target == 0.0]
    assert sum(a.iterations for a in start_attempts) + \
        sum(a.iterations for a in result2.attempts
            if not (a.t_from == 0.0 and a.t_target == 0.0)) == result2.budget_used


# ---- 节点核验报告 -----------------------------------------------------

def _report_family(t):
    n = 6
    diag = 1.0 + t * np.arange(1, n + 1)
    A = np.diag(diag) + 0.05 * t * (np.ones((n, n)) - np.eye(n))
    return LinearSystem(A, np.ones(n))


def _make_report_session(tmp_path=None, budget=3000):
    from numexp import ContinuationSession
    x0 = np.linalg.solve(_report_family(0.0).A, np.ones(6))
    kwargs = dict(criterion=RelativeResidual(tol=1e-4),
                  initial_step=0.5, min_step=1e-3, max_step_iters=5, x0=x0)
    if tmp_path is not None:
        kwargs["session_dir"] = str(tmp_path / "sess")
    return ContinuationSession(_report_family, 0.0, 1.0,
                               lambda c: ConjugateGradient(c),
                               total_budget=budget, **kwargs)


def test_verification_report_confirmed_nodes_independently_checked():
    """有效初值：每个确认节点带参数值、独立重算残差、容差与结论。"""
    session = _make_report_session()
    session.run()
    report = session.verification_report()
    scheme = report["schemes"][0]
    assert scheme["stopped_at"]["kind"] == "end"
    assert scheme["unconfirmed_start"] is None
    nodes = scheme["confirmed_nodes"]
    assert len(nodes) == len(session.schemes[0].nodes) > 2
    for entry in nodes:
        problem = _report_family(entry["t"])
        assert entry["residual"] == pytest.approx(
            problem.residual_norm(session.schemes[0].nodes[entry["index"]].x))
        assert entry["tolerance"]["tol"] == 1e-4
        assert entry["threshold"] == pytest.approx(
            1e-4 * np.linalg.norm(problem.b))
        assert entry["conclusion"] == "confirmed_pass"
        assert entry["residual"] <= entry["threshold"]
    # 起点节点为精确初值：零消耗确认
    assert nodes[0]["t"] == 0.0 and nodes[0]["iterations"] == 0
    # 失败尝试单独成节，不混入确认节点
    assert report["failed_attempts"]
    failed_targets = {(a["t_from"], a["t_target"]) for a in report["failed_attempts"]}
    node_ts = {n["t"] for n in nodes}
    for t_from, t_target in failed_targets:
        if all(a["status"] != "converged"
               for a in report["failed_attempts"]
               if a["t_target"] == t_target):
            assert t_target not in node_ts


def test_verification_report_budget_partition_and_text_json(tmp_path):
    """预算按起点校正/后续推进/失败尝试三类汇总，文本与 JSON 可导出。"""
    session = _make_report_session(tmp_path)
    session.run()
    report = session.verification_report()
    bg = report["budget"]
    assert bg["used"] == (bg["start_confirmation"] + bg["advances"]
                           + bg["failed_attempts"])
    assert bg["used"] + bg["remaining"] == bg["total"]
    assert bg["advances"] == sum(n["iterations"]
                                 for n in report["schemes"][0]["confirmed_nodes"])
    assert bg["failed_attempts"] == sum(a["iterations"]
                                         for a in report["failed_attempts"])
    json_path = tmp_path / "report.json"
    text_path = tmp_path / "report.txt"
    json_text = session.export_report_json(json_path)
    plain_text = session.export_report_text(text_path)
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed == json.loads(json_text)
    assert text_path.read_text(encoding="utf-8") == plain_text
    assert "节点核验报告" in plain_text
    assert "已确认节点" in plain_text and "失败尝试" in plain_text


def test_verification_report_save_load_stable(tmp_path):
    """保存恢复后再次导出：节点数值、结论与预算完全一致。"""
    from numexp import ContinuationSession
    session = _make_report_session(tmp_path, budget=12)
    session.run()  # 预算不足，暂停在后续推进上
    json_before = session.export_report_json()
    text_before = session.export_report_text()
    resumed = ContinuationSession.load(tmp_path / "sess", _report_family,
                                       lambda c: ConjugateGradient(c),
                                       criterion=RelativeResidual(tol=1e-4))
    assert resumed.export_report_json() == json_before
    assert resumed.export_report_text() == text_before
    # 追加预算跑到终点后再次保存/恢复仍一致
    resumed.run(extra_budget=3000)
    resumed.save()
    json_done = resumed.export_report_json()
    resumed2 = ContinuationSession.load(tmp_path / "sess", _report_family,
                                        lambda c: ConjugateGradient(c),
                                        criterion=RelativeResidual(tol=1e-4))
    assert resumed2.export_report_json() == json_done
    assert resumed2.export_report_text() == resumed.export_report_text()


def _start_report_family(t):
    n = 7
    b = 1.0 + np.arange(n)
    diag = 2.0 + np.arange(n) + t
    A = np.diag(diag) + 0.1 * t * (np.ones((n, n)) - np.eye(n))
    return LinearSystem(A, b)


def test_verification_report_invalid_initial_guess_listed_separately():
    """无效初值：起点未确认，初始猜测单独列出、方案停在起点。"""
    from numexp import ContinuationSession
    session = ContinuationSession(_start_report_family, 0.0, 1.0,
                                  lambda c: ConjugateGradient(c),
                                  total_budget=3000,
                                  criterion=RelativeResidual(tol=1e-4),
                                  initial_step=0.5, min_step=1e-3,
                                  max_step_iters=2, x0=np.zeros(7))
    session.run()
    report = session.verification_report()
    scheme = report["schemes"][0]
    assert scheme["confirmed_nodes"] == []
    assert scheme["stopped_at"] == {"kind": "start_unconfirmed", "t": 0.0}
    pending = scheme["unconfirmed_start"]
    assert pending["t"] == 0.0 and pending["status"] == "stalled"
    assert pending["residual"] == pytest.approx(np.linalg.norm(1.0 + np.arange(7)))
    assert scheme["budget"]["start_confirmation"] == 0
    assert scheme["budget"]["advances"] == 0
    assert scheme["budget"]["failed_attempts"] > 0
    assert report["failed_attempts"]
    assert all(a["kind"] == "start_confirmation" for a in report["failed_attempts"])
    assert report["paused_attempts"] == []


def test_verification_report_budget_exhausted_paused_attempt(tmp_path):
    """预算耗尽：暂停尝试单列，恢复续跑后预算账目保持一致。"""
    from numexp import ContinuationSession
    session = ContinuationSession(
        _start_report_family, 0.0, 1.0, lambda c: ConjugateGradient(c),
        total_budget=3, criterion=RelativeResidual(tol=1e-12),
        initial_step=0.5, min_step=1e-3, max_step_iters=10,
        session_dir=str(tmp_path / "sess"))
    session.run()
    report = session.verification_report()
    scheme = report["schemes"][0]
    assert scheme["status"] == "paused"
    assert scheme["stopped_at"]["kind"] == "start_unconfirmed"
    assert scheme["confirmed_nodes"] == []
    assert scheme["unconfirmed_start"]["status"] == "paused"
    assert len(report["paused_attempts"]) == 1
    paused = report["paused_attempts"][0]
    assert paused["iterations"] == 3 and paused["status"] == "paused"
    assert report["failed_attempts"] == []
    assert report["budget"]["used"] == 3 and report["budget"]["remaining"] == 0
    # 暂停尝试的消耗不重复计入失败分类，但已在总消耗中扣减
    assert report["budget"]["start_confirmation"] == 0
    resumed = ContinuationSession.load(tmp_path / "sess", _start_report_family,
                                       lambda c: ConjugateGradient(c),
                                       criterion=RelativeResidual(tol=1e-12))
    resumed.run(extra_budget=3000)
    done = resumed.verification_report()
    bg = done["budget"]
    assert bg["used"] == (bg["start_confirmation"] + bg["advances"]
                           + bg["failed_attempts"])
    assert bg["used"] > 3
    assert done["schemes"][0]["confirmed_nodes"][0]["t"] == 0.0
    # 暂停重开不重复扣减：全部尝试消耗之和等于总消耗
    assert sum(a.iterations for a in resumed.attempts) == bg["used"]


def test_verification_report_singular_system_precondition_failure():
    """奇异系统：前提拒绝列为失败尝试，残差非有限时以 null 导出且 JSON 合法。"""
    from numexp import ContinuationSession

    def singular(t):
        n = 4
        A = np.eye(n)
        A[0, 0] = 0.0  # 奇异且非正定
        return LinearSystem(A, np.ones(n))

    session = ContinuationSession(singular, 0.0, 1.0,
                                  lambda c: ConjugateGradient(c),
                                  total_budget=100,
                                  criterion=RelativeResidual(tol=1e-8))
    session.run()
    report = session.verification_report()
    scheme = report["schemes"][0]
    assert scheme["status"] == "stalled"
    assert scheme["confirmed_nodes"] == []
    assert scheme["unconfirmed_start"] is not None
    failed = report["failed_attempts"]
    assert len(failed) == 1
    assert failed[0]["status"] == "precondition_failed"
    assert failed[0]["iterations"] == 0
    assert failed[0]["residual"] is None  # inf 不出现在 JSON 中
    assert report["budget"]["used"] == 0
    assert report["budget"]["failed_attempts"] == 0
    json_text = session.export_report_json()
    json.loads(json_text)  # 合法 JSON（不依赖非标准 NaN/Infinity）
    plain_text = session.export_report_text()
    assert "失败尝试" in plain_text and "precondition_failed" in plain_text
