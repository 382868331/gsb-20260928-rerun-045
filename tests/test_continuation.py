"""参数连续跟踪：回退、推进、预算、分支比较与暂停重开。"""

import numpy as np
import pytest

from numexp import (Richardson, RelativeResidual, LinearSystem,
                    ContinuationSession)


N = 30
B = np.ones(N)


def make_problem(t):
    """t<=0.4 时 A=I（平稳段，解不变化）；之后最小特征角降到 0.05（难收敛段）。"""
    s = max(0.0, (t - 0.4) / 0.6)
    lam_min = 10.0 ** (-1.3 * s)
    c = np.geomspace(lam_min, 1.0, N)
    return LinearSystem(np.diag(c), B)


def make_solver(criterion):
    return Richardson(omega=1.0, criterion=criterion)


def make_session(**kwargs):
    defaults = dict(param_start=0.0, param_end=1.0,
                    total_budget=40000,
                    criterion=RelativeResidual(tol=1e-6),
                    attempt_budget=200, initial_step=0.6)
    defaults.update(kwargs)
    return ContinuationSession(make_problem, make_solver, **defaults)


def test_reaches_target_with_full_records():
    session = make_session()
    result = session.run()
    path = result.paths[0]
    assert path.completed
    final = path.final_node
    assert final is not None and final.param == pytest.approx(1.0)
    # 每个节点保留实际参数、起点来源与消耗
    for node in path.nodes:
        assert node.source
        assert node.cost >= 0
        # 节点残差是按该节点参数的问题重算的真实残差
        assert node.residual_norm == pytest.approx(
            make_problem(node.param).residual_norm(node.x))
        assert node.residual_norm <= 1e-6 * np.linalg.norm(B) * 1.001
    # 参数单调推进到终点
    params = [n.param for n in path.nodes]
    assert params == sorted(params)
    # 统一预算同时覆盖成功步与失败尝试
    failed = [a for a in path.attempts if not a.accepted]
    assert failed
    assert result.budget_used == sum(a.cost for a in path.attempts)
    assert result.budget_used == path.total_cost


def test_failed_attempt_backtracks_and_never_becomes_node():
    session = make_session()
    path = session.run().paths[0]
    failed = [a for a in path.attempts if not a.accepted]
    assert failed, "大步长直接进入难收敛段应当失败过"
    accepted = [a for a in path.attempts if a.accepted]
    # 节点只来自收敛尝试：失败尝试的残差不进入节点序列
    assert len(path.nodes) == len(accepted)
    for attempt in failed:
        assert attempt.status != "converged"
    # 每次失败后，下一次尝试从上一个已确认节点出发且步长缩小
    for i, attempt in enumerate(path.attempts[:-1]):
        if attempt.accepted:
            continue
        nxt = path.attempts[i + 1]
        assert abs(nxt.step) < abs(attempt.step)
        last_confirmed = max((a.param for a in path.attempts[:i + 1]
                              if a.accepted), default=0.0)
        assert nxt.param == pytest.approx(last_confirmed + nxt.step)


def test_plateau_segment_converges_immediately():
    session = make_session()
    path = session.run().paths[0]
    # 平稳段（t<=0.4）内解不变，推进顺利且消耗极小
    plateau_nodes = [n for n in path.nodes if n.param <= 0.4]
    assert plateau_nodes
    assert all(np.allclose(n.x, np.ones(N)) for n in plateau_nodes)
    assert all(n.cost <= 1 for n in plateau_nodes)


def test_budget_pause_and_resume():
    session = make_session(total_budget=400)
    result = session.run()
    path = result.paths[0]
    assert not path.completed
    assert path.status == "paused"
    # 未到达终点：明确未完成，不能以最近节点代替终点结果
    assert path.final_node is None
    assert result.completed_paths == []
    used_before = result.budget_used
    attempts_before = len(path.attempts)
    assert used_before <= 400
    # 追加预算后继续原路径：原记录保留，预算累计
    result2 = session.run(extra_budget=40000)
    path2 = result2.paths[0]
    assert path2.completed
    assert path2.final_node.param == pytest.approx(1.0)
    assert result2.budget_used > used_before
    assert len(path2.attempts) > attempts_before
    assert result2.budget_used == sum(a.cost for a in path2.attempts)


def test_save_load_resume(tmp_path):
    session = make_session(total_budget=400,
                           session_dir=str(tmp_path / "cont"))
    result = session.run()
    path = result.paths[0]
    assert path.status == "paused"
    used_before = result.budget_used
    attempts_before = len(path.attempts)
    nodes_before = [(n.param, n.source, n.cost) for n in path.nodes]
    # 暂停重开：从会话目录恢复，原节点与预算记录不被覆盖
    restored = ContinuationSession.load(tmp_path / "cont",
                                        make_problem, make_solver,
                                        criterion=RelativeResidual(tol=1e-6))
    assert restored.budget_used == used_before
    rpath = restored.paths[0]
    assert len(rpath.attempts) == attempts_before
    assert [(n.param, n.source, n.cost) for n in rpath.nodes] == nodes_before
    result2 = restored.run(extra_budget=40000)
    path2 = result2.paths[0]
    assert path2.completed
    assert path2.final_node.param == pytest.approx(1.0)
    assert result2.budget_used > used_before
    assert [(n.param, n.source, n.cost)
            for n in path2.nodes][:len(nodes_before)] == nodes_before


def test_branch_from_node_and_compare_at_target():
    session = make_session()
    path = session.run().paths[0]
    assert path.completed
    mid = len(path.nodes) // 2
    main_nodes_before = [(n.param, n.cost) for n in path.nodes]
    main_attempts_before = len(path.attempts)
    main_budget_before = session.budget_used
    # 从旧节点另开方案：原节点与预算记录不被覆盖
    session.branch("main", mid, label="alt", initial_step=0.05)
    result = session.run()
    main, alt = result.paths
    assert [(n.param, n.cost) for n in main.nodes] == main_nodes_before
    assert len(main.attempts) == main_attempts_before
    assert alt.completed
    assert alt.nodes[0].source == f"branch:main#{mid}"
    assert session.budget_used > main_budget_before
    # 两条路径到达相同参数：按真实残差比较
    entries = session.compare_at(1.0)
    assert {e["path"] for e in entries} == {"main", "alt"}
    for entry in entries:
        assert entry["residual_norm"] == pytest.approx(
            make_problem(1.0).residual_norm(entry["x"]))
        assert entry["residual_norm"] <= 1e-6 * np.linalg.norm(B) * 1.001
    assert np.allclose(entries[0]["x"], entries[1]["x"], atol=1e-4)


def test_stalled_when_never_converges():
    # omega=3 对该问题族发散：任何小步都不收敛，路径应明确标记 stalled
    def bad_solver(criterion):
        return Richardson(omega=3.0, criterion=criterion)
    session = ContinuationSession(
        make_problem, bad_solver,
        param_start=0.0, param_end=1.0, total_budget=5000,
        criterion=RelativeResidual(tol=1e-6),
        attempt_budget=100, initial_step=0.5, min_step=0.05)
    path = session.run().paths[0]
    assert path.status == "stalled"
    assert not path.completed
    assert path.final_node is None
    assert any(not a.accepted for a in path.attempts)
