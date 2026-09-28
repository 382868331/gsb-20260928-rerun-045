"""小规模直接求解对照与测试矩阵生成。"""

from __future__ import annotations

import numpy as np

from .problems import LinearSystem


def direct_solve(problem: LinearSystem) -> np.ndarray:
    """小规模稠密直接求解（numpy.linalg.solve），仅用于对照验证。"""
    if problem.n > 500:
        raise ValueError("直接求解对照只允许用于小规模问题")
    return np.linalg.solve(np.asarray(problem.A), problem.b)


def spd_1d_poisson(n: int, kappa: float = 1.0, seed: int = 0):
    """生成 n 阶对称正定三对角矩阵 T 和右端项。

    kappa>1 时通过对角缩放制造病态（条件数约 O(kappa^2 * n^2)）。
    返回 (problem, info)。
    """
    rng = np.random.default_rng(seed)
    main = np.full(n, 2.0)
    if n >= 2:
        main[0] = 1.0 / kappa + 1.0
        main[-1] = 1.0 / kappa + 1.0
    A = np.diag(main) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
    x_exact = rng.standard_normal(n)
    b = A @ x_exact
    return LinearSystem(A, b), {"x_exact": x_exact}


def spd_sparse_laplacian(n: int, seed: int = 1):
    """n 阶稀疏 SPD（一维离散拉普拉斯，scipy 稀疏矩阵）。"""
    from scipy import sparse
    rng = np.random.default_rng(seed)
    A = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    x_exact = rng.standard_normal(n)
    b = A @ x_exact
    return LinearSystem(A, b), {"x_exact": x_exact}


def nonsymmetric_system(n: int = 8, seed: int = 2):
    """非对称正定（特征值为正）系统，用于验证 CG 前提拒绝。"""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)  # SPD
    A[0, 1] += 1.0                # 破坏对称性，仍非奇异
    b = rng.standard_normal(n)
    return LinearSystem(A, b)


def indefinite_system(n: int = 8, seed: int = 3):
    """对称不定系统，用于验证 CG 因非正定被拒绝/breakdown。"""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = rng.standard_normal(n)
    eig[0] = -abs(eig[0]) - 0.5  # 至少一个负特征值
    A = Q @ np.diag(eig) @ Q.T
    A = 0.5 * (A + A.T)
    b = rng.standard_normal(n)
    return LinearSystem(A, b)


def divergent_richardson_problem(n: int = 40, seed: int = 4):
    """使 omega*2 超出收敛区间的 Richardson 发散案例。

    T 特征值范围约 [0, 4]，omega=1 时谱半径 |1 - omega*lambda_max| ~ 3 > 1。
    """
    problem, info = spd_1d_poisson(n, seed=seed)
    return problem, info
