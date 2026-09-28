"""实验案例：良性、病态、不收敛三类。"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from numexp.problems import LinearSystem


def poisson2d(k: int = 8) -> LinearSystem:
    """二维泊松方程五点差分：大型稀疏对称正定，适合 CG。n = k*k。"""
    n = k * k
    # n=k*k 网格的标准五点模板：对角 4，横向相邻 +/-1，纵向相邻 +/-k；
    # 每行的块边界用掩码消去跨行的水平耦合。
    main = 4.0 * np.ones(n)
    horiz = -np.ones(n - 1)
    horiz[k - 1 :: k] = 0.0
    vert = -np.ones(n - k)
    A = sparse.diags(
        [vert, horiz, main, horiz, vert],
        [-k, -1, 0, 1, k],
        shape=(n, n),
        format="csr",
    )
    x_true = np.arange(1, n + 1, dtype=float)
    b = A @ x_true
    return LinearSystem(A, b, x_true=x_true, name=f"poisson2d_{k}x{k}",
                        description="二维五点差分拉普拉斯，稀疏 SPD")


def diagonally_dominant(n: int = 50) -> LinearSystem:
    """严格对角占优的对称系统：Jacobi 保证收敛，CG 也适用。"""
    rng = np.random.default_rng(42)
    B = sparse.random(n, n, density=0.05, random_state=rng, format="csr")
    B = B - sparse.diags(B.diagonal())
    B = 0.5 * (B + B.T)  # 对称化，使其同时满足 CG 前提
    A = sparse.diags(np.asarray(abs(B).sum(axis=1)).ravel() + 2.0) + B
    A = A.tocsr()
    x_true = rng.standard_normal(n)
    b = A @ x_true
    return LinearSystem(A, b, x_true=x_true, name=f"diag_dominant_n{n}",
                        description="严格对角占优，Jacobi 收敛")


def hilbert(n: int = 12) -> LinearSystem:
    """Hilbert 矩阵：对称正定但极度病态，迭代收敛缓慢。"""
    i = np.arange(1, n + 1)
    A = 1.0 / (i[:, None] + i[None, :] - 1)
    x_true = np.ones(n)
    b = A @ x_true
    return LinearSystem(A, b, x_true=x_true, name=f"hilbert_n{n}",
                        description="Hilbert 矩阵，条件数极大")


def indefinite_symmetric(n: int = 20) -> LinearSystem:
    """对称但不定的矩阵（含负特征值）：CG 应拒绝或数值崩溃。"""
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.linspace(1.0, n, n)
    eigs[0] = -3.0
    A = q @ np.diag(eigs) @ q.T
    A = 0.5 * (A + A.T)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    return LinearSystem(A, b, x_true=x_true, name=f"indefinite_n{n}",
                        description="对称不定，不满足 CG 前提")


def nonsymmetric(n: int = 10) -> LinearSystem:
    """非对称矩阵：CG 必须拒绝。"""
    rng = np.random.default_rng(11)
    M = rng.standard_normal((n, n))
    A = M - M.T + 4.0 * np.eye(n)  # 反对称部分使其明确非对称，对角占优保证可解
    x_true = rng.standard_normal(n)
    b = A @ x_true
    return LinearSystem(A, b, x_true=x_true, name=f"nonsymmetric_n{n}",
                        description="非对称矩阵，违反 CG 前提")


def jacobi_divergent(n: int = 6) -> LinearSystem:
    """Jacobi 迭代矩阵谱半径大于 1 的小系统：迭代发散。"""
    # 对称矩阵，仅第一行严重不满足对角占优，整体非奇异且可逆
    A = np.eye(n)
    A[0, 1] = A[1, 0] = 2.0
    A[1, 2] = A[2, 1] = 2.0
    b = np.ones(n)
    x_true = np.linalg.solve(A, b)
    return LinearSystem(A, b, x_true=x_true, name="jacobi_divergent",
                        description="Jacobi 迭代矩阵谱半径>1，迭代发散")


def zero_diagonal(n: int = 5) -> LinearSystem:
    """含零对角元的系统：Jacobi 前提不满足，必须拒绝。"""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((n, n))
    A = 0.5 * (A + A.T) + n * np.eye(n)
    A[0, 0] = 0.0
    b = rng.standard_normal(n)
    return LinearSystem(A, b, name="zero_diagonal",
                        description="含零对角元，违反 Jacobi 前提")
