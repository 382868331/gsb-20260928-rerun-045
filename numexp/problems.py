"""问题层：描述待求解的问题并负责方法前提检查。

第一版只包含大型稀疏线性方程组 A x = b，但接口按通用问题设计，
后续加入特征值、非线性方程等问题时可复用停止条件与结果检查。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from numpy.typing import ArrayLike


class PreconditionError(ValueError):
    """输入不满足求解方法前提条件时抛出。"""


class Problem:
    """所有数值问题的抽象基类。"""

    n: int


class LinearSystem(Problem):
    """线性方程组 A x = b。

    Parameters
    ----------
    A : 二维方阵或线性算子，形状 (n, n)。
    b : 右端项，形状 (n,)。
    """

    def __init__(self, A: ArrayLike, b: ArrayLike):
        b = np.asarray(b, dtype=float)
        is_sparse = hasattr(A, "shape") and hasattr(A, "toarray")
        A = A if is_sparse else np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise PreconditionError("A 必须是方阵，得到形状 " + str(A.shape))
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise PreconditionError("b 的长度必须与 A 的阶数一致")
        self.A = A
        self.b = b
        self.n = A.shape[0]

    def identity(self) -> dict:
        """返回由矩阵和右端项内容确定的问题身份。"""
        return {
            "type": type(self).__name__,
            "n": int(self.n),
            "A_sha256": _fingerprint_matrix(self.A),
            "b_sha256": _fingerprint_vector(self.b),
        }

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """计算 A @ x；支持稀疏矩阵。"""
        return self.A @ x

    def true_residual(self, x: np.ndarray) -> np.ndarray:
        """真实残差 r = b - A x（直接由原矩阵计算，不使用迭代递推值）。"""
        return self.b - self.matvec(x)

    def residual_norm(self, x: np.ndarray, ord: int | float | None = 2) -> float:
        """真实残差的范数。"""
        return float(np.linalg.norm(self.true_residual(x), ord=ord))

    # ---- 方法前提检查 -------------------------------------------------

    def check_symmetric(self, tol: float | None = None) -> None:
        """检查 A 是否（在给定容差下）对称，否则拒绝。"""
        try:
            dense = self.A.toarray() if hasattr(self.A, "toarray") else np.asarray(self.A)
        except Exception as exc:
            raise PreconditionError("无法对该矩阵做对称性检查") from exc
        if dense.ndim != 2:
            raise PreconditionError("无法对非线性算子做对称性检查")
        if tol is None:
            tol = 1e-10 * max(1.0, float(np.max(np.abs(np.diag(dense)))))
        asymmetry = float(np.max(np.abs(dense - dense.T)))
        if asymmetry > tol:
            raise PreconditionError(
                f"矩阵不对称：max|A - A^T| = {asymmetry:.3e} > tol = {tol:.3e}")

    def check_spd(self, tol: float | None = None) -> None:
        """检查 A 对称正定。

        小规模矩阵通过 Cholesky 分解给出确定结论；
        大规模矩阵不做昂贵的稠密分解，先查对称性与对角元符号，
        正定性最终由迭代中遇到非正曲率（CG breakdown）确认。
        """
        self.check_symmetric(tol)
        dense = self.A.toarray() if hasattr(self.A, "toarray") else np.asarray(self.A)
        diag = np.diag(dense)
        if np.any(diag <= 0.0):
            raise PreconditionError("矩阵对角元存在非正值，不可能正定")
        if self.n <= 200:
            try:
                np.linalg.cholesky(dense)
            except np.linalg.LinAlgError as exc:
                raise PreconditionError("Cholesky 分解失败，矩阵非正定") from exc

    def check_nonsingular(self) -> None:
        """检查 A 非奇异（仅用于小规模问题）。"""
        if self.n > 200:
            raise PreconditionError("稠密奇异值检查不允许用于 n > 200 的矩阵")
        dense = self.A.toarray() if hasattr(self.A, "toarray") else np.asarray(self.A)
        if np.linalg.matrix_rank(dense) < self.n:
            raise PreconditionError("矩阵奇异")


def _fingerprint_vector(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": values.shape}).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _fingerprint_matrix(A) -> str:
    digest = hashlib.sha256()
    if hasattr(A, "tocoo"):
        sparse = A.tocoo(copy=True)
        sparse.sum_duplicates()
        nonzero = sparse.data != 0.0
        rows = sparse.row[nonzero]
        cols = sparse.col[nonzero]
        data = sparse.data[nonzero]
        order = np.lexsort((cols, rows))
        rows = rows[order].astype(np.int64, copy=False)
        cols = cols[order].astype(np.int64, copy=False)
        data = np.ascontiguousarray(data[order], dtype=np.float64)
        entries = (rows, cols, data)
        shape = sparse.shape
    else:
        dense = np.asarray(A, dtype=np.float64)
        rows, cols = np.nonzero(dense)
        entries = (rows.astype(np.int64, copy=False),
                   cols.astype(np.int64, copy=False),
                   np.ascontiguousarray(dense[rows, cols], dtype=np.float64))
        shape = dense.shape

    digest.update(json.dumps({"shape": shape}).encode("utf-8"))
    for values in entries:
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()
