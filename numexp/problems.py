"""问题层：线性方程组及其前提校验。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse


class ProblemError(ValueError):
    """问题本身不合法（维度不匹配、非实数等）。"""


@dataclass
class LinearSystem:
    """稀疏（或密集）线性方程组 A x = b。

    Attributes:
        A: 系数矩阵，优先使用 scipy 稀疏矩阵。
        b: 右端项。
        x_true: 可选的已知真解，用于实验中度量误差；不参与求解。
        name/description: 实验标注信息。
    """

    A: "np.ndarray | sparse.spmatrix"
    b: np.ndarray
    x_true: np.ndarray | None = None
    name: str = "unnamed"
    description: str = ""

    def __post_init__(self) -> None:
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        if self.x_true is not None:
            self.x_true = np.asarray(self.x_true, dtype=float).reshape(-1)
        if sparse.issparse(self.A):
            if self.A.format not in ("csr", "csc"):
                self.A = self.A.tocsr()
        else:
            self.A = np.asarray(self.A, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ProblemError("A 必须是方阵")
        n = self.A.shape[0]
        if self.b.shape != (n,):
            raise ProblemError(f"b 的形状与 A 不兼容：A 为 {n} 阶，b 为 {self.b.shape}")
        if not np.all(np.isfinite(self.b)):
            raise ProblemError("b 中含有 NaN 或 Inf")
        if self.x_true is not None and self.x_true.shape != (n,):
            raise ProblemError("x_true 与 A 的维度不匹配")

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def is_sparse(self) -> bool:
        return sparse.issparse(self.A)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """统一的矩阵-向量乘法入口。"""
        if sparse.issparse(self.A):
            return self.A @ x
        return self.A @ x

    def residual(self, x: np.ndarray) -> np.ndarray:
        """真实残差向量 b - A x（独立于求解器内部递推量）。"""
        return self.b - self.matvec(x)

    def fingerprint(self) -> str:
        """问题数据的指纹，检查点续算时用于确认问题未被调换。"""
        import hashlib

        h = hashlib.sha256()
        if sparse.issparse(self.A):
            coo = self.A.tocoo()
            h.update(repr((coo.shape, coo.nnz)).encode())
            h.update(coo.row.tobytes())
            h.update(coo.col.tobytes())
            h.update(coo.data.tobytes())
        else:
            h.update(self.A.tobytes())
        h.update(self.b.tobytes())
        return h.hexdigest()
