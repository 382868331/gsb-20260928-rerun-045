"""小规模参考解：仅用于对照实验，不被迭代求解器调用。

限制为小规模密集系统，避免在框架核心里包一层现成完整求解器来“解题”。
"""
from __future__ import annotations

import numpy as np

from numexp.problems import LinearSystem


def direct_solve(problem: LinearSystem, *, max_n: int = 200) -> np.ndarray:
    """用 LAPACK 直接法求参考解；超过 max_n 拒绝，防止误用为核心求解手段。"""
    if problem.n > max_n:
        raise ValueError(
            f"直接对照解仅支持 n <= {max_n} 的小规模问题，当前 n={problem.n}"
        )
    A = problem.A.toarray() if hasattr(problem.A, "toarray") else np.asarray(problem.A)
    return np.linalg.solve(A, problem.b)
