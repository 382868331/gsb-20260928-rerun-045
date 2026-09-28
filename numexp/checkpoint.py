"""检查点：把迭代状态落盘，支持中断后续算。

矩阵 A 不写入检查点（大型稀疏矩阵可能很大）；续算时由调用方提供同一问题，
通过 fingerprint 校验数据未被调换。
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from numexp.problems import LinearSystem


class Checkpointer:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, problem: LinearSystem) -> Path:
        return self.directory / f"checkpoint_{problem.name}.npz"

    def save(self, solver, problem: LinearSystem, state, records, history, iteration) -> Path:
        payload = {
            "solver_name": np.array(solver.name),
            "fingerprint": np.array(problem.fingerprint()),
            "iteration": np.array(iteration),
            "history": np.asarray(history, dtype=float),
            "state_keys": np.array(list(state.keys())),
        }
        for key, value in state.items():
            payload[f"state_{key}"] = np.asarray(value)
        rec_arrays = [
            (r.iteration, r.residual_norm, r.relative_residual) for r in records
        ]
        payload["rec_iteration"] = np.array([t[0] for t in rec_arrays], dtype=int)
        payload["rec_residual"] = np.array([t[1] for t in rec_arrays], dtype=float)
        payload["rec_relative"] = np.array([t[2] for t in rec_arrays], dtype=float)
        path = self._path(problem)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(path)
        return path

    def load(self, problem: LinearSystem) -> dict | None:
        path = self._path(problem)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        if str(data["fingerprint"]) != problem.fingerprint():
            raise ValueError(f"检查点 {path.name} 的问题指纹不匹配：问题已更改或被调换")
        state = {
            str(key): np.array(data[f"state_{key}"])
            for key in data["state_keys"].tolist()
        }
        records = [
            {
                "iteration": int(it),
                "residual_norm": float(rn),
                "relative_residual": float(rel),
                "metrics": {},
            }
            for it, rn, rel in zip(
                data["rec_iteration"], data["rec_residual"], data["rec_relative"]
            )
        ]
        return {
            "solver_name": str(data["solver_name"]),
            "iteration": int(data["iteration"]),
            "history": data["history"].tolist(),
            "state": state,
            "records": records,
        }

    def clear(self, problem: LinearSystem) -> None:
        path = self._path(problem)
        if path.exists():
            path.unlink()
