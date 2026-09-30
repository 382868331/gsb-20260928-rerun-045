"""检查点：保存/恢复迭代器内部状态以支持续算。

状态用 np.savez 保存为 .npz，元数据（求解器名、迭代号）存为 JSON。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Checkpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data_path = self.path.with_suffix(".npz")
        self.meta_path = self.path.with_suffix(".json")

    def resolved_path(self) -> str:
        return str(self.path.expanduser().absolute())

    def save(self, solver_name: str, iteration: int,
             arrays: dict[str, np.ndarray], *, problem_identity: dict,
             criterion_signature: dict, solver_settings: dict,
             **meta) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_data = self.data_path.with_name("." + self.data_path.stem + ".writing.npz")
        tmp_meta = self.meta_path.with_suffix(".json.tmp")
        np.savez(tmp_data, **arrays)
        payload = {
            "version": 1,
            "solver": solver_name,
            "iteration": int(iteration),
            "problem": problem_identity,
            "criterion": criterion_signature,
            "solver_settings": solver_settings,
            **meta,
        }
        tmp_meta.write_text(json.dumps(payload), encoding="utf-8")
        tmp_data.replace(self.data_path)
        tmp_meta.replace(self.meta_path)

    def exists(self) -> bool:
        return self.data_path.exists() and self.meta_path.exists()

    def load(self, solver_name: str | None = None
             ) -> tuple[int, dict[str, np.ndarray], dict]:
        if not self.exists():
            raise FileNotFoundError(f"检查点不存在：{self.path}")
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != 1:
            raise ValueError("检查点格式不受支持或缺少问题/求解设定身份")
        if solver_name is not None and meta.get("solver") != solver_name:
            raise ValueError(
                f"检查点属于 {meta.get('solver')!r}，不能用于 {solver_name!r}")
        with np.load(self.data_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        iteration = int(meta.get("iteration", 0))
        return iteration, arrays, meta

    def assert_resumable(self, solver_name: str, problem,
                         criterion, solver_settings: dict) -> None:
        """原进度恢复前检查问题、停止条件和求解器设定完全匹配。"""
        _, _, meta = self.load(solver_name)
        if meta.get("problem") != problem.identity():
            raise ValueError(
                "检查点中的线性系统与当前问题不一致，不能恢复原进度；"
                "请改用 continuation='restart' 从旧近似解重新开始")
        if meta.get("criterion") != criterion.signature():
            raise ValueError(
                "检查点中的停止条件与当前停止条件不一致，不能恢复原进度；"
                "请改用 continuation='restart' 从旧近似解重新开始")
        if meta.get("solver_settings") != solver_settings:
            raise ValueError(
                "检查点中的求解器设定与当前设定不一致，不能恢复原进度；"
                "请改用 continuation='restart' 从旧近似解重新开始")

    def approximation(self, problem) -> tuple[int, np.ndarray, dict]:
        """只读方式取出旧近似解；受控重新开始绝不复用求解器内部状态。"""
        iteration, arrays, meta = self.load()
        old_n = meta.get("problem", {}).get("n")
        if old_n is not None and old_n != problem.n:
            raise ValueError(
                f"检查点问题维度为 {old_n}，当前问题维度为 {problem.n}，"
                "旧近似解不能作为新初值")
        if "x" not in arrays:
            raise ValueError("检查点不包含近似解 x，不能重新开始")
        return iteration, np.asarray(arrays["x"], dtype=float), meta

    def remove(self) -> None:
        for p in (self.data_path, self.meta_path):
            if p.exists():
                p.unlink()
