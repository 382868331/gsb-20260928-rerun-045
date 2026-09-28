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

    def save(self, solver_name: str, iteration: int,
             arrays: dict[str, np.ndarray], **meta) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_data = self.data_path.with_name("." + self.data_path.stem + ".writing.npz")
        tmp_meta = self.meta_path.with_suffix(".json.tmp")
        np.savez(tmp_data, **arrays)
        payload = {"solver": solver_name, "iteration": int(iteration), **meta}
        tmp_meta.write_text(json.dumps(payload), encoding="utf-8")
        tmp_data.replace(self.data_path)
        tmp_meta.replace(self.meta_path)

    def exists(self) -> bool:
        return self.data_path.exists() and self.meta_path.exists()

    def load(self, solver_name: str) -> tuple[int, dict[str, np.ndarray], dict]:
        if not self.exists():
            raise FileNotFoundError(f"检查点不存在：{self.path}")
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta.get("solver") != solver_name:
            raise ValueError(
                f"检查点属于 {meta.get('solver')!r}，不能用于 {solver_name!r}")
        with np.load(self.data_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        iteration = int(meta.pop("iteration"))
        meta.pop("solver", None)
        return iteration, arrays, meta

    def remove(self) -> None:
        for p in (self.data_path, self.meta_path):
            if p.exists():
                p.unlink()
