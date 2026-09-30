"""检查点：保存/恢复迭代器内部状态以支持续算。

状态用 np.savez 保存为 .npz，元数据（求解器名、迭代号）存为 JSON。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class CheckpointMismatchError(ValueError):
    """检查点与当前问题或求解设定不一致，拒绝恢复原进度。"""


class Checkpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data_path = self.path.with_suffix(".npz")
        self.meta_path = self.path.with_suffix(".json")

    def save(self, solver_name: str, iteration: int,
             arrays: dict[str, np.ndarray], *, problem_id: str | None = None,
             problem_identity: dict | None = None,
             settings_key: dict | None = None, **meta) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_data = self.data_path.with_name("." + self.data_path.stem + ".writing.npz")
        tmp_meta = self.meta_path.with_suffix(".json.tmp")
        np.savez(tmp_data, **arrays)
        payload = {"solver": solver_name, "iteration": int(iteration),
                   "problem_id": problem_id,
                   "problem_identity": problem_identity,
                   "settings_key": settings_key, **meta}
        tmp_meta.write_text(json.dumps(payload), encoding="utf-8")
        tmp_data.replace(self.data_path)
        tmp_meta.replace(self.meta_path)

    def exists(self) -> bool:
        return self.data_path.exists() and self.meta_path.exists()

    def inspect(self) -> dict:
        """只读检查点元数据，不恢复状态。"""
        if not self.exists():
            raise FileNotFoundError(f"检查点不存在：{self.path}")
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def load(self, solver_name: str) -> tuple[int, dict[str, np.ndarray], dict]:
        meta = self.inspect()
        if meta.get("solver") != solver_name:
            raise ValueError(
                f"检查点属于 {meta.get('solver')!r}，不能用于 {solver_name!r}")
        with np.load(self.data_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        iteration = int(meta.pop("iteration"))
        meta.pop("solver", None)
        return iteration, arrays, meta

    def verify_resume(self, solver_name: str, problem,
                      settings_key: dict) -> dict:
        """恢复原进度前的身份校验，任一不一致立即明确拒绝。

        区分求解器、问题维度、矩阵 A、右端向量 b、停止精度设定，
        避免换问题后误用旧内部状态（例如 CG 的递推残差与搜索方向），
        停在旧解附近却只报告预算耗尽。
        """
        meta = self.inspect()
        if meta.get("solver") != solver_name:
            raise CheckpointMismatchError(
                f"检查点属于求解方法 {meta.get('solver')!r}，"
                f"不能按 {solver_name!r} 的原进度恢复")

        old_ident = meta.get("problem_identity") or {}
        new_ident = problem.identity
        differences = []
        if old_ident.get("n") != new_ident["n"]:
            differences.append(
                f"问题维度 {old_ident.get('n')} -> {new_ident['n']}")
        if old_ident.get("A") is not None and old_ident.get("A") != new_ident["A"]:
            differences.append("矩阵 A 内容不同")
        if old_ident.get("b") is not None and old_ident.get("b") != new_ident["b"]:
            differences.append("右端向量 b 不同")
        if differences:
            raise CheckpointMismatchError(
                "检查点记录的是另一个问题，拒绝恢复原进度："
                + "；".join(differences)
                + "。可改用 continuation='restart' 以旧近似解重新开始")

        old_settings = meta.get("settings_key")
        new_settings = settings_key
        if old_settings is not None and old_settings != new_settings:
            raise CheckpointMismatchError(
                f"停止/精度设定与检查点不一致：旧 {old_settings}，新 {new_settings}。"
                "预算 max_iter 可变，但精度设定不同不能按原进度恢复；"
                "可改用 continuation='restart' 以旧近似解重新开始")
        return meta

    def approximate_solution(self) -> tuple[int, np.ndarray, str | None]:
        """受控接续（重启）时取出旧近似解 x，只用作新迭代的初值。

        允许跨求解方法借用近似解（例如先 Richardson 磨光再交给 CG）；
        不返回 r、p 等与旧问题/旧方法耦合的内部状态。
        返回 (迭代号, x, 旧求解器名)。
        """
        meta = self.inspect()
        with np.load(self.data_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        iteration = int(meta.get("iteration", 0))
        if "x" not in arrays:
            raise ValueError("检查点中没有近似解 x，无法作为初值重启")
        return iteration, np.array(arrays["x"], dtype=float), meta.get("solver")

    def remove(self) -> None:
        for p in (self.data_path, self.meta_path):
            if p.exists():
                p.unlink()
