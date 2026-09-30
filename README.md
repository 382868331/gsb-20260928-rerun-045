# 数值计算实验框架（第一版：大型稀疏线性方程迭代求解）

三层边界设计，后续新增求解方法可直接复用停止条件与结果检查：

- `numexp/problems.py` — **问题层**：`LinearSystem(A, b)`，负责形状校验、
  对称/正定/非奇异等方法前提检查（`PreconditionError` 直接拒绝），
  以及真实残差 `b - Ax` 的计算（支持 scipy 稀疏矩阵）。
- `numexp/criteria.py` — **停止条件层**：`StoppingCriterion` 统一处理
  达到精度（`RelativeResidual` 等）、预算用尽（`BudgetExhausted`）、
  数值异常（NaN/Inf）。与具体求解器解耦。
- `numexp/records.py` — **记录层**：`IterationRecord` 每步记录迭代号与
  **真实残差范数**（基类每步用原矩阵重算，不使用迭代递推残差）；
  `SolveResult.reason` 区分 `converged / budget_exhausted /
  numerical_breakdown / precondition_failed`，只有 `converged` 才算成功，
  `verify()` 独立重算残差做结果检查。
- `numexp/base.py` — 迭代求解器基类：驱动“一步一记录一判定”循环、
  异常分类、检查点存取；新方法只需实现 `validate / initial_state / step`。
- `numexp/cg.py` — 手写共轭梯度（前提 SPD；非正曲率 breakdown 归类为数值异常）。
- `numexp/stationary.py` — 手写 Richardson 迭代（用于不收敛/发散实验）。
- `numexp/checkpoint.py` — npz+json 检查点，原子写入，保存问题身份
  （维度、矩阵 A 与右端向量 b 的内容指纹）与关键求解设定，收敛后自动清理。
  恢复原进度前严格核对：求解方法、A、b、精度容差任一不一致立即抛
  `CheckpointMismatchError` 明确拒绝（`max_iter` 属于预算，允许放宽续算）。
  `solve(..., continuation="restart")` 为受控接续：只借用旧近似解 x 作
  新问题初值，内部状态全部重建、迭代从 0 计数，检查点只读不覆盖；
  跨方法（如 Richardson 磨光后交 CG）也只能走 restart。
- `numexp/reference.py` — 小规模稠密直接求解对照、SPD/稀疏/非对称/不定/病态矩阵生成。

核心求解循环与 CG/Richardson 均为手写实现，直接求解器仅用于小规模对照。

## 运行

```bash
python -m pytest                 # 单元测试
python -m experiments.run_experiments   # 对照/病态/不收敛/前提拒绝/续算实验
```
