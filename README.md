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
- `numexp/checkpoint.py` — npz+json 检查点，原子写入，记录矩阵/右端项
  SHA-256、停止条件数学设定与求解器参数；原进度恢复严格校验身份，
  跨方法或换问题只能受控地从旧近似解重新开始。
- `numexp/reference.py` — 小规模稠密直接求解对照、SPD/稀疏/非对称/不定/病态矩阵生成。
- `numexp/sequence.py` — **问题序列**：`solve_sequence` 依次求解一系列相关但
  不同的问题，把上一问题的近似解作为下一问题的起点（热启动），同时从零向量
  冷启动对照；每次求解独立记录目标问题身份与收敛判断，上一问题的剩余迭代
  状态不会被当作兼容检查点。`describe_carryover` 在矩阵、右端项或容差变化时
  明确说明可沿用的内容（维度不变时仅近似解 x 可作初值，内部状态一律不沿用）。
- `numexp/multistart.py` — **多起点统一预算**：`MultiStartRunner` 让所有路径
  共享一个总迭代预算、按轮分片下发；某条路径失败（数值异常/前提拒绝）或停滞
  （残差长期不改善）即被隔离，不影响其他路径继续；最终选择以当前问题的真实
  残差 `||b - Ax||` 为依据。工作目录中的 manifest 记录每条路径的预算、来源与
  结果，暂停后可用更大预算重开；每条路径使用独立检查点，作为起点的旧检查点
  只读、绝不覆盖。

核心求解循环与 CG/Richardson 均为手写实现，直接求解器仅用于小规模对照。

## 运行

```bash
python -m pytest                 # 单元测试
python -m experiments.run_experiments   # 对照/病态/不收敛/前提拒绝/续算实验
```

## 检查点接续

- **原进度恢复**：默认 `continuation="resume"`。检查点中的问题身份
  （矩阵和右端项内容）、停止条件数学设定、求解器名称及关键参数必须完全一致；
  `max_iter` 是预算，允许用更大预算恢复一次预算耗尽的运行。
- **受控重新开始**：显式传入 `continuation="restart"` 和
  `restart_from=旧检查点`。框架只读取旧近似解 `x`，由当前求解器重建内部状态，
  新运行的第一条记录重新从迭代 0 开始，并用当前问题计算真实残差。
- **记录不覆盖**：新实验应给当前求解器新的 `checkpoint` 目标路径，旧检查点仅作为
  `restart_from` 的只读来源；若直接把源路径同时作为目标路径，也不会回写或删除旧记录。
- 矩阵变化、右端项变化、停止条件变化但仍请求原进度恢复时，会在迭代前抛出明确错误，
  不会沿用旧内部状态后只报告预算耗尽。

## 问题序列与多起点

- **问题序列**：`solve_sequence(problems, solver_factory)` 逐步求解相关问题；
  `solver_factory` 每次返回全新求解器（不带检查点），热启动只传递上一问题的
  近似解 x，新运行从迭代 0 重新记录；`SequenceStep.carryover` 给出本步相对
  上一步"什么变了、什么能沿用"的明确说明。
- **多起点**：`MultiStartRunner(problem, solver_factory, criterion_factory,
  starts, total_budget, workdir)` 统一分配预算；`StartPoint` 支持零向量、
  用户初值或 `restart_from=旧检查点`（只读）三种来源；`MultiStartResult`
  汇报各路径状态（converged / budget_exhausted / failed / stagnant）、
  各自消耗的预算与当前问题真实残差，并按真实残差最小选出最优解。
  中断后用同一 `workdir` 与更大 `total_budget` 重开即可继续，各路径的
  预算、来源与结果都从 manifest 恢复；换问题后不能在同一 workdir 重开。
