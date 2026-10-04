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
- `numexp/reuse.py` — **可沿用性说明**：矩阵、右端项或容差变化时，
  `assess_reuse` 明确列出可沿用内容（旧近似解可热启动）与不可沿用内容
  （求解器内部迭代状态必须重建），并判定是否允许原进度恢复。
- `numexp/series.py` — **问题序列求解**：`solve_problem_series` 对一系列
  相关但不同的问题，分别用上一问题较优近似解热启动和从零冷启动，
  每次求解独立记录目标问题身份与收敛判断，不复用前一问题的迭代状态。
- `numexp/multistart.py` — **多起点统一预算管理**：`MultiStartSession`
  让所有求解路径共享一个迭代预算池；某条路径失败（前提拒绝、数值异常）
  或停滞（`StagnationGuard`）只影响该路径，其余路径继续；最终选择以
  当前问题的真实残差重算为准。会话可暂停并重开：各路径的预算消耗、
  起点来源与结果持久化在会话目录，热启动来源的旧检查点只读不覆盖。

- `numexp/continuation.py` — **参数连续跟踪**：`ContinuationSession` 在用户
  指定的参数起点与终点之间逐步推进问题族 `problem_at(t)`。用户给出的起点
  先在参数起点处独立求解/核验，初值残差、修正消耗与失败/待续原因都保留，
  未满足节点方程与精度时不创建根节点。确认后以上一个已确认节点的解热启动
  下一步；某一步未收敛时记录失败尝试（含实际参数与消耗）、缩小步长后从上
  一个已确认节点重新出发，失败尝试的残差绝不当作已收敛。统一预算覆盖起点
  修正、成功步与失败尝试；可从任一已确认节点另开方案（fork），
  原节点与预算记录不被覆盖；`compare_at(t)` 按该参数的问题重算真实残差
  比较多条路径；未到达终点的路径 `final_solution` 返回 None，不以最近
  结果代替。会话可暂停重开，恢复时校验参数区间、端点问题身份与精度条件，
  并按各节点自身参数重算残差；终点到达和中间/起点失败分开表达。

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

## 多起点与序列求解

- **统一预算**：`MultiStartSession(total_budget=...)` 是全部路径共享的
  迭代池，每条路径按实际消耗扣减；池用尽时未收敛路径标记为 `paused`，
  `run(extra_budget=...)` 追加预算后从各自路径检查点续跑。
- **路径隔离**：单条路径的 `precondition_failed / numerical_breakdown /
  stagnated` 不影响其他路径；`MultiStartResult.best(problem)` 用当前问题
  重算真实残差选出最优路径。
- **暂停重开**：设置 `session_dir` 后每次运行结束自动保存
  `session.json/.npz`（各路径预算、来源、结果）与 `paths/` 下的路径检查点；
  `MultiStartSession.load` 校验目标问题身份一致才恢复，问题变了必须新建会话。
