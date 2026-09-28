# 数值计算实验框架（第一版：大型稀疏线性方程迭代求解）

## 设计边界

框架分三层，新增求解方法时停止条件与结果检查可直接复用：

- **问题层**（numexp/problems.py）：LinearSystem（A x = b），负责维度、有限性等
  基本合法性，并提供独立的 residual(x)（真实残差 b - A x）与数据指纹。
- **求解器层**（numexp/solvers/）：IterativeSolver 抽象基类规定
  validate / init_state / step 三个钩子。子类（当前为手工实现的 Jacobi、CG）
  只负责推进一个迭代步，不决定何时停止，也不包装任何现成完整求解器。
- **记录与裁决层**：
  - records.py：IterationRecord（每步真实残差）、SolveResult、StopReason；
  - stopping.py：可组合的停止条件（精度 / 预算 / 数值异常）；
  - verification.py：脱离求解器内部状态，重算残差并核对“声明是否属实”。

## 停止原因（不一律叫成功）

StopReason 有三种，只有 CONVERGED 是成功：

- CONVERGED：真实（相对）残差达到 rtol/atol；
- BUDGET_EXHAUSTED：达到最大迭代步数仍未达标；
- NUMERICAL_BREAKDOWN：NaN/Inf、除零、CG 负曲率等异常。

## 方法前提（不满足即拒绝）

- Jacobi：对角元必须全部非零，否则抛 MethodPreconditionError；
- CG：必须实对称；小规模密集矩阵额外用 Cholesky 确认正定，非正定直接拒绝。

## 检查点续算

Checkpointer 把迭代状态（不含大矩阵）写入目录下的 npz；续算时校验问题指纹
与求解器名，历史记录无缝拼接，轨迹与一次跑完逐位一致；收敛后自动清理检查点。

    solver = ConjugateGradientSolver(suite, checkpointer=Checkpointer('./ckpt'))
    solver.solve(problem)   # 预算中断
    solver.solve(problem)   # 再次调用即从检查点续算

## 实验与对照

- 小规模（n <= 200）用 numexp.reference.direct_solve（LAPACK）求参考解，仅用于
  对照，框架核心不依赖完整求解器；
- 案例（numexp/cases.py）：稀疏 SPD（二维泊松）、严格对角占优、Hilbert 病态、
  对称不定、非对称、Jacobi 发散、零对角元。

运行：

    python -m experiments.run_experiments
    pytest -q

## 新增求解方法

1. 继承 IterativeSolver，实现 validate / init_state / step；
2. 数值崩溃时抛 IterationBreakdown，前提不满足抛 MethodPreconditionError；
3. 停止条件、真实残差记录、结果复核、检查点全部自动复用。
