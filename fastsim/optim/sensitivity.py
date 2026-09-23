"""灵敏度分析：一次扫描同时拿到「模型输出」与「各参数影响的排序」。

对应 COMSOL 优化模块里的**灵敏度分析**，但这里是**有限差分**实现 ——
之所以可行且够用，正是因为自研求解器快（单次 10² ms 量级）：
COMSOL 里做 N 维中心差分需要 2N 次全模型求解（分钟级/次），
在自研内核里就是 2N × 0.1 s。

为什么用**归一化灵敏度**而不是原始偏导
--------------------------------------
原始偏导 ∂f/∂x 带量纲，跨参数不可比（电流单位 A、流量单位 m³/s）。
归一化灵敏度（弹性系数）无量纲，可直接排序：

    S_i = (∂f/∂x_i) · (x_i / f)

含义：参数 x_i 变化 1% 时，输出 f 变化 S_i %。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = ["SensitivityResult", "sensitivities", "finite_difference_gradient"]


def finite_difference_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    rel_step: float = 1e-4,
    method: str = "central",
) -> np.ndarray:
    """数值梯度。中心差分默认（精度 O(h²)），前向差分 O(h)。"""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    if method == "central":
        for i in range(x.size):
            h = rel_step * max(abs(x[i]), 1e-30)
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            g[i] = (func(xp) - func(xm)) / (2.0 * h)
    elif method == "forward":
        f0 = func(x)
        for i in range(x.size):
            h = rel_step * max(abs(x[i]), 1e-30)
            xp = x.copy()
            xp[i] += h
            g[i] = (func(xp) - f0) / h
    else:
        raise ValueError(f"未知 method: {method!r}（可选 central / forward）")
    return g


@dataclass
class SensitivityResult:
    """灵敏度分析结果。"""

    names: list[str]
    values: np.ndarray              # 基准点参数值
    baseline: float                 # 基准点输出
    gradient: np.ndarray            # 原始偏导 ∂f/∂x
    normalized: np.ndarray          # 归一化灵敏度 S_i = (∂f/∂x_i)(x_i/f)
    n_evaluations: int = 0
    names_used: list[str] = field(default_factory=list)

    def ranking(self, by_abs: bool = True) -> list[tuple[str, float]]:
        """按影响大小排序，返回 [(参数名, 归一化灵敏度), ...]。"""
        order = np.argsort(-np.abs(self.normalized)) if by_abs else np.argsort(
            -self.normalized)
        return [(self.names[i], float(self.normalized[i])) for i in order]

    def table(self, top: int | None = None) -> str:
        """格式化表格（便于直接贴进报告）。"""
        rows = self.ranking()
        if top:
            rows = rows[:top]
        lines = [
            f"{'参数':>14s} {'基准值':>12s} {'∂f/∂x':>12s} {'归一化灵敏度':>14s}  影响",
            "-" * 68,
        ]
        vmax = max((abs(v) for _, v in rows), default=1.0) or 1.0
        for i, (name, s) in enumerate(rows):
            idx = self.names.index(name)
            bar = "█" * int(round(20 * abs(s) / vmax))
            lines.append(
                f"{name:>14s} {self.values[idx]:12.4g} "
                f"{self.gradient[idx]:12.4g} {s:14.4g}  {bar}"
            )
        return "\n".join(lines)


def sensitivities(
    func: Callable[[np.ndarray], float],
    x0: Sequence[float] | np.ndarray,
    names: Sequence[str] | None = None,
    rel_step: float = 1e-4,
    method: str = "central",
) -> SensitivityResult:
    """一次调用完成基准求值与全参数灵敏度分析。

    参数
    ----
    func : callable
        ``f(x) -> float``，x 为参数向量。
    x0 : array
        基准工况。
    names : sequence[str] | None
        参数名（用于排序输出）；缺省用 ``x0``/``x1``/...
    rel_step : float
        相对步长 h = rel_step·max(|x_i|, tiny)。

    ⚠ 步长选择的权衡：太大 → 截断误差（O(h²)）；太小 → 相消误差
    （浮点有效位损失，量级 √ε ≈ 1.5e-8）。默认 1e-4 是常用折中；
    对病态/强非线性函数建议配合 `check_step_robustness()` 复核。
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    if names is None:
        names = [f"x{i}" for i in range(n)]
    names = list(names)
    if len(names) != n:
        raise ValueError(f"names 长度 {len(names)} 与参数个数 {n} 不一致")

    baseline = float(func(x0))
    grad = finite_difference_gradient(func, x0, rel_step=rel_step, method=method)

    n_evals = 1 + (2 * n if method == "central" else n)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(
            np.abs(baseline) > 0.0, grad * x0 / baseline, np.nan
        )

    return SensitivityResult(
        names=names, values=x0, baseline=baseline, gradient=grad,
        normalized=normalized, n_evaluations=n_evals,
    )


def check_step_robustness(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    rel_steps: Sequence[float] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
) -> dict[float, np.ndarray]:
    """在多个步长下重算梯度，用于识别相消误差或强非线性。

    判据：若梯度在某个步长区间内**稳定**（相对变化 < 1%），则该区间可信。
    若随步长单调漂移，说明截断误差未进入渐近区；若先稳定后发散，
    说明进入了相消误差区 —— 应取稳定段中较大的步长。
    """
    out: dict[float, np.ndarray] = {}
    for h in rel_steps:
        out[float(h)] = finite_difference_gradient(func, x, rel_step=h)
    return out
