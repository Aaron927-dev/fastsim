"""参数估计：用实验数据反推模型参数（拟合 + 不确定度 + 可辨识性诊断）。

对应 COMSOL 优化模块的**参数估计**（手册明言「寻找一组参数值，使模型与
测试（目标）之间的差异最小」，求解器为 Levenberg-Marquardt）。

三个容易被忽略但决定结论可靠性的输出
------------------------------------
1. **参数标准差** σ_θ —— 只给点估计而不给不确定度，等于说"我不知道这个数可信多少"
2. **协方差/相关系数矩阵** —— 两参数相关性接近 ±1 时，它们的**和/差**才可辨识，
   单独取值没有意义（典型的 k₁ 与 k₂ 强相关）
3. **J^T J 条件数** —— 病态时拟合"成功"但参数毫无意义

本模块把这三项作为**一等输出**，而不只是返回最优参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import optimize

__all__ = ["FitResult", "fit", "residual_norm"]


@dataclass
class FitResult:
    """参数估计结果（含不确定度与可辨识性诊断）。"""

    theta: np.ndarray                     # 最优参数
    names: list[str]
    sigma: np.ndarray                     # 参数标准差（1σ）
    covariance: np.ndarray                # 协方差矩阵
    correlation: np.ndarray               # 相关系数矩阵
    cost: float                           # 加权残差平方和
    r_squared: float
    residuals: np.ndarray
    n_data: int
    n_params: int
    n_evals: int
    condition_number: float                # cond(J^T J)，病态度量
    converged: bool
    message: str

    @property
    def dof(self) -> int:
        return self.n_data - self.n_params

    def ci(self, k: float = 1.96) -> list[tuple[str, float, float, float]]:
        """置信区间 (名称, 估计值, 下界, 上界)。默认 95% 正态近似。"""
        return [
            (n, float(t), float(t - k * s), float(t + k * s))
            for n, t, s in zip(self.names, self.theta, self.sigma)
        ]

    def report(self) -> str:
        """人类可读的拟合报告（含可辨识性判据）。"""
        lines = [
            f"参数估计结果：{self.n_params} 参数 / {self.n_data} 数据点 / "
            f"自由度 {self.dof}",
            f"  R² = {self.r_squared:.6f}   cost = {self.cost:.6e}   "
            f"迭代 {self.n_evals} 次   {'收敛' if self.converged else '未收敛'}",
            "",
            f"  {'参数':>14s} {'估计值':>14s} {'标准差':>12s} {'95% 区间':>28s}",
            "  " + "-" * 72,
        ]
        for name, val, lo, hi in self.ci():
            lines.append(
                f"  {name:>14s} {val:14.6g} {dict(zip(self.names, self.sigma))[name]:12.4g}"
                f"   [{lo:12.5g}, {hi:12.5g}]"
            )
        lines.append("")
        lines.append(f"  J^T J 条件数 = {self.condition_number:.3e}")
        if self.condition_number > 1e10:
            lines.append("  ⚠ 病态：参数高度相关，单独取值不可信（见下方相关系数）")
        # 报告强相关参数对
        pairs = []
        for i in range(self.n_params):
            for j in range(i + 1, self.n_params):
                if abs(self.correlation[i, j]) > 0.9:
                    pairs.append((self.names[i], self.names[j],
                                  float(self.correlation[i, j])))
        if pairs:
            lines.append("  强相关参数对（|r| > 0.9）：")
            for a, b, r in pairs:
                lines.append(f"    {a} ↔ {b}:  r = {r:+.3f}")
            lines.append("    → 二者只有组合（和/差）可辨识，建议固定其一或改测条件")
        return "\n".join(lines)


def residual_norm(res: np.ndarray) -> float:
    """残差向量的 2-范数。"""
    return float(np.linalg.norm(res))


def fit(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta0: Sequence[float],
    x_data: np.ndarray,
    y_data: np.ndarray,
    names: Sequence[str] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    sigma: np.ndarray | float | None = None,
    method: str = "auto",
    max_nfev: int | None = None,
) -> FitResult:
    """最小二乘参数估计（Levenberg-Marquardt / Trust-Region-Reflective）。

    参数
    ----
    model : callable
        ``model(theta, x) -> y_pred``。应当用**解析/数值 Jacobian 友好**的实现
        （本仓库的求解器都提供解析 Jacobian，但拟合走的是外层残差，
        由 scipy 用有限差分近似，参数个数通常 < 10，代价可接受）。
    theta0 : array
        初值。
    x_data, y_data : array
        实验数据。
    sigma : array | float | None
        各数据点标准差（用于加权）；None 表示等权。
    bounds : (lower, upper) | None
        参数边界（有界时自动改用 TRF 方法）。
        物理参数通常有明确边界（k>0 等），**建议总是给**，否则可能收敛到非物理解。

    返回
    ----
    FitResult（含参数、标准差、协方差、相关系数、条件数）
    """
    theta0 = np.asarray(theta0, dtype=float)
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    n, p = y_data.size, theta0.size
    if names is None:
        names = [f"θ{i}" for i in range(p)]
    names = list(names)
    if len(names) != p:
        raise ValueError(f"names 长度 {len(names)} 与参数个数 {p} 不一致")
    if n <= p:
        raise ValueError(f"数据点 {n} 不多于参数个数 {p}，无法估计（自由度 ≤ 0）")

    if sigma is None:
        w = np.ones(n)
    else:
        s = np.asarray(sigma, dtype=float)
        if s.ndim == 0:
            s = np.full(n, float(s))
        if np.any(s <= 0):
            raise ValueError("sigma 必须全为正")
        w = 1.0 / s

    def residual(theta: np.ndarray) -> np.ndarray:
        return (np.asarray(model(theta, x_data), dtype=float) - y_data) * w

    use_bounds = bounds is not None
    m = method
    if m == "auto":
        m = "trf" if use_bounds else "lm"

    kwargs: dict = {"method": m}
    if use_bounds:
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
        if lo.shape != (p,) or hi.shape != (p,):
            raise ValueError("bounds 两端的长度都须等于参数个数")
        if np.any(lo >= hi):
            raise ValueError("存在下界 ≥ 上界的参数")
        if not (np.all(theta0 > lo) and np.all(theta0 < hi)):
            raise ValueError("初值必须在边界之内（严格）")
        kwargs["bounds"] = (lo, hi)
    if max_nfev:
        kwargs["max_nfev"] = max_nfev

    sol = optimize.least_squares(residual, theta0, **kwargs)

    # ── 协方差与不确定度 ─────────────────────────────────
    # 残差方差 s² = Σr²/(n−p)，协方差 cov = (JᵀJ)⁻¹·s²
    r = residual(sol.x)
    dof = n - p
    s2 = float(np.sum(r**2) / dof) if dof > 0 else float("nan")
    J = sol.jac if sol.jac is not None else optimize.approx_fprime(
        sol.x, lambda t: residual(t), 1e-8 * np.maximum(np.abs(sol.x), 1.0))
    JtJ = J.T @ J
    try:
        cov = np.linalg.inv(JtJ) * s2
        sigma_theta = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        cond = float(np.linalg.cond(JtJ))
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
        sigma_theta = np.full(p, np.nan)
        cond = float("inf")

    # 相关系数矩阵
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        corr = cov / np.outer(d, d)
    corr = np.nan_to_num(corr, nan=0.0)

    # R²（对加权残差口径；等权时即通常意义）
    y_bar = float(np.average(y_data, weights=w**2))
    ss_res = float(np.sum((np.asarray(model(sol.x, x_data)) - y_data) ** 2
                          * w**2))
    ss_tot = float(np.sum((y_data - y_bar) ** 2 * w**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FitResult(
        theta=sol.x, names=names, sigma=sigma_theta, covariance=cov,
        correlation=corr, cost=float(sol.cost), r_squared=r2,
        residuals=r, n_data=n, n_params=p, n_evals=int(sol.nfev),
        condition_number=cond, converged=bool(sol.success),
        message=str(sol.message),
    )
