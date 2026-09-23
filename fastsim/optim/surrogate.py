"""代理模型 + 贝叶斯寻优：用尽量少的高保真调用找到设计最优点。

**这是优化基座的核心价值所在。**

COMSOL 做设计寻优时，每次目标函数评估 = 一次全模型求解（分钟级），
于是"优化"实际退化为"跑几十个工况然后肉眼看趋势"。
本模块的思路是：

    少量高保真采样（如 8+15 次） → 训练代用模型 → 用代用模型做海量评估
    → 用采集函数挑出"最值得再算一次"的点 → 只对那个点调用高保真

因为自研内核单次评估只要 10² ms 量级，**同一套算法可以跑得更多轮、更细**，
从而在同等时间里把设计空间挖得更透。

实现选择
--------
- 高斯过程（GP）：平方指数核，超参数按**对数边缘似然**优化。
  自行实现（不引入 sklearn），保持 fastsim 的 numpy+scipy 轻依赖。
- 采样：拉丁超立方（`scipy.stats.qmc`），保证初值覆盖设计空间。
- 采集函数：**期望改进 EI**（可解析计算，平衡"利用"与"探索"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import optimize
from scipy.stats import norm, qmc

__all__ = [
    "GaussianProcess",
    "latin_hypercube",
    "DesignResult",
    "bayesian_optimize",
]


# ══════════════════════════════════════════════════════════════
# 高斯过程
# ══════════════════════════════════════════════════════════════

class GaussianProcess:
    """轻量高斯过程回归（平方指数核 + 对数边缘似然优化超参数）。

    模型： y(x) ~ GP(m(x), k(x,x'))，k = σ_f²·exp(−½Σ((x_d−x'_d)/ℓ_d)²)
    先验均值 m(x) 取训练集均值（比取 0 更稳健，避免远端无约束地回落到 0）。

    实现要点（这三条都是被实测缺陷逼出来的，缺一不可）
    --------------------------------------------------
    1. **输入按维归一化到 [0,1]**：否则各维量纲不同时，长度尺度在原始坐标下
       不可比 —— 实测未归一化时同一问题两个方向的 ℓ 差 3 万倍（0.013 vs 403）。
    2. **噪声有下界（默认 1e-4，相对目标方差）**：若允许噪声→0，
       边缘似然会被"ℓ→0 完美插值"解支配（此时 K≈(σ_f²+ε)I，拟合残差为零、
       log|K| 又无足够惩罚）。实测该病态使 ℓ 塌到下限、测试 R² 变**负**
       （−0.09 ~ −0.57）——比预测均值还差。
    3. **长度尺度有物理边界**（归一化空间 [0.02, 20]）：防止塌缩到退化极限。

    ⚠ 能力边界（诚实声明）
    ---------------------
    - n 较大时（> ~500）O(n³) 求逆成为瓶颈；本模块面向**昂贵函数**
      （少量采样的场景），这是正确用法。
    - 核函数固定为平方指数（假设光滑）。阶跃/尖峰型景观需要 Matérn 核，
      当前不支持。
    - 样本数 < 2·d 时不做超参数优化（信息不足以可靠估计各向异性 ℓ_d）。
    """

    _LS_LO, _LS_HI = 0.02, 20.0        # 归一化空间中的长度尺度边界
    _SF_LO, _SF_HI = 0.01, 10.0
    _SN_LO, _SN_HI = 1e-6, 0.3         # 噪声标准差（相对目标标准差）

    def __init__(self, length_scale: float | np.ndarray | None = None,
                 sigma_f: float = 1.0, noise: float = 1e-4,
                 optimize_hp: bool = True):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.noise = noise
        self.optimize_hp = optimize_hp
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._x_mean: np.ndarray | None = None
        self._x_span: np.ndarray | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self.noise_: float = noise
        self.nll_: float = np.nan
        self.history_: list[float] = []

    # ── 核与内部工具 ────────────────────────────────────
    @staticmethod
    def _kernel(A: np.ndarray, B: np.ndarray,
                length_scale: np.ndarray, sigma_f: float) -> np.ndarray:
        """平方指数核（输入须已归一化）。A:(n,1,d) B:(1,m,d) → (n,m)"""
        diff = (A[:, None, :] - B[None, :, :]) / length_scale[None, None, :]
        return sigma_f**2 * np.exp(-0.5 * np.sum(diff**2, axis=2))

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_span

    def _neg_log_marginal(self, log_params: np.ndarray) -> float:
        """负对数边缘似然（在归一化输入 / 归一化目标上）。

        参数：log_params = [log ℓ (d 个), log σ_f, log σ_n]
        """
        d = self.Xn.shape[1]
        ls = np.exp(log_params[:d])
        sf = float(np.exp(log_params[d]))
        sn = float(np.exp(log_params[d + 1]))
        K = self._kernel(self.Xn, self.Xn, ls, sf)
        K = K + (sn**2 + 1e-12) * np.eye(self.Xn.shape[0])
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.yn))
        nll = 0.5 * float(self.yn @ alpha) + float(np.sum(np.log(np.diag(L))))
        return nll + 0.5 * self.Xn.shape[0] * np.log(2 * np.pi)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.size:
            raise ValueError(f"X 行数 {X.shape[0]} 与 y 长度 {y.size} 不一致")
        self.X, self.y = X, y

        # 输入按维归一化（尺度无关）
        self._x_mean = X.mean(axis=0)
        span = np.ptp(X, axis=0)
        self._x_span = np.where(span > 0, span, 1.0)
        self.Xn = self._normalize(X)

        # 目标归一化（数值稳定性）
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        self.yn = (y - self._y_mean) / self._y_std

        d = X.shape[1]
        if self.length_scale is None:
            ls0 = np.full(d, 0.3)          # 归一化空间中 0.3 是合理的初值
        elif np.ndim(self.length_scale) == 0:
            ls0 = np.full(d, float(self.length_scale))
        else:
            ls0 = np.asarray(self.length_scale, dtype=float)
        sn0 = max(float(self.noise), self._SN_LO)

        if self.optimize_hp and X.shape[0] >= 2 * d + 1:
            p0 = np.concatenate([np.log(ls0), [np.log(self.sigma_f)], [np.log(sn0)]])
            bounds = ([(np.log(self._LS_LO), np.log(self._LS_HI))] * d
                      + [(np.log(self._SF_LO), np.log(self._SF_HI))]
                      + [(np.log(self._SN_LO), np.log(self._SN_HI))])
            best = None
            # 多起点：边缘似然在 ℓ 上可能多峰，单起点易落到退化解
            for ls_try in (0.15, 0.3, 0.6):
                p_start = p0.copy()
                p_start[:d] = np.log(ls_try)
                try:
                    r = optimize.minimize(
                        self._neg_log_marginal, p_start, method="L-BFGS-B",
                        bounds=bounds, options={"maxiter": 300},
                    )
                except Exception:  # noqa: BLE001
                    continue
                if best is None or r.fun < best.fun:
                    best = r
            if best is not None:
                ls = np.exp(best.x[:d])
                sf = float(np.exp(best.x[d]))
                sn = float(np.exp(best.x[d + 1]))
                self.nll_ = float(best.fun)
                self.history_.append(self.nll_)
            else:
                ls, sf, sn = ls0, self.sigma_f, sn0
        else:
            ls, sf, sn = ls0, self.sigma_f, sn0

        self.length_scale = ls
        self.sigma_f = sf
        self.noise_ = sn

        K = (self._kernel(self.Xn, self.Xn, ls, sf)
             + (sn**2 + 1e-12) * np.eye(X.shape[0]))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, self.yn))
        return self

    def predict(self, Xs: np.ndarray,
                return_std: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """预测。return_std=True 时返回 (mean, std)（std 为后验标准差）。"""
        if self._L is None:
            raise RuntimeError("请先 fit()")
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Xq = self._normalize(Xs)
        ls = np.asarray(self.length_scale, dtype=float)
        Ks = self._kernel(Xq, self.Xn, ls, self.sigma_f)          # (m, n)
        mu = Ks @ self._alpha
        mu = mu * self._y_std + self._y_mean
        if not return_std:
            return mu
        v = np.linalg.solve(self._L, Ks.T)                        # (n, m)
        var = self.sigma_f**2 - np.sum(v**2, axis=0)
        std = np.sqrt(np.clip(var, 0.0, None)) * self._y_std
        return mu, std


# ══════════════════════════════════════════════════════════════
# 采样与寻优
# ══════════════════════════════════════════════════════════════

def latin_hypercube(n_samples: int, bounds: Sequence[tuple[float, float]],
                    seed: int | None = None) -> np.ndarray:
    """拉丁超立方采样，返回 (n_samples, d) 的参数矩阵。"""
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    u = sampler.random(n=n_samples)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return qmc.scale(u, lo, hi)


@dataclass
class DesignResult:
    """设计寻优结果。"""

    best_x: np.ndarray
    best_y: float
    history_x: np.ndarray
    history_y: np.ndarray
    surrogate: GaussianProcess
    names: list[str] = field(default_factory=list)
    minimize: bool = True

    @property
    def n_evaluations(self) -> int:
        return int(self.history_y.size)

    def improvement_trace(self) -> np.ndarray:
        """最优值随评估次数的演化（诊断收敛用）。"""
        sign = 1.0 if self.minimize else -1.0
        return np.minimum.accumulate(sign * self.history_y) * sign

    def report(self) -> str:
        lines = [
            f"设计寻优：{self.n_evaluations} 次高保真评估，"
            f"{'最小化' if self.minimize else '最大化'}目标",
            f"  最优目标值 = {self.best_y:.6g}",
            "  最优参数：",
        ]
        for n, v in zip(self.names or [f"x{i}" for i in range(self.best_x.size)],
                        self.best_x):
            lines.append(f"    {n:>14s} = {v:.6g}")
        tr = self.improvement_trace()
        if tr.size > 1:
            lines.append(f"  收敛轨迹（每 25% 处的最优值）："
                         f"{np.array2string(tr[:: max(1, tr.size // 4)], precision=4)}")
        return "\n".join(lines)


def _expected_improvement(mu: np.ndarray, std: np.ndarray,
                          best: float) -> np.ndarray:
    """期望改进（**纯最大化口径**）：EI = (μ−y*)Φ(z) + σφ(z)，z=(μ−y*)/σ。

    这里只实现最大化版本；最小化在调用处通过取负统一转换
    （`sign = -1 if minimize else 1`，对观测值、μ、best 一并取负）。

    ⚠ 曾经的缺陷：本函数原先带 `minimize` 标志，而调用处已经把 `best`
    取过负了，形成**双重取负** —— 结果是最小化问题被反向当成最大化求解：
    EI 每轮都选中目标值最大的点（实测 177→200→211），比随机搜索还差。
    教训：阈值/符号只在一处翻转，不要跨越函数边界重复翻转。
    """
    best = float(best)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 0, (mu - best) / std, 0.0)
        ei = np.where(std > 0,
                      (mu - best) * norm.cdf(z) + std * norm.pdf(z),
                      0.0)
    return np.nan_to_num(ei, nan=0.0)


def bayesian_optimize(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    n_init: int = 8,
    n_iter: int = 15,
    names: Sequence[str] | None = None,
    seed: int = 0,
    minimize: bool = True,
    candidate_pool: int = 2000,
    gp_kwargs: dict | None = None,
    verbose: bool = False,
) -> DesignResult:
    """代理模型驱动的设计寻优（LHS 初采 + EI 迭代加点）。

    参数
    ----
    func : callable
        ``f(x) -> float``，**高保真**目标函数（昂贵）。
        ⚠ **契约**：传入的 ``x`` 是**一维数组**（一个设计点，形状 ``(d,)``），
        不是批量矩阵。这与本仓库求解器的"向量化批量"风格不同，
        因为目标函数内部通常要跑一次完整仿真、只返回标量。
        若你的目标函数是向量化的（接受 ``(n, d)``），请自行包装：
        ``lambda x: vec_func(x[None, :])[0]``。
    bounds : sequence[(lo, hi)]
        每维取值范围（需给有限区间；设计寻优必须有界）。
    n_init : int
        LHS 初始采样次数。
    n_iter : int
        贝叶斯迭代轮数（每轮 1 次高保真调用）。
    candidate_pool : int
        每轮在候选池上评估 EI 取最大。

    返回
    ----
    DesignResult（含历史、代理模型、收敛轨迹）

    ⚠ 适用性说明：贝叶斯优化的价值来自"评估昂贵"。若目标函数是解析式或
    单次评估 < 1 ms，直接用 `scipy.optimize.differential_evolution`
    做全局搜索更简单可靠。本模块面向的是**每次评估 10 ms–分钟级**的
    真实仿真场景（如本仓库的耦合反应器模型）。
    """
    bounds = list(bounds)
    d = len(bounds)
    if d == 0:
        raise ValueError("bounds 不能为空")
    for lo, hi in bounds:
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"非法区间 ({lo}, {hi})")
    if n_init < 2:
        raise ValueError("n_init 至少为 2")
    if names is None:
        names = [f"x{i}" for i in range(d)]
    names = list(names)
    if len(names) != d:
        raise ValueError(f"names 长度 {len(names)} 与维度 {d} 不一致")

    rng = np.random.default_rng(seed)

    # ── 初采 ────────────────────────────────────────────
    X = latin_hypercube(n_init, bounds, seed=seed)
    y = np.array([float(func(x)) for x in X])

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    gp = GaussianProcess(**(gp_kwargs or {}))
    # 统一转成最大化口径：minimize 时对观测值取负（只在这一处翻转符号）
    sign = -1.0 if minimize else 1.0
    for _ in range(n_iter):
        gp.fit(X, y)
        # 候选池：用 LHS 而非随机均匀（5 维以上随机点会聚簇、覆盖不均）
        Xc = latin_hypercube(candidate_pool, bounds, seed=int(rng.integers(1e9)))
        mu, std = gp.predict(Xc, return_std=True)
        best = float(np.max(sign * y))
        ei = _expected_improvement(sign * mu, std, best)
        x_new = Xc[int(np.argmax(ei))]
        y_new = float(func(x_new))
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
        if verbose:
            print(f"    iter {_ + 1:2d}: f = {y_new:.6g}"
                  f"  (best = {(np.min(y) if minimize else np.max(y)):.6g})")

    gp.fit(X, y)
    best_idx = int(np.argmin(y) if minimize else np.argmax(y))
    return DesignResult(best_x=X[best_idx], best_y=float(y[best_idx]),
                        history_x=X, history_y=y, surrogate=gp,
                        names=names, minimize=minimize)
