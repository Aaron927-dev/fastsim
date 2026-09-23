"""一维对流-扩散-反应（ADR）求解器 —— 化学场基座的核心。

对应 COMSOL 化学反应工程模块的「稀物质传递 + 反应」组合，
但用**有限体积 + 解析 Jacobian + 稀疏线性代数**替代 3D FEM，
目标是把单次求解从分钟级压到毫秒级，从而让参数扫描与优化变得可行。

控制方程（稳态）
----------------
    u dC/dz = D d²C/dz² + R(C)

边界条件（Danckwerts，化工反应器的标准封闭条件）
------------------------------------------------
    入口：u C_in = u C(0) − D dC/dz|₀      （对流进料 + 返混通量）
    出口：dC/dz|_L = 0                      （无扩散出流）

这两条边界保证「停留时间分布封闭」，避免 COMSOL 里常见的
「入口用 Dirichlet 导致返混被忽略」的默认设定误差。

数值方案
--------
- 空间：单元中心有限体积，N 个控制体，界面通量 F = uC − D·dC/dz
- 对流：一阶迎风（u>0）；扩散：中心差分
- 稳态：牛顿法 + **解析 Jacobian**（块三对角稀疏）
- 瞬态：方法线 + scipy BDF（刚性）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, sparse
from scipy.sparse import linalg as splinalg

from .network import ReactionNetwork
from .schemes import SCHEMES, peclet_factor

__all__ = ["ADR1D", "analytic_first_order_dispersion"]


def analytic_first_order_dispersion(
    z: np.ndarray | float,
    length: float,
    velocity: float,
    dispersion: float,
    k: float,
    c_in: float = 1.0,
    inlet_bc: str = "danckwerts",
) -> np.ndarray:
    """一维轴向扩散模型 + 一级反应 + 出口零梯度的解析解。

        u dC/dz = D d²C/dz² − k C

    入口边界有两种口径：
    - ``inlet_bc="danckwerts"``（默认，化工标准封闭条件）
          u C_in = u C(0) − D C'(0)     对流进料 + 返混通量
    - ``inlet_bc="dirichlet"``（COMSOL 稀物质传递 Inflow 默认行为）
          C(0) = C_in

    出口统一为 C'(L) = 0。

    ⚠ 两种入口在低 Pe（强返混）下结果差异显著，对标时必须声明用的是哪一种。
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if dispersion <= 0.0:
        # 纯平推流极限：两种入口条件给出同一结果
        return c_in * np.exp(-k * z / velocity)

    Pe = velocity * length / dispersion
    Da = k * length / velocity
    a = np.sqrt(1.0 + 4.0 * Da / Pe)
    m1 = 0.5 * Pe * (1.0 + a)      # ≥ 0，大 Pe 时极大
    m2 = 0.5 * Pe * (1.0 - a)      # ≤ 0，恒非正

    # ── 数值稳定形式 ─────────────────────────────────────────
    # 朴素写法 C(ζ)=A·e^{m1ζ}+B·e^{m2ζ} 在大 Pe 下 e^{m1} 会溢出
    # （实测 Pe=1.6e6 时直接 nan）。
    # 改用按出口缩放的等价形式：
    #     C(ζ) = α·e^{−m1(1−ζ)} + β·e^{m2·ζ}
    # 两项指数均 ≤ 0，全 Pe 区间有界；α 与 β 的关系含 e^{m2−m1}=e^{−Pe·a}，
    # 大 Pe 时正确下溢为 0（对应入口边界层的物理极限）。
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        exp_shift = float(np.exp(np.clip(m2 - m1, -700.0, 0.0)))   # e^{m2−m1}
        exp_m2 = float(np.exp(np.clip(m2, -700.0, 0.0)))           # e^{m2}
        exp_neg_m1 = float(np.exp(np.clip(-m1, -700.0, 0.0)))      # e^{−m1}

        # 出口零梯度: m1·α + m2·β·e^{m2} = 0  →  α = coef·β
        coef = -(m2 * exp_m2 / m1) if m1 != 0.0 else 0.0

        if inlet_bc == "danckwerts":
            # 入口返混: α·e^{−m1}(1 + m1/Pe) + β(1 − m2/Pe) = c_in
            denom = coef * exp_neg_m1 * (1.0 + m1 / Pe) + (1.0 - m2 / Pe)
        elif inlet_bc == "dirichlet":
            # 入口定浓度: α·e^{−m1} + β = c_in
            denom = 1.0 + coef * exp_neg_m1
        else:
            raise ValueError(f"未知 inlet_bc: {inlet_bc!r}（可选 danckwerts / dirichlet）")

        beta = c_in / denom
        alpha = coef * beta

        zeta = z / length
        return alpha * np.exp(np.clip(-m1 * (1.0 - zeta), -700.0, 0.0)) + beta * np.exp(
            np.clip(m2 * zeta, -700.0, 0.0)
        )


@dataclass
class ADR1D:
    """一维对流-扩散-反应模型。

    参数
    ----
    network : ReactionNetwork
        反应网络（提供 R(C) 与解析 Jacobian）。
    length : float
        反应器长度 (m)。
    velocity : float
        表观流速 u (m/s)。用 ``fastsim.flow.profile`` 的剖面均值即可。
    dispersion : float | array
        轴向扩散系数 D (m²/s)。可为标量或长度 N 的数组（非均匀弥散）。
    n_cells : int
        控制体数目。
    """

    network: ReactionNetwork
    length: float
    velocity: float
    dispersion: float = 0.0
    n_cells: int = 200
    inlet_bc: str = "danckwerts"
    convection_scheme: str = "power_law"

    # 计算得到
    _dz: float = field(default=0.0, init=False, repr=False)
    _z: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_cells < 3:
            raise ValueError("n_cells 至少为 3")
        if self.length <= 0.0:
            raise ValueError("length 必须为正")
        if self.inlet_bc not in ("danckwerts", "dirichlet"):
            raise ValueError(
                f"未知 inlet_bc: {self.inlet_bc!r}（可选 danckwerts / dirichlet）"
            )
        if self.convection_scheme not in SCHEMES:
            raise ValueError(
                f"未知 convection_scheme: {self.convection_scheme!r}（可选 {SCHEMES}）"
            )
        self._dz = self.length / self.n_cells
        self._z = (np.arange(self.n_cells) + 0.5) * self._dz

    # ── 基础量 ─────────────────────────────────────────────
    @property
    def z(self) -> np.ndarray:
        """控制体中心坐标 (m)。"""
        return self._z

    @property
    def dz(self) -> float:
        return self._dz

    def _dispersion_cell(self) -> np.ndarray:
        d = np.asarray(self.dispersion, dtype=float)
        if d.ndim == 0:
            return np.full(self.n_cells, float(d))
        if d.shape[0] != self.n_cells:
            # 允许给节点值 → 插值到单元中心
            return np.interp(self._z, np.linspace(0.0, self.length, d.shape[0]), d)
        return d

    # ── 面系数（残差与 Jacobian 共用，保证两者一致）────────
    def _face_coeffs(self) -> np.ndarray:
        """内部面（i 与 i+1 之间）的扩散导纳系数 k_d = D_face·A(|P|)/dz。

        按 A(|P|) 框架（见 schemes.py）：面上净通量
            F = u·c_upwind − k_d·(c_{i+1} − c_i)
        A≡1 即一阶迎风；默认幂律可显著降低数值扩散。
        """
        N, dz = self.n_cells, self._dz
        if N < 2:
            return np.zeros(0)
        D = self._dispersion_cell()
        D_face = 0.5 * (D[:-1] + D[1:])
        A_f = peclet_factor(self.velocity * dz / D_face, self.convection_scheme)
        return D_face * A_f / dz

    # ── 残差与 Jacobian ────────────────────────────────────
    def residual(self, c_flat: np.ndarray, c_in: np.ndarray) -> np.ndarray:
        """稳态残差（长度 N·n_species）。Newton 用，也可独立检查守恒。

        统一的面通量形式（与 ADR2D 同一离散口径）：
            res[i] = F_hi(i) − F_lo(i) − src·dz
            F_hi(i) = u·C_upwind − k_d[i]·(C_{i+1} − C_i)
            F_lo(i) = u·C_upwind − k_d[i−1]·(C_i − C_{i−1})
        """
        n_s, N, dz = self.network.n_species, self.n_cells, self._dz
        u = self.velocity
        D = self._dispersion_cell()
        kd = self._face_coeffs()
        C = c_flat.reshape(N, n_s)

        R = np.array([self.network.rates(C[i]) for i in range(N)])   # (N, n_rxn)
        src = R @ self.network.nu_net.T                              # (N, n_species) = dC/dt

        F_hi = np.zeros_like(C)
        F_lo = np.zeros_like(C)
        if N > 1:
            # 内部面 i+1/2：迎风取上游单元
            cup = np.where(u >= 0.0, C[:-1], C[1:])
            Fz_int = u * cup - kd[:, None] * (C[1:] - C[:-1])
            F_hi[:-1] = Fz_int
            F_lo[1:] = Fz_int

        # 入口面（i=0 的下侧面）
        if self.inlet_bc == "danckwerts":
            F_lo[0] = u * c_in                       # 总通量 = u·c_in
        elif N >= 3:
            # Dirichlet：二次 ghost-cell 还原边界梯度（二阶）
            grad = (3.0 * (C[0] - c_in) - (C[1] - c_in) / 3.0) / dz
            F_lo[0] = u * c_in - D[0] * grad
        else:
            F_lo[0] = u * c_in - (D[0] / (0.5 * dz)) * (C[0] - c_in)

        # 出口面（i=N-1 的上侧面）：零梯度 → 无扩散通量
        F_hi[-1] = u * C[-1]

        return (F_hi - F_lo - src * dz).ravel()

    def jacobian(self, c_flat: np.ndarray) -> sparse.csr_matrix:
        """稳态残差的解析 Jacobian（稀疏，块三对角）。

        ⚠ 必须与 `residual` 使用**同一** k_d（含同一 A(|P|) 因子），
        否则牛顿迭代会失效（残差与导数不自洽）。
        """
        n_s, N, dz = self.network.n_species, self.n_cells, self._dz
        u = self.velocity
        D = self._dispersion_cell()
        kd = self._face_coeffs()
        C = c_flat.reshape(N, n_s)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        def push(i: int, j: int, block: np.ndarray) -> None:
            for a in range(n_s):
                for b in range(n_s):
                    v = block[a, b]
                    if v != 0.0:
                        rows.append(i * n_s + a)
                        cols.append(j * n_s + b)
                        vals.append(v)

        eye = np.eye(n_s)
        for i in range(N):
            Jrxn = self.network.jacobian(0.0, C[i])       # d(src)/dC
            src_jac = Jrxn * dz

            # ── 上侧面 i+1/2（i < N−1）───────────────────
            if i < N - 1:
                k = kd[i]
                if u >= 0.0:
                    push(i, i, (u + k) * eye)
                    push(i, i + 1, -k * eye)
                else:
                    push(i, i, k * eye)
                    push(i, i + 1, (u - k) * eye)
            else:
                push(i, i, u * eye)          # 出口零梯度：只有对流

            # ── 下侧面 i−1/2 ─────────────────────────────
            if i > 0:
                k = kd[i - 1]
                if u >= 0.0:
                    push(i, i - 1, -(u + k) * eye)
                    push(i, i, k * eye)
                else:
                    push(i, i - 1, -k * eye)
                    push(i, i, -(u - k) * eye)
            elif self.inlet_bc == "danckwerts":
                pass                          # 入口总通量与 C 无关
            elif N >= 3:
                # 二次 ghost-cell：F_lo = u·c_in − D·grad
                #   grad = [3(C0−c_in) − (C1−c_in)/3]/dz
                #   d(−D·grad)/dC0 = −3D/dz；d/dC1 = +D/(3dz)
                push(i, i, -3.0 * D[0] / dz * eye)
                push(i, i + 1, D[0] / (3.0 * dz) * eye)
            else:
                push(i, i, -(D[0] / (0.5 * dz)) * eye)

            # ── 反应源 ───────────────────────────────────
            push(i, i, -src_jac)

        return sparse.csr_matrix((vals, (rows, cols)), shape=(N * n_s, N * n_s))

    # ── 求解 ───────────────────────────────────────────────
    def _residual_scale(self, c_in: np.ndarray) -> float:
        """残差的物理量级参考值（以**入口浓度**定标），与 ADR2D 同一口径。"""
        return self._residual_scale_from(max(float(np.abs(c_in).max()), 1e-30))

    def _residual_scale_from(self, c_mag: float) -> float:
        """残差量级参考值，以给定浓度幅值定标。

        ⚠ 为什么必须能按「当前浓度」定标：对**入口浓度为零、原位生成**的物种，
        若固定用 c_in 定标，c_ref 会塌缩到 1e-30，收敛判据随之失去意义。
        """
        D = self._dispersion_cell()
        c_ref = max(float(c_mag), 1e-30)
        conv = abs(self.velocity) * c_ref
        diff = float(np.abs(D).max()) * c_ref / self._dz
        return max(conv, diff, 1e-30)

    def solve_steady(
        self,
        c_in: np.ndarray,
        c_guess: np.ndarray | None = None,
        tol: float = 1e-9,
        max_iter: int = 60,
    ) -> np.ndarray:
        """稳态求解，返回形状 (N, n_species) 的浓度剖面。

        参数
        ----
        tol : float
            **相对**收敛容差（相对 `_residual_scale()` 归一），默认 1e-9。

        用 Newton-Krylov（稀疏解析 Jacobian）；失败时回退到 scipy 的 hybr。
        """
        c_in = np.asarray(c_in, dtype=float)
        if c_in.shape != (self.network.n_species,):
            raise ValueError(
                f"c_in 形状应为 ({self.network.n_species},)，收到 {c_in.shape}"
            )
        n_s, N = self.network.n_species, self.n_cells

        if c_guess is None:
            # 初值：沿程按一级衰减的量级给梯度，帮助 Newton 收敛
            tau = self.length / max(self.velocity, 1e-30)
            decay = np.exp(-np.linspace(0.0, 1.0, N))[:, None]
            guess = np.tile(c_in, (N, 1)) * decay
        else:
            guess = np.asarray(c_guess, dtype=float).copy()

        x0 = guess.ravel()

        # 主路径：阻尼牛顿 + 稀疏解析 Jacobian。
        # 注意 scipy 的 root(method="krylov") 会**忽略**传入的 jac，
        # 走它等于把解析 Jacobian 白写了，故不用它做主求解器。
        # 第三个参数是**相对容差**（见 _damped_newton 文档）。
        x = self._damped_newton(x0, c_in, tol, max_iter)

        res = float(np.linalg.norm(self.residual(x, c_in), ord=np.inf))
        # 外层判据同样按**当前解**的浓度幅值定标（不能用 c_in ——
        # 对入口浓度为零的原位生成物种会塌缩到 1e-30，误判未收敛并触发稠密回退）
        c_mag = max(float(np.abs(c_in).max()), float(np.abs(x).max()), 1e-30)
        thr = tol * self._residual_scale_from(c_mag)
        if not np.isfinite(res) or res > 100.0 * thr:
            n_unknown = x.size
            if n_unknown <= 20000:
                sol = optimize.root(
                    lambda v: self.residual(v, c_in), x, method="hybr", tol=tol
                )
                if np.isfinite(sol.fun).all():
                    res_alt = np.linalg.norm(sol.fun, ord=np.inf)
                    if res_alt < res:
                        x = sol.x
        out = x.reshape(N, n_s)
        self._warn_if_unphysical(out, c_in)
        return out

    def min_concentration(self, profile: np.ndarray) -> float:
        """解的最小浓度，用于物理有效性自检。"""
        return float(np.asarray(profile).min())

    def _warn_if_unphysical(self, profile: np.ndarray, c_in: np.ndarray) -> None:
        """负浓度护栏（与 ADR2D 同一口径）。"""
        import warnings

        ref = max(float(np.abs(c_in).max()), 1e-30)
        cmin = float(profile.min())
        if cmin < -1e-6 * ref:
            warnings.warn(
                f"解中出现负浓度 (min = {cmin:.4g}，入口浓度 {ref:.4g})。"
                "常见原因：边界通量超过传质所能供给的极限，"
                "或弥散系数/网格不足以解析边界层。",
                UserWarning,
                stacklevel=2,
            )

    def _damped_newton(
        self, x0: np.ndarray, c_in: np.ndarray, tol: float, max_iter: int
    ) -> np.ndarray:
        """阻尼牛顿。第三个参数是**相对容差**（与 ADR2D 同口径）。

        收敛判据：`|f| < tol · max(|f₀|, 尺度(当前浓度))` ——
        取两者较大值以同时覆盖「高活性/小尺度」与「入口浓度为零」两种病态。
        """
        x = x0.copy()
        f0: float | None = None
        for _ in range(max_iter):
            # ⚠ f 必须在循环内**每次重算**。曾把它提到循环外（为取 f0），
            #   结果收敛检查永远比对初始残差 → 牛顿跑满 max_iter 空转，
            #   实测把 27 s 的测试拖到 88 s（线性问题本该一步收敛）。
            f = self.residual(x, c_in)
            norm = float(np.linalg.norm(f, ord=np.inf))
            if f0 is None:
                f0 = max(float(np.abs(f).max()), 1e-300)
            c_mag = max(float(np.abs(x).max()), 1e-30)
            if norm < tol * max(f0, self._residual_scale_from(c_mag)):
                break
            J = self.jacobian(x).tocsc()
            try:
                dx = splinalg.spsolve(J, -f)
            except Exception:  # noqa: BLE001
                break
            if not np.all(np.isfinite(dx)):
                break
            alpha = 1.0
            for _ in range(30):
                xn = x + alpha * dx
                if np.all(xn > -1e-12) and np.linalg.norm(
                    self.residual(xn, c_in), ord=np.inf
                ) < norm:
                    break
                alpha *= 0.5
            x = x + alpha * dx
        return x

    def solve_transient(
        self,
        c_in: np.ndarray,
        c0: np.ndarray | None = None,
        t_end: float = 10.0,
        n_out: int = 50,
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """瞬态求解（方法线 + BDF）。返回值 (t, C)，C 形状 (n_out, N, n_species)。"""
        from scipy.integrate import solve_ivp

        c_in = np.asarray(c_in, dtype=float)
        n_s, N, dz = self.network.n_species, self.n_cells, self._dz
        u = self.velocity
        D = self._dispersion_cell()

        if c0 is None:
            x0 = np.tile(c_in, (N, 1)).ravel()
        else:
            x0 = np.asarray(c0, dtype=float).ravel()

        kd = self._face_coeffs()

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            """瞬态右端 dC/dt = −(F_hi − F_lo)/dz + src（与稳态残差同口径）。"""
            C = x.reshape(N, n_s)
            R = np.array([self.network.rates(C[i]) for i in range(N)])
            src = R @ self.network.nu_net.T

            F_hi = np.zeros_like(C)
            F_lo = np.zeros_like(C)
            if N > 1:
                cup = np.where(u >= 0.0, C[:-1], C[1:])
                Fz_int = u * cup - kd[:, None] * (C[1:] - C[:-1])
                F_hi[:-1] = Fz_int
                F_lo[1:] = Fz_int

            if self.inlet_bc == "danckwerts":
                F_lo[0] = u * c_in
            elif N >= 3:
                # 与稳态残差同口径：二次 ghost-cell 还原边界梯度
                grad = (3.0 * (C[0] - c_in) - (C[1] - c_in) / 3.0) / dz
                F_lo[0] = u * c_in - D[0] * grad
            else:
                F_lo[0] = u * c_in - (D[0] / (0.5 * dz)) * (C[0] - c_in)
            F_hi[-1] = u * C[-1]

            return (-(F_hi - F_lo) / dz + src).ravel()

        t_eval = np.linspace(0.0, t_end, n_out)
        sol = solve_ivp(rhs, (0.0, t_end), x0, method="BDF", t_eval=t_eval,
                        rtol=rtol, atol=atol)
        return sol.t, sol.y.T.reshape(len(sol.t), N, n_s)

    # ── 评价 ───────────────────────────────────────────────
    def conversion(self, profile: np.ndarray, species: str, c_in: float) -> float:
        """按出口浓度计算转化率。"""
        idx = self.network.species.index(species)
        c_out = profile[-1, idx]
        return float(1.0 - c_out / c_in) if c_in > 0 else float("nan")

    def residence_time(self) -> float:
        """空塔停留时间 τ = L/u (s)。"""
        return self.length / self.velocity
