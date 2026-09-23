"""二维对流-扩散-反应（ADR2D）—— 化学场基座的二维扩展。

一维模型只能描述沿程变化；管式/环隙反应器的真实物理包含**径向梯度**
（壁面浓度边界层、电极表面反应、非均匀速度剖面），必须升到二维。

控制方程（轴对称，r-z 平面）
---------------------------
    u_z ∂c/∂z + u_r ∂c/∂r = (1/r)∂/∂r( r·D_r·∂c/∂r ) + ∂/∂z( D_z·∂c/∂z ) + R(c)

平面模式（x-y）去掉 1/r 因子，两者共用同一套装配代码
（轴对称 → 面权重含 2πr；平面 → 面权重为常数）。

⚠️ **单位约定：浓度必须是 SI 的 mol/m³**
--------------------------------------
守恒式要求「浓度 × 体积流量 = 摩尔流量」，而壁面通量是 mol·m⁻²·s⁻¹
（法拉第定律的输出）。只有浓度取 mol/m³ 时
    mol/m³ × m³/s = mol/s
才与 mol/m²/s × m² = mol/s 在同一量纲上守恒。

因此若动力学速率常数来自文献（惯例是 **M 基**），二级常数必须换算：
    k[mol/m³ 基] = k[M 基] / 1000      （见 `mechanisms.k_second_order_to_SI`）
一级常数（1/s）无需换算。忘记这一步会让反应速率差 1000 倍。

数值方案
--------
- 空间：单元中心有限体积、结构化网格、**守恒格式**
- 对流：一阶迎风（**已知精度上限：advection-dominated 时仅一阶**）
- 扩散：中心差分；轴/壁面用 ghost-cell，入口 Dirichlet 用**二次 ghost-cell**
  （对二次分布可精确还原边界梯度 → 二阶通量）
- 稳态：阻尼牛顿 + **稀疏解析 Jacobian**（5 点模板 × 物质块）
- 瞬态：方法线 + scipy BDF

⚠️ **速度场必须满足 ∇·u = 0**
---------------------------
本求解器采用**守恒形式** ∇·(u·c) = ∇·(D∇c) + R，这是变速度场的物理正确形式
（质量守恒），在离散上等价于 `u·∇c + c·∇·u`。
因此若传入的速度场不满足 ∇·u = 0，会**凭空多出一个体积源 c·∇·u**。
用 `divergence_free_violation()` 自检：返回值应 < 1e-6，否则先修正速度场。

（这个坑是在 MMS 验证中被抓出来的：一个非无散的径向速度场
`u_r = u₀·r/R` 使离散系统与 PDE 不一致，误差高达 45% 且**随网格加密上升** ——
正是"离散系统收敛到了另一个问题"的典型特征。）

边界条件（**只保留已实现且被验证过的选项**，避免存在静默错误的支路）
------------------------------------------------------------------
+-------------------+-----------------------------------------------+
| z = 0 入口        | ``danckwerts``（总通量 = u·c_in）/ ``dirichlet`` |
| z = L 出口        | 零梯度（唯一选项）                             |
| r = R 外壁        | ``zero_flux``（默认）或 ``wall_flux`` 指定通量   |
| r = r_inner > 0   | ``zero_flux``（默认）或 ``wall_flux_inner``      |
| r = 0（轴对称）   | 自然满足（面权重为 0，对应 ∂c/∂r = 0）          |
+-------------------+-----------------------------------------------+

``r_inner > 0`` 即**环隙几何**：内管外壁可作电极面，与
``fastsim.echem.SecondaryCurrent2D`` 的几何完全对齐 ——
后者算出的局部电流密度经法拉第换算即可作为 ``wall_flux_inner``，
构成「电场 → 界面通量 → 浓度场」的完整闭环（见 examples/）。

**``wall_flux`` 是电化学耦合接口**：电极反应给出的摩尔通量
（mol·m⁻²·s⁻¹，正值为流入流体域）直接作为 r=R 的边界源，
即可把电流分布与化学场连起来（P3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import optimize, sparse
from scipy.sparse import linalg as splinalg

from .network import ReactionNetwork
from .schemes import SCHEMES, peclet_factor

__all__ = ["ADR2D"]


@dataclass
class ADR2D:
    """二维反应-传质模型。

    参数
    ----
    network : ReactionNetwork
        反应网络（提供 R(c) 与解析 Jacobian）。
    length, radius : float
        z 方向长度与 r 方向尺寸 (m)：轴对称时为半径，平面时为宽度。
    velocity_z, velocity_r : float | (nr,) | (nz, nr)
        速度分量 (m/s)。解析速度剖面由 ``fastsim.flow`` 给出，无需解 Navier-Stokes。
    dispersion : float | (D_r, D_z) | 数组
        弥散/扩散系数 (m²/s)。
    mode : str
        ``"axisymmetric"``（r-z）或 ``"planar"``（x-y）。
    wall_flux : callable | None
        ``f(z) -> array(nz,) 或 (nz, ns)``，外壁 r=R 处的摩尔通量
        (mol·m⁻²·s⁻¹)，正值表示流入流体域。**按 r=R 精确求值**。
    extra_source : callable | None
        ``f(R_grid, Z_grid) -> (nz, nr, ns)`` 额外体积源，用于 MMS 验证。
    """

    network: ReactionNetwork
    length: float
    radius: float
    velocity_z: float | np.ndarray = 0.0
    velocity_r: float | np.ndarray = 0.0
    dispersion: float | tuple = 1e-9
    n_z: int = 80
    n_r: int = 32
    mode: str = "axisymmetric"
    inlet_bc: str = "danckwerts"
    r_inner: float = 0.0
    convection_scheme: str = "power_law"
    wall_flux: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    wall_flux_inner: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    extra_source: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None

    _dz: float = field(default=0.0, init=False, repr=False)
    _dr: float = field(default=0.0, init=False, repr=False)
    _z: np.ndarray = field(default=None, init=False, repr=False)
    _r: np.ndarray = field(default=None, init=False, repr=False)
    _rg: np.ndarray = field(default=None, init=False, repr=False)
    _zg: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_z < 2 or self.n_r < 2:
            raise ValueError("n_z 与 n_r 至少为 2")
        if self.length <= 0.0 or self.radius <= 0.0:
            raise ValueError("length 与 radius 必须为正")
        if not (0.0 <= self.r_inner < self.radius):
            raise ValueError("需 0 ≤ r_inner < radius")
        if self.mode not in ("axisymmetric", "planar"):
            raise ValueError(f"未知 mode: {self.mode!r}")
        if self.inlet_bc not in ("danckwerts", "dirichlet"):
            raise ValueError(
                f"未知 inlet_bc: {self.inlet_bc!r}（可选 danckwerts / dirichlet）"
            )
        if self.convection_scheme not in SCHEMES:
            raise ValueError(
                f"未知 convection_scheme: {self.convection_scheme!r}（可选 {SCHEMES}）"
            )
        # 预计算并按值检查，避免求解中途才发现形状不符
        self._dz = self.length / self.n_z
        self._dr = (self.radius - self.r_inner) / self.n_r
        self._z = (np.arange(self.n_z) + 0.5) * self._dz
        self._r = self.r_inner + (np.arange(self.n_r) + 0.5) * self._dr
        self._rg, self._zg = np.meshgrid(self._r, self._z)
        self._as_field(self.velocity_z, "velocity_z")
        self._as_field(self.velocity_r, "velocity_r")
        self._dispersion_fields()

    # ── 基本量 ─────────────────────────────────────────────
    @property
    def z(self) -> np.ndarray:
        return self._z

    @property
    def r(self) -> np.ndarray:
        return self._r

    @property
    def dz(self) -> float:
        return self._dz

    @property
    def dr(self) -> float:
        return self._dr

    def _weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A_rface[nr+1], A_zface[nr], V[nr])。"""
        if self.mode == "axisymmetric":
            r_face = self.r_inner + np.arange(self.n_r + 1) * self._dr
            a_rface = 2.0 * np.pi * r_face * self._dz
            a_zface = 2.0 * np.pi * self._r * self._dr
            vol = 2.0 * np.pi * self._r * self._dr * self._dz
        else:
            a_rface = np.full(self.n_r + 1, self._dz)
            a_zface = np.full(self.n_r, self._dr)
            vol = np.full(self.n_r, self._dr * self._dz)
        return a_rface, a_zface, vol

    def _as_field(self, val, name: str) -> np.ndarray:
        a = np.asarray(val, dtype=float)
        if a.ndim == 0:
            return np.full((self.n_z, self.n_r), float(a))
        if a.ndim == 1:
            if a.shape[0] != self.n_r:
                raise ValueError(
                    f"{name} 一维输入长度应为 n_r={self.n_r}，收到 {a.shape[0]}"
                )
            return np.tile(a[None, :], (self.n_z, 1))
        if a.ndim == 2:
            if a.shape != (self.n_z, self.n_r):
                raise ValueError(
                    f"{name} 二维输入形状应为 ({self.n_z},{self.n_r})，收到 {a.shape}"
                )
            return a
        raise ValueError(f"{name} 维度不支持: {a.ndim}")

    def _dispersion_fields(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.dispersion
        if isinstance(d, (tuple, list)):
            if len(d) != 2:
                raise ValueError("dispersion 元组需为 (D_r, D_z)")
            return self._as_field(d[0], "D_r"), self._as_field(d[1], "D_z")
        f = self._as_field(d, "dispersion")
        return f, f

    def velocity_fields(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (u_z, u_r)，各为 (nz, nr)，供外部检查。"""
        return (
            self._as_field(self.velocity_z, "velocity_z"),
            self._as_field(self.velocity_r, "velocity_r"),
        )

    def velocity_divergence(self) -> np.ndarray:
        """离散的 ∇·u 场 (nz, nr)，与求解器实际使用的面通量口径一致。

        轴对称： ∇·u = (1/r)∂(r·u_r)/∂r + ∂u_z/∂z
        平面：   ∇·u = ∂u_r/∂r + ∂u_z/∂z
        """
        nz, nr = self.n_z, self.n_r
        uz, ur = self.velocity_fields()
        a_rface, a_zface, vol = self._weights()

        # 面速度（内部面取算术平均，边界取单侧）
        uz_hi = np.empty((nz, nr))
        uz_lo = np.empty((nz, nr))
        uz_hi[:-1] = 0.5 * (uz[:-1] + uz[1:])
        uz_hi[-1] = uz[-1]
        uz_lo[0] = uz[0]
        uz_lo[1:] = 0.5 * (uz[:-1] + uz[1:])

        ur_hi = np.empty((nz, nr))
        ur_lo = np.empty((nz, nr))
        ur_hi[:, :-1] = 0.5 * (ur[:, :-1] + ur[:, 1:])
        ur_hi[:, -1] = ur[:, -1]
        ur_lo[:, 0] = ur[:, 0]
        ur_lo[:, 1:] = 0.5 * (ur[:, :-1] + ur[:, 1:])

        div = (uz_hi - uz_lo) * a_zface[None, :]
        div = div + ur_hi * a_rface[1:][None, :] - ur_lo * a_rface[:-1][None, :]
        return div / vol[None, :]

    def divergence_free_violation(self) -> float:
        """无量纲的「非无散」程度：max|∇·u| · L_char / U_char。

        > 1e-6 说明速度场不满足质量守恒，求解器会因使用守恒形式 ∇·(uc)
        而引入一个**虚假的体积源 c·∇·u**（物理上正确，但通常不是使用者想要的）。
        """
        uz, ur = self.velocity_fields()
        u_char = max(float(np.abs(uz).max()), float(np.abs(ur).max()), 1e-30)
        d = np.abs(self.velocity_divergence()).max()
        return float(d * self.length / u_char)

    # ── 通量装配（残差与 Jacobian 共用）────────────────────
    def _face_fluxes(
        self, C: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算四个面的通量场。

        返回 (Fz_hi, Fz_lo, Fr_hi, Fr_lo)，各为 (nz, nr, ns)，单位面积的摩尔通量。
        Fz_hi(i) 是单元 i 的上侧面（z 更大一侧），Fz_lo(i) 是下侧面。
        """
        nz, nr, ns = self.n_z, self.n_r, self.network.n_species
        dz, dr = self._dz, self._dr
        uz, ur = self.velocity_fields()
        D_r, D_z = self._dispersion_fields()
        c_in = np.asarray(self._c_in_ref, dtype=float)

        Fz_hi = np.zeros_like(C)
        Fz_lo = np.zeros_like(C)

        # 轴向内部面：face k 位于单元 k 与 k+1 之间
        # 通量按 A(|P|) 框架组装（见 schemes.py）：
        #   Fz = (D/dz)·A(|P|)·(c_i − c_{i+1}) + max(u,0)·c_i + min(u,0)·c_{i+1}
        # A≡1 即一阶迎风；幂律/指数格式可大幅降低数值扩散。
        if nz > 1:
            uzf = 0.5 * (uz[:-1] + uz[1:])
            Dz_f = 0.5 * (D_z[:-1] + D_z[1:])
            A_z = peclet_factor(uzf * dz / Dz_f, self.convection_scheme)
            pos = uzf >= 0.0
            cup = np.where(pos[:, :, None], C[:-1], C[1:])
            Fz_int = (
                uzf[:, :, None] * cup
                - (Dz_f * A_z / dz)[:, :, None] * (C[1:] - C[:-1])
            )
            Fz_hi[:-1] = Fz_int          # 单元 i 的上侧面
            Fz_lo[1:] = Fz_int           # 单元 i+1 的下侧面

        # 轴向入口面：总通量（Danckwerts）或 Dirichlet
        if self.inlet_bc == "danckwerts":
            Fz_lo[0] = uz[0][:, None] * c_in[None, :]
        else:
            cin_f = np.broadcast_to(c_in[None, :], (nr, ns))
            if nz >= 3:
                # 二次 ghost-cell 外推：对二次分布可精确还原边界梯度 → 二阶通量。
                # 梯度 = [3(c_0−c_w) − (c_1−c_w)/3]/dz
                grad = (
                    3.0 * (C[0] - cin_f) - (C[1] - cin_f) / 3.0
                ) / dz
            else:
                # 网格过粗时退回半格（线性 ghost-cell），此时只有一阶梯度
                grad = (C[0] - cin_f) / (0.5 * dz)
            Fz_lo[0] = uz[0][:, None] * cin_f - (D_z[0])[:, None] * grad

        # 轴向出口面：零梯度 → 只有对流项
        Fz_hi[-1] = uz[-1][:, None] * C[-1]

        Fr_hi = np.zeros_like(C)
        Fr_lo = np.zeros_like(C)

        # 径向内部面：face k 位于单元 k 与 k+1 之间
        if nr > 1:
            urf = 0.5 * (ur[:, :-1] + ur[:, 1:])
            Dr_f = 0.5 * (D_r[:, :-1] + D_r[:, 1:])
            A_r = peclet_factor(urf * dr / Dr_f, self.convection_scheme)
            cup = np.where((urf >= 0.0)[:, :, None], C[:, :-1], C[:, 1:])
            Fr_int = (
                urf[:, :, None] * cup
                - (Dr_f * A_r / dr)[:, :, None] * (C[:, 1:] - C[:, :-1])
            )
            Fr_hi[:, :-1] = Fr_int
            Fr_lo[:, 1:] = Fr_int

        # 内侧 r=r_inner：
        #   r_inner == 0 → 轴对称中心线，面权重为 0（自然对称），通量项无贡献
        #   r_inner > 0  → 实体壁面（可作电极面）：零通量或指定摩尔通量
        #
        # ⚠ 内外侧符号相反（面法线方向不同）：
        #   外侧 r=R 的域外法线为 +r̂ → 流入域的通量对应 F_r < 0，故取 −j
        #   内侧 r=r_inner 的外法线为 −r̂ → 流入域的通量对应 F_r > 0，故取 +j
        if self.r_inner > 0.0 and self.wall_flux_inner is not None:
            Fr_lo[:, 0] = self._parse_wall_flux(self.wall_flux_inner, "wall_flux_inner")
        # 外侧 r=R：零通量或指定摩尔通量（正值流入域 → +r 方向通量为负）
        if self.wall_flux is not None:
            Fr_hi[:, -1] = -self._parse_wall_flux(self.wall_flux, "wall_flux")
        return Fz_hi, Fz_lo, Fr_hi, Fr_lo

    def _parse_wall_flux(self, fn, name: str) -> np.ndarray:
        """解析壁面通量回调，返回 (nz, ns)。

        壁面通量按壁面位置**精确求值**（传入 z 坐标）；若在单元中心
        r = R−dr/2 求值会引入 O(dr) 误差、破坏二阶收敛。

        多物质体系下必须返回 (nz, ns) 形状：一维数组会被广播到所有物质
        （产物也会被消耗 → 负浓度），这里显式拒绝而不是静默广播。
        """
        ns = self.network.n_species
        jw = np.asarray(fn(self._z), dtype=float)
        if jw.ndim == 1:
            if ns > 1:
                raise ValueError(
                    f"{name} 返回了一维数组，但本模型有 {ns} 个物质。"
                    "一维数组会被广播到**所有**物质（产物也会被消耗 → 负浓度）。"
                    f"请返回形状 (n_z, {ns}) 的数组，按物质分别给出通量；"
                    "不需要参与界面反应的物质请填 0。"
                )
            return jw[:, None]
        if jw.shape != (self.n_z, ns):
            raise ValueError(
                f"{name} 返回形状 {jw.shape}，应为 ({self.n_z}, {ns})"
            )
        return jw

    # 求解时注入的入口浓度（供 _face_fluxes 使用）
    _c_in_ref: np.ndarray = field(default=None, init=False, repr=False)

    # ── 残差 ───────────────────────────────────────────────
    def residual(self, x: np.ndarray, c_in: np.ndarray) -> np.ndarray:
        """稳态残差（长度 n_z·n_r·n_species）。"""
        nz, nr, ns = self.n_z, self.n_r, self.network.n_species
        C = x.reshape(nz, nr, ns)
        self._c_in_ref = np.asarray(c_in, dtype=float)
        a_rface, a_zface, vol = self._weights()

        Fz_hi, Fz_lo, Fr_hi, Fr_lo = self._face_fluxes(C)

        res = (Fz_hi - Fz_lo) * a_zface[None, :, None]
        res = res + Fr_hi * a_rface[1:][None, :, None] - Fr_lo * a_rface[:-1][None, :, None]

        src = self.network.source_batch(C.reshape(-1, ns)).reshape(nz, nr, ns)
        if self.extra_source is not None:
            src = src + np.asarray(self.extra_source(self._rg, self._zg), dtype=float)
        res = res - src * vol[None, :, None]
        return res.ravel()

    # ── 解析 Jacobian ──────────────────────────────────────
    def jacobian(self, x: np.ndarray, c_in: np.ndarray | None = None) -> sparse.csr_matrix:
        """稀疏解析 Jacobian（5 点模板 × 物质块）。**全向量化构造**。

        性能说明：早期版本用逐单元 Python 双重循环（nz×nr 次）拼装 COO，
        在 60×16 网格上单次装配约 0.1 s；Newton 迭代上百次时总耗时达分钟级
        （实测某工况 84 s），完全违背"快速响应"的定位。
        这里改为按**面**批量构造系数数组，只保留 O(1) 次 numpy 操作与一次 COO 装配。
        """
        nz, nr, ns = self.n_z, self.n_r, self.network.n_species
        dz, dr = self._dz, self._dr
        C = x.reshape(nz, nr, ns)
        uz, ur = self.velocity_fields()
        D_r, D_z = self._dispersion_fields()
        a_rface, a_zface, vol = self._weights()

        # 逐单元标量系数（对角项）与四个邻块系数
        diag = np.zeros((nz, nr))
        c_im1 = np.zeros((nz, nr))      # (i,j) → (i−1,j)
        c_ip1 = np.zeros((nz, nr))      # (i,j) → (i+1,j)
        c_jm1 = np.zeros((nz, nr))      # (i,j) → (i,j−1)
        c_jp1 = np.zeros((nz, nr))      # (i,j) → (i,j+1)

        # ── 轴向内部面（(i) 与 (i+1) 之间），面通量为
        #    F = uzf·C_up − A(|P|)·(Dz/dz)(C_{i+1} − C_i)
        if nz > 1:
            uzf = 0.5 * (uz[:-1] + uz[1:])
            Dzf = 0.5 * (D_z[:-1] + D_z[1:])
            # ★ 与 residual 用同一 A(|P|)——不一致会让牛顿迭代失效
            Azf = peclet_factor(uzf * dz / Dzf, self.convection_scheme)
            az = a_zface[None, :]
            pos = uzf >= 0.0
            d_i_self = np.where(pos, uzf + Dzf * Azf / dz, Dzf * Azf / dz) * az
            d_i_next = np.where(pos, -Dzf * Azf / dz, uzf - Dzf * Azf / dz) * az
            # 单元 i+1 的偏导系数（res[i+1] 取负号）
            d_n_self = np.where(pos, Dzf * Azf / dz, -uzf + Dzf * Azf / dz) * az
            d_n_prev = np.where(pos, -(uzf + Dzf * Azf / dz), -Dzf * Azf / dz) * az
            diag[:-1] += d_i_self
            c_ip1[:-1] += d_i_next
            diag[1:] += d_n_self
            c_im1[1:] += d_n_prev

        # ── 轴向边界 ────────────────────────────────────────
        if self.inlet_bc == "dirichlet":
            if nz >= 3:
                diag[0] += 3.0 * D_z[0] / dz * a_zface
                c_ip1[0] += -D_z[0] / (3.0 * dz) * a_zface
            else:
                diag[0] += 2.0 * D_z[0] / dz * a_zface
        # danckwerts 入口：总通量与 C 无关
        # 出口零梯度：res[−1] += uz[−1]·C[−1]·az → 对角
        diag[-1] += uz[-1] * a_zface

        # ── 径向内部面 ─────────────────────────────────────
        if nr > 1:
            urf = 0.5 * (ur[:, :-1] + ur[:, 1:])
            Drf = 0.5 * (D_r[:, :-1] + D_r[:, 1:])
            Arf = peclet_factor(urf * dr / Drf, self.convection_scheme)
            ar_hi = a_rface[1:-1][None, :]        # 面 j+1 的面积，(nr−1,)
            pos_r = urf >= 0.0
            e_j_self = np.where(pos_r, urf + Drf * Arf / dr, Drf * Arf / dr) * ar_hi
            e_j_next = np.where(pos_r, -Drf * Arf / dr, urf - Drf * Arf / dr) * ar_hi
            e_n_self = np.where(pos_r, Drf * Arf / dr, -urf + Drf * Arf / dr) * ar_hi
            e_n_prev = np.where(pos_r, -(urf + Drf * Arf / dr), -Drf * Arf / dr) * ar_hi
            diag[:, :-1] += e_j_self
            c_jp1[:, :-1] += e_j_next
            diag[:, 1:] += e_n_self
            c_jm1[:, 1:] += e_n_prev

        # ── 反应源（全 ns×ns 块）────────────────────────────
        J_react = self.network.jacobian_batch(C.reshape(-1, ns))
        J_react = J_react.reshape(nz, nr, ns, ns) * vol[None, :, None, None]

        # ── 组装 COO ────────────────────────────────────────
        cells = np.arange(nz * nr)
        base = (cells * ns)[:, None, None]                     # (N,1,1)
        r_off = np.arange(ns)[None, :, None]                   # (1,ns,1)
        c_off = np.arange(ns)[None, None, :]                   # (1,1,ns)
        shape3 = (nz * nr, ns, ns)

        rows = [np.broadcast_to(base + r_off, shape3).ravel()]
        cols = [np.broadcast_to(base + c_off, shape3).ravel()]
        vals = [(-J_react.reshape(nz * nr, ns, ns)).ravel()]

        # 标量对角项（coef × eye）
        diag_flat = diag.ravel()
        base_r = (cells * ns)[:, None] + np.arange(ns)[None, :]   # (N, ns)
        rows.append(base_r.ravel())
        cols.append(base_r.ravel())
        vals.append(np.repeat(diag_flat, ns))

        # 四个邻块（均为 coef × eye）
        def add_offset(coef: np.ndarray, di: int, dj: int) -> None:
            if di == 0 and dj == 0:
                return
            ii, jj = np.meshgrid(np.arange(nz), np.arange(nr), indexing="ij")
            ti, tj = ii + di, jj + dj
            m = (ti >= 0) & (ti < nz) & (tj >= 0) & (tj < nr)
            if not m.any():
                return
            src_cells = (ii[m] * nr + jj[m])
            dst_cells = (ti[m] * nr + tj[m])
            rr = (src_cells[:, None] * ns + np.arange(ns)[None, :]).ravel()
            cc = (dst_cells[:, None] * ns + np.arange(ns)[None, :]).ravel()
            vv = np.repeat(coef[m], ns)
            rows.append(rr); cols.append(cc); vals.append(vv)

        add_offset(c_im1, -1, 0)
        add_offset(c_ip1, +1, 0)
        add_offset(c_jm1, 0, -1)
        add_offset(c_jp1, 0, +1)

        n = nz * nr * ns
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )

    # ── 求解 ───────────────────────────────────────────────
    def min_concentration(self, profile: np.ndarray) -> float:
        """解的最小浓度，用于物理有效性自检。"""
        return float(np.asarray(profile).min())

    def _residual_scale(self, c_in: np.ndarray) -> float:
        """残差的物理量级参考值（以**入口浓度**定标）。"""
        return self._residual_scale_from(max(float(np.abs(c_in).max()), 1e-30))

    def _residual_scale_from(self, c_mag: float) -> float:
        """残差的物理量级参考值，以给定浓度幅值定标。

        ⚠ 为什么必须能按「当前浓度」定标：对**入口浓度为零、原位生成**的物种
        （如电极生成的 H₂O₂/O₃），若固定用 c_in 定标，c_ref 会塌缩到 1e-30，
        收敛判据随之失去意义 —— 表现为解已正确收敛却报「未收敛」，
        并触发稠密回退（大网格下直接内存溢出）。实测案例见
        bench/validate_vs_naturecomms.py。
        """
        uz, ur = self.velocity_fields()
        D_r, D_z = self._dispersion_fields()
        a_rface, a_zface, _ = self._weights()
        c_ref = max(float(c_mag), 1e-30)
        a_max = max(float(a_zface.max()), float(a_rface.max()))
        u_max = max(float(np.abs(uz).max()), float(np.abs(ur).max()))
        d_max = max(float(D_r.max()), float(D_z.max()))
        h_min = min(self._dz, self._dr)
        conv = u_max * c_ref * a_max
        diff = d_max * c_ref / h_min * a_max
        return max(conv, diff, 1e-30)

    def solve_steady(
        self,
        c_in: np.ndarray,
        c_guess: np.ndarray | None = None,
        tol: float = 1e-9,
        max_iter: int = 80,
    ) -> np.ndarray:
        """稳态求解，返回 (n_z, n_r, n_species)。

        参数
        ----
        tol : float
            **相对**收敛容差（相对 `_residual_scale_from()` 归一），默认 1e-9。

        若解出现**显著负浓度**，会发出 `UserWarning` 而不是静默返回 ——
        负浓度几乎总是意味着边界条件病态（最常见：壁面通量超过了传质极限，
        即"极限电流"被突破，见 examples/demo_tubular_electrode.py）。
        """
        c_in = np.asarray(c_in, dtype=float)
        ns = self.network.n_species
        if c_in.shape != (ns,):
            raise ValueError(f"c_in 形状应为 ({ns},)，收到 {c_in.shape}")
        self._c_in_ref = c_in

        if c_guess is None:
            guess = np.tile(c_in, (self.n_z, self.n_r, 1))
        else:
            guess = np.asarray(c_guess, dtype=float).copy()

        # ⚠ `_damped_newton` 的第三个参数是**相对容差 tol**（它内部按
        #    max(|f₀|, 尺度(当前浓度)) 自行折算绝对值）。
        #    此前把按 c_in 折算出的绝对阈值传了进去 —— 对入口浓度为零的
        #    物种该阈值小到 ~1e-42，牛顿空转满 max_iter（实测 7200 未知量
        #    耗 231 s），而线性问题本应一步收敛。
        x = self._damped_newton(guess.ravel(), c_in, tol, max_iter)

        res = float(np.linalg.norm(self.residual(x, c_in), ord=np.inf))
        # 外层收敛判据同样必须按**当前解**的浓度幅值定标（不能用 c_in）
        c_mag = max(float(np.abs(c_in).max()), float(np.abs(x).max()), 1e-30)
        thr = tol * self._residual_scale_from(c_mag)
        if not np.isfinite(res) or res > 100.0 * thr:
            x = self._try_dense_fallback(x, c_in, res, tol)

        out = x.reshape(self.n_z, self.n_r, ns)
        self._warn_if_unphysical(out, c_in)
        return out

    # 稠密回退的规模上限：hybr 会构造 (N·ns)² 的稠密 Jacobian，
    # 16 万未知量 → 191 GB，直接内存溢出。超过上限就不回退，
    # 而是保留牛顿结果并明确报告未收敛（由调用方决定是否接受）。
    _DENSE_FALLBACK_MAX_UNKNOWN = 20000

    def _try_dense_fallback(self, x, c_in, res_cur, tol):
        n_unknown = x.size
        if n_unknown > self._DENSE_FALLBACK_MAX_UNKNOWN:
            import warnings

            warnings.warn(
                f"Newton 未收敛，且未知量数 {n_unknown} 超过稠密回退上限 "
                f"{self._DENSE_FALLBACK_MAX_UNKNOWN}（否则需 "
                f"{(n_unknown*8*2/1e9):.0f} GB 内存）。"
                "已跳过回退，返回当前迭代结果 —— 请减小网格或放宽 tol。",
                UserWarning, stacklevel=3,
            )
            return x
        sol = optimize.root(
            lambda v: self.residual(v, c_in), x, method="hybr", tol=tol
        )
        if np.isfinite(sol.fun).all() and np.linalg.norm(
                sol.fun, ord=np.inf) < res_cur:
            return sol.x
        return x

    def _warn_if_unphysical(self, profile: np.ndarray, c_in: np.ndarray) -> None:
        """负浓度护栏：显著负值几乎总意味着边界条件病态，必须让使用者知道。"""
        import warnings

        ref = max(float(np.abs(c_in).max()), 1e-30)
        cmin = float(profile.min())
        if cmin < -1e-6 * ref:
            warnings.warn(
                f"解中出现负浓度 (min = {cmin:.4g}，入口浓度 {ref:.4g})。"
                "常见原因：壁面/边界通量超过了传质所能供给的极限"
                "（极限电流被突破），或弥散系数/网格不足以解析边界层。"
                "请核对 wall_flux 是否小于 k_L·c_bulk，或加密网格。",
                UserWarning,
                stacklevel=2,
            )

    def _damped_newton(
        self, x0: np.ndarray, c_in: np.ndarray, tol: float, max_iter: int
    ) -> np.ndarray:
        """阻尼牛顿。

        收敛判据：`|f| < tol · max(|f₀|, 尺度(当前浓度))`。
        取两者较大值是为了同时覆盖两种病态：
          · 只用物理尺度 —— 高活性电极在小槽压下即通过大电流，尺度极小，
            判据严到不可达
          · 只用初始残差 —— 热启动 / 入口浓度为零（原位生成）时 |f₀| 或
            浓度尺度本身极小，同样不可达
        """
        x = x0.copy()
        f = self.residual(x, c_in)
        f0 = max(float(np.abs(f).max()), 1e-300)
        for _ in range(max_iter):
            norm = float(np.linalg.norm(f, ord=np.inf))
            c_mag = max(float(np.abs(x).max()), 1e-30)
            thr = tol * max(f0, self._residual_scale_from(c_mag))
            if norm < thr:
                break
            J = self.jacobian(x, c_in).tocsc()
            try:
                dx = splinalg.spsolve(J, -f)
            except Exception:  # noqa: BLE001
                break
            if not np.all(np.isfinite(dx)):
                break
            alpha = 1.0
            f_new = None
            for _ in range(30):
                f_try = self.residual(x + alpha * dx, c_in)
                if np.linalg.norm(f_try, ord=np.inf) < norm:
                    f_new = f_try
                    break
                alpha *= 0.5
            x = x + alpha * dx
            f = f_new if f_new is not None else self.residual(x, c_in)
        return x

    def solve_transient(
        self,
        c_in: np.ndarray,
        c0: np.ndarray | None = None,
        t_end: float = 10.0,
        n_out: int = 20,
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """瞬态求解。返回 (t, C)，C 形状 (n_out, n_z, n_r, n_species)。"""
        from scipy.integrate import solve_ivp

        c_in = np.asarray(c_in, dtype=float)
        nz, nr, ns = self.n_z, self.n_r, self.network.n_species
        _, _, vol = self._weights()
        self._c_in_ref = c_in

        x0 = (
            np.tile(c_in, (nz, nr, 1)).ravel()
            if c0 is None
            else np.asarray(c0, dtype=float).ravel()
        )

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            res = self.residual(x, c_in).reshape(nz, nr, ns)
            return (-res / vol[None, :, None]).ravel()

        t_eval = np.linspace(0.0, t_end, n_out)
        sol = solve_ivp(rhs, (0.0, t_end), x0, method="BDF", t_eval=t_eval,
                        rtol=rtol, atol=atol)
        return sol.t, sol.y.T.reshape(len(sol.t), nz, nr, ns)

    # ── 评价 ───────────────────────────────────────────────
    def section_average(
        self, field: np.ndarray, weight: str = "flow"
    ) -> tuple[np.ndarray, np.ndarray]:
        """截面平均，返回 (z, 平均值[nz, ns])。

        与一维模型对比时必须用**流量加权**（``flow``）而非面积加权 ——
        抛物线剖面下两者差异显著。
        """
        uz, _ = self.velocity_fields()
        _, _, vol = self._weights()
        if weight == "area":
            w = np.broadcast_to(vol[None, :], (self.n_z, self.n_r))
        elif weight == "flow":
            w = np.abs(uz) * vol[None, :]
        else:
            raise ValueError("weight 应为 'area' 或 'flow'")
        w3 = w[:, :, None]
        avg = np.sum(field * w3, axis=1) / np.sum(w3, axis=1)
        return self._z.copy(), avg

    def molar_flow_out(self, profile: np.ndarray) -> np.ndarray:
        """出口摩尔流量 (mol/s)，按物质。"""
        uz, _ = self.velocity_fields()
        _, a_zface, _ = self._weights()
        return np.sum(uz[-1][:, None] * profile[-1] * a_zface[:, None], axis=0)

    def molar_flow_in(self, c_in: np.ndarray) -> np.ndarray:
        """入口摩尔流量 (mol/s)，按物质。"""
        uz, _ = self.velocity_fields()
        _, a_zface, _ = self._weights()
        return np.sum(uz[0][:, None] * np.asarray(c_in)[None, :] * a_zface[:, None], axis=0)
