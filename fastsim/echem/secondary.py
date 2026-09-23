"""电化学层：环隙反应器的电流分布（一次/二次 + 固相欧姆降）。

对应 COMSOL 电化学模块的「电流分布，一次/二次接口」，但聚焦环境
电化学反应器最常用的构型——**管状阳极 + 同心筒状阴极的环隙**
（与 ECO 反应器一致），并把三个工程维度显式参数化：

- **电极特性（动力学）**：`ElectrodeKinetics`（j₀ / αₐ / α_c / E_eq）
- **电极特性（固相导电）**：`sheet_conductance_*`（σ_eff·A_cross，S·m）
- **水质特性**：电导率 κ（随离子强度/温度变化，0.1 M Na₂SO₄ ≈ 1 S/m）

控制方程
--------
液相电荷守恒（一次 + 二次均成立）：

    ∇·(κ∇φ_l) = 0        （R_a ≤ r ≤ R_c，0 ≤ z ≤ L）

电极边界（二次分布，Butler-Volmer）：

    j = j₀·[exp(αₐ F η /RT) − exp(−α_c F η /RT)]，  η = φ_s(z) − φ_l − E_eq

固相（可选，`sheet_conductance` 给定时启用）：

    G·d²φ_s/dz² = j(z)·P        （P = 电极周长），  G = σ_eff·A_cross
    z=0 端：φ_s = V（集流体）；z=L 端：绝缘

**为什么需要固相方程**：纯环隙 + 理想导体电极时，z 方向严格均匀（几何对称、
无端部效应），电流分布恒为常数 —— 这不是"算得准"，而是"没有不均匀的来源"。
真实电极有固相欧姆降，导致馈电端电流密度最大、远端最小，这是放大设计的
核心风险（βL 判据）。

数值方案
--------
- 单元中心有限体积，轴对称面权重 2πr（与 ADR2D 同套几何口径）
- 电极面：half-cell 电导（一次）或 BV（二次），Newton 解析 Jacobian
- 固相：一维有限差分，与液相双向耦合（φ_s 影响 j，j 影响 φ_s）
- z 端面绝缘（非集流体端）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

__all__ = ["ElectrodeKinetics", "SecondaryCurrent2D", "CurrentSolution"]

F_CONST = 96485.33     # C/mol
R_GAS = 8.31446        # J/(mol·K)


@dataclass
class ElectrodeKinetics:
    """电极动力学参数（BV）。不同电极材料 = 一组不同参数。

    常见参考量级（25 °C 水处理体系，仅作量级参考，实测为准）：
    - DSA（Ti/RuIrOₓ）析氧阳极：j₀ ~ 1e-6–1e-4 A/m²（OER 慢动力学）
    - BDD：j₀ ~ 1e-5 A/m² 量级
    - 不锈钢阴极析氢：j₀ ~ 1e-4–1e-3 A/m²
    αₐ+α_c 不必为 1（非对称传递系数常见）。
    """

    j0: float                   # A/m²，交换电流密度
    alpha_a: float = 0.5        # 阳极传递系数
    alpha_c: float = 0.5        # 阴极传递系数
    e_eq: float = 0.0           # V，平衡电位（vs 同一参考）
    n_electrons: int = 1

    def current_density(self, overpotential: float, T: float) -> float:
        """BV 电流密度（A/m²，正值 = 阳极氧化流出电极）。"""
        f = self.n_electrons * F_CONST / (R_GAS * T)
        x = np.clip(f * overpotential, -60.0, 60.0)   # 防 exp 溢出
        return self.j0 * (np.exp(self.alpha_a * x) - np.exp(-self.alpha_c * x))

    def conductance_per_area(self, T: float) -> float:
        """线性化面电导 g = nF·j₀·(αₐ+α_c)/RT（小信号极限，A·m⁻²·V⁻¹）。"""
        f = self.n_electrons * F_CONST / (R_GAS * T)
        return self.j0 * f * (self.alpha_a + self.alpha_c)


@dataclass
class CurrentSolution:
    """电流分布求解结果。"""

    phi: np.ndarray                 # (nz, nr) 液相电位
    j_anode: np.ndarray             # (nz,) 阳极局部电流密度（A/m²，正 = 氧化）
    j_cathode: np.ndarray           # (nz,) 阴极局部电流密度（A/m²，负 = 还原）
    i_total: float                  # A，总电流
    v_cell: float                   # V，槽压
    z: np.ndarray                   # (nz,) 单元中心坐标
    r: np.ndarray                   # (nr,)
    phi_s_anode: np.ndarray | None = None    # (nz,) 阳极固相电位（理想导体时 None）
    phi_s_cathode: np.ndarray | None = None
    x: np.ndarray | None = None              # 完整未知量向量（供热启动复用）

    @property
    def j_anode_cv(self) -> float:
        """阳极电流分布变异系数（均匀性指标）。"""
        j = self.j_anode
        return float(np.std(j) / np.mean(j))

    @property
    def apparent_resistance(self) -> float:
        """表观槽阻 V/I（Ω，含欧姆 + 动力学 + 固相）。"""
        return self.v_cell / self.i_total if self.i_total != 0 else float("inf")

    @property
    def feed_end_ratio(self) -> float:
        """馈电端与远端电流密度之比（放大设计的核心风险指标）。

        理想导体电极为 1.0；固相电导越低，该比值越大。
        """
        return float(self.j_anode[0] / self.j_anode[-1]) if self.j_anode[-1] else float("inf")


@dataclass
class SecondaryCurrent2D:
    """环隙二次电流分布模型（管状阳极在内，筒状阴极在外）。

    参数
    ----
    r_inner, r_outer : float
        阳极外半径 / 阴极内半径 (m)。
    length : float
        电极长度 (m)。
    kappa : float
        电解液电导率 (S/m) —— **水质参数**（0.1 M Na₂SO₄ ≈ 1.0–1.1）。
    anode, cathode : ElectrodeKinetics | None
        电极动力学 —— **电极特性参数**。`None` = 一次分布（纯欧姆）。
    sheet_conductance_anode / sheet_conductance_cathode : float | None
        电极**固相轴向电导** G = σ_eff · A_cross (S·m) —— **电极特性参数**。
        `None` 表示理想导体（等电位），此时沿 z 严格均匀；
        给定有限值时启用固相电位 φ_s(z)，出现**轴向电流不均匀**
        （馈电端电流密度最大），对应 βL 判据。

    模型局限（诚实声明）
    -------------------
    - 电极边界等距、κ 均匀、无气泡遮挡（气泡改变 κ 的影响未建模）
    - 集流体仅在 z=0 端；双端馈电未实现
    - 无端部效应（电极覆盖整个 z 范围）
    """

    r_inner: float
    r_outer: float
    length: float
    kappa: float
    anode: ElectrodeKinetics | None = None
    cathode: ElectrodeKinetics | None = None
    sheet_conductance_anode: float | None = None
    sheet_conductance_cathode: float | None = None
    feed_end: str = "z0"
    n_z: int = 60
    n_r: int = 16
    temperature: float = 298.15

    _dz: float = field(default=0.0, init=False, repr=False)
    _dr: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.r_inner < self.r_outer):
            raise ValueError("需 0 < r_inner < r_outer")
        if self.length <= 0.0 or self.kappa <= 0.0:
            raise ValueError("length 与 kappa 必须为正")
        if self.n_z < 4 or self.n_r < 3:
            raise ValueError("n_z ≥ 4、n_r ≥ 3")
        if self.anode is None and self.cathode is None:
            self.mode = "primary"
        elif self.anode is not None and self.cathode is not None:
            self.mode = "secondary"
        else:
            raise ValueError("anode 与 cathode 要么都给（二次），要么都不给（一次）")
        for g in (self.sheet_conductance_anode, self.sheet_conductance_cathode):
            if g is not None and g <= 0.0:
                raise ValueError("固相电导必须为正")
        if self.feed_end != "z0":
            raise ValueError("当前仅支持 feed_end='z0'（集流体在 z=0 端）")
        self._dz = self.length / self.n_z
        self._dr = (self.r_outer - self.r_inner) / self.n_r

    # ── 几何与未知量布局 ───────────────────────────────────
    def _grid(self):
        z = (np.arange(self.n_z) + 0.5) * self._dz
        r = self.r_inner + (np.arange(self.n_r) + 0.5) * self._dr
        return z, r

    def _areas(self):
        """(A_rface[nr+1], A_zface[nr])：径向/轴向面面积（轴对称 2πr）。"""
        r_face = self.r_inner + np.arange(self.n_r + 1) * self._dr
        a_r = 2.0 * np.pi * r_face * self._dz
        _, r = self._grid()
        a_z = 2.0 * np.pi * r * self._dr
        return a_r, a_z

    @property
    def _has_sheet_a(self) -> bool:
        return self.sheet_conductance_anode is not None

    @property
    def _has_sheet_c(self) -> bool:
        return self.sheet_conductance_cathode is not None

    @property
    def _n_liquid(self) -> int:
        return self.n_z * self.n_r

    @property
    def _n_unknown(self) -> int:
        return (self._n_liquid
                + (self.n_z if self._has_sheet_a else 0)
                + (self.n_z if self._has_sheet_c else 0))

    def ohmic_resistance_analytic(self) -> float:
        """径向欧姆电阻解析值 R = ln(R_c/R_a)/(2πκL)（细长几何，忽略端部）。"""
        return np.log(self.r_outer / self.r_inner) / (
            2.0 * np.pi * self.kappa * self.length
        )

    # ── 电极面电流 ─────────────────────────────────────────
    def _electrode_face(self, kin, v_s, phi_cell, face: str, T: float):
        """电极面电流（A，流入溶液为正）与其对 (φ_l, φ_s) 的偏导。

        v_s : float | ndarray
            固相电位：理想导体时为标量，含固相欧姆降时为 (nz,) 数组。

        返回 (I[nz], dI_dphi_l[nz], dI_dphi_s[nz] | None)

        动力学作用在**电极表面**处的液相电位上，而未知量是首个单元**中心**
        的 φ_l，两者相差半个单元的欧姆降：

            φ_l,surf = φ_l,cell + (Δr/2)·j/κ      （j = 流入溶液的面电流密度）

        一次分布分支已经用半单元电导 g = κA/(Δr/2) 体现了这段电阻；
        二次分布分支若直接用 φ_l,cell，就丢掉它，引入**一阶**空间误差：
        实测 n_r=32 时电流偏高 3.1%，且随 n_r 以 O(Δr) 收敛，而非 O(Δr²)
        （见 bench/fastsim_radial_convergence.py 的实测与预测对照）。
        """
        a_r, _ = self._areas()
        area = a_r[0] if face == "anode" else a_r[-1]

        if self.mode == "primary":
            g = self.kappa * area / (0.5 * self._dr)
            return g * (v_s - phi_cell), np.full_like(phi_cell, -g), None

        f = kin.n_electrons * F_CONST / (R_GAS * T)
        half_ohm = 0.5 * self._dr / self.kappa      # m²·A⁻¹

        def bv(x_eta):
            xx = np.clip(f * x_eta, -60.0, 60.0)
            j = kin.j0 * (np.exp(kin.alpha_a * xx) - np.exp(-kin.alpha_c * xx))
            dj = kin.j0 * f * (kin.alpha_a * np.exp(kin.alpha_a * xx)
                               + kin.alpha_c * np.exp(-kin.alpha_c * xx))
            return j, dj

        # 隐式方程 η + half_ohm·j(η) = v_s − φ_l,cell − e_eq。
        # 左端对 η 严格单调（dj/dη > 0），从 η = rhs 出发牛顿单调收敛。
        rhs = np.asarray(v_s - phi_cell - kin.e_eq, dtype=float)
        eta = rhs.copy()
        for _ in range(6):
            j, dj = bv(eta)
            step = (eta + half_ohm * j - rhs) / (1.0 + half_ohm * dj)
            eta = eta - step
            if np.all(np.abs(step) <= 1e-14 * (1.0 + np.abs(eta))):
                break
        j, dj = bv(eta)
        # dη/dφ_s = +scale，dη/dφ_l = −scale（对隐式关系求导）
        scale = 1.0 / (1.0 + half_ohm * dj)
        I = j * area
        dI_dphi_l = -dj * area * scale
        dI_dphi_s = dj * area * scale
        return I, dI_dphi_l, dI_dphi_s

    # ── 装配 ───────────────────────────────────────────────
    def _assemble(self, x: np.ndarray, v_cell: float):
        """残差与稀疏 Jacobian。未知量 x = [φ_l, (φ_s_a), (φ_s_c)]。"""
        nz, nr, T = self.n_z, self.n_r, self.temperature
        nl = self._n_liquid
        phi = x[:nl].reshape(nz, nr)

        off_a = nl
        off_c = nl + (nz if self._has_sheet_a else 0)
        phi_s_a = x[off_a:off_a + nz] if self._has_sheet_a else v_cell
        phi_s_c = x[off_c:off_c + nz] if self._has_sheet_c else 0.0

        a_r, a_z = self._areas()
        res = np.zeros(self._n_unknown)
        rows, cols, vals = [], [], []

        def blk(rr, cc, vv):
            """追加一个 COO 块。vv 允许标量（自动广播）或与 rr 同形状的数组。"""
            rr = np.asarray(rr, dtype=int).ravel()
            cc = np.asarray(cc, dtype=int).ravel()
            if rr.size != cc.size:
                raise ValueError(f"行列索引长度不一致: {rr.size} vs {cc.size}")
            vv = np.asarray(vv, dtype=float)
            if vv.ndim == 0 or vv.size == 1:
                vv = np.full(rr.size, float(vv.ravel()[0]))
            else:
                vv = vv.ravel()
                if vv.size != rr.size:
                    raise ValueError(
                        f"值长度 {vv.size} 与索引长度 {rr.size} 不一致"
                    )
            rows.append(rr)
            cols.append(cc)
            vals.append(vv)

        # ── 液相内部面（径向 + 轴向）────────────────────────
        idx = np.arange(nl).reshape(nz, nr)
        g_r = self.kappa * a_r[1:-1] / self._dr            # (nr−1,)
        g_z = self.kappa * a_z / self._dz                  # (nr,)
        g_z_int = np.broadcast_to(g_z, (nz - 1, nr))

        res_l = np.zeros((nz, nr))
        res_l[:, 1:] += g_r[None, :] * (phi[:, :-1] - phi[:, 1:])
        res_l[:, :-1] += g_r[None, :] * (phi[:, 1:] - phi[:, :-1])
        res_l[1:, :] += g_z[None, :] * (phi[:-1, :] - phi[1:, :])
        res_l[:-1, :] += g_z[None, :] * (phi[1:, :] - phi[:-1, :])

        gr_b = np.broadcast_to(g_r, (nz, nr - 1))
        blk(idx[:, 1:], idx[:, :-1], gr_b)
        blk(idx[:, 1:], idx[:, 1:], -gr_b)
        blk(idx[:, :-1], idx[:, 1:], gr_b)
        blk(idx[:, :-1], idx[:, :-1], -gr_b)
        blk(idx[1:, :], idx[:-1, :], g_z_int)
        blk(idx[1:, :], idx[1:, :], -g_z_int)
        blk(idx[:-1, :], idx[1:, :], g_z_int)
        blk(idx[:-1, :], idx[:-1, :], -g_z_int)

        # ── 电极面（液相侧）────────────────────────────────
        I_a, dIa_l, dIa_s = self._electrode_face(
            self.anode, phi_s_a, phi[:, 0], "anode", T)
        I_c, dIc_l, dIc_s = self._electrode_face(
            self.cathode, phi_s_c, phi[:, -1], "cathode", T)
        res_l[:, 0] += I_a
        res_l[:, -1] += I_c
        blk(idx[:, 0], idx[:, 0], dIa_l)
        blk(idx[:, -1], idx[:, -1], dIc_l)
        # 仅当该电极的固相电位是未知量时才写耦合一列
        if self._has_sheet_a and dIa_s is not None:
            blk(idx[:, 0], off_a + np.arange(nz), dIa_s)
        if self._has_sheet_c and dIc_s is not None:
            blk(idx[:, -1], off_c + np.arange(nz), dIc_s)

        res[:nl] = res_l.ravel()

        # ── 固相方程（如启用）──────────────────────────────
        # 固相节点 i 的电流守恒（控制体长度 dz）：
        #   G·(φ_{i−1} − 2φ_i + φ_{i+1})/dz = I_i    （I_i = 流入溶液的电流）
        dz = self._dz
        if self._has_sheet_a:
            G = self.sheet_conductance_anode
            r_s = np.zeros(nz)
            if nz > 2:
                r_s[1:-1] = (
                    G * (x[off_a:off_a + nz - 2] - 2.0 * x[off_a + 1:off_a + nz - 1]
                         + x[off_a + 2:off_a + nz]) / dz
                    - I_a[1:-1]
                )
            r_s[0] = x[off_a] - v_cell                       # 集流体 Dirichlet
            r_s[-1] = G * (x[off_a + nz - 2] - x[off_a + nz - 1]) / dz - I_a[-1]
            res[off_a:off_a + nz] = r_s

            # Jacobian：固相自耦合
            kk = off_a + np.arange(nz)
            if nz > 2:
                blk(kk[1:-1], kk[:-2], G / dz)
                blk(kk[1:-1], kk[1:-1], -2.0 * G / dz - dIa_s[1:-1])
                blk(kk[1:-1], kk[2:], G / dz)
            blk(kk[0], kk[0], 1.0)
            blk(kk[-1], kk[-2], G / dz)
            blk(kk[-1], kk[-1], -G / dz - dIa_s[-1])
            # 固相 → 液相耦合
            blk(kk[1:-1], idx[1:-1, 0], -dIa_l[1:-1])
            blk(kk[-1], idx[-1, 0], -dIa_l[-1])

        if self._has_sheet_c:
            G = self.sheet_conductance_cathode
            r_s = np.zeros(nz)
            if nz > 2:
                r_s[1:-1] = (
                    G * (x[off_c:off_c + nz - 2] - 2.0 * x[off_c + 1:off_c + nz - 1]
                         + x[off_c + 2:off_c + nz]) / dz
                    - I_c[1:-1]
                )
            r_s[0] = x[off_c] - 0.0                          # 阴极集流体 0 V
            r_s[-1] = G * (x[off_c + nz - 2] - x[off_c + nz - 1]) / dz - I_c[-1]
            res[off_c:off_c + nz] = r_s

            kk = off_c + np.arange(nz)
            if nz > 2:
                blk(kk[1:-1], kk[:-2], G / dz)
                blk(kk[1:-1], kk[1:-1], -2.0 * G / dz - dIc_s[1:-1])
                blk(kk[1:-1], kk[2:], G / dz)
            blk(kk[0], kk[0], 1.0)
            blk(kk[-1], kk[-2], G / dz)
            blk(kk[-1], kk[-1], -G / dz - dIc_s[-1])
            blk(kk[1:-1], idx[1:-1, -1], -dIc_l[1:-1])
            blk(kk[-1], idx[-1, -1], -dIc_l[-1])

        J = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self._n_unknown, self._n_unknown),
        )
        return res, J, I_a, I_c

    # ── 求解 ───────────────────────────────────────────────
    def _initial_guess(self, v_cell: float) -> np.ndarray:
        nz, nr = self.n_z, self.n_r
        r = self.r_inner + (np.arange(nr) + 0.5) * self._dr
        frac = np.log(r / self.r_inner) / np.log(self.r_outer / self.r_inner)
        phi = v_cell * (1.0 - frac)[None, :] * np.ones(nz)[:, None]
        x = [phi.ravel()]
        if self._has_sheet_a:
            x.append(np.full(nz, v_cell))
        if self._has_sheet_c:
            x.append(np.zeros(nz))
        return np.concatenate(x)

    def _scale(self, v_cell: float) -> float:
        return max(self.kappa * abs(v_cell) * 2.0 * np.pi * self.r_inner, 1e-30)

    def solve(self, v_cell: float, tol: float = 1e-10,
              max_iter: int = 60,
              x_guess: np.ndarray | None = None) -> CurrentSolution:
        """给定槽压求解。v_cell = V_anode − V_cathode（阴极集流体取 0）。

        收敛判据取两个尺度的**较宽松者**：
            |f| < tol · max(|f₀|, κ·V·2πr_inner)
        单用哪一个都会失效：
        · 只用物理尺度 κ·V·2πr —— 高活性电极在很小槽压下即通过大电流，
          该尺度极小，判据严到不可达（实测残差 1e-13 仍被判未收敛）
        · 只用初始残差 |f₀| —— 热启动（x_guess 已接近解）时 |f₀| 本身就极小，
          同样把判据压到不可达
        """
        nz, nr = self.n_z, self.n_r
        nl = self._n_liquid
        x = (self._initial_guess(v_cell) if x_guess is None
             else np.array(x_guess, dtype=float, copy=True))
        thr = tol * self._scale(v_cell)
        f_ref = None

        for _ in range(max_iter):
            res, J, _, _ = self._assemble(x, v_cell)
            f_norm = float(np.abs(res).max())
            if f_ref is None:
                f_ref = max(f_norm, 1e-300)
                thr = tol * max(f_ref, self._scale(v_cell))
            if f_norm < thr:
                break
            try:
                dx = splinalg.spsolve(J.tocsc(), -res)
            except Exception:  # noqa: BLE001
                break
            if not np.all(np.isfinite(dx)):
                break
            alpha = 1.0
            for _ in range(30):
                cand = x + alpha * dx
                r2, _, _, _ = self._assemble(cand, v_cell)
                if np.abs(r2).max() < f_norm:
                    break
                alpha *= 0.5
            x = x + alpha * dx

        res, _, I_a, I_c = self._assemble(x, v_cell)
        f_final = float(np.abs(res).max())
        if not np.isfinite(f_final) or f_final > thr:
            raise RuntimeError(
                f"电流分布求解未收敛（|f_final|={f_final:.3e} > 阈值 {thr:.3e}，"
                f"初始残差 {f_ref:.3e}）"
            )

        phi = x[:nl].reshape(nz, nr)
        a_r, _ = self._areas()
        off_c = nl + (nz if self._has_sheet_a else 0)
        z, r = self._grid()
        return CurrentSolution(
            phi=phi,
            j_anode=I_a / a_r[0],
            j_cathode=I_c / a_r[-1],
            i_total=float(np.sum(I_a)),
            v_cell=v_cell,
            z=z, r=r,
            phi_s_anode=x[nl:off_c] if self._has_sheet_a else None,
            phi_s_cathode=x[off_c:off_c + nz] if self._has_sheet_c else None,
            x=x.copy(),
        )

    def solve_at_current(self, i_target: float, v_max: float = 20.0) -> CurrentSolution:
        """恒流模式：二分槽压使总电流匹配 i_target。

        区间搜索用「**欧姆电阻给出的真实下界**起步 + 倍增」：
        达到目标电流所需槽压必 ≥ i_target·R_ohm，从这里倍增不会跳到
        强极化溢出区。（早期版本用线性化 BV 估计电阻，对小 j₀ 电极会
        严重高估——实测某工况估出 3068 Ω，试探电压直接给到 1227 V 而发散。）
        """
        if i_target <= 0.0:
            raise ValueError("i_target 必须为正")

        v_lo = max(i_target * self.ohmic_resistance_analytic(), 1e-9)
        sol_lo = self.solve(v_lo)
        if sol_lo.i_total >= i_target:
            return sol_lo

        v_hi = v_lo
        sol_hi = sol_lo
        for _ in range(80):
            v_hi = v_hi * 2.0
            if v_hi > v_max:
                raise ValueError(
                    f"槽压上限 {v_max} V 下最大电流 {sol_hi.i_total:.4g} A "
                    f"< 目标 {i_target} A"
                )
            # 热启动：电压递增时上一步的解是很好的初值
            sol_hi = self.solve(v_hi, x_guess=sol_hi.x)
            if sol_hi.i_total >= i_target:
                break

        # 二分（同样全程热启动；25 次已给到 3e-8 相对精度，远超需要）
        x_cache = sol_lo.x
        for _ in range(25):
            mid = 0.5 * (v_lo + v_hi)
            sol = self.solve(mid, x_guess=x_cache)
            x_cache = sol.x
            if sol.i_total < i_target:
                v_lo = mid
            else:
                v_hi = mid
        return self.solve(v_hi, x_guess=x_cache)

    # ── 评价 ───────────────────────────────────────────────
    def mid_section_mask(self, frac: float = 0.25) -> np.ndarray:
        """中段索引掩码（避开馈电端与远端，与解析解对比用）。"""
        n = self.n_z
        lo, hi = int(frac * n), int((1.0 - frac) * n)
        return np.arange(lo, max(hi, lo + 1))

    def sheet_resistance(self, electrode: str) -> float | None:
        """电极固相总电阻 (Ω)：R = L/G。理想导体返回 None。"""
        G = (self.sheet_conductance_anode if electrode == "anode"
             else self.sheet_conductance_cathode)
        return self.length / G if G else None
