"""流场基座：解析速度剖面 + 传质/弥散关联式。

设计判断（见 docs/COMSOL四基座自研替代可行性评估.md §2.3）：
本类反应器是规则几何（环隙 / 圆管 / 平板）且 Re 不高，
**不需要求解 Navier-Stokes**，用层流解析剖面 + 关联式即可，
把 CFD 的自由度从"场求解"降到"代数求值"（微秒级）。

同时给出轴向弥散系数（Taylor-Aris），这是把流场接到
``fastsim.chemistry.ADR1D`` 的 dispersion 参数的桥梁 ——
否则 ADR 里的 D 只能靠猜，这是 COMSOL 用户常见的隐性误差源。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AnnulusFlow",
    "PipeFlow",
    "taylor_aris_dispersion",
    "sherwood_graetz",
    "sherwood_dittus_boelter",
]

# 25 °C 水的物性（可覆盖）
RHO_W = 997.0        # kg/m³
MU_W = 8.90e-4       # Pa·s
NU_W = MU_W / RHO_W  # m²/s


@dataclass
class AnnulusFlow:
    """同心环隙层流（ECO 反应器的典型构型：管状阳极 + 外筒阴极）。

    u(r) = (1/4μ)(−dp/dz)[ R_o² − r² + (R_o² − R_i²)/ln(R_o/R_i) · ln(r/R_o) ]

    平均流速与压降的关系：
        u_mean = (1/8μ)(−dp/dz)[ R_o² + R_i² − (R_o² − R_i²)/ln(R_o/R_i) ]
    """

    r_inner: float          # m，内管外半径（阳极）
    r_outer: float          # m，外管内半径（阴极面）
    length: float           # m
    mu: float = MU_W        # Pa·s
    rho: float = RHO_W      # kg/m³

    @property
    def gap(self) -> float:
        return self.r_outer - self.r_inner

    @property
    def area(self) -> float:
        """流通截面积 (m²)。"""
        return np.pi * (self.r_outer**2 - self.r_inner**2)

    @property
    def hydraulic_diameter(self) -> float:
        """水力直径 D_h = 4A/P = 2(R_o − R_i)。"""
        return 2.0 * self.gap

    def mean_velocity(self, q_vol: float) -> float:
        """由体积流量 (m³/s) 得平均流速 (m/s)。"""
        return q_vol / self.area

    def pressure_gradient(self, u_mean: float) -> float:
        """由平均流速反推所需压降梯度 −dp/dz (Pa/m)。"""
        ri, ro = self.r_inner, self.r_outer
        geom = ro**2 + ri**2 - (ro**2 - ri**2) / np.log(ro / ri)
        return 8.0 * self.mu * u_mean / geom

    def profile(self, u_mean: float, n: int = 101) -> tuple[np.ndarray, np.ndarray]:
        """返回 (r, u(r))，用于检查速度分布的均匀性。"""
        ri, ro = self.r_inner, self.r_outer
        r = np.linspace(ri, ro, n)
        dpdz = self.pressure_gradient(u_mean)
        u = (dpdz / (4.0 * self.mu)) * (
            ro**2 - r**2 + (ro**2 - ri**2) / np.log(ro / ri) * np.log(r / ro)
        )
        return r, u

    def max_velocity(self, u_mean: float) -> float:
        """峰值流速（在 u'(r)=0 处）。"""
        r, u = self.profile(u_mean, n=2001)
        return float(u.max())

    def reynolds(self, u_mean: float) -> float:
        """Re = ρ u D_h / μ。"""
        return self.rho * u_mean * self.hydraulic_diameter / self.mu

    def regime(self, u_mean: float) -> str:
        """流态判据（环隙层流临界 Re 约 2000，取保守值）。"""
        re = self.reynolds(u_mean)
        if re < 2000.0:
            return "层流"
        if re < 4000.0:
            return "过渡"
        return "湍流"

    def residence_time(self, u_mean: float) -> float:
        return self.length / u_mean


@dataclass
class PipeFlow:
    """圆管层流（Hagen-Poiseuille）。u(r) = 2u_mean (1 − (r/R)²)。"""

    radius: float
    length: float
    mu: float = MU_W
    rho: float = RHO_W

    @property
    def area(self) -> float:
        return np.pi * self.radius**2

    @property
    def hydraulic_diameter(self) -> float:
        return 2.0 * self.radius

    def mean_velocity(self, q_vol: float) -> float:
        return q_vol / self.area

    def pressure_gradient(self, u_mean: float) -> float:
        return 8.0 * self.mu * u_mean / self.radius**2

    def profile(self, u_mean: float, n: int = 101) -> tuple[np.ndarray, np.ndarray]:
        r = np.linspace(0.0, self.radius, n)
        return r, 2.0 * u_mean * (1.0 - (r / self.radius) ** 2)

    def max_velocity(self, u_mean: float) -> float:
        return 2.0 * u_mean

    def reynolds(self, u_mean: float) -> float:
        return self.rho * u_mean * self.hydraulic_diameter / self.mu

    def residence_time(self, u_mean: float) -> float:
        return self.length / u_mean


def taylor_aris_number(length: float, u_mean: float, d_char: float, d_mol: float) -> float:
    """Taylor 弥散数的倒数（无量纲）：Ta = u·a²/(D·L)。

    物理含义：径向扩散时间 τ_rad = a²/D 与停留时间 τ_res = L/u 之比
        Ta = τ_rad / τ_res  （差一个 1/2 因子，量级一致）

    Taylor-Aris 公式成立要求 **Ta ≪ 1**（径向上浓度剖面先被扩散抹平，
    对流展宽才可由一维弥散模型描述）。Ta ≫ 1 时该公式给出非物理的大弥散系数。
    """
    return u_mean * d_char**2 / (d_mol * length)


TAYLOR_ARIS_TA_LIMIT = 0.1


def taylor_aris_dispersion(
    radius: float,
    u_mean: float,
    d_mol: float,
    length: float,
    shape: str = "pipe",
    strict: bool = True,
) -> float:
    """Taylor-Aris 轴向弥散系数 (m²/s)。

    层流管内：D_ax = D_mol + u²a²/(48 D_mol)

    参数
    ----
    length : float
        反应段长度 (m)。**必需** —— 用于判定 Taylor 区是否成立；
        缺了它就无法判断公式是否可用，而该公式在区外会给出非物理结果。
    strict : bool
        默认 True：Ta > 0.1 时**报错**而不是返回不可信的值。
        显式传 False 可强制取值（调用方须自行承担外推风险并注明）。

    ⚠ 为什么必须守卫：本函数在非 Taylor 区会放大到分子扩散的 10⁸ 倍量级
    （实测某环隙工况 u=8 mm/s、a=20 mm、D=1e-9 → D_ax=0.53 m²/s，Pe→0），
    这种数值一旦静默流入下游 ADR 求解器，会把近平推流的反应器算成全混流。
    """
    if d_mol <= 0.0:
        raise ValueError("d_mol 必须为正")
    if length <= 0.0:
        raise ValueError("length 必须为正")

    d_char = radius
    ta = taylor_aris_number(length, u_mean, d_char, d_mol)
    if strict and ta > TAYLOR_ARIS_TA_LIMIT:
        raise ValueError(
            f"不在 Taylor 弥散适用域：Ta = u·a²/(D·L) = {ta:,.1f}，"
            f"要求 ≪ 1（阈值 {TAYLOR_ARIS_TA_LIMIT}）。\n"
            "  含义：径向扩散时间远长于停留时间，浓度剖面来不及被抹平，\n"
            "  一维弥散模型不成立 —— 此时应直接求解 2D 对流-扩散，\n"
            "  或改用分子扩散系数 D_mol 作为下限并明确标注为降级处理。\n"
            f"  若确需强制取值，显式传 strict=False 并注明外推风险。"
        )
    return d_mol + (u_mean**2 * d_char**2) / (48.0 * d_mol)


def sherwood_graetz(re: float, sc: float, d: float, length: float) -> float:
    """层流入口段传质（Graetz/Leveque 解）：Sh = 1.62 (Re·Sc·d/L)^(1/3)。

    适用域：Re·Sc·d/L > 10（否则用充分发展值 Sh→3.66）。
    """
    gz = re * sc * d / length
    if gz <= 10.0:
        return 3.66
    return 1.62 * gz ** (1.0 / 3.0)


def sherwood_dittus_boelter(re: float, sc: float) -> float:
    """湍流充分发展传质：Sh = 0.023 Re^0.8 Sc^0.33。

    ⚠ 适用域：Re > 10⁴（Dittus-Boelter 标准区间），超出范围不应外推。
    """
    if re < 1.0e4:
        raise ValueError(
            f"Dittus-Boelter 仅适用于 Re>1e4（收到 Re={re:.3g}）；"
            "过渡区请改用层流入口段关联式或直接求解"
        )
    return 0.023 * re**0.8 * sc**0.33


def mass_transfer_coefficient(sh: float, d_mol: float, d_char: float) -> float:
    """由 Sherwood 数得传质系数 k_L (m/s)。"""
    return sh * d_mol / d_char
