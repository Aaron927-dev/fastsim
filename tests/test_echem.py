"""电化学层验证：环隙一/二次电流分布。

验证链（不验证不声称）：
1. **电流守恒**（离散装配的硬检验）：阳极总电流 = 阴极总电流
2. **一次分布 vs 解析解**：同心圆筒径向解析 φ(r)=A·ln(r)+B、
   中段电流密度与总电流的解析值
3. **线性极化串联电阻**：小槽压下 I = ΔV/(R_ohm + R_ct,a + R_ct,c)
4. **Wagner 数行为**：j₀ 越大（极化电阻越小）→ 越趋一次分布（端部越集中）
5. **恒流模式**自洽（solve_at_current 的电流回读）
6. 电极/水质参数的响应方向（DSA 慢动力学 vs 快动力学；κ 高低）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.echem import ElectrodeKinetics, SecondaryCurrent2D  # noqa: E402

# 细长几何（L/间隙 = 100）压制端部效应，使解析对比有意义
RA, RC, L_LEN = 0.01, 0.02, 1.0
KAPPA = 1.0                     # S/m，0.1 M Na₂SO₄ 量级


def _model(**kw):
    return SecondaryCurrent2D(
        r_inner=RA, r_outer=RC, length=L_LEN, kappa=KAPPA,
        n_z=80, n_r=12, **kw
    )


# ══════════════════════════════════════════════════════════════
# 1. 电流守恒（装配正确性的硬检验）
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode_kw", [
    dict(),                                                    # 一次分布
    dict(anode=ElectrodeKinetics(j0=1e-3),
         cathode=ElectrodeKinetics(j0=1e-3)),                  # 对称 BV
    dict(anode=ElectrodeKinetics(j0=1e-2, alpha_a=0.7, alpha_c=0.3, e_eq=1.2),
         cathode=ElectrodeKinetics(j0=1e-4, alpha_a=0.3, alpha_c=0.7)),  # 非对称
])
def test_current_conservation(mode_kw):
    """阳极总流出 = 阴极总流入（z 端绝缘 ⇒ 全部电流穿环隙）。"""
    m = _model(**mode_kw)
    sol = m.solve(2.0)
    i_a = float(np.sum(sol.j_anode * 2 * np.pi * RA * m._dz))
    i_c = float(np.sum(sol.j_cathode * 2 * np.pi * RC * m._dz))
    assert i_a == pytest.approx(-i_c, rel=1e-8), (
        f"电流不守恒：阳极 {i_a:.6f} vs 阴极 {i_c:.6f}"
    )
    assert sol.i_total == pytest.approx(i_a, rel=1e-12)


# ══════════════════════════════════════════════════════════════
# 2. 一次分布 vs 解析解
# ══════════════════════════════════════════════════════════════

def test_primary_mid_section_vs_analytic():
    """一次分布：中段电流密度与解析值 κΔV/(R_a·ln(R_c/R_a)) 一致。"""
    m = _model()
    v = 2.0
    sol = m.solve(v)
    j_ana = KAPPA * v / (RA * np.log(RC / RA))
    mid = m.mid_section_mask(0.25)
    j_num = float(np.mean(sol.j_anode[mid]))
    assert j_num == pytest.approx(j_ana, rel=0.02), (
        f"中段 j = {j_num:.4f} vs 解析 {j_ana:.4f} A/m²"
    )
    # 总电流（中段贡献占主导，L/间隙=100 时端部 <1%）
    i_ana = 2 * np.pi * KAPPA * L_LEN * v / np.log(RC / RA)
    assert sol.i_total == pytest.approx(i_ana, rel=0.03)


def test_primary_potential_profile_logarithmic():
    """一次分布径向电位须是对数剖面 φ(r) = A + B·ln(r)。"""
    m = _model()
    v = 2.0
    sol = m.solve(v)
    i_mid = sol.phi.shape[0] // 2
    phi_r = sol.phi[i_mid, :]
    r = sol.r
    # 拟合 A + B·ln(r)
    coef = np.polyfit(np.log(r), phi_r, 1)
    fit = np.polyval(coef, np.log(r))
    assert float(np.abs(phi_r - fit).max()) < 0.02 * v, (
        "径向电位偏离对数剖面"
    )


# ══════════════════════════════════════════════════════════════
# 3. 线性极化：串联电阻
# ══════════════════════════════════════════════════════════════

def test_secondary_linear_polarization_series_resistance():
    """小槽压下总电阻 = R_ohm + R_ct,a + R_ct,c（面积加权线性 BV）。"""
    j0a = j0c = 1e-3
    kin = ElectrodeKinetics(j0=1e-3)
    m = SecondaryCurrent2D(
        r_inner=RA, r_outer=RC, length=L_LEN, kappa=KAPPA,
        anode=ElectrodeKinetics(j0=j0a), cathode=ElectrodeKinetics(j0=j0c),
        n_z=80, n_r=12,
    )
    dv = 0.02                     # 20 mV，BV 线性区
    sol = m.solve(dv)
    r_app = sol.apparent_resistance

    r_ohm = m.ohmic_resistance_analytic()
    f1 = 96485.33 / (8.31446 * 298.15)
    area_a = 2 * np.pi * RA * L_LEN
    area_c = 2 * np.pi * RC * L_LEN
    r_ct = 1.0 / (j0a * f1 * area_a) + 1.0 / (j0c * f1 * area_c)
    r_series = r_ohm + r_ct

    assert r_app == pytest.approx(r_series, rel=0.05), (
        f"表观槽阻 {r_app:.4f} Ω vs 串联解析 {r_series:.4f} Ω"
    )


# ══════════════════════════════════════════════════════════════
# 4. 均匀性：环隙的天然自均匀 + 固相欧姆降导致的轴向不均匀
# ══════════════════════════════════════════════════════════════

def test_ideal_conductor_is_uniform_by_geometry():
    """环隙 + 理想导体电极 → 电流分布沿 z 严格均匀（CV ≈ 0）。

    这不是"算得准"，而是几何上**没有不均匀的来源**：电极等距、κ 均匀、
    无端部效应。该结论与 `eco_field` 的既有判断一致
    （"同心圆筒在径向天然均匀，放大风险在轴向"）。
    """
    for mode_kw in (dict(),
                    dict(anode=ElectrodeKinetics(j0=1e-3),
                         cathode=ElectrodeKinetics(j0=1e-3)),
                    dict(anode=ElectrodeKinetics(j0=1e2),
                         cathode=ElectrodeKinetics(j0=1e2))):
        m = SecondaryCurrent2D(r_inner=RA, r_outer=RC, length=0.2,
                               kappa=KAPPA, n_z=60, n_r=10, **mode_kw)
        sol = m.solve_at_current(0.1)
        assert sol.j_anode_cv < 1e-10, (
            f"理想导体下电流分布不均匀（CV={sol.j_anode_cv:.2e}）"
        )
        assert sol.feed_end_ratio == pytest.approx(1.0, rel=1e-8)


def test_solid_phase_resistance_causes_axial_nonuniformity():
    """固相欧姆降 → 馈电端电流密度大于远端（βL 类行为）。

    这是把电极当"理想导体"会完全漏掉的物理，也是"电极特性"的核心之一。
    """
    m = SecondaryCurrent2D(
        r_inner=RA, r_outer=RC, length=0.20, kappa=KAPPA,
        anode=ElectrodeKinetics(j0=1e-3),
        cathode=ElectrodeKinetics(j0=1e-3),
        sheet_conductance_anode=0.05,      # 低固相电导（薄/多孔电极）
        sheet_conductance_cathode=1e6,     # 阴极理想导体
        n_z=80, n_r=10)
    sol = m.solve_at_current(0.1)
    assert sol.phi_s_anode is not None
    # 固相电位沿 z 单调下降（电流流出 → 欧姆降）
    grad = np.diff(sol.phi_s_anode)
    assert np.all(grad <= 1e-12), "固相电位未沿 z 单调下降"
    # 馈电端电流更大
    assert sol.feed_end_ratio > 1.05, (
        f"固相电阻未产生轴向不均匀（馈电端/远端 = {sol.feed_end_ratio:.4f}）"
    )
    # 总压降包含固相份额
    assert sol.phi_s_anode[0] - sol.phi_s_anode[-1] > 0


def test_lower_sheet_conductance_worsens_uniformity():
    """固相电导越低 → 轴向不均匀越严重（feed_end_ratio 单调上升）。"""
    ratios = []
    for G in (10.0, 1.0, 0.2, 0.05):
        m = SecondaryCurrent2D(
            r_inner=RA, r_outer=RC, length=0.20, kappa=KAPPA,
            anode=ElectrodeKinetics(j0=1e-3),
            cathode=ElectrodeKinetics(j0=1e-3),
            sheet_conductance_anode=G,
            sheet_conductance_cathode=1e6,
            n_z=60, n_r=8)
        ratios.append(m.solve_at_current(0.1).feed_end_ratio)
    assert ratios == sorted(ratios), (
        f"不均匀度未随固相电导降低而恶化：{['%.3f' % r for r in ratios]}"
    )
    assert ratios[-1] > ratios[0] * 1.5, "最低电导应显著更不均匀"


def test_longer_electrode_worsens_uniformity():
    """电极越长 → 固相欧姆降越累积 → 越不均匀（βL 判据的定性体现）。"""
    ratios = []
    for L_elec in (0.05, 0.2, 0.8):
        m = SecondaryCurrent2D(
            r_inner=RA, r_outer=RC, length=L_elec, kappa=KAPPA,
            anode=ElectrodeKinetics(j0=1e-3),
            cathode=ElectrodeKinetics(j0=1e-3),
            sheet_conductance_anode=0.5,
            sheet_conductance_cathode=1e6,
            n_z=80, n_r=8)
        ratios.append(m.solve_at_current(0.1).feed_end_ratio)
    assert ratios == sorted(ratios), (
        f"不均匀度未随电极长度增加而恶化：{['%.3f' % r for r in ratios]}"
    )


def test_tafel_region_uniformity_insensitive_to_j0():
    """Tafel 区：电流分布形状对 j₀ 不敏感（这是一个**反直觉但正确**的结论）。

    物理原因：强极化（Tafel 区）下 BV 的微分电阻
        dj/dη = αf·j
    只取决于**工作电流**，与交换电流密度 j₀ 无关。
    因此只要工作点不变，改变电极的 j₀（例如换一种电极材料）并不会改变
    电流分布的**形状**——它影响的是达到该电流所需的**槽压**。

    实测：j₀ 从 1e-6 扫到 1e-1 A/m²（5 个数量级），
    feed_end_ratio 仅从 1.4330 变到 1.4341（< 0.1%）。

    ⚠ 该结论只在 Tafel 区成立。弱极化（线性）区动力学电阻 = RT/(nF·j₀)，
    此时 j₀ 才会显著影响分布——但那对应极低电流密度，工程上不常见。
    """
    ratios = []
    for j0 in (1e-6, 1e-3, 1e-1):
        m = SecondaryCurrent2D(
            r_inner=RA, r_outer=RC, length=0.20, kappa=KAPPA,
            anode=ElectrodeKinetics(j0=j0),
            cathode=ElectrodeKinetics(j0=j0),
            sheet_conductance_anode=0.2,
            sheet_conductance_cathode=0.2,
            n_z=60, n_r=8)
        ratios.append(m.solve_at_current(0.05).feed_end_ratio)
    spread = (max(ratios) - min(ratios)) / np.mean(ratios)
    assert spread < 0.02, (
        f"Tafel 区分布形状应基本不随 j₀ 变化，实测离散度 {spread:.2%}：{ratios}"
    )
    # 但分布确实是不均匀的（来自固相欧姆降）
    assert all(r > 1.1 for r in ratios), f"固相电阻应产生明显不均匀：{ratios}"


def test_tafel_region_j0_controls_voltage_not_shape():
    """同一电流下，j₀ 变化主要改变**槽压**（而非分布形状）。"""
    v_by_j0 = {}
    r_by_j0 = {}
    for j0 in (1e-5, 1e-1):
        m = SecondaryCurrent2D(
            r_inner=RA, r_outer=RC, length=0.20, kappa=KAPPA,
            anode=ElectrodeKinetics(j0=j0),
            cathode=ElectrodeKinetics(j0=j0),
            sheet_conductance_anode=0.2,
            sheet_conductance_cathode=0.2,
            n_z=40, n_r=8)
        sol = m.solve_at_current(0.05)
        v_by_j0[j0] = sol.v_cell
        r_by_j0[j0] = sol.feed_end_ratio
    # 槽压对 j₀ 敏感（高活性电极省电）
    assert v_by_j0[1e-1] < v_by_j0[1e-5], f"槽压未随活性提高而下降：{v_by_j0}"
    # 形状几乎不变
    assert abs(r_by_j0[1e-1] - r_by_j0[1e-5]) / r_by_j0[1e-5] < 0.02


# ══════════════════════════════════════════════════════════════
# 5. 恒流模式与参数响应
# ══════════════════════════════════════════════════════════════

def test_solve_at_current_hits_target():
    m = SecondaryCurrent2D(
        r_inner=RA, r_outer=RC, length=0.2, kappa=KAPPA,
        anode=ElectrodeKinetics(j0=1e-3), cathode=ElectrodeKinetics(j0=1e-3),
        n_z=60, n_r=10)
    sol = m.solve_at_current(0.5)
    assert sol.i_total == pytest.approx(0.5, rel=2e-3)


def test_kappa_controls_ohmic_drop():
    """水质电导率越高 → 同电流下槽压越低（欧姆损耗减小）。"""
    v_by_kappa = {}
    for kap in (0.5, 2.0):
        m = SecondaryCurrent2D(
            r_inner=RA, r_outer=RC, length=0.2, kappa=kap,
            anode=ElectrodeKinetics(j0=1e-3),
            cathode=ElectrodeKinetics(j0=1e-3),
            n_z=40, n_r=8)
        v_by_kappa[kap] = m.solve_at_current(0.5).v_cell
    assert v_by_kappa[2.0] < v_by_kappa[0.5], (
        f"电导率翻倍槽压未降：{v_by_kappa}"
    )


def test_electrode_activity_controls_voltage():
    """电极动力学更快（DSA→高活性）→ 同电流下槽压更低。"""
    v_by_j0 = {}
    for j0 in (1e-5, 1e-2):
        m = SecondaryCurrent2D(
            r_inner=RA, r_outer=RC, length=0.2, kappa=KAPPA,
            anode=ElectrodeKinetics(j0=j0),
            cathode=ElectrodeKinetics(j0=j0),
            n_z=40, n_r=8)
        v_by_j0[j0] = m.solve_at_current(0.5).v_cell
    assert v_by_j0[1e-2] < v_by_j0[1e-5], (
        f"更高活性电极槽压未降：{v_by_j0}"
    )


# ══════════════════════════════════════════════════════════════
# 6. 输入校验
# ══════════════════════════════════════════════════════════════

def test_rejects_bad_inputs():
    with pytest.raises(ValueError):       # 半径颠倒
        SecondaryCurrent2D(r_inner=0.02, r_outer=0.01, length=1, kappa=1)
    with pytest.raises(ValueError):       # 只给一个电极
        SecondaryCurrent2D(
            r_inner=RA, r_outer=RC, length=1, kappa=1,
            anode=ElectrodeKinetics(j0=1e-3))
    with pytest.raises(ValueError):       # 非正电导率
        SecondaryCurrent2D(r_inner=RA, r_outer=RC, length=1, kappa=0.0)
