"""电-化耦合验证：法拉第通量 + 环隙几何 + 闭环一致性。

验证链：
1. 法拉第定律的量级与线性（flux ∝ j，反比于 n）
2. 电流效率的线性响应与边界校验
3. 环隙 ADR2D：内壁通量的质量守恒（入口 + 壁面生成 = 出口）
4. 环隙 ADR2D 退化：r_inner → 0 应回到实心管结果
5. **闭环**：电流分布 → 通量 → 浓度场，端到端量级自洽
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.chemistry import ADR2D, ReactionNetwork  # noqa: E402
from fastsim.couple import F_CONST, ElectrochemicalFlux, faradaic_flux  # noqa: E402
from fastsim.echem import ElectrodeKinetics, SecondaryCurrent2D  # noqa: E402

# 环隙几何（与 SecondaryCurrent2D 对齐）
RA, RC, L_LEN, KAPPA = 0.01, 0.02, 0.20, 1.0


# ══════════════════════════════════════════════════════════════
# 1–2. 法拉第定律
# ══════════════════════════════════════════════════════════════

def test_faradaic_flux_magnitude():
    """通量 = j·CE/(n·F)：用 1 A/m²、n=6（O₃）、CE=1 手算核对。"""
    flux = faradaic_flux(1.0, n_electrons=6, current_efficiency=1.0)
    expect = 1.0 / (6.0 * F_CONST)
    assert flux == pytest.approx(expect, rel=1e-12)
    # 量级感受：1 A/m² 满效率生成 O₃ ≈ 1.7e-6 mol·m⁻²·s⁻¹
    assert 1e-6 < expect < 1e-5


def test_faradaic_flux_linear_in_j_and_inverse_in_n():
    j = np.array([0.1, 1.0, 10.0])
    f = faradaic_flux(j, n_electrons=2)
    assert np.allclose(f / j, f[0] / j[0]), "通量应与 j 成正比"
    assert faradaic_flux(1.0, 4) == pytest.approx(faradaic_flux(1.0, 2) / 2.0)


def test_faradaic_flux_rejects_bad_efficiency():
    with pytest.raises(ValueError):
        faradaic_flux(1.0, 2, current_efficiency=0.0)
    with pytest.raises(ValueError):
        faradaic_flux(1.0, 2, current_efficiency=1.5)
    with pytest.raises(ValueError):
        faradaic_flux(1.0, 0)


# ══════════════════════════════════════════════════════════════
# 3. 环隙几何的质量守恒
# ══════════════════════════════════════════════════════════════

def _annulus_model(flux_inner=None, u=0.01, ns_species="A->A"):
    net = ReactionNetwork()
    net.add(ns_species, k=0.0)
    return ADR2D(
        network=net, length=L_LEN, radius=RC, r_inner=RA,
        velocity_z=u, velocity_r=0.0, dispersion=1e-9,
        n_z=60, n_r=12, mode="axisymmetric", inlet_bc="danckwerts",
        wall_flux_inner=flux_inner,
    )


def test_annulus_no_flux_mass_conservation():
    """环隙 + 无通量 + 无反应：出入口摩尔流量相等。"""
    m = _annulus_model()
    c_in = np.array([1.0])
    prof = m.solve_steady(c_in)
    fin = m.molar_flow_in(c_in)[0]
    fout = m.molar_flow_out(prof)[0]
    assert fout == pytest.approx(fin, rel=1e-10)
    # 且浓度处处等于入口
    assert np.allclose(prof[:, :, 0], 1.0, rtol=1e-10)


def test_annulus_inner_wall_flux_conserved():
    """内壁恒定通量：入口 + 壁面生成 = 出口（守恒性硬检验）。"""
    j_flux = 2.0e-6                     # mol·m⁻²·s⁻¹，流入流体域
    m = _annulus_model(flux_inner=lambda z: np.full_like(z, j_flux))
    c_in = np.array([1.0])
    prof = m.solve_steady(c_in)

    fin = m.molar_flow_in(c_in)[0]
    fout = m.molar_flow_out(prof)[0]

    # 壁面总面积（内壁，轴对称）：2π·r_inner·L
    area = 2.0 * np.pi * RA * L_LEN
    generated = j_flux * area
    assert (fout - fin) == pytest.approx(generated, rel=1e-6), (
        f"入口 {fin:.6e} + 壁面 {generated:.6e} ≠ 出口 {fout:.6e}"
    )


def test_annulus_inner_flux_raises_concentration():
    """内壁生成通量应提升出口浓度（方向性检验）。"""
    m0 = _annulus_model()
    p0 = m0.solve_steady(np.array([1.0]))[-1, :, 0].mean()

    m1 = _annulus_model(flux_inner=lambda z: np.full_like(z, 2.0e-6))
    p1 = m1.solve_steady(np.array([1.0]))[-1, :, 0].mean()
    assert p1 > p0, f"壁面生成未提升浓度：{p1:.6f} vs {p0:.6f}"


def test_annulus_rejects_flux_when_no_inner_wall():
    """r_inner = 0（实心管）时给内壁通量应被忽略（无内壁面）。"""
    net = ReactionNetwork()
    net.add("A -> A", k=0.0)
    m = ADR2D(network=net, length=L_LEN, radius=RC, r_inner=0.0,
              velocity_z=0.01, dispersion=1e-9, n_z=30, n_r=8,
              mode="axisymmetric",
              wall_flux_inner=lambda z: np.full_like(z, 1e-5))
    c_in = np.array([1.0])
    prof = m.solve_steady(c_in)
    # 中心线不是实体壁面 → 无通量注入 → 浓度保持
    assert np.allclose(prof[:, :, 0], 1.0, rtol=1e-10), "r=0 处不应有壁面通量"


def test_annulus_rejects_bad_geometry():
    net = ReactionNetwork()
    net.add("A -> A", k=0.0)
    with pytest.raises(ValueError):
        ADR2D(network=net, length=0.2, radius=0.01, r_inner=0.02)  # 内径 > 外径
    with pytest.raises(ValueError):
        ADR2D(network=net, length=0.2, radius=0.01, r_inner=-0.001)


# ══════════════════════════════════════════════════════════════
# 4. 电流分布 → 通量（ElectrochemicalFlux）
# ══════════════════════════════════════════════════════════════

def _current_solution():
    m = SecondaryCurrent2D(
        r_inner=RA, r_outer=RC, length=L_LEN, kappa=KAPPA,
        anode=ElectrodeKinetics(j0=1e-3),
        cathode=ElectrodeKinetics(j0=1e-3),
        n_z=40, n_r=10)
    return m.solve_at_current(0.1)


def test_electrochemical_flux_builds_per_species():
    """按物质构造通量：生成取正、消耗取负。"""
    net = ReactionNetwork()
    net.add("O3 + A -> P", k=1.0)
    sol = _current_solution()

    ecf = ElectrochemicalFlux(
        current_solution=sol,
        species_map={"O3": (6, 0.05)},          # 阳极生成 O₃
        consumed_map={"A": (2, 0.02)},          # 阳极同时氧化 A
        face_radius=RA,
    )
    idx = {s: i for i, s in enumerate(net.species)}
    flux = ecf.build(idx)
    assert flux.shape == (sol.j_anode.size, net.n_species)
    assert np.all(flux[:, idx["O3"]] > 0), "O₃ 应为正通量（生成）"
    assert np.all(flux[:, idx["A"]] < 0), "A 应为负通量（消耗）"
    assert np.allclose(flux[:, idx["P"]], 0.0), "P 不参与界面反应"


def test_electrochemical_flux_scales_with_efficiency():
    """电流效率线性缩放通量。"""
    net = ReactionNetwork()
    net.add("O3 -> O3", k=0.0)
    sol = _current_solution()
    idx = {s: i for i, s in enumerate(net.species)}

    f_lo = ElectrochemicalFlux(sol, {"O3": (6, 0.02)}, face_radius=RA).build(idx)
    f_hi = ElectrochemicalFlux(sol, {"O3": (6, 0.10)}, face_radius=RA).build(idx)
    ratio = f_hi[:, idx["O3"]] / f_lo[:, idx["O3"]]
    assert np.allclose(ratio, 5.0, rtol=1e-10)


def test_electrochemical_flux_rejects_unknown_species():
    net = ReactionNetwork()
    net.add("O3 -> O3", k=0.0)
    sol = _current_solution()
    ecf = ElectrochemicalFlux(sol, {"NOPE": (1, 0.5)}, face_radius=RA)
    with pytest.raises(KeyError, match="NOPE"):
        ecf.build({s: i for i, s in enumerate(net.species)})


def test_electrochemical_flux_total_rates_requires_radius():
    net = ReactionNetwork()
    net.add("O3 -> O3", k=0.0)
    sol = _current_solution()
    ecf = ElectrochemicalFlux(sol, {"O3": (6, 0.05)})   # 未给 face_radius
    with pytest.raises(ValueError, match="face_radius"):
        ecf.total_rates({s: i for i, s in enumerate(net.species)})


# ══════════════════════════════════════════════════════════════
# 5. 闭环一致性
# ══════════════════════════════════════════════════════════════

def test_closed_loop_ec_to_chemistry():
    """端到端闭环：电场 → 界面通量 → 浓度场，量级自洽。

    检验的是"物理量守恒链条"而非精度：
    阳极总生成速率（mol/s，由总电流按法拉第定律算）
    必须等于化学场中该物质的净累积速率（mol/s，由出入口 + 壁面通量算）。
    """
    sol = _current_solution()
    ce, n_e = 0.08, 6                                # O₃：8% 电流效率、6 电子

    net = ReactionNetwork()
    net.add("O3 -> O3", k=0.0)                       # 仅输运，不反应
    idx = {s: i for i, s in enumerate(net.species)}

    ecf = ElectrochemicalFlux(sol, {"O3": (n_e, ce)}, face_radius=RA)
    wall = ecf.build(idx)

    m = ADR2D(
        network=net, length=L_LEN, radius=RC, r_inner=RA,
        velocity_z=0.01, velocity_r=0.0, dispersion=1e-9,
        n_z=sol.j_anode.size, n_r=12, mode="axisymmetric",
        inlet_bc="danckwerts",
        wall_flux_inner=lambda z: wall,
    )
    # 入口不含 O₃（由电极原位生成）
    c_in = np.array([0.0])
    prof = m.solve_steady(c_in)

    # 化学场：出口摩尔流量 = 壁面注入总量
    fout = m.molar_flow_out(prof)[0]
    area = 2.0 * np.pi * RA * L_LEN
    wall_total = float(np.mean(wall[:, idx["O3"]])) * area

    assert fout == pytest.approx(wall_total, rel=1e-5), (
        f"出口 {fout:.6e} mol/s ≠ 壁面注入 {wall_total:.6e} mol/s"
    )

    # 电场侧：总电流按法拉第定律换算，应与化学场一致
    i_total = sol.i_total
    from_current = i_total * ce / (n_e * F_CONST)
    assert from_current == pytest.approx(wall_total, rel=1e-5), (
        f"电流换算 {from_current:.6e} ≠ 壁面注入 {wall_total:.6e} mol/s"
    )


def test_closed_loop_responds_to_current():
    """闭环响应：电流增大 → O₃ 生成增多 → 出口浓度升高（单调）。"""
    c_outs = []
    for i_target in (0.05, 0.15):
        sol = SecondaryCurrent2D(
            r_inner=RA, r_outer=RC, length=L_LEN, kappa=KAPPA,
            anode=ElectrodeKinetics(j0=1e-3),
            cathode=ElectrodeKinetics(j0=1e-3),
            n_z=30, n_r=8).solve_at_current(i_target)

        net = ReactionNetwork()
        net.add("O3 -> O3", k=0.0)
        idx = {s: i for i, s in enumerate(net.species)}
        wall = ElectrochemicalFlux(
            sol, {"O3": (6, 0.08)}, face_radius=RA).build(idx)

        m = ADR2D(network=net, length=L_LEN, radius=RC, r_inner=RA,
                  velocity_z=0.01, dispersion=1e-9, n_z=sol.j_anode.size,
                  n_r=8, mode="axisymmetric",
                  wall_flux_inner=lambda z, w=wall: w)
        prof = m.solve_steady(np.array([0.0]))
        c_outs.append(float(m.molar_flow_out(prof)[0]))

    assert c_outs[1] > c_outs[0] * 1.2, (
        f"电流提高未显著增加 O₃ 产量：{c_outs}"
    )
