"""fastsim 化学场 / 流场内核的正确性验证。

验证原则（不验证不声称）：
1. 解析 Jacobian vs 有限差分 Jacobian —— 逐元素对比
2. ADR 稳态解 vs 轴向扩散模型解析解 —— 一级反应严格解
3. 两个极限：Pe→∞ 应退化为平推流；Pe→0 应退化为全混流
4. 流场解析剖面自洽性 —— 面积加权均速 = 输入均速；Poiseuille 极限
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.chemistry import ADR1D, ReactionNetwork, analytic_first_order_dispersion  # noqa: E402
from fastsim.chemistry.network import parse_equation  # noqa: E402
from fastsim.flow import (  # noqa: E402
    TAYLOR_ARIS_TA_LIMIT,
    AnnulusFlow,
    PipeFlow,
    sherwood_graetz,
    taylor_aris_dispersion,
    taylor_aris_number,
)

TOL_JAC = 1e-6


# ══════════════════════════════════════════════════════════════
# 1. 反应网络
# ══════════════════════════════════════════════════════════════

def test_parse_equation_basic():
    r, p = parse_equation("2 A + B -> C")
    assert r == {"A": 2.0, "B": 1.0}
    assert p == {"C": 1.0}
    # 省略系数与 Unicode 箭头
    r2, p2 = parse_equation("A → B + 2 C")
    assert r2 == {"A": 1.0}
    assert p2 == {"B": 1.0, "C": 2.0}


def test_mass_action_rates_single_first_order():
    net = ReactionNetwork()
    net.add("A -> B", k=2.0)
    c = net.initial(A=3.0, B=1.0)
    d = net(0.0, c)
    # dA/dt = -k A = -6 ; dB/dt = +6
    assert d[net.species.index("A")] == pytest.approx(-6.0)
    assert d[net.species.index("B")] == pytest.approx(6.0)


def test_mass_action_rates_second_order():
    net = ReactionNetwork()
    net.add("2 A -> B", k=0.5)
    c = net.initial(A=4.0, B=0.0)
    # r = k A² = 0.5*16 = 8 ; dA/dt = -2r = -16 ; dB/dt = +8
    d = net(0.0, c)
    assert d[net.species.index("A")] == pytest.approx(-16.0)
    assert d[net.species.index("B")] == pytest.approx(8.0)


def test_analytic_jacobian_matches_finite_difference():
    """解析 Jacobian 逐元素对有限差分 —— 这是刚性求解器能收敛的前提。"""
    net = ReactionNetwork()
    net.add("A + B -> C", k=1.3)
    net.add("C -> A", k=0.07)
    net.add("2 A -> D", k=0.9)
    net.add("B + C -> E", k=2.1)

    rng = np.random.default_rng(0)
    for _ in range(5):
        c = rng.uniform(0.1, 5.0, net.n_species)

        def f(x):
            return net(0.0, x)

        J_an = net.jacobian(0.0, c)
        J_fd = np.zeros_like(J_an)
        h = 1e-7
        for m in range(net.n_species):
            e = np.zeros(net.n_species)
            e[m] = h
            J_fd[:, m] = (f(c + e) - f(c - e)) / (2.0 * h)

        err = np.abs(J_an - J_fd).max() / max(np.abs(J_fd).max(), 1e-30)
        assert err < TOL_JAC, f"Jacobian 相对误差 {err:.3e} 超限"


# ══════════════════════════════════════════════════════════════
# 2. ADR 稳态 vs 解析解
# ══════════════════════════════════════════════════════════════

def _single_first_order_net(k: float) -> ReactionNetwork:
    net = ReactionNetwork()
    net.add("A -> B", k=k)
    return net


@pytest.mark.parametrize("D", [1e-4, 5e-5, 2e-5])
def test_adr_steady_vs_analytic(D):
    """ADR 稳态解 vs 轴向扩散模型解析解（Danckwerts 边界）。"""
    L, u, k, c_in = 0.20, 0.01, 0.05, 1.0
    net = _single_first_order_net(k)
    m = ADR1D(network=net, length=L, velocity=u, dispersion=D, n_cells=400)
    prof = m.solve_steady(np.array([c_in, 0.0]))

    ia = net.species.index("A")
    c_num = prof[:, ia]
    c_ana = analytic_first_order_dispersion(m.z, L, u, D, k, c_in)

    err = np.abs(c_num - c_ana).max() / c_in
    # 一阶迎风在 400 单元下的离散误差；<1% 即为通过
    assert err < 0.01, f"D={D}: 与解析解最大偏差 {err:.2%}"


def test_adr_pfr_limit():
    """Pe→∞（D→0）时退化为平推流：C_out = C_in·exp(−kτ)。"""
    L, u, k, c_in = 0.20, 0.01, 0.05, 1.0
    tau = L / u
    net = _single_first_order_net(k)
    m = ADR1D(network=net, length=L, velocity=u, dispersion=1e-12, n_cells=800)
    prof = m.solve_steady(np.array([c_in, 0.0]))
    c_out = prof[-1, net.species.index("A")]
    assert c_out == pytest.approx(c_in * np.exp(-k * tau), rel=2e-3)


def test_adr_cstr_limit():
    """Pe→0（强返混）时逼近全混流：C_out = C_in/(1+kτ)。"""
    L, u, k, c_in = 0.20, 0.01, 0.05, 1.0
    tau = L / u
    D = u * L / 0.02                      # Pe = 0.02，强返混
    net = _single_first_order_net(k)
    m = ADR1D(network=net, length=L, velocity=u, dispersion=D, n_cells=400)
    prof = m.solve_steady(np.array([c_in, 0.0]))
    c_out = prof[-1, net.species.index("A")]
    c_cstr = c_in / (1.0 + k * tau)
    assert c_out == pytest.approx(c_cstr, rel=0.05)


def test_adr_conserves_mass_without_reaction():
    """无反应时应守恒：积分质量流量出入口相等。"""
    L, u, c_in = 0.20, 0.01, 1.0
    net = ReactionNetwork()
    net.add("A -> B", k=0.0)               # 零速率 = 无反应，仅登记物质
    m = ADR1D(network=net, length=L, velocity=u, dispersion=1e-6, n_cells=200)
    prof = m.solve_steady(np.array([c_in, 0.0]))
    ia = net.species.index("A")
    ib = net.species.index("B")
    assert prof[-1, ia] == pytest.approx(c_in, rel=1e-6)
    assert prof[-1, ib] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(prof[:, ia] + prof[:, ib], c_in, rtol=1e-6)


def test_adr_serial_reaction_conversion_matches_analytic():
    """串联反应 A→B→C：A 的衰减仍是一级，应与解析解一致（检验多物质耦合）。"""
    L, u, k1, c_in = 0.20, 0.02, 0.10, 1.0
    net = ReactionNetwork()
    net.add("A -> B", k=k1)
    net.add("B -> C", k=0.5)
    m = ADR1D(network=net, length=L, velocity=u, dispersion=1e-6, n_cells=600)
    prof = m.solve_steady(np.array([c_in, 0.0, 0.0]))

    tau = L / u
    cA_ana = c_in * np.exp(-k1 * tau)
    assert prof[-1, net.species.index("A")] == pytest.approx(cA_ana, rel=3e-3)
    # B 的解析解：C_in k1/(k2-k1) (e^{-k1 τ} - e^{-k2 τ})
    k2 = 0.5
    cB_ana = c_in * k1 / (k2 - k1) * (np.exp(-k1 * tau) - np.exp(-k2 * tau))
    assert prof[-1, net.species.index("B")] == pytest.approx(cB_ana, rel=1e-2)
    # 总质量守恒（无生成/消耗的总量）
    total = prof[-1].sum()
    assert total == pytest.approx(c_in, rel=1e-6)


def test_adr_transient_approaches_steady_state():
    """瞬态积分到足够长时间应收敛到稳态解。"""
    L, u, D, k, c_in = 0.10, 0.01, 2e-5, 0.05, 1.0
    net = _single_first_order_net(k)
    m = ADR1D(network=net, length=L, velocity=u, dispersion=D, n_cells=80)
    c_in_vec = np.array([c_in, 0.0])
    steady = m.solve_steady(c_in_vec)

    t_end = 8.0 * m.residence_time()
    t, C = m.solve_transient(c_in_vec, t_end=t_end, n_out=8)
    final = C[-1]
    err = np.abs(final[:, 0] - steady[:, 0]).max()
    assert err < 5e-4, f"瞬态未收敛到稳态，偏差 {err:.2e}"


def test_adr_steady_residual_is_zero():
    """稳态解代回残差应接近机器精度（自洽性检查）。"""
    L, u, D, k, c_in = 0.20, 0.01, 5e-5, 0.05, 1.0
    net = _single_first_order_net(k)
    m = ADR1D(network=net, length=L, velocity=u, dispersion=D, n_cells=200)
    c_in_vec = np.array([c_in, 0.0])
    prof = m.solve_steady(c_in_vec)
    res = m.residual(prof.ravel(), c_in_vec)
    assert np.abs(res).max() < 1e-8


@pytest.mark.parametrize("D", [1e-4, 5e-5])
def test_adr_dirichlet_inlet_vs_analytic(D):
    """Dirichlet 入口（COMSOL Inflow 默认口径）对解析解。"""
    L, u, k, c_in = 0.20, 0.01, 0.05, 1.0
    net = _single_first_order_net(k)
    m = ADR1D(network=net, length=L, velocity=u, dispersion=D,
              n_cells=400, inlet_bc="dirichlet")
    prof = m.solve_steady(np.array([c_in, 0.0]))
    c_num = prof[:, net.species.index("A")]
    c_ana = analytic_first_order_dispersion(m.z, L, u, D, k, c_in,
                                           inlet_bc="dirichlet")
    err = np.abs(c_num - c_ana).max() / c_in
    assert err < 0.01, f"D={D}: Dirichlet 入口与解析解偏差 {err:.2%}"


def test_two_inlet_bcs_differ_and_converge_at_high_pe():
    """两种入口口径在低 Pe 下应有显著差异，高 Pe 下应收敛到同一结果。

    这是对标时必须声明边界条件口径的原因。
    """
    L, u, k, c_in = 0.20, 0.01, 0.05, 1.0
    net = _single_first_order_net(k)

    # 低 Pe（强返混）：两种口径差异显著
    D_low = u * L / 2.0
    a_dan = analytic_first_order_dispersion(
        np.array([L]), L, u, D_low, k, c_in, inlet_bc="danckwerts")
    a_dir = analytic_first_order_dispersion(
        np.array([L]), L, u, D_low, k, c_in, inlet_bc="dirichlet")
    rel_gap = abs(a_dan[0] - a_dir[0]) / a_dir[0]
    assert rel_gap > 0.05, f"低 Pe 下两口径差异仅 {rel_gap:.1%}，应显著不同"

    # 低 Pe 下数值解须分别贴合各自的解析解
    for bc, ref in (("danckwerts", a_dan[0]), ("dirichlet", a_dir[0])):
        m = ADR1D(network=net, length=L, velocity=u, dispersion=D_low,
                  n_cells=800, inlet_bc=bc)
        prof = m.solve_steady(np.array([c_in, 0.0]))
        got = prof[-1, net.species.index("A")]
        assert got == pytest.approx(ref, rel=0.02), f"{bc}: {got} vs {ref}"

    # 高 Pe（近平推流）：两口径收敛到同一结果 exp(-kτ)
    D_high = 1e-9
    tau = L / u
    for bc in ("danckwerts", "dirichlet"):
        m = ADR1D(network=net, length=L, velocity=u, dispersion=D_high,
                  n_cells=800, inlet_bc=bc)
        prof = m.solve_steady(np.array([c_in, 0.0]))
        got = prof[-1, net.species.index("A")]
        assert got == pytest.approx(c_in * np.exp(-k * tau), rel=2e-3), bc


def test_adr_rejects_unknown_inlet_bc():
    net = _single_first_order_net(0.05)
    with pytest.raises(ValueError):
        ADR1D(network=net, length=0.2, velocity=0.01, inlet_bc="neumann")


@pytest.mark.parametrize("D", [1e-8, 1e-10, 1e-12, 1e-14])
def test_analytic_solution_stable_at_high_pe(D):
    """大 Pe 下解析解必须有限（不得溢出/NaN），且收敛到平推流结果。

    背景：朴素形式 C(ζ)=A·e^{m1ζ}+B·e^{m2ζ} 在 Pe=1.6e6 时 e^{m1} 溢出，
    实测直接返回 nan，导致"与解析解对标"这一步失效。
    """
    L, u, k, c_in = 0.20, 0.01, 0.05, 1.0
    z = np.linspace(0.0, L, 50)
    expect = c_in * np.exp(-k * L / u)
    for bc in ("danckwerts", "dirichlet"):
        c = analytic_first_order_dispersion(z, L, u, D, k, c_in, inlet_bc=bc)
        assert np.all(np.isfinite(c)), f"D={D} {bc}: 出现非有限值"
        assert c[-1] == pytest.approx(expect, rel=1e-3), f"D={D} {bc}: 未退化到平推流"


def test_analytic_solution_matches_at_moderate_pe():
    """中等 Pe 下两种入口口径的差异应被正确表征（交叉校验稳定形式与原式）。"""
    L, u, k, c_in, D = 0.20, 0.01, 0.05, 1.0, 5e-5
    z = np.linspace(0.0, L, 30)
    c_dan = analytic_first_order_dispersion(z, L, u, D, k, c_in, inlet_bc="danckwerts")
    c_dir = analytic_first_order_dispersion(z, L, u, D, k, c_in, inlet_bc="dirichlet")
    assert np.all(np.isfinite(c_dan)) and np.all(np.isfinite(c_dir))
    # 两种口径必须不同（低 Pe 下），且都落在物理区间 (0, c_in]
    assert np.abs(c_dan - c_dir).max() > 1e-3
    for c in (c_dan, c_dir):
        assert np.all(c > 0.0) and np.all(c <= c_in + 1e-12)
    # 出口值应与数值解一致（在容差内）
    net = _single_first_order_net(k)
    for bc, ref in (("danckwerts", c_dan[-1]), ("dirichlet", c_dir[-1])):
        m = ADR1D(network=net, length=L, velocity=u, dispersion=D,
                  n_cells=800, inlet_bc=bc)
        prof = m.solve_steady(np.array([c_in, 0.0]))
        assert prof[-1, 0] == pytest.approx(ref, rel=0.02), bc


# ══════════════════════════════════════════════════════════════
# 3. 流场
# ══════════════════════════════════════════════════════════════

def test_pipe_profile_area_average_equals_mean():
    """抛物线剖面的面积加权平均必须等于输入均速。"""
    pf = PipeFlow(radius=0.005, length=0.2)
    u_mean = 0.01
    r, u = pf.profile(u_mean, n=4001)
    a = np.trapezoid(u * 2.0 * np.pi * r, r) / pf.area
    assert a == pytest.approx(u_mean, rel=1e-3)
    assert pf.max_velocity(u_mean) == pytest.approx(2.0 * u_mean)


def test_annulus_profile_area_average_equals_mean():
    """环隙层流剖面的面积加权平均必须等于输入均速（这是解析式正确性的硬检验）。"""
    af = AnnulusFlow(r_inner=0.01, r_outer=0.03, length=0.20)
    u_mean = 0.02
    r, u = af.profile(u_mean, n=20001)
    a = np.trapezoid(u * 2.0 * np.pi * r, r) / af.area
    assert a == pytest.approx(u_mean, rel=2e-3), f"面积平均 {a} vs 目标 {u_mean}"


def test_annulus_small_gap_reduces_to_parallel_plate():
    """小间隙极限下环隙解须退化为平行板 Poiseuille：−dp/dz = 12μ·u/h²。

    （不用 r_inner→0 检验：ln(r_o/r_i) 增长极慢，数值上无法真正到达圆管极限，
    用解析可严格证明的小间隙极限才是有效检验。）
    """
    mu = 8.90e-4
    h = 1.0e-3
    r_mid = 0.02
    af = AnnulusFlow(
        r_inner=r_mid - h / 2.0, r_outer=r_mid + h / 2.0, length=0.2, mu=mu
    )
    u_mean = 0.01
    expect = 12.0 * mu * u_mean / h**2          # 平行板解析值
    assert af.pressure_gradient(u_mean) == pytest.approx(expect, rel=5e-3)


def test_annulus_approaches_poiseuille_monotonically():
    """内径减小时压降梯度应单调趋近圆管值，且始终不低于它。"""
    mu = 8.90e-4
    r_out = 0.005
    u_mean = 0.01
    poiseuille = PipeFlow(radius=r_out, length=0.2, mu=mu).pressure_gradient(u_mean)
    vals = [
        AnnulusFlow(r_inner=ri, r_outer=r_out, length=0.2, mu=mu).pressure_gradient(u_mean)
        for ri in (0.002, 0.001, 1e-4)
    ]
    assert all(v >= poiseuille for v in vals), "环隙压降不应低于同外径圆管"
    assert vals == sorted(vals, reverse=True), "内径减小时压降应单调下降趋近圆管值"


def test_annulus_hydraulic_diameter():
    af = AnnulusFlow(r_inner=0.01, r_outer=0.03, length=0.2)
    assert af.hydraulic_diameter == pytest.approx(0.04)
    assert af.gap == pytest.approx(0.02)


def test_annulus_regime_and_residence_time():
    af = AnnulusFlow(r_inner=0.01, r_outer=0.03, length=0.20)
    u = af.mean_velocity(2.0e-5)          # 20 mL/s = 2e-5 m³/s
    assert u > 0
    assert af.regime(u) in {"层流", "过渡", "湍流"}
    assert af.residence_time(u) == pytest.approx(0.20 / u)


def test_taylor_aris_exceeds_molecular():
    """Taylor 弥散必须大于分子扩散（对流展宽是净增项）。"""
    # 取细管 + 低流速，使 Ta = u·a²/(D·L) = 0.025 ≪ 1，落在适用域内
    d_mol = 1.0e-9
    a, u, length = 5.0e-5, 1.0e-3, 0.1
    d_ax = taylor_aris_dispersion(radius=a, u_mean=u, d_mol=d_mol, length=length)
    assert d_ax > d_mol
    expect = d_mol + (u**2 * a**2) / (48.0 * d_mol)
    assert d_ax == pytest.approx(expect, rel=1e-12)
    assert taylor_aris_number(length, u, a, d_mol) < 0.1


def test_taylor_aris_guards_applicability_domain():
    """非 Taylor 区必须报错，而不是返回非物理的大弥散系数。

    这个守卫来自一次真实事故：环隙工况（u=8 mm/s、a=20 mm、D=1e-9）
    曾算出 D_ax = 0.53 m²/s（分子扩散的 5×10⁸ 倍），Pe→0，
    会把近平推流的反应器算成全混流。
    """
    from fastsim.flow import taylor_aris_dispersion as tad

    # 环隙工况：Ta = 0.008·0.02²/(1e-9·0.2) ≈ 1.6e4 ≫ 1
    with pytest.raises(ValueError, match="不在 Taylor 弥散适用域"):
        tad(radius=0.02, u_mean=0.00796, d_mol=1e-9, length=0.2)

    # 显式关闭守卫时才允许取值（外推风险由调用方承担）
    val = tad(radius=0.02, u_mean=0.00796, d_mol=1e-9, length=0.2, strict=False)
    assert val > 1e-2, "非 Taylor 区该公式的确会给出很大的值——这正是必须守卫的原因"


def test_taylor_aris_rejects_nonphysical():
    with pytest.raises(ValueError):
        taylor_aris_dispersion(radius=0.005, u_mean=0.01, d_mol=0.0, length=0.2)
    with pytest.raises(ValueError):
        taylor_aris_dispersion(radius=0.005, u_mean=0.01, d_mol=1e-9, length=0.0)


def test_sherwood_graetz_falls_back_to_fully_developed():
    """低 Graetz 数（Re·Sc·d/L ≤ 10）应回落到充分发展值 3.66，而不是继续外推。"""
    # Re·Sc·d/L = 1·10·0.01/1.0 = 0.1 ≤ 10 → 应取 3.66
    assert sherwood_graetz(re=1.0, sc=10.0, d=0.01, length=1.0) == pytest.approx(3.66)
    # 边界附近（gz=10）仍是 3.66；超过后按 1/3 次幂增长
    assert sherwood_graetz(re=10.0, sc=100.0, d=0.01, length=1.0) == pytest.approx(3.66)
    assert sherwood_graetz(re=100.0, sc=1000.0, d=0.01, length=1.0) > 3.66


def test_sherwood_dittus_boelter_guards_range():
    """Dittus-Boelter 超出适用域必须报错，禁止静默外推。"""
    assert sherwood_graetz(re=100.0, sc=1000.0, d=0.04, length=0.2) > 3.66
    from fastsim.flow import sherwood_dittus_boelter as sdb

    with pytest.raises(ValueError):
        sdb(re=5.0e3, sc=1000.0)          # 低于 Re=1e4 的适用下限
    assert sdb(re=2.0e4, sc=1000.0) > 0
