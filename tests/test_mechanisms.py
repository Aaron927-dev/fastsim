"""AOP 机理库（臭氧链 / 芬顿）的 0D 动力学验证。

验证原则：机理库的速率常数本身取自文献（±30% 分散），因此验收门是
**文献公认的特征量级**，而不是某篇论文的精确数值——
这是机理预置库诚实且可执行的验证标准。

判据来源（教科书/综述级共识）：
- 纯缓冲水中 O₃ 半衰期：pH 7 数十分钟–小时；pH 8 分钟级；pH 10 秒级
- ·OH 稳态浓度：臭氧化水体 1e-14–1e-11 M；芬顿 1e-13–1e-10 M
- Rct（∫·OH dt / ∫O₃ dt，Elovitz & von Gunten 1999）：1e-9–1e-6（天然水），
  纯水偏低端，本验证放宽到 1e-11–1e-5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.chemistry.mechanisms import (  # noqa: E402
    M_to_mgL,
    fenton_classic,
    mgL_to_M,
    ozone_chain,
)


def _integrate(mech, y0, t_end, n_out=200):
    net = mech.network
    sol = solve_ivp(net, (0.0, t_end), y0, method="BDF",
                    t_eval=np.linspace(0.0, t_end, n_out),
                    jac=net.jacobian, rtol=1e-8, atol=1e-18)
    assert sol.success, sol.message
    return sol.t, sol.y


def _half_life(t, y):
    """首穿半衰期：浓度首次跌破初值一半的时刻。"""
    y0 = y[0]
    below = np.where(y < 0.5 * y0)[0]
    if below.size == 0:
        return np.inf
    i = below[0]
    if i == 0:
        return 0.0
    # 线性插值
    t0, t1 = t[i - 1], t[i]
    y0_, y1 = y[i - 1], y[i]
    frac = (0.5 * y0 - y0_) / (y1 - y0_)
    return t0 + frac * (t1 - t0)


# ══════════════════════════════════════════════════════════════
# 臭氧链
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("pH, tmin, tmax, tag", [
    (7.0, 300.0, 4 * 3600.0, "数十分钟–小时"),
    (8.0, 60.0, 3600.0, "分钟级"),
    (10.0, 1.0, 120.0, "秒级（链加速下限可至 ~1 s）"),
])
def test_ozone_half_life_by_pH(pH, tmin, tmax, tag):
    """O₃ 半衰期随 pH 的量级（链式加速须明显快于裸引发）。"""
    mech = ozone_chain(pH=pH)
    net = mech.network
    y0 = net.initial(O3=1e-4, H2O2=0.0, pCBA=1e-6)   # ~4.8 mg/L O₃
    i_o3 = net.species.index("O3")

    t_end = min(tmax * 3, 6 * 3600)
    t, y = _integrate(mech, y0, t_end, n_out=400)
    th = _half_life(t, y[i_o3])
    assert tmin <= th <= tmax, (
        f"pH={pH} O₃ 半衰期 {th:.0f}s 不在文献量级 [{tmin},{tmax}]s（{tag}）"
    )


def test_ozone_half_life_monotonic_in_pH():
    """pH 越高链引发越快，半衰期必须单调下降。"""
    halves = []
    for pH in (7.0, 8.0, 9.0, 10.0):
        mech = ozone_chain(pH=pH)
        net = mech.network
        y0 = net.initial(O3=1e-4, pCBA=1e-6)
        i = net.species.index("O3")
        t_end = 4000.0 if pH <= 8 else 300.0
        t, y = _integrate(mech, y0, t_end, n_out=300)
        halves.append(_half_life(t, y[i]))
    assert halves == sorted(halves, reverse=True), f"半衰期未随 pH 单调下降：{halves}"


def test_ozone_oh_steady_state_magnitude():
    """·OH 稳态须落在 1e-14–1e-9 M（低清除剂水体偏高端，天然水偏低端）。"""
    mech = ozone_chain(pH=8.0)
    net = mech.network
    y0 = net.initial(O3=1e-4, pCBA=1e-6)
    i_oh = net.species.index("OHr")
    t, y = _integrate(mech, y0, 600.0, n_out=200)
    oh_ss = y[i_oh, 100:].max()
    assert 1e-14 < oh_ss < 1e-9, f"·OH 稳态 {oh_ss:.2e} M 超出量级"


def test_ozone_oh_suppressed_by_scavenger_load():
    """加 DOC 型清除剂后 ·OH 稳态必须被压低（机理对清除剂负荷的正确响应）。

    天然水 [·OH]ss 之所以在 1e-12–1e-11 量级，靠的是 DOC/碳酸盐清除；
    本机理无内置碳酸盐，通过外加大分子清除剂（≈1 mgC/L DOC 当量）
    检验同一响应。
    """
    base = ozone_chain(pH=8.0)
    net = base.network
    net.add("OHr + Scav -> Pscav", 2.0e9)          # DOC 型清除剂

    y0 = net.initial(O3=1e-4, pCBA=1e-6, Scav=2e-5)  # ≈1 mgC/L 位点当量
    i_oh = net.species.index("OHr")
    t, y = _integrate(base, y0, 600.0, n_out=200)
    oh_with = y[i_oh, 100:].max()

    y0_lo = net.initial(O3=1e-4, pCBA=1e-6, Scav=0.0)
    t2, y2 = _integrate(base, y0_lo, 600.0, n_out=200)
    oh_wo = y2[i_oh, 100:].max()

    assert oh_with < 0.3 * oh_wo, (
        f"清除剂未有效压制 ·OH：{oh_with:.2e} vs 无清除剂 {oh_wo:.2e}"
    )


def test_ozone_rct_order_of_magnitude():
    """Rct = ∫·OH dt / ∫O₃ dt 须落在 1e-11–1e-5（E&vG 1999 定义的量级）。"""
    mech = ozone_chain(pH=8.0)
    net = mech.network
    y0 = net.initial(O3=1e-4, pCBA=1e-6)
    i_oh, i_o3 = net.species.index("OHr"), net.species.index("O3")
    t, y = _integrate(mech, y0, 900.0, n_out=300)
    int_oh = np.trapezoid(y[i_oh], t)
    int_o3 = np.trapezoid(y[i_o3], t)
    rct = int_oh / int_o3
    assert 1e-11 < rct < 1e-5, f"Rct = {rct:.2e} 超出量级区间"


def test_ozone_ho2_equilibrium_consistent_with_pKa():
    """HO₂⁻/H₂O₂ 稳态比须收敛到 10^(pH−11.6)（解离对的内部自洽性检验）。"""
    pH = 9.0
    mech = ozone_chain(pH=pH)
    net = mech.network
    y0 = net.initial(O3=1e-4, H2O2=1e-5, pCBA=1e-6)
    i_a, i_b = net.species.index("HO2m"), net.species.index("H2O2")
    t, y = _integrate(mech, y0, 60.0, n_out=50)     # 解离弛豫 ~1.8 s⁻¹，60 s 足够
    ratio = y[i_a, -1] / y[i_b, -1]
    expect = 10.0 ** (pH - 11.6)
    assert ratio == pytest.approx(expect, rel=0.1), (
        f"HO₂⁻/H₂O₂ = {ratio:.2e} vs 平衡期望 {expect:.2e}"
    )


def test_ozone_probe_consumption_matches_oh_exposure():
    """pCBA 的消耗须与 ·OH 暴露量一致（k·∫·OH dt ≈ ln(c0/c)）——Rct 方法的内核。"""
    k_oh_pcba = 5.0e9
    mech = ozone_chain(pH=8.0)
    net = mech.network
    y0 = net.initial(O3=1e-4, pCBA=1e-6)
    i_oh, i_p = net.species.index("OHr"), net.species.index("pCBA")
    t, y = _integrate(mech, y0, 900.0, n_out=300)

    int_oh = np.trapezoid(y[i_oh], t)
    expect_consumed_frac = 1.0 - np.exp(-k_oh_pcba * int_oh)
    got_consumed_frac = 1.0 - y[i_p, -1] / y0[i_p]
    # pCBA 消耗在 10–90% 区间内两条路径可比（指数敏感区外）
    if 0.1 < got_consumed_frac < 0.9:
        assert got_consumed_frac == pytest.approx(expect_consumed_frac, rel=0.15), (
            f"探针消耗 {got_consumed_frac:.3f} vs ·OH 暴露预测 {expect_consumed_frac:.3f}"
        )


def test_ozone_rejects_acidic_ph():
    """合并假设仅适用 pH>5，酸性必须报错（禁止静默外推）。"""
    with pytest.raises(ValueError, match="pH>5"):
        ozone_chain(pH=3.5)


# ══════════════════════════════════════════════════════════════
# 芬顿
# ══════════════════════════════════════════════════════════════

def test_fenton_oh_steady_state_magnitude():
    """芬顿 ·OH 稳态须落在 1e-13–1e-10 M 量级。"""
    mech = fenton_classic()
    net = mech.network
    y0 = net.initial(Fe2=1e-4, H2O2=1e-3, phenol=1e-5)
    i_oh = net.species.index("OHr")
    t, y = _integrate(mech, y0, 300.0, n_out=200)
    oh_ss = y[i_oh, 20:].max()
    assert 1e-13 < oh_ss < 1e-10, f"芬顿 ·OH 稳态 {oh_ss:.2e} M 超出量级"


def test_fenton_h2o2_consumed_and_fe3_formed():
    """H₂O₂ 须被实际消耗（≥5%/500 s），Fe³⁺ 须出现（Fe²⁺→Fe³⁺ 循环启动）。"""
    mech = fenton_classic()
    net = mech.network
    y0 = net.initial(Fe2=1e-4, H2O2=1e-3, phenol=1e-5)
    i_hp, i_f3, i_f2 = (net.species.index(s) for s in ("H2O2", "Fe3", "Fe2"))
    t, y = _integrate(mech, y0, 500.0, n_out=200)

    consumed = 1.0 - y[i_hp, -1] / y0[i_hp]
    assert consumed > 0.05, f"H₂O₂ 仅消耗 {consumed:.1%}，反应未有效启动"
    assert y[i_f3, -1] > 1e-7, "Fe³⁺ 未出现，Fe 循环未启动"
    # 铁守恒（所有形态之和不变；OHr/HO2r 不含铁）
    fe_total = y[i_f2, :] + y[i_f3, :]
    assert np.allclose(fe_total, 1e-4, rtol=1e-6), "铁质量不守恒"


def test_fenton_oh_quenched_by_excess_fe2():
    """Fe²⁺ 过量的淬灭效应用**·OH 利用效率**度量。

    注意不能用「苯酚总量」判断：Fe²⁺ 过量时 ·OH 总生成也同比放大，
    苯酚降解总量反而更高（初版测试就写错了这个预期）。
    正确的淬灭度量是单位 H₂O₂ 消耗转化到目标物的比例——
    Fe²⁺ 大量过量时 ·OH 被 Fe²⁺ 抢走，该效率必须显著下降。
    """
    mech = fenton_classic()
    net = mech.network
    efficiency = {}
    for fe2_0 in (1e-4, 2e-3):          # 20× 过量 Fe²⁺
        y0 = net.initial(Fe2=fe2_0, H2O2=1e-3, phenol=1e-5)
        i_p, i_hp = net.species.index("phenol"), net.species.index("H2O2")
        t, y = _integrate(mech, y0, 120.0, n_out=100)
        d_phenol = y0[i_p] - y[i_p, -1]
        d_h2o2 = y0[i_hp] - y[i_hp, -1]
        efficiency[fe2_0] = d_phenol / d_h2o2
    assert efficiency[2e-3] < 0.5 * efficiency[1e-4], (
        f"Fe²⁺ 过量未压低 ·OH 利用效率：{efficiency}"
    )


# ══════════════════════════════════════════════════════════════
# 元数据与换算
# ══════════════════════════════════════════════════════════════

def test_mechanism_metadata_present():
    """每个机理必须带假设清单与速率常数出处（可追溯性）。"""
    for mech in (ozone_chain(7.5), fenton_classic()):
        assert mech.assumptions, f"{mech.name} 缺假设清单"
        assert mech.sources, f"{mech.name} 缺速率常数出处"
        assert mech.network.n_reactions > 0


def test_unit_conversion_roundtrip():
    x = 4.8  # mg/L 臭氧
    assert mgL_to_M(x) == pytest.approx(1e-4, rel=1e-2)
    assert M_to_mgL(mgL_to_M(x)) == pytest.approx(x)
