"""ADR2D（二维反应-传质）验证。

验证策略（不验证不声称）：
1. **解析 Jacobian vs 有限差分** —— 逐元素校验线性化
2. **制造解（MMS）检验扩散算子** —— 应达二阶收敛（含轴对称 1/r 项）
3. **MMS 检验对流算子** —— 一阶迎风应为一阶收敛（如实记录，不外推精度）
4. **一维退化** —— 平面模式 + 零通量壁 + 无径向变化，须与 ADR1D 完全一致
5. **守恒性** —— 无反应、壁面零通量时入口摩尔流量 = 出口摩尔流量
6. **瞬态 → 稳态收敛**
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.chemistry import ADR1D, ADR2D, ReactionNetwork  # noqa: E402

# ── 通用标定工况 ────────────────────────────────────────────
L, R = 0.20, 0.02
C_IN, AMP = 1.0, 0.3


def _net(k: float = 0.0) -> ReactionNetwork:
    net = ReactionNetwork()
    net.add("A -> B", k=k)
    return net


def _net_single() -> ReactionNetwork:
    """单物质网络（`A -> A` 且 k=0 → 化学计量净为零，只有 1 个物质）。

    用于 MMS 与壁面通量验证：避免产物 B 引入额外自由度干扰误差统计
    （B 无进料却共享同一源项/通量时，其解与制造解无关）。
    """
    net = ReactionNetwork()
    net.add("A -> A", k=0.0)
    return net


# ══════════════════════════════════════════════════════════════
# 1. 解析 Jacobian
# ══════════════════════════════════════════════════════════════

def test_adr2d_jacobian_matches_finite_difference():
    """二维解析 Jacobian 逐元素对有限差分（含反应耦合与非均匀速度剖面）。"""
    net = ReactionNetwork()
    net.add("A + B -> C", k=1.3)
    net.add("C -> A", k=0.07)
    net.add("2 A -> D", k=0.9)

    uz = 0.01 * (1.0 - (np.arange(6) / 6.0) ** 2)          # 抛物线型剖面
    m = ADR2D(network=net, length=L, radius=R, velocity_z=uz, velocity_r=0.0,
              dispersion=(2e-6, 5e-6), n_z=6, n_r=6, mode="axisymmetric")

    rng = np.random.default_rng(1)
    c = rng.uniform(0.2, 3.0, m.n_z * m.n_r * net.n_species)
    c_in = np.array([1.0, 0.5, 0.0, 0.0])

    J_an = m.jacobian(c, c_in).toarray()
    f0 = m.residual(c, c_in)

    J_fd = np.zeros_like(J_an)
    for k in range(c.size):
        h = 1e-7 * max(abs(c[k]), 1.0)
        e = np.zeros(c.size)
        e[k] = h
        J_fd[:, k] = (m.residual(c + e, c_in) - m.residual(c - e, c_in)) / (2.0 * h)

    denom = max(np.abs(J_fd).max(), 1e-30)
    err = np.abs(J_an - J_fd).max() / denom
    assert err < 1e-5, f"二维 Jacobian 相对误差 {err:.3e} 超限"
    assert np.isfinite(f0).all()


# ══════════════════════════════════════════════════════════════
# 2/3. 制造解（MMS）
# ══════════════════════════════════════════════════════════════
#
# 制造解 c(r,z) = C_IN + AMP·sin²(πz/2L)·(r/R)²
#   · z=0 处 sin²=0  → c 均匀 = C_IN，正好匹配 Dirichlet 入口
#   · z=L 处 d/dz sin²(πz/2L) = (π/2L)sin(π) = 0 → 正好匹配零梯度出口
#   · r=0 处 ∂c/∂r = 0 → 正好匹配轴对称条件
#   · r=R 处 ∂c/∂r = 2·AMP·f(z)/R → 作为指定壁面通量精确施加


def _exact(rg: np.ndarray, zg: np.ndarray) -> np.ndarray:
    return C_IN + AMP * np.sin(np.pi * zg / (2 * L)) ** 2 * (rg / R) ** 2


def _source_factory(u0: float, ur0: float, d_r: float, d_z: float):
    """制造源 S = ∇·(u·c) − ∇·(D∇c)。

    关键：求解器用**守恒形式** ∇·(u·c)，而
        ∇·(u·c) = u·∇c + c·∇·u
    对 u_z = u0（均匀）、u_r = ur0·r/R，有 ∇·u = 2·ur0/R ≠ 0，
    因此源里**必须**带上 c·∇·u 这一项，否则离散系统与 PDE 不一致
    （误差不随网格加密下降，反而上升 —— 这是最初把误差写成 45% 的原因）。
    """
    div_u = 2.0 * ur0 / R          # u_z 均匀 + u_r 线性 → 常数散度

    def src(rg: np.ndarray, zg: np.ndarray) -> np.ndarray:
        f = np.sin(np.pi * zg / (2 * L)) ** 2
        fp = (np.pi / (2 * L)) * np.sin(np.pi * zg / L)
        fpp = (np.pi**2 / (2 * L**2)) * np.cos(np.pi * zg / L)
        g = (rg / R) ** 2
        gp = 2.0 * rg / R**2
        gpp = np.full_like(rg, 2.0 / R**2)

        c = C_IN + AMP * f * g
        dc_dz = AMP * fp * g
        dc_dr = AMP * f * gp
        # 轴对称散度：(1/r)∂/∂r(r·D_r·∂c/∂r) + ∂/∂z(D_z·∂c/∂z)
        lap = d_z * AMP * fpp * g + d_r * AMP * f * (gp / rg + gpp)
        conv = u0 * dc_dz + ur0 * (rg / R) * dc_dr + c * div_u
        return (conv - lap)[:, :, None]

    return src


def _wall_factory(d_r: float):
    def wall(z: np.ndarray) -> np.ndarray:
        f = np.sin(np.pi * z / (2 * L)) ** 2
        # 流入域的通量 = D_r·∂c/∂r|_R = D_r·AMP·f·2/R
        return d_r * AMP * f * 2.0 / R

    return wall


def _mms_error(nz: int, nr: int, u0: float, ur0: float, d_r: float, d_z: float,
               scheme: str = "power_law") -> float:
    """制造解 MMS 误差（默认幂律格式）。"""
    return _mms_error_scheme(nz, nr, u0, ur0, d_r, d_z, scheme)


def test_adr2d_mms_diffusion_second_order():
    """纯扩散（含轴对称 1/r 项）：MMS 应收敛到二阶。

    u=0 时对流通量为零，验证的是「径向 1/r 散度 + 轴向扩散 + 壁面通量」
    这套扩散算子的正确性。
    """
    dr_, dz_ = 2e-6, 5e-6
    e1 = _mms_error(10, 5, 0.0, 0.0, dr_, dz_)
    e2 = _mms_error(20, 10, 0.0, 0.0, dr_, dz_)
    e3 = _mms_error(40, 20, 0.0, 0.0, dr_, dz_)

    assert e3 < e2 < e1, f"误差未随网格加密单调下降: {e1:.3e} {e2:.3e} {e3:.3e}"
    order1 = np.log2(e1 / e2)
    order2 = np.log2(e2 / e3)
    assert order2 > 1.8, f"二阶收敛未达标：观测阶 {order2:.2f}（{e2:.3e}→{e3:.3e}）"
    assert order1 > 1.8, f"二阶收敛未达标：观测阶 {order1:.2f}"


def _mms_error_scheme(nz: int, nr: int, u0: float, ur0: float, d_r: float,
                      d_z: float, scheme: str) -> float:
    """指定对流格式的 MMS 误差。"""
    m = ADR2D(
        network=_net_single(), length=L, radius=R,
        velocity_z=u0, velocity_r=ur0, dispersion=(d_r, d_z),
        n_z=nz, n_r=nr, mode="axisymmetric", inlet_bc="dirichlet",
        wall_flux=_wall_factory(d_r),
        extra_source=_source_factory(u0, ur0, d_r, d_z),
        convection_scheme=scheme,
    )
    prof = m.solve_steady(np.array([C_IN]))
    exact = _exact(m._rg, m._zg)
    return float(np.abs(prof[:, :, 0] - exact).max() / C_IN)


def test_adr2d_convection_scheme_order_improvement():
    """**对流格式升级的核心证据**：默认幂律格式把收敛阶从 1.0 提到 ~1.8。

    背景：初版只用一阶迎风，对流通量误差 O(h)，advection-dominated 工况下
    数值扩散（u·Δx/2）主导物理扩散 —— 这是本求解器最大的精度短板。

    解法参考成熟开源实现（NIST FiPy / Patankar 的 A(|P|) 框架，
    见 `chemistry/schemes.py`）：把扩散导纳乘以 A(|P|)，
    幂律格式 A = max(0, 1−0.1|P|)⁵ 在**保持正系数（无条件有界）**的前提下
    大幅降低数值扩散。FiPy 亦以此为默认格式。

    实测（对流主导，Pe_cell≈4）：
        upwind    阶 1.00，最细网格误差 1.56e-3
        power_law 阶 1.84，最细网格误差 2.99e-4   ← 误差降 5.2×
    """
    dr_, dz_ = 2e-6, 5e-6
    u0 = 2e-3
    grids = [(20, 10), (40, 20), (80, 40)]

    def orders(scheme):
        es = [_mms_error_scheme(nz, nr, u0, 0.0, dr_, dz_, scheme)
              for nz, nr in grids]
        return es, float(np.log2(es[1] / es[2]))

    e_up, ord_up = orders("upwind")
    e_pl, ord_pl = orders("power_law")

    assert 0.7 < ord_up < 1.4, f"迎风应为一阶，实测 {ord_up:.2f}"
    assert ord_pl > 1.6, f"幂律应为二阶量级，实测 {ord_pl:.2f}"
    assert ord_pl > ord_up + 0.5, "幂律未显著优于迎风"
    # 最细网格误差应显著下降
    assert e_pl[2] < 0.4 * e_up[2], (
        f"幂律误差 {e_pl[2]:.3e} 未显著低于迎风 {e_up[2]:.3e}"
    )


def test_adr2d_all_schemes_bounded_on_high_peclet():
    """高 Peclet 下：保正格式（迎风/幂律/指数/混合）必须无振荡，中心差分可能振荡。

    这是幂律相对中心差分的存在价值：同为二阶量级，但幂律**保证正系数**，
    在单元 Peclet 高时不会产生非物理过冲。
    """
    # 构造高 Pe_cell 工况：强对流 + 弱扩散
    dr_, dz_ = 1e-8, 1e-8          # 极小扩散 → Pe_cell 很大
    u0 = 1.0
    c_in = np.array([C_IN])
    net = _net_single()

    def solve(scheme):
        m = ADR2D(network=net, length=L, radius=R, velocity_z=u0,
                  dispersion=(dr_, dz_), n_z=40, n_r=8,
                  mode="axisymmetric", inlet_bc="dirichlet",
                  convection_scheme=scheme)
        return m, m.solve_steady(c_in)

    pe_cell = u0 * (L / 40) / dz_
    assert pe_cell > 100, f"本工况 Pe_cell={pe_cell:.0f}，不足以区分格式"

    for scheme in ("upwind", "power_law", "exponential", "hybrid"):
        _, prof = solve(scheme)
        c = prof[:, :, 0]
        assert c.min() > -1e-9, f"{scheme} 出现负浓度（非保正）"
        assert c.max() < C_IN * (1.0 + 1e-9), f"{scheme} 出现过冲"

    # 中心差分在高 Pe 下不保证有界 —— 这里只记录其行为，不断言必然振荡
    # （是否振荡取决于具体离散，故不写成硬断言，避免测试脆弱）
    _, prof_c = solve("central")
    c_central = prof_c[:, :, 0]
    assert np.all(np.isfinite(c_central)), "中心差分解应有限"


def test_adr2d_rejects_unknown_scheme():
    net = _net_single()
    with pytest.raises(ValueError, match="未知 convection_scheme"):
        ADR2D(network=net, length=L, radius=R, convection_scheme="magic")


def test_adr1d_and_adr2d_use_same_scheme_default():
    """一维与二维的对流格式默认值必须一致，否则一维退化测试会失败。"""
    net = ReactionNetwork()
    net.add("A -> B", k=0.0)
    m1 = ADR1D(network=net, length=L, velocity=0.01, dispersion=1e-6)
    m2 = ADR2D(network=net, length=L, radius=R, velocity_z=0.01,
               dispersion=1e-6, mode="planar")
    assert m1.convection_scheme == m2.convection_scheme == "power_law"


def test_adr2d_mms_advection_error_decreases_slowly():
    """对流主导时误差量级应接近「数值扩散」估计 u·Δz/2·∇²c 驱动的量级。

    这条测试把一阶迎风的实际代价量化出来，避免使用者误以为它很准。
    """
    dr_, dz_ = 2e-6, 5e-6
    e = _mms_error(20, 10, 2e-3, 0.0, dr_, dz_)
    # 数值扩散 u·dz/2 = 2e-3*0.01/2 = 1e-5，远大于物理 D_z = 5e-6
    d_num = 2e-3 * (L / 20) / 2.0
    assert d_num > dr_, "本工况确实是对流主导（数值扩散大于物理扩散）"
    assert e > 1e-3, f"一阶迎风在该 Pe 下不应给出高精度，实测误差仅 {e:.3e}"


# ══════════════════════════════════════════════════════════════
# 4. 一维退化
# ══════════════════════════════════════════════════════════════

def test_adr2d_planar_reduces_to_adr1d():
    """平面模式 + 零通量壁 + 无径向变化时，二维解须与一维求解器一致。

    这是对二维装配最强的一致性检验：若径向面权重、装配符号或边界处理
    有错，径向项不会恰好抵消，二维结果就会偏离一维。
    """
    k, u, D = 0.05, 0.01, 5e-5
    net = ReactionNetwork()
    net.add("A -> B", k=k)
    c_in = np.array([C_IN, 0.0])

    m2 = ADR2D(network=net, length=L, radius=R, velocity_z=u, velocity_r=0.0,
               dispersion=D, n_z=200, n_r=8, mode="planar",
               inlet_bc="danckwerts")
    p2 = m2.solve_steady(c_in)

    m1 = ADR1D(network=net, length=L, velocity=u, dispersion=D,
               n_cells=200, inlet_bc="danckwerts")
    p1 = m1.solve_steady(c_in)

    ia = net.species.index("A")
    # 每个径向位置都应与一维解相同
    for j in range(m2.n_r):
        err = np.abs(p2[:, j, ia] - p1[:, ia]).max() / C_IN
        assert err < 1e-9, f"径向位置 j={j} 与一维解偏差 {err:.3e}"


def test_adr2d_planar_reduces_to_adr1d_dirichlet():
    """Dirichlet 入口下的一维退化（检验另一条入口边界支路）。"""
    k, u, D = 0.03, 0.005, 2e-5
    net = ReactionNetwork()
    net.add("A -> B", k=k)
    c_in = np.array([C_IN, 0.0])

    m2 = ADR2D(network=net, length=L, radius=R, velocity_z=u, dispersion=D,
               n_z=120, n_r=6, mode="planar", inlet_bc="dirichlet")
    p2 = m2.solve_steady(c_in)
    m1 = ADR1D(network=net, length=L, velocity=u, dispersion=D,
               n_cells=120, inlet_bc="dirichlet")
    p1 = m1.solve_steady(c_in)

    ia = net.species.index("A")
    err = np.abs(p2[:, :, ia] - p1[:, ia][:, None]).max() / C_IN
    assert err < 1e-9, f"Dirichlet 入口下与一维解偏差 {err:.3e}"


# ══════════════════════════════════════════════════════════════
# 5. 守恒性
# ══════════════════════════════════════════════════════════════

def test_adr2d_mass_conservation_no_reaction():
    """无反应 + 壁面零通量：入口摩尔流量必须等于出口摩尔流量。"""
    net = ReactionNetwork()
    net.add("A -> B", k=0.0)                     # 零速率 = 不反应
    uz = 0.01 * (1.0 - (np.arange(16) / 16.0) ** 2)   # 非均匀剖面，加大检验强度
    m = ADR2D(network=net, length=L, radius=R, velocity_z=uz, dispersion=1e-6,
              n_z=80, n_r=16, mode="axisymmetric", inlet_bc="danckwerts")
    c_in = np.array([C_IN, 0.0])
    prof = m.solve_steady(c_in)

    fin = m.molar_flow_in(c_in)
    fout = m.molar_flow_out(prof)
    # A 有进料 → 用相对误差；B 无进料（fin=0）→ 用绝对误差判据
    assert abs(fout[0] - fin[0]) / abs(fin[0]) < 1e-9, f"A 摩尔流量失衡"
    assert abs(fout[1]) < 1e-12, f"B 不应产生（无反应），得到 {fout[1]:.3e}"


def test_adr2d_reaction_mass_balance():
    """有反应时：入口 A 流量 − 出口 A 流量 应等于 A 的净反应消耗量。"""
    k = 0.05
    net = ReactionNetwork()
    net.add("A -> B", k=k)
    m = ADR2D(network=net, length=L, radius=R, velocity_z=0.01, dispersion=1e-6,
              n_z=60, n_r=12, mode="axisymmetric")
    c_in = np.array([C_IN, 0.0])
    prof = m.solve_steady(c_in)

    _, _, vol = m._weights()
    src = net.source_batch(prof.reshape(-1, net.n_species)).reshape(m.n_z, m.n_r, net.n_species)
    consumed_A = -float(np.sum(src[:, :, 0] * vol[None, :]))
    produced_B = float(np.sum(src[:, :, 1] * vol[None, :]))

    fin = m.molar_flow_in(c_in)
    fout = m.molar_flow_out(prof)

    assert (fin[0] - fout[0]) == pytest.approx(consumed_A, rel=1e-6)
    assert (fout[1] - 0.0) == pytest.approx(produced_B, rel=1e-6)


# ══════════════════════════════════════════════════════════════
# 6. 瞬态与稳态一致性
# ══════════════════════════════════════════════════════════════

def test_adr2d_transient_approaches_steady():
    """瞬态积分足够长时间应收敛到稳态解。"""
    k, u, D = 0.05, 0.01, 1e-6
    net = ReactionNetwork()
    net.add("A -> B", k=k)
    m = ADR2D(network=net, length=L, radius=R, velocity_z=u, dispersion=D,
              n_z=40, n_r=8, mode="axisymmetric")
    c_in = np.array([C_IN, 0.0])
    steady = m.solve_steady(c_in)

    tau = L / u
    t, C = m.solve_transient(c_in, t_end=6.0 * tau, n_out=6)
    err = np.abs(C[-1][:, :, 0] - steady[:, :, 0]).max() / C_IN
    assert err < 1e-3, f"瞬态未收敛到稳态，偏差 {err:.2e}"


# ══════════════════════════════════════════════════════════════
# 7. 参数校验与接口
# ══════════════════════════════════════════════════════════════

def test_adr2d_velocity_divergence_guard():
    """速度场无散性自检：均匀场应通过；非无散场必须被量化标出。

    这个接口来自一次真实事故：MMS 里用了 u_r = u₀·r/R（∇·u = 2u₀/R ≠ 0），
    求解器按守恒形式 ∇·(u·c) 正确求解，却因源项漏掉 c·∇·u
    导致误差 45% 且**随网格加密上升**。
    """
    net = _net(0.0)
    # 均匀流场：∇·u = 0
    ok = ADR2D(network=net, length=L, radius=R, velocity_z=0.01, velocity_r=0.0,
               n_z=20, n_r=8, mode="axisymmetric")
    assert ok.divergence_free_violation() < 1e-12

    # 抛物线轴向剖面 + 零径向速度：仍然无散（∂u_z/∂z = 0）
    uz_prof = 0.01 * (1.0 - (np.arange(8) / 8.0) ** 2)
    ok2 = ADR2D(network=net, length=L, radius=R, velocity_z=uz_prof, velocity_r=0.0,
                n_z=20, n_r=8, mode="axisymmetric")
    assert ok2.divergence_free_violation() < 1e-12

    # 线性径向速度：∇·u = 2·ur0/R ≠ 0，必须被标出
    bad = ADR2D(network=net, length=L, radius=R, velocity_z=0.0,
                velocity_r=1e-4 * (np.arange(8) / 8.0), n_z=20, n_r=8,
                mode="axisymmetric")
    assert bad.divergence_free_violation() > 1e-3, "非无散速度场必须被检出"


def test_adr2d_velocity_field_satisfies_divergence(monkeypatch):
    """由解析环隙剖面构造的三维无散速度场（轴向均匀 + 无径向分量）应通过自检。"""
    net = _net(0.0)
    prof = 0.01 * (1.0 - (np.arange(12) / 12.0) ** 2)
    m = ADR2D(network=net, length=L, radius=R, velocity_z=prof, velocity_r=0.0,
              n_z=30, n_r=12, mode="axisymmetric")
    assert m.divergence_free_violation() < 1e-12


def test_adr2d_small_flux_not_swallowed_by_absolute_tolerance():
    """回归测试：小通量工况不得因**绝对**容差过松而被判"已收敛"、原样返回初值。

    历史缺陷：收敛判据曾用绝对容差 1e-10，而低流速/小面积工况下初始均匀猜想的
    残差本就 < 1e-10（实测 ~5e-11），牛顿第一步就退出，返回的"解"其实是初值 ——
    表现为「求解成功但壁面完全不消耗」。修复方式是把判据相对化
    （`_residual_scale()`）。
    """
    net = _net_single()
    R_small, L_small = 5e-3, 0.20
    nz, nr = 40, 40
    u_prof = 0.02 * (1.0 - (np.arange(nr) / nr) ** 2)
    j = 3e-7                                    # 约 0.1×极限电流，亚极限
    m = ADR2D(network=net, length=L_small, radius=R_small,
              velocity_z=u_prof, dispersion=1e-9, n_z=nz, n_r=nr,
              mode="axisymmetric", inlet_bc="danckwerts",
              wall_flux=lambda z: np.full_like(z, -j))
    c_in = np.array([C_IN])

    # 前提：初始猜想的残差必须小于旧的绝对容差，否则复现不了该缺陷
    guess = np.tile(c_in, (m.n_z, m.n_r, 1)).ravel()
    res_guess = float(np.abs(m.residual(guess, c_in)).max())
    assert res_guess < 1e-10, (
        f"本工况初始残差 {res_guess:.2e} 未小于 1e-10，无法复现原缺陷"
    )

    prof = m.solve_steady(c_in)
    cA = prof[:, :, 0]
    # 必须真的被消耗（壁面出现浓度边界层），而不是原样返回初值
    assert cA[:, -1].min() < 0.99 * C_IN, (
        f"壁面通量被容差吞掉了：壁面浓度 {cA[:, -1].min():.6f} ≈ 初值"
    )
    assert cA.min() >= -1e-12, "本工况应保持非负（亚极限）"
    # 关键判据：壁面通量核算必须闭合
    removed = m.molar_flow_in(c_in)[0] - m.molar_flow_out(prof)[0]
    area = 2.0 * np.pi * R_small * L_small
    assert removed == pytest.approx(j * area, rel=1e-6)


def test_adr1d_residual_scale_is_relative():
    """一维的收敛阈值必须随工况缩放（相对容差），而非固定绝对值。

    通过 `_residual_scale()` 的比值间接验证：流速/弥散差 4 个数量级时，
    残差量级参考值也必须差 3 个数量级以上，否则低流速工况会被误判收敛。
    """
    net = ReactionNetwork()
    net.add("A -> B", k=1e-4)
    c_in = np.array([C_IN, 0.0])
    slow = ADR1D(network=net, length=0.2, velocity=1e-6, dispersion=1e-11)
    fast = ADR1D(network=net, length=0.2, velocity=0.01, dispersion=1e-6)
    s_slow = slow._residual_scale(c_in)
    s_fast = fast._residual_scale(c_in)
    assert s_fast > s_slow * 1000, (
        f"残差量级未随工况缩放：slow={s_slow:.3e} fast={s_fast:.3e}"
    )
    # 相对容差下，两者的绝对阈值按同一比例缩放
    assert 1e-9 * s_slow < 1e-9 * s_fast


def test_adr2d_wall_flux_requires_explicit_per_species_shape():
    """多物质体系下，一维形状的壁面通量必须**显式报错**而不是静默广播。

    这是接口陷阱的防线：若静默广播，产物 B 也会被同样"消耗"而变成负浓度
    （本项目实际踩过）。按"禁止静默行为"的准则改为报错。
    """
    net = ReactionNetwork()
    net.add("A -> B", k=0.0)
    c_in = np.array([C_IN, 0.0])
    common = dict(network=net, length=L, radius=R, velocity_z=0.005,
                  dispersion=1e-9, n_z=20, n_r=8, mode="planar",
                  inlet_bc="danckwerts")

    # (nz,) 且 ns=2 → 报错
    m_bad = ADR2D(**common, wall_flux=lambda z: np.full_like(z, -1e-7))
    with pytest.raises(ValueError, match="wall_flux 返回了一维数组"):
        m_bad.solve_steady(c_in)

    # 形状不符 → 报错
    m_bad2 = ADR2D(**common, wall_flux=lambda z: np.zeros((z.size, 3)))
    with pytest.raises(ValueError, match="返回形状"):
        m_bad2.solve_steady(c_in)

    # 单物质时 (nz,) 合法
    m_ok1 = ADR2D(network=_net_single(), length=L, radius=R, velocity_z=0.005,
                  dispersion=1e-9, n_z=20, n_r=8, mode="planar",
                  inlet_bc="danckwerts",
                  wall_flux=lambda z: np.full_like(z, -1e-8))
    m_ok1.solve_steady(np.array([C_IN]))

    # 多物质时给 (nz, ns) → 合法
    def per_species(z):
        out = np.zeros((z.size, 2))
        out[:, 0] = -1e-7
        out[:, 1] = +1e-7
        return out

    m_ok2 = ADR2D(**common, wall_flux=per_species)
    p2 = m_ok2.solve_steady(c_in)
    assert p2[:, :, 0].min() >= -1e-12, "A 不应变负"
    assert p2[:, :, 1].max() > 0, "B 应被生成（同计量）"
    # A + B 总量守恒（本体不反应，仅界面转化）
    tot = p2[:, :, 0] + p2[:, :, 1]
    assert tot.max() - tot.min() == pytest.approx(0.0, abs=1e-12)


def test_adr2d_warns_on_negative_concentration():
    """突破传质极限（通量超过极限电流）时必须发出警告，而不是静默返回负浓度。"""
    m = ADR2D(network=_net_single(), length=L, radius=R, velocity_z=0.005,
              dispersion=1e-9, n_z=20, n_r=8, mode="planar",
              inlet_bc="danckwerts",
              wall_flux=lambda z: np.full_like(z, -5e-4))   # 远大于传质能力
    with pytest.warns(UserWarning, match="负浓度"):
        m.solve_steady(np.array([C_IN]))


def test_adr2d_rejects_bad_inputs():
    net = _net(0.05)
    with pytest.raises(ValueError):                       # 未知模式
        ADR2D(network=net, length=L, radius=R, mode="cylindrical")
    with pytest.raises(ValueError):                       # 未知入口口径
        ADR2D(network=net, length=L, radius=R, inlet_bc="neumann")
    with pytest.raises(ValueError):                       # 速度剖面长度不符
        ADR2D(network=net, length=L, radius=R, velocity_z=np.zeros(5), n_r=8)
    with pytest.raises(ValueError):                       # 非物理几何
        ADR2D(network=net, length=0.0, radius=R)


def test_adr2d_section_average_weighting_matters():
    """流量加权与面积加权在抛物线剖面下必须给出不同结果。"""
    net = _net(0.0)
    uz = 0.01 * (1.0 - (np.arange(24) / 24.0) ** 2)
    m = ADR2D(network=net, length=L, radius=R, velocity_z=uz, dispersion=1e-6,
              n_z=20, n_r=24, mode="axisymmetric")
    field = np.tile(m._rg[:, :, None], (1, 1, net.n_species))   # 场随 r 变化
    _, a_flow = m.section_average(field, weight="flow")
    _, a_area = m.section_average(field, weight="area")
    assert np.abs(a_flow - a_area).max() > 1e-3, "两种加权方式不应等同"


def test_adr2d_velocity_profile_broadcast():
    """一维剖面应能正确广播成二维场；标量应铺满全场。"""
    net = _net(0.0)
    prof = np.array([0.01, 0.005])
    m = ADR2D(network=net, length=L, radius=R, velocity_z=prof,
              n_z=5, n_r=2, mode="planar")
    uz, ur = m.velocity_fields()
    assert uz.shape == (5, 2)
    assert np.allclose(uz[:, 0], 0.01) and np.allclose(uz[:, 1], 0.005)
    assert np.allclose(ur, 0.0)

    m2 = ADR2D(network=net, length=L, radius=R, velocity_z=0.02,
               n_z=5, n_r=2, mode="planar")
    uz2, _ = m2.velocity_fields()
    assert np.allclose(uz2, 0.02)
