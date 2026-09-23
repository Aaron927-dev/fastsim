"""三方同题比对：COMSOL（轴对称环隙）× fastsim × 一维解析解。

几何（三者完全相同的物理问题）
    管状阳极 r = r_inner（在内），筒状阴极 r = r_outer（在外），
    电极覆盖整个轴向长度 L，两端绝缘。

解析解（一维、理想导体电极；径向电流沿 ln r 分布）
    设 I' 为单位长度电流（A/m）：
        j_i = I'/(2π r_i),   j_o = I'/(2π r_o)
        φ_l(r) = φ_l(r_i) − I'·ln(r/r_i)/(2πκ)
        V = I'·R' + (1/f)·[asinh(I'/(4π r_i j₀)) + asinh(I'/(4π r_o j₀))]
        f = αF/RT,   R' = ln(r_o/r_i)/(2πκ)
    一次分布（无动力学）时退化为 I' = V/R'。
    这是标量方程，求根即得全部量，无自由参数。

读数方式（COMSOL 侧）
    不用电极电流后处理变量，而是从 φ_l 的**径向剖面**反解 I'：
        φ_l(r) = A − I'·ln(r)/(2πκ)  →  对 ln r 线性拟合，斜率 = −I'/2πκ
    这样得到的 I' 与解析解同一物理量，比对无歧义。
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # bench/_comsol_env.py
sys.path.insert(0, str(_HERE.parent))   # fastsim 包

from _comsol_env import setup_comsol  # noqa: E402

_COMSOL_ROOT = setup_comsol()

import numpy as np  # noqa: E402

from fastsim.echem.secondary import ElectrodeKinetics, SecondaryCurrent2D  # noqa: E402

R_I, R_O, L_LEN = 0.005, 0.010, 0.20
KAPPA, V_CELL = 1.0, 2.0
ALPHA, T_K, I0 = 0.5, 298.15, 1e-3
R_GAS, F_CONST = 8.31446, 96485.33
F_DIMLESS = ALPHA * F_CONST / (R_GAS * T_K)          # αF/RT ≈ 19.46 1/V
R_PRIME = np.log(R_O / R_I) / (2.0 * np.pi * KAPPA)   # Ω·m
BND_INNER, BND_OUTER = 1, 4


def P(*a):
    print(*a, flush=True)


def analytic_current(secondary: bool, v=V_CELL):
    """一维环隙精确解：求 I'（A/m）。"""
    if not secondary:
        return v / R_PRIME

    def g(i):
        return (i * R_PRIME
                + (1.0 / F_DIMLESS) * (np.arcsinh(i / (4 * np.pi * R_I * I0))
                                       + np.arcsinh(i / (4 * np.pi * R_O * I0))))

    lo, hi = 1e-15, 1e9
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if g(mid) < v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def comsol_run(secondary: bool, divs=32):
    import mph
    client = mph.start(cores=2)
    model = client.create("ann" + ("_sec" if secondary else "_pri"))
    m = model.java
    comp = m.component().create("comp1", True)
    g = comp.geom().create("geom1", 2)
    try:
        g.axisymmetric(True)
        P(f"  [轴对称已启用]")
    except Exception as e:  # noqa: BLE001
        P(f"  ⚠ axisymmetric() 失败: {type(e).__name__}: {e}")
    rec = g.create("r1", "Rectangle")
    rec.set("pos", [R_I, 0.0])
    rec.set("size", [R_O - R_I, L_LEN])
    g.run()

    m.param().set("kappa", f"{KAPPA}[S/m]")
    m.param().set("Vcell", f"{V_CELL}[V]")
    m.param().set("i0", f"{I0}[A/m^2]")

    if secondary:
        phys = comp.physics().create("cd", "SecondaryCurrentDistribution", "geom1")
        ice = phys.feature("ice1")
        ice.set("sigmal_mat", "userdef")
        ice.set("sigmal", "kappa")
        for tagn, bnd, v in (("es_a", BND_INNER, "Vcell"), ("es_c", BND_OUTER, "0")):
            es = phys.feature().create(tagn, "ElectrodeSurface", 1)
            es.selection().set([bnd])
            es.set("BoundaryCondition", "ElectricPotential")
            es.set("phisext0", v)
            er = es.feature("er1")
            er.set("ElectrodeKinetics", "ButlerVolmer")
            er.set("i0Type", "userdef")
            er.set("i0", "i0")
            er.set("alphaa", f"{ALPHA}")
            er.set("alphac", f"{ALPHA}")
            er.set("Eeq_mat", "userdef"); er.set("Eeq_ref", "0[V]"); er.set("Eeq", "0[V]")
        phi_expr = "phil"
        ptag = "cd"
    else:
        phys = comp.physics().create("ec", "ConductiveMedia", "geom1")
        phys.feature("cucn1").set("sigma_mat", "userdef")
        phys.feature("cucn1").set("sigma", "kappa")
        phys.feature().create("gnd1", "Ground", 1).selection().set([BND_OUTER])
        pot = phys.feature().create("pot1", "ElectricPotential", 1)
        pot.selection().set([BND_INNER])
        pot.set("V0", "Vcell")
        phi_expr = "V"
        ptag = "ec"

    mesh = comp.mesh().create("mesh1")
    try:
        sz = mesh.feature("size")
    except Exception:  # noqa: BLE001
        sz = mesh.feature().create("size", "Size")
    sz.set("hmax", f"{(R_O - R_I)/divs:.4e}")
    mesh.run()
    m.study().create("std1")
    m.study("std1").create("stat", "Stationary").activate(ptag, True)
    m.sol().create("sol1")
    m.sol("sol1").createAutoSequence("std1")
    m.sol("sol1").runAll()

    phi = np.asarray(model.evaluate(phi_expr, "V"), dtype=float).ravel()
    r_n = np.asarray(model.evaluate("r", "m"), dtype=float).ravel()
    z_n = np.asarray(model.evaluate("z", "m"), dtype=float).ravel()

    for nm, s in ((f"r=r_i(边界{BND_INNER})", r_n <= R_I + 1e-12),
                  (f"r=r_o(边界{BND_OUTER})", r_n >= R_O - 1e-12),
                  ("z=0", z_n <= 1e-12), (f"z=L", z_n >= L_LEN - 1e-12)):
        if s.sum():
            P(f"    {nm:18s} n={s.sum():5d}  φ∈[{phi[s].min():+.6f}, {phi[s].max():+.6f}] "
              f"跨度={phi[s].max()-phi[s].min():.2e}")
    return phi, r_n, z_n


def comsol_current(phi, r_n, z_n):
    """从 φ_l 的径向剖面反解 I'（对 ln r 线性拟合）。"""
    mid = (z_n > 0.3 * L_LEN) & (z_n < 0.7 * L_LEN)
    lr, pv = [], []
    edges = np.linspace(np.log(R_I), np.log(R_O), 12)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = mid & (np.log(r_n) >= lo) & (np.log(r_n) < hi)
        if s.sum():
            lr.append(0.5 * (lo + hi))
            pv.append(phi[s].mean())
    slope = np.polyfit(np.array(lr), np.array(pv), 1)[0]
    return -slope * 2.0 * np.pi * KAPPA


def main():
    P("=" * 84)
    P("环隙参数：r_i=5 mm, r_o=10 mm, L=0.20 m, κ=1 S/m, α=0.5, T=298.15 K, V=2 V")
    P(f"  f=αF/RT={F_DIMLESS:.4f} 1/V,  R'=ln(r_o/r_i)/(2πκ)={R_PRIME:.6f} Ω·m")
    P("=" * 84)

    for secondary in (False, True):
        name = "二次分布（BV 动力学, i0=1e-3）" if secondary else "一次分布（纯欧姆）"
        P("")
        P("-" * 84)
        P(f"### {name}")
        P("-" * 84)
        iP = analytic_current(secondary)
        j_i_a = iP / (2 * np.pi * R_I)
        j_o_a = iP / (2 * np.pi * R_O)
        I_a = iP * L_LEN
        P(f"  解析：I'={iP:.6f} A/m   j_i={j_i_a:.4f} A/m²   j_o={j_o_a:.4f} A/m²   "
          f"I_total={I_a:.6f} A")

        # COMSOL
        try:
            phi, r_n, z_n = comsol_run(secondary)
            iP_c = comsol_current(phi, r_n, z_n)
            zspan_i = (np.mean(phi[np.abs(r_n - R_I) <= 1e-12])
                       if (np.abs(r_n - R_I) <= 1e-12).sum() else np.nan)
            P(f"  COMSOL：I'={iP_c:.6f} A/m   "
              f"j_i={iP_c/(2*np.pi*R_I):.4f}   j_o={iP_c/(2*np.pi*R_O):.4f}   "
              f"I_total={iP_c*L_LEN:.6f} A   "
              f"偏差 {(iP_c-iP)/iP*100:+.2f}%")
        except Exception as e:  # noqa: BLE001
            P(f"  COMSOL 失败：{type(e).__name__}: {e}")
            iP_c = np.nan

        # fastsim
        try:
            sim = SecondaryCurrent2D(
                r_inner=R_I, r_outer=R_O, length=L_LEN, kappa=KAPPA,
                anode=(ElectrodeKinetics(j0=I0, alpha_a=ALPHA, alpha_c=ALPHA)
                       if secondary else None),
                cathode=(ElectrodeKinetics(j0=I0, alpha_a=ALPHA, alpha_c=ALPHA)
                         if secondary else None),
                n_z=120, n_r=32, temperature=T_K,
            )
            sol = sim.solve(V_CELL)
            j_i_f = float(np.mean(sol.j_anode))
            j_o_f = float(np.mean(sol.j_cathode))
            iP_f = j_i_f * 2 * np.pi * R_I
            P(f"  fastsim：I'={iP_f:.6f} A/m  j_i={j_i_f:.4f}   j_o={j_o_f:.4f}   "
              f"I_total={sol.i_total:.6f} A   "
              f"偏差 {(iP_f-iP)/iP*100:+.2f}%")
            P(f"          轴向均匀性：j_i 跨度 = {sol.j_anode.max()-sol.j_anode.min():.3e} A/m²"
              f"（解析应 =0）")
        except Exception as e:  # noqa: BLE001
            P(f"  fastsim 失败：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
