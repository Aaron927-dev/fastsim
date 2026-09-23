"""COMSOL 复现基准（最终版）：平行板电池的电流分布 vs 一维精确解。

边界编号真值（实测，非假设）
    Rectangle(pos=[0,0], size=[gap, L]) →
        1 = x=0   （左，长边 0.2 m）
        2 = y=0   （下，短边 0.01 m）
        3 = y=L   （上，短边 0.01 m）
        4 = x=gap （右，长边 0.2 m）
    因此平行板电池的两个电极边是 1 与 4。

判据
    (a) ec（纯欧姆，无动力学）：φ(x) = V·(1 − x/gap)，与 y 无关 → 应严格线性。
    (b) SCD + Butler-Volmer：一维两点边值精确解
            V = j·gap/κ + 2·η(j),   η(j) = (R·T/αF)·asinh(j/(2·i0))
            φ_l(x) = (V − η) − j·x/κ
        对三个数量级的 i0 各测一次，避免「调参调出来」的假象。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench/_comsol_env.py

from _comsol_env import setup_comsol  # noqa: E402

_COMSOL_ROOT = setup_comsol()

import numpy as np  # noqa: E402
import mph  # noqa: E402

GAP, L = 0.010, 0.20
KAPPA, V_CELL = 1.0, 2.0
ALPHA, T_K = 0.5, 298.15
R_GAS, F_CONST = 8.314, 96485.33
BND_A, BND_C = 1, 4                     # 阳极 x=0，阴极 x=gap
I0_LIST = (1e-3, 1e-1, 1.0)
MESH_DIVS = (16, 32, 64)                # ec 网格收敛扫描


def P(*a):
    print(*a, flush=True)


def mesh_and_solve(m, comp, phys_tag, divs=32):
    mesh = comp.mesh().create("mesh1")
    try:
        sz = mesh.feature("size")
    except Exception:  # noqa: BLE001
        sz = mesh.feature().create("size", "Size")
    sz.set("hmax", f"{min(GAP, L)/divs:.4e}")
    mesh.run()
    m.study().create("std1")
    m.study("std1").create("stat", "Stationary").activate(phys_tag, True)
    m.sol().create("sol1")
    m.sol("sol1").createAutoSequence("std1")
    m.sol("sol1").runAll()


def new_model(client, tag):
    model = client.create(tag)
    m = model.java
    comp = m.component().create("comp1", True)
    g = comp.geom().create("geom1", 2)
    rec = g.create("r1", "Rectangle")
    rec.set("pos", [0.0, 0.0])
    rec.set("size", [GAP, L])
    g.run()
    return model, m, comp


def run_ec(client, divs):
    model, m, comp = new_model(client, f"ec_{divs}")
    m.param().set("kappa", f"{KAPPA}[S/m]")
    m.param().set("Vcell", f"{V_CELL}[V]")
    ec = comp.physics().create("ec", "ConductiveMedia", "geom1")
    ec.feature("cucn1").set("sigma_mat", "userdef")
    ec.feature("cucn1").set("sigma", "kappa")
    ec.feature().create("gnd1", "Ground", 1).selection().set([BND_C])
    pot = ec.feature().create("pot1", "ElectricPotential", 1)
    pot.selection().set([BND_A])
    pot.set("V0", "Vcell")
    mesh_and_solve(m, comp, "ec", divs)
    return (np.asarray(model.evaluate("V", "V"), dtype=float).ravel(),
            np.asarray(model.evaluate("x", "m"), dtype=float).ravel(),
            np.asarray(model.evaluate("y", "m"), dtype=float).ravel())


def run_scd(client, i0, tag):
    model, m, comp = new_model(client, tag)
    m.param().set("kappa", f"{KAPPA}[S/m]")
    m.param().set("Vcell", f"{V_CELL}[V]")
    m.param().set("i0", f"{i0}[A/m^2]")
    cd = comp.physics().create("cd", "SecondaryCurrentDistribution", "geom1")
    ice = cd.feature("ice1")
    ice.set("sigmal_mat", "userdef")
    ice.set("sigmal", "kappa")
    for tagn, bnd, v in (("es_a", BND_A, "Vcell"), ("es_c", BND_C, "0")):
        es = cd.feature().create(tagn, "ElectrodeSurface", 1)
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
    mesh_and_solve(m, comp, "cd")
    return (np.asarray(model.evaluate("phil", "V"), dtype=float).ravel(),
            np.asarray(model.evaluate("x", "m"), dtype=float).ravel(),
            np.asarray(model.evaluate("y", "m"), dtype=float).ravel())


def one_d(i0):
    f = ALPHA * F_CONST / (R_GAS * T_K)
    lo, hi = 1e-12, 1e12
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if mid * GAP / KAPPA + 2.0 / f * np.arcsinh(mid / (2 * i0)) < V_CELL:
            lo = mid
        else:
            hi = mid
    j = 0.5 * (lo + hi)
    return j, (1.0 / f) * np.arcsinh(j / (2 * i0))


def mid_y_slice(phi, x_n, y_n):
    my = (y_n > 0.4 * L) & (y_n < 0.6 * L)
    xs, vs = [], []
    for lo in np.arange(0.0, GAP, GAP / 10):
        s = my & (x_n >= lo) & (x_n < lo + GAP / 10)
        if s.sum():
            xs.append(lo + GAP / 20)
            vs.append(phi[s].mean())
    return np.array(xs), np.array(vs)


def y_span(phi, x_n, y_n):
    mx = (x_n > 0.4 * GAP) & (x_n < 0.6 * GAP)
    bands = [phi[mx & (y_n >= lo) & (y_n < lo + L / 8)].mean()
             for lo in np.arange(0.0, L, L / 8)
             if (mx & (y_n >= lo) & (y_n < lo + L / 8)).sum()]
    return float(max(bands) - min(bands))


def main():
    client = mph.start(cores=2)

    P("=" * 84)
    P("(a) ec 纯欧姆平行板 —— 靶 φ(x)=V(1−x/gap)；同时做网格收敛扫描")
    P("=" * 84)
    P(f"  {'每 gap 单元数':>14} {'x=0 边':>12} {'x=gap 边':>12} "
      f"{'拟合斜率':>12} {'RMS vs 解析':>14} {'y 跨度':>12}")
    for divs in MESH_DIVS:
        try:
            phi, x_n, y_n = run_ec(client, divs)
        except Exception as e:  # noqa: BLE001
            P(f"  {divs:>14}  求解失败：{type(e).__name__}")
            continue
        xs, vs = mid_y_slice(phi, x_n, y_n)
        ana = V_CELL * (1.0 - xs / GAP)
        rms = float(np.sqrt(np.mean((vs - ana) ** 2)))
        slope = float(np.polyfit(xs, vs, 1)[0])
        P(f"  {divs:>14} {phi[x_n <= 1e-9].mean():>12.6f} "
          f"{phi[x_n >= GAP-1e-9].mean():>12.6f} {slope:>12.2f} "
          f"{rms:>14.3e} {y_span(phi, x_n, y_n):>12.3e}")

    P("")
    P("=" * 84)
    P("(b) SCD + Butler-Volmer —— 靶为含欧姆降+动力学过电位的一维精确解")
    P("=" * 84)
    P(f"  {'i0 (A/m²)':>10} {'j解析':>10} {'j实测':>10} {'偏差':>9} "
      f"{'η解析':>9} {'η_a实测':>9} {'η_c实测':>9} {'线性RMS':>10} {'y跨度':>10}")
    for i0 in I0_LIST:
        try:
            phi, x_n, y_n = run_scd(client, i0, f"scd_{i0:g}".replace("-", "m"))
        except Exception as e:  # noqa: BLE001
            j_a, eta_a = one_d(i0)
            P(f"  {i0:>10.3g}  ✗ 求解失败（{type(e).__name__}）—— "
              f"解析 j={j_a:.4f}, η={eta_a:.5f}；需用延拓/斜坡")
            continue
        j_a, eta_a = one_d(i0)
        xs, vs = mid_y_slice(phi, x_n, y_n)
        slope = np.polyfit(xs, vs, 1)[0]
        j_m = -slope * KAPPA
        phi_an = float(np.mean(phi[x_n <= 1e-9]))
        phi_ca = float(np.mean(phi[x_n >= GAP - 1e-9]))
        lin = float(np.sqrt(np.mean((vs - np.polyval(np.polyfit(xs, vs, 1), xs)) ** 2)))
        P(f"  {i0:>10.3g} {j_a:>10.4f} {j_m:>10.4f} "
          f"{(j_m-j_a)/j_a*100:>8.2f}% {eta_a:>9.5f} {V_CELL-phi_an:>9.5f} "
          f"{phi_ca:>9.5f} {lin:>10.2e} {y_span(phi, x_n, y_n):>10.2e}")

    P("")
    P("=" * 80)
    P("边界编号真值（实测）：1=x=0(左)  2=y=0(下)  3=y=L(上)  4=x=gap(右)")
    P("=" * 80)


if __name__ == "__main__":
    main()
