"""诚实的耗时对比：同一环隙问题，fastsim vs COMSOL。

口径说明（避免夸大）
    · COMSOL 计时只取 `sol1.runAll()`（稳态求解），不含启动、几何、物理场搭建；
      后者在本机实测约 25–35 s，若计入则差距更大，故**不计入**。
    · COMSOL 网格 hmax = gap/32（约 26k 节点）；fastsim n_r=32 / n_z=120（约 4.1k 未知量）。
      两者精度已在 threeway_annulus.py 中比对：fastsim 0.01%，COMSOL 1.9%。
      —— 即 fastsim 用**更少未知量**取得**更高精度**，对比不偏袒 fastsim。
    · fastsim 重复 20 次取均值（求解毫秒级，单次计时噪声大）。
"""
import sys
import time
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
N_Z, N_R = 120, 32


def time_fastsim(reps=20):
    sim = SecondaryCurrent2D(
        r_inner=R_I, r_outer=R_O, length=L_LEN, kappa=KAPPA,
        anode=ElectrodeKinetics(j0=I0, alpha_a=ALPHA, alpha_c=ALPHA),
        cathode=ElectrodeKinetics(j0=I0, alpha_a=ALPHA, alpha_c=ALPHA),
        n_z=N_Z, n_r=N_R, temperature=T_K,
    )
    sim.solve(V_CELL)                       # 预热
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sol = sim.solve(V_CELL)
        ts.append(time.perf_counter() - t0)
    n_unknown = sim._n_unknown
    return float(np.mean(ts)), float(np.min(ts)), n_unknown, float(sol.i_total)


def time_comsol():
    import mph
    client = mph.start(cores=2)
    model = client.create("timing")
    m = model.java
    comp = m.component().create("comp1", True)
    g = comp.geom().create("geom1", 2)
    g.axisymmetric(True)
    rec = g.create("r1", "Rectangle")
    rec.set("pos", [R_I, 0.0])
    rec.set("size", [R_O - R_I, L_LEN])
    g.run()
    m.param().set("kappa", f"{KAPPA}[S/m]")
    m.param().set("Vcell", f"{V_CELL}[V]")
    m.param().set("i0", f"{I0}[A/m^2]")
    cd = comp.physics().create("cd", "SecondaryCurrentDistribution", "geom1")
    ice = cd.feature("ice1")
    ice.set("sigmal_mat", "userdef")
    ice.set("sigmal", "kappa")
    for tagn, bnd, v in (("es_a", 1, "Vcell"), ("es_c", 4, "0")):
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
        er.set("Eeq_mat", "userdef"); es.feature("er1").set("Eeq_ref", "0[V]")
        es.feature("er1").set("Eeq", "0[V]")
    mesh = comp.mesh().create("mesh1")
    try:
        sz = mesh.feature("size")
    except Exception:  # noqa: BLE001
        sz = mesh.feature().create("size", "Size")
    sz.set("hmax", f"{(R_O - R_I)/32:.4e}")
    mesh.run()
    m.study().create("std1")
    m.study("std1").create("stat", "Stationary").activate("cd", True)
    m.sol().create("sol1")
    m.sol("sol1").createAutoSequence("std1")

    t0 = time.perf_counter()
    m.sol("sol1").runAll()
    t_solve = time.perf_counter() - t0

    nnode = int(m.mesh("mesh1").getNumVertex())
    return t_solve, nnode


def main():
    print("=" * 78)
    print("同一环隙二次电流分布问题的耗时对比")
    print(f"  r_i={R_I} m, r_o={R_O} m, L={L_LEN} m, κ={KAPPA} S/m, i0={I0} A/m², V={V_CELL} V")
    print("=" * 78)

    mean_t, min_t, n_unk, i_tot = time_fastsim()
    print(f"  fastsim : 均值 {mean_t*1e3:8.2f} ms   最快 {min_t*1e3:8.2f} ms   "
          f"未知量 {n_unk}   I_total={i_tot:.6f} A")

    try:
        t_c, n_node = time_comsol()
        print(f"  COMSOL  : 求解 {t_c:8.3f} s          节点 {n_node}   "
              f"（仅 sol1.runAll()，不含启动/建模）")
        print("")
        print(f"  → 求解耗时比 = {t_c/mean_t:.0f} ×   "
              f"（未知量比 = {n_node/n_unk:.1f} ×）")
    except Exception as e:  # noqa: BLE001
        print(f"  COMSOL 计时失败：{type(e).__name__}: {e}")
    print("=" * 78)


if __name__ == "__main__":
    main()
