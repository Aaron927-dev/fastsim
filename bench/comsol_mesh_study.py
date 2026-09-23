"""COMSOL 网格加密研究（保守网格梯度，附即时输出）。

目的：验证「COMSOL 相对解析解的 0.98% 偏差来自网格分辨率不足」这一假设。
入口边界层 δ = D/u = 3e-6 m，占 L 的 1% —— 若该假设成立，
加密后偏差应持续下降并趋近 Dirichlet 解析解。

⚠ 上一版把 hmax 设到 5.9e-7，二维自由三角形网格要 8 万单元导致内存不足；
本版限制在 ~3 万单元以内。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench/_comsol_env.py

from _comsol_env import setup_comsol  # noqa: E402

_COMSOL_ROOT = setup_comsol()

import numpy as np  # noqa: E402
import mph  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastsim.chemistry import analytic_first_order_dispersion  # noqa: E402


def P(*a):
    print(*a, flush=True)


SRC = (r"D:\COMSOL\COMSOL60\Multiphysics\applications\COMSOL_Multiphysics"
       r"\Chemical_Engineering\transport_and_adsorption.mph")

L_CH, U0, DCOEF = 3.0e-4, 1.0e-2, 3.0e-8
KRATE, CIN = U0 / L_CH, 1.0
P(f"Pe = {U0*L_CH/DCOEF:.0f}  Da = {KRATE*L_CH/U0:.3f}  "
  f"入口边界层 δ/L = {DCOEF/U0/L_CH:.2%}")
P("构建模型…")

client = mph.start(cores=2)
model = client.load(SRC)
m = model.java
comp = m.component("comp1")
comp.physics("gb").active(False)
comp.physics("tds").feature("fl1").active(False)
for k, v in (("D", f"{DCOEF}[m^2/s]"), ("u0", f"{U0}[m/s]"),
             ("kr", f"{KRATE}[1/s]"), ("cin", f"{CIN}[mol/m^3]")):
    m.param().set(k, v)
p = comp.physics("tds")
cdm = p.feature("cdm1")
cdm.set("u_src", "userdef")
cdm.set("u", ["0", f"{U0}[m/s]", "0"])
cdm.set("DiffusionCoefficientSource", "mat")
cdm.set("D_c", "D")
p.feature("conc1").set("c0", "cin")
r = p.feature().create("reac1", "Reactions", 2)
r.selection().set([1])
r.set("R_c", "-kr*c")
m.study("std1").feature("time").set("tlist", "range(0,50,200)")
P("模型就绪，开始网格加密扫描")

mesh = comp.mesh("mesh1")
mtags = [str(x) for x in mesh.feature().tags()]
P(f"网格特征: {mtags}")
if "size" not in mtags:
    mesh.feature().create("size", "Size")
    mtags.append("size")

P(f"\n{'hmax(m)':>10s} {'单元数':>9s} {'耗时(s)':>8s} {'vs Dirichlet':>13s} "
  f"{'vs Danckwerts':>14s} {'判定':>12s}")
P("-" * 74)

for hmax in (None, 2.0e-5, 5.0e-6, 2.0e-6):
    tag = "默认(自动)" if hmax is None else f"{hmax:.0e}"
    try:
        if hmax is not None:
            mesh.feature("size").set("hmax", hmax)
        mesh.run()
        n_el = int(mesh.getNumElem("domain"))
    except Exception as exc:  # noqa: BLE001
        P(f"{tag:>10s}  网格失败: {str(exc).splitlines()[-1][:55]}")
        continue

    try:
        t0 = time.perf_counter()
        m.sol("sol1").runAll()
        dt = time.perf_counter() - t0
        c_n = np.asarray(model.evaluate("c", "mol/m^3", inner="last"),
                         dtype=float).ravel()
        y_n = np.asarray(model.evaluate("y", "m", inner="last"),
                         dtype=float).ravel()
    except Exception as exc:  # noqa: BLE001
        P(f"{tag:>10s}  求解失败: {str(exc).splitlines()[-1][:55]}")
        continue

    z = y_n - y_n.min()
    o = np.argsort(z)
    z, c = z[o], c_n[o]
    e_dir = float(np.abs(c - analytic_first_order_dispersion(
        z, L_CH, U0, DCOEF, KRATE, CIN, inlet_bc="dirichlet")).max() / CIN)
    e_dan = float(np.abs(c - analytic_first_order_dispersion(
        z, L_CH, U0, DCOEF, KRATE, CIN, inlet_bc="danckwerts")).max() / CIN)
    verdict = "Dirichlet ✓" if e_dir < e_dan else "Danckwerts"
    P(f"{tag:>10s} {n_el:9d} {dt:8.2f} {e_dir:13.4%} {e_dan:14.4%} {verdict:>12s}")

P("\n判据：若 vs Dirichlet 的偏差随加密持续下降并最终小于 vs Danckwerts，")
P("      则证实此前假设 —— COMSOL 偏差源于入口边界层未被网格分辨。")
