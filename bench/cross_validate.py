"""COMSOL ↔ fastsim ↔ 解析解 三方对标。

这是 P5 的核心交付：把「自研求解器 vs COMSOL」的偏差**量化**，
而不只是声称"量级可信"。

COMSOL 参考解由 bench/comsol_reference.py 生成（基于官方可用模型改造）。
问题：1D 对流-扩散-反应，Dirichlet 入口 + 零梯度出口
      Pe = uL/D = 100，Da = kL/u = 1
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.chemistry import (  # noqa: E402
    ADR1D,
    ReactionNetwork,
    analytic_first_order_dispersion,
)

NPZ = ROOT / "bench" / "comsol_reference.npz"
if not NPZ.exists():
    raise SystemExit("缺少 COMSOL 参考解，请先运行 bench/comsol_reference.py")

ref = np.load(NPZ)
ref_coord, ref_c = ref["coord"], ref["c"]

# 与 comsol_reference.py 保持一致的工况
U0 = 1.0e-2
DCOEF = 3.0e-8
L_CH = 3.0e-4
KRATE = U0 / L_CH
CIN = 1.0

print("=" * 78)
print("COMSOL  ↔  fastsim  ↔  解析解   三方对标")
print("=" * 78)
print(f"工况：L={L_CH} m, u={U0} m/s, D={DCOEF} m²/s, k={KRATE:.4g} 1/s, C_in={CIN}")
print(f"      Pe = uL/D = {U0*L_CH/DCOEF:.4g}，Da = kL/u = {KRATE*L_CH/U0:.4g}")
print(f"      COMSOL 侧：2D 矩形 + 均匀流速 + Dirichlet 入口 + Outflow 出口")
print(f"                 （侧壁 NoFlux；无径向变化 ⇒ 应等价于 1D）")

# 把 COMSOL 的坐标平移到 z ∈ [0, L]
z_ref = ref_coord - ref_coord.min()
o = np.argsort(z_ref)
z_ref, c_ref = z_ref[o], ref_c[o]

print(f"\nCOMSOL 参考：{z_ref.size} 个采样点，"
      f"c 从 {c_ref[0]:.6f} 降到 {c_ref[-1]:.6f}")

# ── fastsim ─────────────────────────────────────────────
net = ReactionNetwork()
net.add("A -> B", k=KRATE)


def run_fastsim(n_cells: int, bc: str):
    m = ADR1D(network=net, length=L_CH, velocity=U0, dispersion=DCOEF,
              n_cells=n_cells, inlet_bc=bc)
    prof = m.solve_steady(np.array([CIN, 0.0]))
    return m.z, prof[:, net.species.index("A")]


# 先判断 COMSOL 用的是哪种入口口径：两种都算，看谁更近
print(f"\n── 判定 COMSOL 的入口口径 ──")
cand = {}
for bc in ("dirichlet", "danckwerts"):
    c_ana = analytic_first_order_dispersion(
        z_ref, L_CH, U0, DCOEF, KRATE, CIN, inlet_bc=bc)
    cand[bc] = float(np.abs(c_ref - c_ana).max() / CIN)
    print(f"  与 {bc:11s} 解析解的最大偏差: {cand[bc]:.4%}")
bc_best = min(cand, key=lambda k: cand[k])
print(f"  → COMSOL 采用: {bc_best}")

# ── 网格收敛：fastsim 自己加密看是否收敛到解析解 ──────────
print(f"\n── fastsim 网格收敛（入口口径 {bc_best}）──")
print(f"  {'单元数':>8s} {'与解析解 L∞ 偏差':>18s} {'与 COMSOL L∞ 偏差':>18s}")
rows = []
for n in (100, 200, 400, 800, 1600):
    zf, cf = run_fastsim(n, bc_best)
    c_ana_f = analytic_first_order_dispersion(
        zf, L_CH, U0, DCOEF, KRATE, CIN, inlet_bc=bc_best)
    e_ana = float(np.abs(cf - c_ana_f).max() / CIN)
    c_ref_on_zf = np.interp(zf, z_ref, c_ref)
    e_com = float(np.abs(cf - c_ref_on_zf).max() / CIN)
    rows.append((n, e_ana, e_com))
    print(f"  {n:8d} {e_ana:18.4%} {e_com:18.4%}")

n_fine, e_ana_fine, e_com_fine = rows[-1]
print(f"\n── 结论 ──")
print(f"  最细网格（{n_fine} 单元）：")
print(f"    fastsim vs 解析解  L∞ 偏差 = {e_ana_fine:.4%}")
print(f"    fastsim vs COMSOL  L∞ 偏差 = {e_com_fine:.4%}")
print(f"    COMSOL  vs 解析解  L∞ 偏差 = {cand[bc_best]:.4%}")

# COMSOL 与 fastsim 的出口值对比
m = ADR1D(network=net, length=L_CH, velocity=U0, dispersion=DCOEF,
          n_cells=800, inlet_bc=bc_best)
prof = m.solve_steady(np.array([CIN, 0.0]))
c_out_fast = float(prof[-1, 0])
c_out_com = float(c_ref[-1])
c_ana_out = float(analytic_first_order_dispersion(
    np.array([L_CH]), L_CH, U0, DCOEF, KRATE, CIN, inlet_bc=bc_best)[0])
print(f"\n  出口浓度三方对照：")
print(f"    COMSOL   = {c_out_com:.6f}")
print(f"    fastsim  = {c_out_fast:.6f}")
print(f"    解析解   = {c_ana_out:.6f}")
print(f"    fastsim 相对 COMSOL 偏差 = "
      f"{abs(c_out_fast-c_out_com)/c_out_com:.4%}")
print("=" * 78)
