"""用官方可用模型改造出 ADR 对标基准（COMSOL 侧参考解）。

思路：不从零建模型（此前反复失败），而是**基于官方能跑通的 tds 模型改造**：
  transport_and_adsorption.mph  → 关闭吸附场，改成纯对流-扩散-反应

然后：COMSOL 解 vs fastsim 解 vs 解析解 三方对比。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench/ 同级的 _comsol_env

from _comsol_env import comsol_applications_dir, setup_comsol  # noqa: E402

_COMSOL_ROOT = setup_comsol()

import numpy as np  # noqa: E402
import mph  # noqa: E402

SRC = comsol_applications_dir(_COMSOL_ROOT) / "COMSOL_Multiphysics" / \
    "Chemical_Engineering" / "transport_and_adsorption.mph"

# ── 目标工况（与 fastsim 对齐）────────────────────────────────
# ⚠ 几何带长度单位：rectangle pos=[0,-0.1] size=[0.1,0.3] 的单位是 **mm**
#   → 实际 x∈[0, 1e-4] m, y∈[-1e-4, 2e-4] m（长边沿 y，L=3e-4 m）
L_CH = 3.0e-4        # m，沿流动方向的长度
U0 = 1.0e-2          # m/s
DCOEF = 3.0e-8       # m²/s  → Pe = uL/D = 100
KRATE = U0 / L_CH    # 1/s    → Da = kL/u = 1
CIN = 1.0            # mol/m³
FLOW_AXIS = "y"      # 由几何（长边沿 y）与入口/出口边界位置确定

client = mph.start(cores=2)
model = client.load(SRC)
m = model.java
comp = m.component("comp1")

# 几何：x∈[0,0.1], y∈[-0.1,0.2]
print("=== 改造官方模型 ===")

# ① 关闭吸附物理场（General Form Boundary PDE）
try:
    comp.physics("gb").active(False)
    print("  已关闭 gb（吸附表面 PDE）")
except Exception as exc:  # noqa: BLE001
    print("  关闭 gb 失败:", str(exc).splitlines()[-1][:80])

# ② 参数
m.param().set("D", f"{DCOEF}[m^2/s]")
m.param().set("u0", f"{U0}[m/s]")
m.param().set("kr", f"{KRATE}[1/s]")
m.param().set("cin", f"{CIN}[mol/m^3]")

# ③ 传递属性：速度 + 扩散
p = comp.physics("tds")
cdm = p.feature("cdm1")
cdm.set("u_src", "userdef")
_uvec = [f"{U0}[m/s]", "0", "0"] if FLOW_AXIS == "x" else ["0", f"{U0}[m/s]", "0"]
cdm.set("u", _uvec)
cdm.set("DiffusionCoefficientSource", "mat")
cdm.set("D_c", "D")

# ④ 入口浓度
p.feature("conc1").set("c0", "cin")

# ⑤ 侧壁通量：直接**停用**该边界条件 → 该边界回落到默认 NoFlux（等效绝缘）
#   （FluxBoundary 的属性只有 FluxType/J0/kc/cb，没有直接给"零通量"的入口）
try:
    p.feature("fl1").active(False)
    print("  已停用 fl1（侧壁回落为 NoFlux）")
except Exception as exc:  # noqa: BLE001
    print("  停用 fl1 失败:", str(exc).splitlines()[-1][:80])

# ⑥ 加一阶反应
# ⚠ create(tag, type, edim) 的第三个参数是**单元维度**：
#    2D 中域 = 2，边界 = 1，点 = 0；1D 中域 = 1，边界 = 0
reac = p.feature().create("reac1", "Reactions", 2)
reac.selection().set([1])
reac.set("R_c", "-kr*c")
print(f"  reac1 selection = {list(reac.selection().entities())}")

# ⑦ 瞬态跑到稳态
# ⚠ COMSOL 的 range 语法是 range(start, **step**, stop) —— 中间是步长
st = m.study("std1")
st.feature("time").set("tlist", "range(0,500,2000)")
print(f"  tlist = {st.feature('time').getString('tlist')}")

# ⑧ 重新求解
comp.mesh("mesh1").run()
t0 = time.perf_counter()
m.sol("sol1").runAll()
print(f"  求解耗时 {time.perf_counter()-t0:.1f} s")

# ── 提取解：直接在网格节点上取值（mph.evaluate）────────────────
# 瞬态有多步解，取最后一步（稳态）—— 用 inner='last'
c_nodes = np.asarray(model.evaluate("c", "mol/m^3", inner="last"),
                     dtype=float).ravel()
x_nodes = np.asarray(model.evaluate("x", "m", inner="last"),
                     dtype=float).ravel()
y_nodes = np.asarray(model.evaluate("y", "m", inner="last"),
                     dtype=float).ravel()
print(f"\n  网格节点数 = {c_nodes.size}")
print(f"  c 范围 [{c_nodes.min():.6f}, {c_nodes.max():.6f}] mol/m³")
print(f"  x 范围 [{x_nodes.min():.4f}, {x_nodes.max():.4f}] m")
print(f"  y 范围 [{y_nodes.min():.4f}, {y_nodes.max():.4f}] m")

# ── 判断流动方向 & 提取沿程剖面 ──────────────────────────
# 沿流动方向浓度应单调下降；比较 x 与 y 方向的变化幅度
order_x = np.argsort(x_nodes)
_sx = np.array_split(c_nodes[order_x], 20)
var_x = float(np.std([s.mean() for s in _sx]))
order_y = np.argsort(y_nodes)
_sy = np.array_split(c_nodes[order_y], 20)
var_y = float(np.std([s.mean() for s in _sy]))
print(f"\n  沿 x 分段的均值离散度 = {var_x:.6f}")
print(f"  沿 y 分段的均值离散度 = {var_y:.6f}")
flow_axis = "x" if var_x > var_y else "y"
print(f"  → 浓度沿 **{flow_axis}** 变化更显著（即流动方向）")

# 沿流动方向取一条近中心的剖面
if flow_axis == "x":
    coord, along = x_nodes, c_nodes
    other = y_nodes
else:
    coord, along = y_nodes, c_nodes
    other = x_nodes
mid = 0.5 * (other.min() + other.max())
sel = np.abs(other - mid) < 0.15 * (other.max() - other.min())
cs = np.argsort(coord[sel])
prof_coord = coord[sel][cs]
prof_c = along[sel][cs]
# 按坐标分箱平均（网格节点在中心线附近可能有多列）
bins = np.linspace(prof_coord.min(), prof_coord.max(), 41)
idx = np.digitize(prof_coord, bins)
bx, by = [], []
for k in range(1, len(bins)):
    msk = idx == k
    if msk.any():
        bx.append(0.5 * (bins[k - 1] + bins[k]))
        by.append(float(prof_c[msk].mean()))
bx, by = np.array(bx), np.array(by)

print(f"\n=== COMSOL 参考解（沿 {flow_axis} 的截面平均剖面，末时刻）===")
print(f"  {'坐标(m)':>10s} {'c(mol/m³)':>12s}")
for k in range(0, len(bx), max(1, len(bx) // 10)):
    print(f"  {bx[k]:10.4f} {by[k]:12.6f}")
print(f"  入口端 c = {by[0]:.6f}   出口端 c = {by[-1]:.6f}")

# 保存供后续与 fastsim 对比
np.savez(Path(__file__).resolve().parent / "comsol_reference.npz",
         coord=bx, c=by, axis=flow_axis,
         L=float(bx[-1] - bx[0]), c_in=float(by[0]))
print(f"\n  已保存 bench/comsol_reference.npz")
print(f"  工况：u0={U0} m/s, D={DCOEF} m²/s, k={KRATE:.4g} 1/s, L={L_CH} m")
print(f"        Pe = uL/D = {U0*L_CH/DCOEF:.4g}，Da = kL/u = {KRATE*L_CH/U0:.4g}")
print(f"        入口口径 = Dirichlet（Concentration c0），出口 = Outflow（零梯度）")
