"""端到端示例：管式反应器 + 壁面电化学反应（二维轴对称）。

展示「二维」相对一维的价值，并演示**传质极限（极限电流）**判据：
- 一维模型只给截面平均浓度，**看不到壁面附近的浓度边界层**
- 电极反应发生在壁面，其速率上限由传质能力决定；壁面浓度是判据，
  只有二维能算出

同时演示求解器的**负浓度护栏**：当壁面通量超过极限电流时，
解会出现负浓度（非物理），求解器会发出警告而不是静默返回。

运行：python examples/demo_tubular_electrode.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.chemistry import ADR1D, ADR2D, ReactionNetwork  # noqa: E402
from fastsim.flow import mass_transfer_coefficient, sherwood_graetz  # noqa: E402

# ── 工况 ────────────────────────────────────────────────────
R_TUBE, L_TUBE = 5.0e-3, 0.20        # m
U_MEAN = 0.01                        # m/s
D_MOL = 1.0e-9                       # m²/s
C_IN = 1.0                           # mol/m³
NU_W = 8.9e-7                        # m²/s，25 °C 水的运动黏度

N_Z, N_R = 120, 36

print("=" * 76)
print("管式反应器 + 壁面电极反应（二维轴对称）")
print("=" * 76)

# ── 1. 几何与流场 ───────────────────────────────────────────
r_centers = (np.arange(N_R) + 0.5) * (R_TUBE / N_R)
u_profile = 2.0 * U_MEAN * (1.0 - (r_centers / R_TUBE) ** 2)
tau = L_TUBE / U_MEAN

print(f"\n[1] 几何与流场")
print(f"    管径 Ø{2*R_TUBE*1e3:.0f} mm，长 {L_TUBE*1e3:.0f} mm，停留时间 τ = {tau:.1f} s")
print(f"    平均流速 {U_MEAN*1e3:.1f} mm/s，峰值 {u_profile.max()*1e3:.1f} mm/s"
      f"（抛物线，峰值/均值 = {u_profile.max()/U_MEAN:.2f}）")
print(f"    径向扩散时间 R²/D = {R_TUBE**2/D_MOL:,.0f} s ≫ τ → 边界层未充分发展（二维必需）")

# ── 2. 传质极限估算（Graetz 关联式）──────────────────────────
d_h = 2.0 * R_TUBE
re = U_MEAN * d_h / NU_W
sc = NU_W / D_MOL
sh = sherwood_graetz(re, sc, d_h, L_TUBE)
k_l = mass_transfer_coefficient(sh, D_MOL, d_h)
j_lim = k_l * C_IN

print(f"\n[2] 传质极限估算（Graetz 入口段关联式）")
print(f"    Re = {re:,.1f}，Sc = {sc:,.0f}，Re·Sc·d/L = {re*sc*d_h/L_TUBE:,.0f}")
print(f"    Sh = {sh:.2f} → 传质系数 k_L = {k_l*1e6:.3f} µm/s")
print(f"    **极限电流通量 j_lim = k_L·c_bulk = {j_lim:.3e} mol·m⁻²·s⁻¹**")
print(f"    （关联式按均匀壁面条件给出平均值，是工程用量级判据）")

# ── 3. 二维求解（亚极限通量）────────────────────────────────
net = ReactionNetwork()
net.add("A -> B", k=0.0)

J_USE = 0.5 * j_lim
J_USE = 0.5 * j_lim


def wall_flux_pair(j: float):
    """按物质给壁面通量：A 被消耗，B 按电极反应计量生成。

    ⚠ 不要返回 (nz,) 形状 —— 它会被广播到**所有**物质，导致产物 B 也
    被"消耗"而变成负浓度（这是本项目踩过的一个接口陷阱）。
    """
    def f(z: np.ndarray) -> np.ndarray:
        out = np.zeros((z.size, 2))
        out[:, 0] = -j      # A：阴极还原消耗
        out[:, 1] = +j      # B：同计量生成
        return out

    return f


m2 = ADR2D(network=net, length=L_TUBE, radius=R_TUBE,
           velocity_z=u_profile, velocity_r=0.0, dispersion=D_MOL,
           n_z=N_Z, n_r=N_R, mode="axisymmetric", inlet_bc="danckwerts",
           wall_flux=wall_flux_pair(J_USE))
c_in = np.array([C_IN, 0.0])

print(f"\n[3] 二维求解（{N_Z}×{N_R} = {N_Z*N_R} 单元）")
print(f"    取 j = {J_USE/j_lim:.1f} × j_lim = {J_USE:.3e}（亚极限工况）")
print(f"    速度场无散性偏差 = {m2.divergence_free_violation():.2e}（应 < 1e-6）")
t0 = time.perf_counter()
prof2 = m2.solve_steady(c_in)
dt2 = time.perf_counter() - t0
print(f"    耗时 {dt2*1e3:.1f} ms")

cA = prof2[:, :, 0]
_, avg = m2.section_average(prof2, weight="flow")
in_flow = m2.molar_flow_in(c_in)[0]
out_flow = m2.molar_flow_out(prof2)[0]
removed = in_flow - out_flow
area_wall = 2.0 * np.pi * R_TUBE * L_TUBE
expect = J_USE * area_wall

print(f"\n[4] 结果")
print(f"    截面平均出口浓度（流量加权） C_out = {avg[-1, 0]:.5f}"
      f"，总脱除率 {removed/in_flow:.2%}")
print(f"    壁面通量核算：j·A_wall = {expect:.4e} mol/s，"
      f"与脱除量偏差 {abs(removed-expect)/expect:.2e}")
print(f"    A 最小浓度 = {cA.min():.5f}（应为正）"
      f"；B 最大浓度 = {prof2[:,:,1].max():.5f}（产物，应等于脱除量/体积流量）")
print(f"    A+B 总量守恒（无本体反应，仅界面转化）："
      f"偏差 {abs((prof2[:,:,0]+prof2[:,:,1]).max()-(prof2[:,:,0]+prof2[:,:,1]).min()):.2e}")

# ── 5. 通量扫描：暴露传质极限 ───────────────────────────────
print(f"\n[5] 通量扫描：传质极限在哪里？")
print(f"    {'j / j_lim':>10s} {'A 最小浓度':>12s} {'壁面/中心':>10s} {'物理性':>10s}")
for ratio in (0.2, 0.5, 0.8, 1.0, 1.3, 1.6):
    j = ratio * j_lim
    mm = ADR2D(network=net, length=L_TUBE, radius=R_TUBE,
               velocity_z=u_profile, velocity_r=0.0, dispersion=D_MOL,
               n_z=N_Z, n_r=N_R, mode="axisymmetric", inlet_bc="danckwerts",
               wall_flux=wall_flux_pair(j))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pp = mm.solve_steady(c_in)
    cA_p = pp[:, :, 0]
    cmin = float(cA_p.min())
    ok = "正常" if cmin > -1e-6 else "⚠ 非物理"
    warned = "  (已警告)" if caught else ""
    print(f"    {ratio:10.1f} {cmin:12.5f} {cA_p[-1,-1]/cA_p[-1,0]:10.4f} {ok:>10s}{warned}")

print(f"    → j 超过 j_lim 后壁面浓度转负：**模型在告诉你传质供不上了**。")
print(f"      这正是电化学里「极限电流」的定义 —— 一维模型给不出这个判据。")

# ── 6. 一维对照 ─────────────────────────────────────────────
net1 = ReactionNetwork()
net1.add("A -> B", k=0.0)
m1 = ADR1D(network=net1, length=L_TUBE, velocity=U_MEAN,
           dispersion=D_MOL, n_cells=N_Z, inlet_bc="danckwerts")
t0 = time.perf_counter()
prof1 = m1.solve_steady(c_in)
dt1 = time.perf_counter() - t0
print(f"\n[6] 一维对照（均匀速度、无壁面通量，{N_Z} 单元）：耗时 {dt1*1e3:.2f} ms")
print(f"    一维出口 = {prof1[-1, 0]:.5f}（无壁面反应，故不衰减）")
print(f"    二维出口 = {avg[-1, 0]:.5f}  差 {abs(prof1[-1,0]-avg[-1,0])/prof1[-1,0]:.2%}")

print(f"\n[7] 径向浓度边界层（二维独有信息）")
print(f"    {'z/L':>6s} {'中心 c(r=0)':>12s} {'壁面 c(r→R)':>12s} {'壁面/中心':>10s}")
for frac in (0.05, 0.25, 0.5, 0.75, 1.0):
    i = min(int(frac * N_Z) - 1, N_Z - 1)
    print(f"    {frac:6.2f} {cA[i,0]:12.5f} {cA[i,-1]:12.5f} {cA[i,-1]/cA[i,0]:10.4f}")

print("\n" + "=" * 76)
