"""端到端示例：环隙管式反应器的一维反应-传质快速求解。

展示四基座中「流场 → 化学场」的串联：
  1. 环隙层流解析剖面给出均速（无需 CFD）
  2. Taylor-Aris 给出轴向弥散系数（把速度剖面的对流展宽算进来）
  3. ADR 求解器解稳态反应-传质（无需 FEM）
  4. 与解析解对标，并给出停留时间/转化率/流态判据

运行：python examples/demo_annulus_reactor.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.chemistry import (  # noqa: E402
    ADR1D,
    ReactionNetwork,
    analytic_first_order_dispersion,
)
from fastsim.flow import (  # noqa: E402
    TAYLOR_ARIS_TA_LIMIT,
    AnnulusFlow,
    mass_transfer_coefficient,
    sherwood_graetz,
    taylor_aris_dispersion,
    taylor_aris_number,
)

# ── 工况：管状阳极 Ø2cm × 20cm，外筒 Ø6cm，0.1 M Na₂SO₄，20 mL/s ──
R_INNER, R_OUTER, LENGTH = 0.01, 0.03, 0.20
Q_VOL = 2.0e-5          # m³/s = 20 mL/s
D_MOL = 1.0e-9          # m²/s，小分子在水中的典型扩散系数
K_RATE = 0.05           # 1/s，一级反应速率常数
C_IN = 1.0              # 归一化入口浓度

print("=" * 74)
print("环隙管式反应器 · 一维反应-传质快速求解")
print("=" * 74)

# ── 1. 流场（解析，微秒级）──────────────────────────────────
flow = AnnulusFlow(r_inner=R_INNER, r_outer=R_OUTER, length=LENGTH)
u = flow.mean_velocity(Q_VOL)
re = flow.reynolds(u)
print(f"\n[1] 流场（解析剖面，无需求解 Navier-Stokes）")
print(f"    环隙 {R_INNER*100:.1f}–{R_OUTER*100:.1f} cm，间隙 {flow.gap*100:.0f} mm，"
      f"水力直径 {flow.hydraulic_diameter*1000:.0f} mm")
print(f"    流量 {Q_VOL*1e6:.0f} mL/s → 均速 {u*1e3:.3f} mm/s，"
      f"Re = {re:.1f}（{flow.regime(u)}）")
print(f"    空塔停留时间 τ = {flow.residence_time(u):.1f} s")
print(f"    峰值流速 / 均速 = {flow.max_velocity(u)/u:.3f}"
      f"（平行板极限 1.5，圆管极限 2.0）")

# ── 2. 传质与弥散 ───────────────────────────────────────────
d_char = flow.hydraulic_diameter
sc = 8.90e-4 / 997.0 / D_MOL
sh = sherwood_graetz(re, sc, d_char, LENGTH)
k_l = mass_transfer_coefficient(sh, D_MOL, d_char)

# Taylor-Aris 有适用域：Ta = u·a²/(D·L) 必须 ≪ 1（径向扩散远快于停留时间）。
# 本工况 a = 2 cm、D = 1e-9 → Ta 极大，公式会给出非物理的大弥散系数，
# 因此这里走"降级处理 + 显式标注"，而不是静默取值。
a_char = R_OUTER - R_INNER
ta = taylor_aris_number(LENGTH, u, a_char, D_MOL)
in_taylor = ta <= TAYLOR_ARIS_TA_LIMIT
print(f"\n[2] 传质与弥散")
print(f"    Sc = {sc:,.0f}，Graetz 数 = {re*sc*d_char/LENGTH:,.1f} → Sh = {sh:.2f}")
print(f"    传质系数 k_L = {k_l*1e6:.3f} µm/s")
print(f"    Taylor 数 Ta = u·a²/(D·L) = {ta:,.1f}（适用域要求 ≤ {TAYLOR_ARIS_TA_LIMIT}）")
if in_taylor:
    d_ax = taylor_aris_dispersion(a_char, u, D_MOL, LENGTH)
    print(f"    在 Taylor 区 → D_ax = {d_ax:.3e} m²/s（分子扩散的 {d_ax/D_MOL:.0f} 倍）")
    disp_note = "Taylor-Aris"
else:
    # 降级：改用分子扩散作下限，并明确标注模型局限
    d_ax = D_MOL
    print(f"    ⚠ **不在 Taylor 区**：径向扩散时间 a²/D = {a_char**2/D_MOL:,.0f} s，"
          f"远长于停留时间 {flow.residence_time(u):.1f} s")
    print(f"      → 一维弥散模型本身不成立（浓度剖面在径向上未抹平）")
    print(f"      → 降级处理：取分子扩散 D_mol = {D_MOL:.1e} m²/s 作为**下限**")
    print(f"      → 严谨做法是求解 2D 对流-扩散；下方结果按「近平推流」解读")
    disp_note = "分子扩散（降级）"
pe = u * LENGTH / d_ax
print(f"    Peclet 数 Pe = {pe:,.1f} → {'近平推流' if pe > 50 else '显著返混'}（弥散口径：{disp_note}）")

# ── 3. 反应-传质求解（数值，毫秒级）─────────────────────────
net = ReactionNetwork()
net.add("A -> B", k=K_RATE)
model = ADR1D(network=net, length=LENGTH, velocity=u,
              dispersion=d_ax, n_cells=400)
t0 = time.perf_counter()
prof = model.solve_steady(np.array([C_IN, 0.0]))
dt = time.perf_counter() - t0
cA = prof[:, net.species.index("A")]

print(f"\n[3] 反应-传质求解（{model.n_cells} 单元，{net.n_species} 物质）")
print(f"    耗时 {dt*1e3:.2f} ms")
print(f"    出口浓度 C_A(L) = {cA[-1]:.6f}，转化率 = {model.conversion(prof, 'A', C_IN):.4%}")

# ── 4. 与解析解对标 ─────────────────────────────────────────
ref = analytic_first_order_dispersion(model.z, LENGTH, u, d_ax, K_RATE, C_IN,
                                      inlet_bc="danckwerts")
err = np.abs(cA - ref).max() / C_IN
print(f"\n[4] 对标解析解（轴向扩散模型 + Danckwerts 边界）")
print(f"    剖面最大偏差 {err:.4%}  {'✓' if err < 0.02 else '✗ 超出容差'}")

# ── 5. 设计洞见 ─────────────────────────────────────────────
tau = flow.residence_time(u)
da = K_RATE * tau
print(f"\n[5] 设计洞见")
print(f"    Damköhler 数 Da = kτ = {da:.3f}")
print(f"    · Pe = {pe:,.0f} 说明该流速下接近平推流，轴向返混不是瓶颈")
print(f"    · 要提升转化率，优先延长停留时间（降流量）或提高 k，"
      f"而非改善轴向混合")
print(f"    · 若 Da ≪ 1（当前 Da = {da:.2f}），单级转化率上限约 "
      f"{1-np.exp(-da):.1%}，需串联多级或延长反应段")

print("\n" + "=" * 74)
