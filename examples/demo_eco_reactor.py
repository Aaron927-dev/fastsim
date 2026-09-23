"""端到端闭环示例：管式电催化臭氧反应器（电-化耦合）。

完整链路
--------
    ① 电场        SecondaryCurrent2D：内管阳极 + 外筒阴极的环隙
                  → 局部电流密度 j(z)，考虑电极动力学与固相欧姆降
    ② 界面        法拉第定律：O₃ 生成通量 = j·CE/(n·F)
    ③ 降阶        完整臭氧链（0D 机理库）→ Rct → 二级动力学 k₂
    ④ 化学场      ADR2D：环隙流动 + O₃/污染物的输运与反应
    ⑤ 评价        降解率、比能耗 E_EO、速度

为什么需要第 ③ 步（QSSA 降阶）
-----------------------------
完整臭氧链（本仓库 `mechanisms.ozone_chain`）的速率常数跨度 1e-3 ~ 1e10，
直接放进二维场求解会导致极端刚性。环境工程的标准做法是用**准稳态近似**
消去自由基：

    d[S]/dt = −k_{·OH,S} · [·OH]ss · [S] = −(k_{·OH,S} · Rct) · [O₃] · [S]

其中 Rct = ∫[·OH]dt / ∫[O₃]dt（Elovitz & von Gunten 1999）。
本示例的 Rct **由完整机理库独立算出**，而不是拍脑袋给一个数 ——
这样降阶模型与完整机理是同一套物理。

运行：python examples/demo_eco_reactor.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.chemistry import ADR2D, ReactionNetwork  # noqa: E402
from fastsim.chemistry.mechanisms import (  # noqa: E402
    k_second_order_to_SI,
    mgL_to_M,
    ozone_chain,
)
from fastsim.couple import F_CONST, ElectrochemicalFlux  # noqa: E402
from fastsim.echem import ElectrodeKinetics, SecondaryCurrent2D  # noqa: E402
from fastsim.flow import AnnulusFlow  # noqa: E402

# ══ 工况 ═══════════════════════════════════════════════════════
R_ANODE, R_CATHODE = 0.010, 0.020      # m
L_REACTOR = 0.20                       # m
Q_VOL = 5.0e-6                         # m³/s = 5 mL/s（停留时间 ≈38 s，接近实际接触时间）
I_TARGET = 0.20                        # A
CE_O3 = 0.05                           # O₃ 电流效率（5%，工程典型偏低值）
PH = 8.0
K_OH_POLLUTANT = 5.0e9                 # M⁻¹s⁻¹，pCBA 型探针
C_POLL_M = 1.0e-6                      # M（≈ 0.16 mg/L）
C_POLL_IN = C_POLL_M * 1e3             # → mol/m³（ADR2D 的 SI 单位）
KAPPA = 1.0                            # S/m，0.1 M Na₂SO₄

# 浓度单位约定：ADR2D 用 SI 的 mol/m³（见其模块文档）。
to_uM = 1e3                            # mol/m³ → µM
def o3_mgL(c_mol_m3: float) -> float:
    """mol/m³ → mg/L。因为 1 g/m³ = 1 mg/L，故直接乘摩尔质量 (g/mol)。"""
    return c_mol_m3 * 48.0
print("=" * 78)
print("管式电催化臭氧反应器 · 电-化耦合闭环仿真")
print("=" * 78)

# ══ ① 电场：电流分布 ═══════════════════════════════════════════
t0 = time.perf_counter()
echem = SecondaryCurrent2D(
    r_inner=R_ANODE, r_outer=R_CATHODE, length=L_REACTOR, kappa=KAPPA,
    anode=ElectrodeKinetics(j0=1e-3),        # 钛基氧化物阳极（OER 慢动力学）
    cathode=ElectrodeKinetics(j0=1e-2),      # 不锈钢阴极
    sheet_conductance_anode=2.0,             # 多孔钛壁固相电导（S·m）
    sheet_conductance_cathode=1e6,           # 阴极理想导体
    n_z=60, n_r=10)
sol_ec = echem.solve_at_current(I_TARGET)
t_ec = time.perf_counter() - t0

area_anode = 2 * np.pi * R_ANODE * L_REACTOR
print(f"\n[①] 电场（环隙二次分布 + 固相欧姆降）")
print(f"    几何：阳极 Ø{2*R_ANODE*1e3:.0f} mm × {L_REACTOR*1e3:.0f} mm，"
      f"阴极 Ø{2*R_CATHODE*1e3:.0f} mm，间隙 {(R_CATHODE-R_ANODE)*1e3:.0f} mm")
print(f"    总电流 {sol_ec.i_total:.4f} A，槽压 {sol_ec.v_cell:.3f} V，"
      f"表观槽阻 {sol_ec.apparent_resistance:.3f} Ω")
print(f"    阳极面积 {area_anode*1e4:.2f} cm²，平均 j = "
      f"{sol_ec.i_total/area_anode:.2f} A/m²")
print(f"    电流分布：馈电端/远端 = {sol_ec.feed_end_ratio:.3f}"
      f"（>1 来自固相欧姆降），CV = {sol_ec.j_anode_cv:.3f}")
print(f"    固相电位降 {sol_ec.phi_s_anode[0]-sol_ec.phi_s_anode[-1]:.4f} V")
print(f"    耗时 {t_ec*1e3:.1f} ms")

# ══ ② 界面：法拉第通量 ═════════════════════════════════════════
# 注意 1：反应网络必须在构造 wall 通量**之前**定稿 —— 后续 add() 会引入新物种，
#         使壁面通量数组的列数与网络不匹配（易错点，显式说明）。
# 注意 2：**污染物不设直接阳极氧化通量**。原因：污染物体相浓度仅 1e-6 M，
#         其直接氧化的速率受**传质控制**，上限约 k_L·c_bulk ≈ 4.5e-12 mol·m⁻²·s⁻¹
#         （k_L 由 Graetz 关联式估出）。若强行按电流效率给固定通量
#         （例如按 2% 效率 → 1.65e-6 mol·m⁻²·s⁻¹），会比传质极限高 5 个数量级，
#         壁面浓度被拉成负值、求解严重受阻 —— 这是本项目实测踩过的坑。
#         正确的做法是 Robin 边界（通量 = k_L·c），超出当前固定通量接口的能力，
#         故本示例以**间接氧化（O₃/·OH）为主路径**（这也是 ECO 的实际主导机制）。
net = ReactionNetwork()
net.add("O3 + P -> Prod", k=1.0)          # 二级（·OH 贡献），稍后覆盖
net.add("O3 -> Prod0", k=1e-3)            # O₃ 自身一级衰减，稍后覆盖
idx = {s: i for i, s in enumerate(net.species)}
ecf = ElectrochemicalFlux(
    current_solution=sol_ec,
    species_map={"O3": (6, CE_O3)},       # 3H₂O → O₃ + 6H⁺ + 6e⁻
    face_radius=R_ANODE,
)
wall = ecf.build(idx)
rates = ecf.total_rates(idx)

print(f"\n[②] 界面（法拉第定律）")
print(f"    O₃ 生成：CE = {CE_O3:.0%}，n = 6 → {rates['O3']:.3e} mol/s")
print(f"    O₃ 通量范围 [{wall[:,idx['O3']].min():.3e}, "
      f"{wall[:,idx['O3']].max():.3e}] mol·m⁻²·s⁻¹")
print(f"    （污染物走间接氧化路径；其直接阳极氧化受传质控制，"
      f"上限 ≈4.5e-12 mol·m⁻²·s⁻¹，见上方注释）")

# ══ ③ 降阶：完整臭氧链 → Rct ═══════════════════════════════════
t0 = time.perf_counter()
mech = ozone_chain(pH=PH)
net0 = mech.network
y0 = net0.initial(O3=1e-4, pCBA=1e-6)
sol0 = solve_ivp(net0, (0.0, 900.0), y0, method="BDF", jac=net0.jacobian,
                 t_eval=np.linspace(0.0, 900.0, 300), rtol=1e-8, atol=1e-18)
i_oh, i_o3 = net0.species.index("OHr"), net0.species.index("O3")
int_oh = float(np.trapezoid(sol0.y[i_oh], sol0.t))
int_o3 = float(np.trapezoid(sol0.y[i_o3], sol0.t))
rct = int_oh / int_o3
k2_M = K_OH_POLLUTANT * rct            # M⁻¹s⁻¹（文献惯例）
k2_SI = k_second_order_to_SI(k2_M)     # → m³·mol⁻¹·s⁻¹（ADR2D 的 SI 单位）
k_o3_decay = float(np.log(2.0) / 771.0)  # pH 7 下的 O₃ 一级衰减（1/s，无需换算）
t_mech = time.perf_counter() - t0

print(f"\n[③] 降阶（完整臭氧链 0D → QSSA）")
print(f"    臭氧链 {net0.n_reactions} 条基元反应（速率常数跨度 1e-3–1e10）")
print(f"    Rct = ∫[·OH]dt/∫[O₃]dt = {rct:.3e}")
print(f"    → 污染物二级速率常数 k₂ = k_{{·OH,P}}·Rct = {k2_M:.3e} M⁻¹s⁻¹")
print(f"      **单位换算**（M 基 → SI）: k₂ = {k2_SI:.4g} m³·mol⁻¹·s⁻¹"
      f"（÷1000；见 mechanisms.k_second_order_to_SI）")
print(f"    （该 Rct 与量级区间 1e-11–1e-5 相符，见 tests/test_mechanisms.py）")
print(f"    耗时 {t_mech*1e3:.0f} ms")

# ══ ④ 化学场：环隙反应器 ═══════════════════════════════════════
net.set_k(0, k2_SI)                       # SI 单位的二级速率常数
net.set_k(1, k_o3_decay)                  # O₃ 自身衰减（一级，无需换算）

flow = AnnulusFlow(r_inner=R_ANODE, r_outer=R_CATHODE, length=L_REACTOR)
u = flow.mean_velocity(Q_VOL)
tau = flow.residence_time(u)

nz_chem = sol_ec.j_anode.size
m_chem = ADR2D(
    network=net, length=L_REACTOR, radius=R_CATHODE, r_inner=R_ANODE,
    velocity_z=u, velocity_r=0.0, dispersion=2e-9,
    n_z=nz_chem, n_r=16, mode="axisymmetric", inlet_bc="danckwerts",
    wall_flux_inner=lambda z: wall,
)
c_in = np.zeros(net.n_species)
c_in[idx["P"]] = C_POLL_IN                # 入口只含污染物（O₃ 由电极原位生成）

t0 = time.perf_counter()
import warnings  # noqa: E402

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    prof = m_chem.solve_steady(c_in)
t_chem = time.perf_counter() - t0

i_o3c, i_p = net.species.index("O3"), net.species.index("P")
_, avg = m_chem.section_average(prof, weight="flow")

print(f"\n[④] 化学场（环隙流动 + 反应-传质）")
print(f"    流量 {Q_VOL*1e6:.0f} mL/s → 均速 {u*1e3:.2f} mm/s，"
      f"停留时间 τ = {tau:.1f} s，Re = {flow.reynolds(u):.0f}（{flow.regime(u)}）")
print(f"    网格 {nz_chem}×16 = {nz_chem*16} 单元，{net.n_species} 物质")
print(f"    无散性自检 {m_chem.divergence_free_violation():.1e}")
print(f"    耗时 {t_chem*1e3:.1f} ms")
if caught:
    print(f"    ⚠ 求解器告警 {len(caught)} 条：{str(caught[0].message)[:60]}...")

# ══ ⑤ 评价 ═════════════════════════════════════════════════════
p_in = C_POLL_IN
p_out = float(avg[-1, i_p])
degradation = 1.0 - p_out / p_in
o3_max = float(prof[:, :, i_o3c].max())
o3_out = float(avg[-1, i_o3c])

# 比能耗 E_EO (kWh·m⁻³·order⁻¹)：电化学高级氧化的标准指标
E_EO = (sol_ec.v_cell * sol_ec.i_total * tau) / (1e-3 * np.log(p_in / p_out)) / 3.6e6

print(f"\n[⑤] 结果")
print(f"    污染物去除率 {degradation:.2%}"
      f"（入口 {C_POLL_M*1e6:.2f} → 出口 {p_out/to_uM*1e6:.3f} µM）")
print(f"    O₃ 出口 {o3_out*to_uM:.4g} µM = {o3_mgL(o3_out):.4g} mg/L，"
      f"峰值 {o3_max*to_uM:.4g} µM = {o3_mgL(o3_max):.4g} mg/L")
print(f"    O₃ 投加量（按流量归一）= {o3_mgL(rates['O3']/Q_VOL):.4g} mg/L")
print(f"    比能耗 E_EO = {E_EO:.3f} kWh·m⁻³·order⁻¹")
print(f"    （含 O₃ 生成与固相欧姆降）")

# 质量平衡核对
in_flow = m_chem.molar_flow_in(c_in)
out_flow = m_chem.molar_flow_out(prof)
wall_o3_total = rates["O3"]
print(f"    质量核对：O₃ 壁面注入 {wall_o3_total:.3e} mol/s，"
      f"出口带出 {out_flow[i_o3c]:.3e} mol/s")
o3_consumed = wall_o3_total - out_flow[i_o3c]
print(f"             差值 {o3_consumed:.3e} mol/s 即为反应消耗 + 衰减")

# ══ 速度小结 ═══════════════════════════════════════════════════
t_total = t_ec + t_chem
print(f"\n[速度] 端到端 {t_total*1e3:.0f} ms"
      f"（电场 {t_ec*1e3:.0f} ms + 化学场 {t_chem*1e3:.0f} ms）")
print(f"       含机理降阶 {t_mech*1e3:.0f} ms（可缓存复用，不计入单次仿真）")
print(f"       —— 同等问题在 COMSOL 3D 中通常需要分钟到小时级建模与求解")

print("\n说明：本示例的 Rct 由完整臭氧链独立算出，因此降阶模型与完整机理同源；")
print("      但二维场中仍使用二级降阶动力学（完整链的刚性不适合直接进场求解）。")
print("=" * 78)
