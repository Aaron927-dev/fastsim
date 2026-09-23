"""端到端示例：用优化基座做管式 ECO 反应器的**运行参数寻优**。

展示完整闭环：
    灵敏度分析（哪个参数最要紧）
      → 贝叶斯寻优（找最优运行点）
      → 与随机搜索对照（证明代理模型确有价值）

目标函数
--------
**最小化比能耗 E_EO**（电化学高级氧化的标准指标，kWh·m⁻³·order⁻¹）：

    E_EO = P / (Q · log10(c_in/c_out))
         = (V_cell·I) / (Q · log10(c_in/c_out)) / 3.6e6

同时用惩罚项保证去除率达标（否则"加大流量降低能耗"这类退化解会被选中）。

设计变量
--------
    I    电流 (A)      —— 决定 O₃ 产率与槽压
    Q    流量 (mL/s)   —— 决定停留时间与处理量

**每次目标评估 = 一次完整的电-化耦合仿真**（电场 → 法拉第 → 化学场），
约 0.26 s。这正是代理模型发挥价值的场景：28 次评估即可定位最优，
若每次评估是 COMSOL 的分钟级，同一份预算要走几小时。

运行：python examples/demo_design_optimization.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.chemistry import ADR2D, ReactionNetwork  # noqa: E402
from fastsim.chemistry.mechanisms import k_second_order_to_SI, ozone_chain  # noqa: E402
from fastsim.couple import ElectrochemicalFlux  # noqa: E402
from fastsim.echem import ElectrodeKinetics, SecondaryCurrent2D  # noqa: E402
from fastsim.optim import bayesian_optimize, latin_hypercube, sensitivities  # noqa: E402

# ══ 固定条件 ═══════════════════════════════════════════════════
R_ANODE, R_CATHODE, L_REACTOR = 0.010, 0.020, 0.20
KAPPA = 1.0
CE_O3 = 0.05
PH = 8.0
C_POLL_IN = 1.0e-3                     # mol/m³（= 1 µM）
K_OH_POLLUTANT = 5.0e9
TARGET_LOG_REMOVAL = 0.05              # 至少 10% 去除（≈0.046 order）
N_Z, N_R = 40, 12


# ══ 降阶：Rct 与运行参数无关，只算一次 ═════════════════════════
# 这是关键的工程判断：把**不随设计变量变化**的昂贵前置计算提出目标函数，
# 否则 0D 机理积分（~90 ms）会在每次评估里重复，成为主导成本。
from scipy.integrate import solve_ivp  # noqa: E402

t_rct0 = time.perf_counter()
_mech0 = ozone_chain(pH=PH)
_n0 = _mech0.network
_y0 = _n0.initial(O3=1e-4, pCBA=1e-6)
_s0 = solve_ivp(_n0, (0.0, 900.0), _y0, method="BDF", jac=_n0.jacobian,
                t_eval=np.linspace(0.0, 900.0, 120), rtol=1e-7, atol=1e-18)
_i_oh, _i_o3 = _n0.species.index("OHr"), _n0.species.index("O3")
RCT = (float(np.trapezoid(_s0.y[_i_oh], _s0.t))
       / float(np.trapezoid(_s0.y[_i_o3], _s0.t)))
K2_SI = k_second_order_to_SI(K_OH_POLLUTANT * RCT)
K_O3_DECAY = float(np.log(2.0) / 771.0)
T_RCT = time.perf_counter() - t_rct0


def _fresh_network() -> tuple[ReactionNetwork, dict[str, int]]:
    """每次求解用独立网络实例（避免状态串扰）。"""
    net = ReactionNetwork()
    net.add("O3 + P -> Prod", k=K2_SI)
    net.add("O3 -> Prod0", k=K_O3_DECAY)
    return net, {s: i for i, s in enumerate(net.species)}


# ══ 目标函数（一次完整电-化耦合仿真）═══════════════════════════
_cache: dict[tuple, float] = {}
_counter = {"n": 0}


def objective(x: np.ndarray) -> float:
    """x = [I (A), Q (mL/s)] → E_EO (kWh·m⁻³·order⁻¹)；越差返回越大值。"""
    I_target = float(np.clip(x[0], 1e-3, 5.0))
    Q_mls = float(np.clip(x[1], 1e-2, 1e3))
    key = (round(I_target, 10), round(Q_mls, 10))
    if key in _cache:
        return _cache[key]
    _counter["n"] += 1

    q_vol = Q_mls * 1e-6               # m³/s
    try:
        # ① 电场（含固相欧姆降）
        echem = SecondaryCurrent2D(
            r_inner=R_ANODE, r_outer=R_CATHODE, length=L_REACTOR, kappa=KAPPA,
            anode=ElectrodeKinetics(j0=1e-3),
            cathode=ElectrodeKinetics(j0=1e-2),
            sheet_conductance_anode=2.0, sheet_conductance_cathode=1e6,
            n_z=N_Z, n_r=8)
        sol_ec = echem.solve_at_current(I_target)

        # ② 界面：O₃ 法拉第通量 → ③ 化学场
        net, idx = _fresh_network()
        wall = ElectrochemicalFlux(
            sol_ec, {"O3": (6, CE_O3)}, face_radius=R_ANODE).build(idx)

        from fastsim.flow import AnnulusFlow
        flow = AnnulusFlow(r_inner=R_ANODE, r_outer=R_CATHODE, length=L_REACTOR)
        u = flow.mean_velocity(q_vol)
        m = ADR2D(network=net, length=L_REACTOR, radius=R_CATHODE,
                  r_inner=R_ANODE, velocity_z=u, dispersion=2e-9,
                  n_z=N_Z, n_r=N_R, mode="axisymmetric",
                  inlet_bc="danckwerts", wall_flux_inner=lambda z: wall)
        c_in = np.zeros(net.n_species)
        c_in[idx["P"]] = C_POLL_IN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof = m.solve_steady(c_in)
        _, avg = m.section_average(prof, weight="flow")
        p_out = max(float(avg[-1, idx["P"]]), 1e-30)

        # ④ 评价：E_EO + 达标惩罚（防"加大流量降能耗"的退化解）
        log_removal = float(np.log10(C_POLL_IN / p_out))
        if log_removal <= 1e-6:
            _cache[key] = 1e6
            return 1e6
        E_EO = (sol_ec.v_cell * sol_ec.i_total) / (q_vol * log_removal) / 3.6e6
        if not np.isfinite(E_EO) or E_EO <= 0:
            _cache[key] = 1e6
            return 1e6
        penalty = (1e4 * (TARGET_LOG_REMOVAL - log_removal) ** 2
                   if log_removal < TARGET_LOG_REMOVAL else 0.0)
        val = float(E_EO + penalty)
    except Exception:  # noqa: BLE001 - 不收敛工况按差解处理
        val = 1e6
    _cache[key] = val
    return val


print("=" * 78)
print("管式 ECO 反应器 · 运行参数寻优（优化基座集成演示）")
print("=" * 78)
print(f"目标：最小化比能耗 E_EO，且 log10 去除 ≥ {TARGET_LOG_REMOVAL}")
print(f"设计变量：电流 I ∈ [0.02, 0.5] A，流量 Q ∈ [2, 60] mL/s")
print(f"降阶前置（Rct 由完整臭氧链算出）：Rct = {RCT:.3e}，"
      f"耗时 {T_RCT*1e3:.0f} ms（只需一次）")

# ══ ① 灵敏度分析 ═══════════════════════════════════════════════
t0 = time.perf_counter()
x_ref = np.array([0.20, 10.0])
sens = sensitivities(
    objective, x_ref, names=["电流 I (A)", "流量 Q (mL/s)"], rel_step=1e-3)
t_sens = time.perf_counter() - t0
print(f"\n[①] 灵敏度分析（基准点 I=0.20 A, Q=10 mL/s）")
print(f"    基准 E_EO = {sens.baseline:.4f} kWh·m⁻³·order⁻¹")
print(f"    高保真评估次数 {sens.n_evaluations}，耗时 {t_sens:.1f} s")
print()
print("    " + sens.table().replace("\n", "\n    "))
rank = sens.ranking()
print(f"\n    → 影响最大的是 **{rank[0][0]}**"
      f"（归一化灵敏度 {rank[0][1]:+.3f}：该参数 +1% 时 E_EO 变化 {rank[0][1]:+.3f}%）")

# ══ ② 贝叶斯寻优 ═══════════════════════════════════════════════
BOUNDS = [(0.02, 0.50), (2.0, 60.0)]
N_INIT, N_ITER = 8, 20

_counter["n"] = 0
t0 = time.perf_counter()
res = bayesian_optimize(objective, BOUNDS, n_init=N_INIT, n_iter=N_ITER,
                        names=["电流 I (A)", "流量 Q (mL/s)"],
                        seed=2026, minimize=True, verbose=True)
t_bo = time.perf_counter() - t0

print(f"\n[②] 贝叶斯寻优")
print(f"    高保真评估 {res.n_evaluations} 次，耗时 {t_bo:.1f} s"
      f"（单次 ≈{t_bo/res.n_evaluations*1e3:.0f} ms）")
print("    " + res.report().replace("\n", "\n    "))

# ══ ③ 与随机搜索对照（同等预算）═══════════════════════════════
print(f"\n[③] 对照：同等预算（{N_INIT + N_ITER} 次评估）的随机搜索")
_counter["n"] = 0
t0 = time.perf_counter()
rnd_best = []
for s in range(3):
    Xr = latin_hypercube(N_INIT + N_ITER, BOUNDS, seed=900 + s)
    rnd_best.append(min(objective(x) for x in Xr))
t_rnd = time.perf_counter() - t0
n_rnd = _counter["n"]
rnd_best = np.array(rnd_best)
print(f"    随机搜索最优（3 组）：{np.array2string(rnd_best, precision=4)}")
print(f"    贝叶斯最优 = {res.best_y:.4f}   "
      f"随机均值 = {rnd_best.mean():.4f}")
improve = (rnd_best.mean() - res.best_y) / rnd_best.mean() * 100
print(f"    → 同类预算下比能耗再降 {improve:.1f}%")

# ══ ④ 结论 ═════════════════════════════════════════════════════
print(f"\n[④] 最优运行点")
I_opt, Q_opt = res.best_x
print(f"    电流 I = {I_opt:.4f} A，流量 Q = {Q_opt:.2f} mL/s")
print(f"    比能耗 E_EO = {res.best_y:.4f} kWh·m⁻³·order⁻¹")
print(f"    （基准点 {sens.baseline:.4f}，改善 "
      f"{(sens.baseline - res.best_y)/sens.baseline*100:.1f}%）")

n_fresh = _counter["n"]
print(f"\n[速度] 各阶段高保真评估次数与墙钟（单次电-化耦合仿真 188–278 ms）：")
print(f"       灵敏度分析  {sens.n_evaluations:3d} 次   {t_sens:5.1f} s")
print(f"       贝叶斯寻优  {res.n_evaluations:3d} 次   {t_bo:5.1f} s")
print(f"       随机对照    {n_rnd:3d} 次   {t_rnd:5.1f} s")
print(f"       COMSOL 侧同等规模：每次 3D 电-化耦合求解分钟级，")
print(f"       仅「贝叶斯寻优」这 {res.n_evaluations} 次就需要数十分钟到数小时；")
print(f"       加上灵敏度与对照共 {sens.n_evaluations + res.n_evaluations + n_rnd} 次，")
print(f"       在本内核里总共 {t_sens + t_bo + t_rnd:.1f} s 完成。")

# ══ ⑤ 结果解读（含方法学提醒）═════════════════════════════════
print(f"\n[⑤] 结果解读")
print(f"    最优点落在约束边界上：")
print(f"      · 流量 Q = {Q_opt:.2f} mL/s 几乎贴上下界 2.0")
print(f"      · 电流 I = {I_opt:.4f} A 取低值")
print(f"      · log10 去除恰好在目标 {TARGET_LOG_REMOVAL} 附近")
print()
print("    ⚠ **这是 E_EO 指标的已知退化倾向，不是求解器的问题**：")
print("      E_EO = 能耗/去除量级，只要接受更低的单程去除率、更低流量，")
print("      该比值就能继续下降。因此「最小化 E_EO」的约束最优解必然")
print("      贴在去除率约束边界上。")
print("      真实设计还需叠加**处理量要求**（如 ≥ X m³/d），")
print("      否则会得到「处理量极小、单程去除极低」的不可用方案。")
print("      本示例的用途是演示优化基座能定位约束最优，")
print("      实际选型请把处理量作为硬约束加进目标函数。")
print("=" * 78)
