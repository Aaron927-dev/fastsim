"""fastsim ADR 求解器 vs COMSOL 对标基准。

同一个物理问题两边各解一次：

    u dC/dz = D d²C/dz² − k C       （一维对流-扩散-反应，一级）
    入口：Danckwerts（u C_in = u C(0) − D C'(0)）或 Dirichlet（C(0)=C_in）
    出口：C'(L) = 0

对比三件事：
1. 精度：fastsim / COMSOL / 解析解 三者的浓度剖面偏差
2. 耗时：模型构建 + 网格 + 求解 + 取值 的端到端墙钟时间
3. 口径：COMSOL 的 Inflow 边界到底是哪种语义 —— 用两种解析解判定，
   再让 fastsim 用同一口径对标（不靠猜，也不静默将就）

用法：python bench/bench_adr_vs_comsol.py
"""

from __future__ import annotations

import os
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

# ── 对标工况（与 tests 里通过的解析解标定工况一致）──────────────
L, U0, DCOEF, KRATE, CIN = 0.20, 0.01, 5e-5, 0.05, 1.0
N_CELLS = 400


def new_network() -> ReactionNetwork:
    net = ReactionNetwork()
    net.add("A -> B", k=KRATE)
    return net


def run_fastsim(inlet_bc: str) -> tuple[np.ndarray, np.ndarray, float]:
    net = new_network()
    m = ADR1D(
        network=net, length=L, velocity=U0, dispersion=DCOEF,
        n_cells=N_CELLS, inlet_bc=inlet_bc,
    )
    c_in = np.array([CIN, 0.0])
    t0 = time.perf_counter()
    prof = m.solve_steady(c_in)
    dt = time.perf_counter() - t0
    return m.z, prof[:, net.species.index("A")], dt


def run_comsol():
    """建 COMSOL 模型并求解。返回 (x_nodes, c_profile, t_build, t_solve, meta)。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench/_comsol_env.py
    from _comsol_env import setup_comsol

    setup_comsol()
    import mph

    client = mph.start(cores=2)
    t_build0 = time.perf_counter()

    model = client.create("bench_adr")
    m = model.java
    comp = m.component().create("comp1", True)

    for tag, expr in (
        ("L", f"{L}[m]"),
        ("u0", f"{U0}[m/s]"),
        ("Dc", f"{DCOEF}[m^2/s]"),
        ("kr", f"{KRATE}[1/s]"),
        ("cin", f"{CIN}[mol/m^3]"),
    ):
        m.param().set(tag, expr)

    g = comp.geom().create("geom1", 1)
    g.create("i1", "Interval").set("coord", [0.0, L])
    g.run()

    p = comp.physics().create("tds", "DilutedSpecies", "geom1")

    # 传递属性：速度场（COMSOL 内部按 3 维存储）+ 扩散系数
    cdm = p.feature("cdm1")
    cdm.set("u_src", "userdef")
    cdm.set("u", [f"{U0}[m/s]", "0", "0"])
    cdm.set("DiffusionCoefficientSource", "chem")   # chem = 用户自定义 D_c
    cdm.set("D_c", f"{DCOEF}[m^2/s]")

    # 反应项：R_c = −kr·c
    reac = p.feature().create("reac1", "Reactions", 1)
    reac.set("R_c", "-kr*c")

    # 边界：入口 = 1，出口 = 2
    inflow = p.feature().create("in1", "Inflow", 0)
    inflow.selection().set([1])
    inflow.set("c0", "cin")                    # 属性名是 c0
    bc_type = inflow.getString("BoundaryConditionType")

    outflow = p.feature().create("out1", "Outflow", 0)
    outflow.selection().set([2])

    # 网格：对齐 fastsim 的单元数
    mesh = comp.mesh().create("mesh1")
    try:
        mesh.feature("size").set("hmax", L / N_CELLS)
    except Exception:  # noqa: BLE001
        pass
    mesh.run()

    t_build = time.perf_counter() - t_build0

    # 求解：必须「显式建 Stationary 步 → activate 物理场 → study.run()」。
    # 踩过的坑（记录下来避免重复）：
    #   · createAutoSequences("tds") 在本环境不产生任何研究步 → 方程根本没求解，
    #     读回的"解"其实是初值（表现为 c≡0）
    #   · sol.runAll() 之后 mph.evaluate 仍报 "solution has not been computed"
    # 故取值改走 Java 的 Eval 节点。
    m.study().create("std1")
    step = m.study("std1").create("stat", "Stationary")
    step.activate("tds", True)

    t_solve0 = time.perf_counter()
    m.study("std1").run()
    t_solve = time.perf_counter() - t_solve0

    def java_eval(expr: str) -> np.ndarray:
        """取全场节点值。

        注意：getReal() 返回的是「每个求值点一个列表」，即 [[v1],[v2],...]，
        取 out[0] 只会拿到**第一个点** —— 必须展平全部点。
        """
        num = m.result().numerical().create(f"ev_{expr}", "Eval")
        num.set("expr", expr)
        num.set("data", "dset1")
        out = num.getReal()
        return np.array([np.asarray(col, dtype=float).ravel() for col in out]).ravel()

    xs = java_eval("x")
    vals = java_eval("c")
    if xs.size != vals.size:
        raise RuntimeError(f"坐标与浓度点数不符：{xs.size} vs {vals.size}")

    meta = {
        "comsol_version": client.version,
        "n_nodes": int(xs.size),
        "inflow_bc_type": bc_type,
        "mesh_hmax": L / N_CELLS,
    }
    return xs, vals, t_build, t_solve, meta


def main() -> int:
    print("=" * 78)
    print("fastsim ADR  vs  COMSOL  对标基准")
    print("=" * 78)
    print(f"工况: L={L} m, u={U0} m/s, D={DCOEF} m²/s, k={KRATE} 1/s, C_in={CIN}")
    print(f"网格: 两侧均为 {N_CELLS} 单元\n")

    # ── COMSOL ──────────────────────────────────────────────
    try:
        xc, cc, t_build, t_solve, meta = run_comsol()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("[COMSOL] 对标未完成（fastsim 侧结论不受影响）")
        return 1

    # ── 判定 COMSOL 的入口口径：与两种解析解各自比一次 ────────
    print("── 判定 COMSOL Inflow 的物理口径 ──")
    print(f"  COMSOL 报告 BoundaryConditionType = {meta['inflow_bc_type']}")
    err_by_bc: dict[str, float] = {}
    for bc in ("danckwerts", "dirichlet"):
        ref = analytic_first_order_dispersion(xc, L, U0, DCOEF, KRATE, CIN, inlet_bc=bc)
        err_by_bc[bc] = float(np.abs(cc - ref).max() / CIN)
        print(f"  与 {bc:11s} 解析解的最大偏差: {err_by_bc[bc]:.4%}")
    bc_best = min(err_by_bc, key=lambda k: err_by_bc[k])
    print(f"  → 判定 COMSOL 采用: {bc_best}"
          f"（偏差 {err_by_bc[bc_best]:.4%}，另一种 {err_by_bc['danckwerts' if bc_best=='dirichlet' else 'dirichlet']:.4%}）\n")

    # ── fastsim：用同一口径 ──────────────────────────────────
    z, cf, t_fast = run_fastsim(bc_best)
    ref_on_z = analytic_first_order_dispersion(z, L, U0, DCOEF, KRATE, CIN, inlet_bc=bc_best)
    err_fast = float(np.abs(cf - ref_on_z).max() / CIN)

    print("── 精度 ──")
    print(f"  fastsim  与解析解最大偏差 {err_fast:.4%}")
    print(f"  COMSOL   与解析解最大偏差 {err_by_bc[bc_best]:.4%}")
    lo, hi = z[0], z[-1]
    sel = (xc >= lo) & (xc <= hi)          # 只在 fastsim 有效区间内插值，避免端点外推
    err_cross = float(np.abs(np.interp(xc[sel], z, cf) - cc[sel]).max() / CIN)
    print(f"  两者最大互差            {err_cross:.4%}\n")

    # ── 耗时 ────────────────────────────────────────────────
    t_total = t_build + t_solve
    print("── 耗时 ──")
    print(f"  fastsim  求解            {t_fast*1e3:8.2f} ms")
    print(f"  COMSOL   模型构建        {t_build:8.2f} s")
    print(f"  COMSOL   求解            {t_solve:8.2f} s")
    print(f"  COMSOL   端到端          {t_total:8.2f} s  （网格节点 {meta['n_nodes']}）")
    print(f"  → 端到端加速比 {t_total / t_fast:,.0f} ×   纯求解加速比 {t_solve / t_fast:,.0f} ×\n")

    print("── 出口浓度三方对照 ──")
    print(f"  解析解   {ref_on_z[-1]:.6f}")
    print(f"  fastsim  {cf[-1]:.6f}")
    print(f"  COMSOL   {cc[-1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
