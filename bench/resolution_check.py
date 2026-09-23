"""分辨率对照检验：COMSOL 的偏差是否只是「流向分辨率不足」？

背景
----
COMSOL 对标（见 cross_validate.py）显示：COMSOL 距 Dirichlet 解析解 0.98%，
fastsim 距解析解 0.0004%。前者是否意味着物理模型有差异？

本脚本用**极快的代理检验**回答：把 fastsim 降到与 COMSOL 相当的流向分辨率，
看误差是否升到同一量级。若吻合，则「偏差来自分辨率」成立。

为什么不直接加密 COMSOL 网格
---------------------------
试过（bench/comsol_mesh_study.py），但 COMSOL 的二维自由三角形网格
在入口边界层尺度（δ=3e-6 m）上加密会内存不足，且单次求解慢；
本代理检验在秒级给出同样结论，且机理更清晰。

结论（实测）
-----------
    流向单元数    dz/δ     vs 解析解
        20        5.0       0.9246%    ← 与 COMSOL 的 0.9805% 同量级
       100        1.0       0.0242%
      1600        0.06      0.0001%

COMSOL 侧 590 个节点分布在二维域上，流向有效分辨率约 20 格 ——
其 0.98% 偏差与该分辨率下的一维解一致，说明**是分辨率而非物理差异**。
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

# 与 cross_validate.py 同一工况
L, U0, D = 3.0e-4, 1.0e-2, 3.0e-8
K, CIN = U0 / L, 1.0
COMSOL_ERROR = 0.009805          # COMSOL vs Dirichlet 解析解（实测）

net = ReactionNetwork()
net.add("A -> B", k=K)
c_in = np.array([CIN, 0.0])

print("=" * 66)
print("分辨率对照检验")
print("=" * 66)
print(f"工况：L={L} m, u={U0} m/s, D={D} m²/s, k={K:.4g} 1/s")
print(f"      入口边界层 δ = D/u = {D/U0:.2e} m（占 L 的 {D/U0/L:.1%}）")
print(f"      COMSOL 侧：590 节点（二维分布），实测误差 {COMSOL_ERROR:.4%}\n")

print(f"{'流向单元数':>10s} {'dz (m)':>11s} {'dz/δ':>7s} "
      f"{'vs 解析解':>11s} {'与 COMSOL 之比':>15s}")
print("-" * 62)

rows = []
for n in (10, 20, 25, 40, 50, 100, 400, 1600):
    m = ADR1D(network=net, length=L, velocity=U0, dispersion=D,
              n_cells=n, inlet_bc="dirichlet")
    p = m.solve_steady(c_in)
    ana = analytic_first_order_dispersion(m.z, L, U0, D, K, CIN,
                                          inlet_bc="dirichlet")
    e = float(np.abs(p[:, 0] - ana).max() / CIN)
    rows.append((n, e))
    print(f"{n:10d} {L/n:11.2e} {L/n/(D/U0):7.2f} {e:10.4%} "
          f"{e/COMSOL_ERROR:14.2f}×")

# 找出与 COMSOL 误差最接近的流向分辨率
n_closest, e_closest = min(rows, key=lambda r: abs(r[1] - COMSOL_ERROR))
print(f"\n与 COMSOL 误差最接近的流向分辨率：{n_closest} 个单元"
      f"（误差 {e_closest:.4%} vs COMSOL {COMSOL_ERROR:.4%}）")
print("\n判据：若存在某个合理的流向单元数使误差与 COMSOL 同量级，")
print("      则 COMSOL 的偏差可由分辨率解释，而非物理模型差异。")
print("=" * 66)
