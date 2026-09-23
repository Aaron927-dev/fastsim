"""输出 ADR2D 的 MMS 网格收敛表，用于写入文档（可复现的量化证据）。

用法：python bench/mms_convergence.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "tests"))
from fastsim.chemistry.schemes import check_bounded_scheme  # noqa: E402
from test_adr2d import _mms_error  # noqa: E402

L, R = 0.20, 0.02


def table(name: str, u0: float, ur0: float, dr_: float, dz_: float, grids,
          scheme: str = "power_law") -> None:
    print(f"\n### {name}   (u_z={u0}, u_r={ur0}, D_r={dr_}, D_z={dz_}, "
          f"格式={scheme})")
    print(f"{'网格 (nz×nr)':>14s} {'h=Δz':>10s} {'误差':>12s} {'观测阶':>8s}")
    prev_e = None
    prev_h = None
    for nz, nr in grids:
        e = _mms_error(nz, nr, u0, ur0, dr_, dz_, scheme)
        h = L / nz
        order = "" if prev_e is None else f"{np.log2(prev_e / e):.2f}"
        print(f"{f'{nz}×{nr}':>14s} {h:10.5f} {e:12.3e} {order:>8s}")
        prev_e, prev_h = e, h


print("=" * 62)
print("ADR2D 制造解（MMS）网格收敛表")
print("=" * 62)

table("扩散主导（含轴对称 1/r 项）", 0.0, 0.0, 2e-6, 5e-6,
      [(10, 5), (20, 10), (40, 20), (80, 40)])

print("\n### 对流主导：对流格式对比  (u_z=2e-3, D_r=2e-6, D_z=5e-6, Pe_cell≈4)")
print(f"{'格式':>12s} {'20×10':>11s} {'40×20':>11s} {'80×40':>11s} {'观测阶':>8s} {'有界':>6s}")
print("-" * 66)
for scheme in ("upwind", "power_law", "exponential", "hybrid", "central"):
    es = [_mms_error(nz, nr, 2e-3, 0.0, 2e-6, 5e-6, scheme)
          for nz, nr in ((20, 10), (40, 20), (80, 40))]
    order = np.log2(es[1] / es[2])
    bounded = "是" if check_bounded_scheme(scheme) else "否"
    print(f"{scheme:>12s} {es[0]:11.3e} {es[1]:11.3e} {es[2]:11.3e} "
          f"{order:8.2f} {bounded:>6s}")

print("\n说明：")
print("  · 扩散主导达二阶 → 轴对称 1/r 散度、壁面通量、入口二次 ghost-cell 均正确")
print("  · 迎风仅一阶（数值扩散 u·Δz/2 主导）；幂律/指数提升到 ~1.8 且保持有界")
print("  · 「有界」= 正系数条件（稳态无伪振荡）；中心差分在高 Peclet 不保证")
print("  · 格式定义参考 NIST FiPy / Patankar 的 A(|P|) 框架，见 chemistry/schemes.py")
print("  · 径向速度取 0：轴对称充分发展流的唯一无散且 u_r(R)=0 的径向速度")
