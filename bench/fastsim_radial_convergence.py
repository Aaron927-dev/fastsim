"""解释 fastsim 二次分布 +3% 偏差的来源：径向离散误差（可闭式预测）。

假设
    环隙里单位长度的径向欧姆电阻精确值 R' = ln(r_o/r_i)/(2πκ)。
    均匀网格的有限体积格式把电阻求和为
        R'_FV = Σ 段长 / (2πκ · 该段面半径)
    而不是积分 ln。两者之比是可闭式计算的，与 n_r 有关。
    若 fastsim 的偏差 ≈ (R'/R'_FV − 1)，则该偏差是**纯离散误差**，
    随 n_r → ∞ 应按 O(dr²) 收敛到 0。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # fastsim 包

from fastsim.echem.secondary import ElectrodeKinetics, SecondaryCurrent2D  # noqa: E402

R_I, R_O, L_LEN = 0.005, 0.010, 0.20
KAPPA, V_CELL = 1.0, 2.0
ALPHA, T_K, I0 = 0.5, 298.15, 1e-3
R_GAS, F_CONST = 8.31446, 96485.33
F_DIMLESS = ALPHA * F_CONST / (R_GAS * T_K)
R_PRIME = np.log(R_O / R_I) / (2.0 * np.pi * KAPPA)


def analytic_ip(secondary=True, v=V_CELL):
    if not secondary:
        return v / R_PRIME

    def g(i):
        return (i * R_PRIME
                + (1.0 / F_DIMLESS) * (np.arcsinh(i / (4 * np.pi * R_I * I0))
                                       + np.arcsinh(i / (4 * np.pi * R_O * I0))))
    lo, hi = 1e-15, 1e9
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if g(mid) < v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def r_prime_fv(n_r: int) -> float:
    """均匀 FV 网格上径向电阻的**离散求和**值。"""
    dr = (R_O - R_I) / n_r
    r_face = R_I + np.arange(n_r + 1) * dr          # 面半径，n_r+1 个
    # 半层（边界→首/末单元中心）+ 内部面
    r_half = 0.5 * dr
    tot = r_half / r_face[0]
    for j in range(1, n_r):
        tot += dr / r_face[j]
    tot += r_half / r_face[-1]
    return tot / (2.0 * np.pi * KAPPA)


def main():
    iP_a1 = analytic_ip(False)
    iP_a2 = analytic_ip(True)
    print("=" * 88)
    print("环隙径向离散误差检验  (r_i=5 mm, r_o=10 mm, κ=1 S/m)")
    print(f"  精确 R' = ln(r_o/r_i)/(2πκ) = {R_PRIME:.8f} Ω·m")
    print(f"  解析一次 I' = {iP_a1:.6f} A/m,   解析二次 I' = {iP_a2:.6f} A/m")
    print("=" * 88)
    print(f"{'n_r':>6} {'R_FV/R_exact':>14} {'预测偏差(一次)':>16} "
          f"{'实测偏差(一次)':>16} {'预测偏差(二次)':>16} {'实测偏差(二次)':>16} {'j_i 跨度':>10}")

    for n_r in (8, 16, 32, 64, 128, 256):
        ratio = r_prime_fv(n_r) / R_PRIME
        pred1 = (1.0 / ratio - 1.0) * 100.0
        pred2 = (iP_a2 * (1.0 / ratio) - iP_a2) / iP_a2 * 100.0
        # 更好的预测：以 R_FV 替代 R' 重解一维方程（含动力学）
        def g(i):
            return (i * r_prime_fv(n_r)
                    + (1.0 / F_DIMLESS) * (np.arcsinh(i / (4 * np.pi * R_I * I0))
                                           + np.arcsinh(i / (4 * np.pi * R_O * I0))))
        lo, hi = 1e-15, 1e9
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            if g(mid) < V_CELL:
                lo = mid
            else:
                hi = mid
        iP_pred2 = 0.5 * (lo + hi)
        pred2 = (iP_pred2 - iP_a2) / iP_a2 * 100.0

        res = {}
        for sec, iP_a in ((False, iP_a1), (True, iP_a2)):
            sim = SecondaryCurrent2D(
                r_inner=R_I, r_outer=R_O, length=L_LEN, kappa=KAPPA,
                anode=(ElectrodeKinetics(j0=I0, alpha_a=ALPHA, alpha_c=ALPHA)
                       if sec else None),
                cathode=(ElectrodeKinetics(j0=I0, alpha_a=ALPHA, alpha_c=ALPHA)
                         if sec else None),
                n_z=40, n_r=n_r, temperature=T_K,
            )
            sol = sim.solve(V_CELL)
            j_i = float(np.mean(sol.j_anode))
            iP = j_i * 2 * np.pi * R_I
            res[sec] = ((iP - iP_a) / iP_a * 100.0,
                        float(sol.j_anode.max() - sol.j_anode.min()))
        print(f"{n_r:>6} {ratio:>14.6f} {pred1:>16.3f} {res[False][0]:>16.3f} "
              f"{pred2:>16.3f} {res[True][0]:>16.3f} {res[True][1]:>10.2e}")

    print("=" * 88)
    print("若「预测偏差」与「实测偏差」逐行吻合，则 +3% 是纯径向离散误差，非模型错误。")
    print("=" * 88)


if __name__ == "__main__":
    main()
