"""电-化耦合：把电流分布接到化学场的界面通量。

闭环结构：

    SecondaryCurrent2D        faradaic_flux           ADR2D
    ────────────────────  →  ────────────────  →  ──────────────────
    电极特性(动力学/固相)      法拉第定律           壁面生成/消耗通量
    水质特性(κ)               电流效率              → 浓度场 + AOP 反应
    几何(环隙)
      ↓ j(z) A/m²                ↓ mol·m⁻²·s⁻¹

法拉第定律
----------
    J_species = j · CE / (n·F)      [mol·m⁻²·s⁻¹]

其中 j 为局部电流密度 (A/m²)，CE 为生成该物质的电流效率 (0–1)，
n 为该反应的电子数，F = 96485 C/mol。

常用 n 值（生成反应）
---------------------
- ·OH（水氧化）：n = 1
- H₂O₂（氧还原，2e⁻）：n = 2
- O₃（水氧化，6e⁻）：n = 6
- 金属沉积 Mⁿ⁺：n = 电荷数
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["F_CONST", "faradaic_flux", "ElectrochemicalFlux"]

F_CONST = 96485.33     # C/mol

# 常见生成反应的电子数（供 faradaic_flux 直接引用）
N_ELECTRONS = {
    "OH": 1,        # H₂O → ·OH + H⁺ + e⁻
    "H2O2": 2,      # O₂ + 2H⁺ + 2e⁻ → H₂O₂
    "O3": 6,        # 3H₂O → O₃ + 6H⁺ + 6e⁻
    "H2": 2,        # 2H⁺ + 2e⁻ → H₂
    "ClO": 2,       # Cl⁻ + H₂O → ClO⁻ + 2H⁺ + 2e⁻
}


def faradaic_flux(j: np.ndarray | float, n_electrons: int,
                  current_efficiency: float = 1.0) -> np.ndarray:
    """法拉第定律：局部电流密度 → 物质的摩尔通量 (mol·m⁻²·s⁻¹)。

    参数
    ----
    j : array | float
        局部电流密度 (A/m²)。
    n_electrons : int
        生成该物质的电子数（见 N_ELECTRONS）。
    current_efficiency : float
        电流效率 (0–1]。默认 1.0（理论上限）。
        真实体系必须先测 CE —— 它通常远小于 1，且随电流密度变化。

    返回
    ----
    摩尔通量 (mol·m⁻²·s⁻¹)，正负号与 j 一致（正 = 阳极氧化生成）。
    """
    if not (0.0 < current_efficiency <= 1.0):
        raise ValueError(f"电流效率须在 (0, 1]，收到 {current_efficiency}")
    if n_electrons <= 0:
        raise ValueError("电子数必须为正")
    return np.asarray(j, dtype=float) * current_efficiency / (n_electrons * F_CONST)


@dataclass
class ElectrochemicalFlux:
    """把 `SecondaryCurrent2D` 的电流分布转成 `ADR2D` 的壁面通量。

    参数
    ----
    current_solution : CurrentSolution
        `SecondaryCurrent2D` 的解（含 j(z)）。
    species_map : dict[str, tuple[int, float]]
        {物种名: (电子数, 电流效率)} —— 该物种由电流驱动生成（正通量）。
        例：``{"O3": (6, 0.05)}`` 表示 O₃ 以 5% 电流效率、6 电子生成。
    consumed_map : dict[str, tuple[int, float]] | None
        {物种名: (电子数, 库仑效率)} —— 该物种在电极上被**消耗**（负通量）。
        例：``{"phenol": (2, 0.1)}``。

    说明
    ----
    `ADR2D.wall_flux_inner` 约定：**正值 = 流入流体域**。
    因此阳极生成的物种取正号，被消耗的物种取负号。
    """

    current_solution: object
    species_map: dict[str, tuple[int, float]]
    consumed_map: dict[str, tuple[int, float]] | None = None
    face: str = "inner"
    face_radius: float = 0.0      # 电极面半径 (m)，用于总面积/总速率换算

    def j_local(self) -> np.ndarray:
        """电极局部电流密度 (A/m²)。"""
        cs = self.current_solution
        return np.asarray(cs.j_anode if self.face == "inner" else cs.j_cathode)

    def build(self, species_index: dict[str, int]) -> np.ndarray:
        """构造 (nz, ns) 的壁面通量数组。

        参数
        ----
        species_index : dict[str, int]
            物种名 → 在 `ReactionNetwork` 中的索引。
        """
        j = self.j_local()
        nz = j.size
        ns = len(species_index)
        out = np.zeros((nz, ns))

        for name, (n_e, ce) in self.species_map.items():
            if name not in species_index:
                raise KeyError(
                    f"物种 {name!r} 不在反应网络中；现有：{sorted(species_index)}"
                )
            out[:, species_index[name]] += faradaic_flux(j, n_e, ce)

        if self.consumed_map:
            for name, (n_e, ce) in self.consumed_map.items():
                if name not in species_index:
                    raise KeyError(
                        f"物种 {name!r} 不在反应网络中；现有：{sorted(species_index)}"
                    )
                out[:, species_index[name]] -= faradaic_flux(j, n_e, ce)

        return out

    def total_rates(self, species_index: dict[str, int]) -> dict[str, float]:
        """各物种的总生成速率 (mol/s)，用于快速核对量级。

        需先设置 `face_radius`（电极面半径）。
        """
        if self.face_radius <= 0.0:
            raise ValueError("请先设置 face_radius（电极面半径，m）")
        z = np.asarray(self.current_solution.z, dtype=float)
        dz = float(z[1] - z[0]) if z.size > 1 else 1.0
        area = 2.0 * np.pi * self.face_radius * dz     # 每个 z 单元的面面积
        flux = self.build(species_index)
        return {
            name: float(np.sum(flux[:, i]) * area)
            for name, i in species_index.items()
        }
