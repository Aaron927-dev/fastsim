"""电化学层：电流分布（对应 COMSOL 电化学模块的一/二次分布接口）。

工程参数化：电极特性（`ElectrodeKinetics`）× 水质特性（电导率 κ）。
"""

from __future__ import annotations

from .secondary import CurrentSolution, ElectrodeKinetics, SecondaryCurrent2D

__all__ = [
    "ElectrodeKinetics",
    "SecondaryCurrent2D",
    "CurrentSolution",
]
