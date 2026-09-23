"""电-化耦合：电流分布 → 界面通量 → 化学场。"""

from __future__ import annotations

from .ec_chem import F_CONST, N_ELECTRONS, ElectrochemicalFlux, faradaic_flux

__all__ = ["faradaic_flux", "ElectrochemicalFlux", "N_ELECTRONS", "F_CONST"]
