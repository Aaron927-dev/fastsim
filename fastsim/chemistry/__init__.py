"""化学场基座：反应网络 + 对流-扩散-反应求解（一维/二维）。

**本层与具体化学体系无关**：反应网络由使用者通过反应式字符串提供，
框架不内置任何机理。不同体系的基元反应互不通用，各自独立建网：

  · 电催化臭氧化（ECO）的一套基元反应
  · CO₂ 还原（CO2RR）的另一套基元反应
  · 其它任何体系

两者只需分别构造 `ReactionNetwork` 实例即可，无需也不应混用同一套机理。
框架只负责把「化学计量 + 速率常数」翻译成 R(c) 与解析 Jacobian——
这正是 COMSOL 化学反应工程模块中「输入反应式自动生成动力学」的那一步。
"""

from __future__ import annotations

from .adr import ADR1D, analytic_first_order_dispersion
from .adr2d import ADR2D
from .mechanisms import (
    Mechanism,
    fenton_classic,
    k_second_order_to_SI,
    mgL_to_M,
    ozone_chain,
)
from .network import Reaction, ReactionNetwork, parse_equation
from .schemes import SCHEMES, describe_scheme, peclet_factor

__all__ = [
    "ADR1D",
    "ADR2D",
    "ReactionNetwork",
    "Reaction",
    "parse_equation",
    "analytic_first_order_dispersion",
    "Mechanism",
    "ozone_chain",
    "fenton_classic",
    "mgL_to_M",
    "k_second_order_to_SI",
    "SCHEMES",
    "peclet_factor",
    "describe_scheme",
]
