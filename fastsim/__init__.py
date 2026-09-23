"""fastsim —— 自研快速仿真内核。

定位：用「降维 + 解析 + 解析 Jacobian」替代 COMSOL 全 3D FEM 的参数扫描，
把单次求解从分钟级压到毫秒级，换取设计寻优与趋势探索的可行性。

四基座对应关系：
    fastsim.echem    ← COMSOL 电化学模块（电流分布 / 固相欧姆降）
    fastsim.chemistry← COMSOL 化学反应工程模块（AOP 机理 / 反应-传质）
    fastsim.flow     ← COMSOL CFD 模块（解析速度剖面 / 传质关联式）
    fastsim.couple   ← 电-化耦合（电流 → 法拉第通量 → 浓度场）
    （优化基座：待建）

高保真校验仍由 COMSOL 承担（见 bench/），本内核不声称替代 COMSOL。
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
