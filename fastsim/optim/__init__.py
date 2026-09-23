"""优化基座：灵敏度分析 + 参数估计 + 代理模型寻优。

对应 COMSOL 优化模块的三类能力：
    - 参数优化设计  → `surrogate.bayesian_optimize`（设计寻优）
    - 参数估计      → `estimate.fit`（用实验数据反推模型参数）
    - 灵敏度分析    → `sensitivity.sensitivities`（参数影响排序）
（形状/拓扑优化需动网格与密度模型，不在本内核范围内，留给 COMSOL。）
"""

from __future__ import annotations

from .estimate import FitResult, fit
from .sensitivity import (
    SensitivityResult,
    check_step_robustness,
    finite_difference_gradient,
    sensitivities,
)
from .surrogate import DesignResult, GaussianProcess, bayesian_optimize, latin_hypercube

__all__ = [
    "fit",
    "FitResult",
    "sensitivities",
    "SensitivityResult",
    "check_step_robustness",
    "finite_difference_gradient",
    "bayesian_optimize",
    "DesignResult",
    "GaussianProcess",
    "latin_hypercube",
]
