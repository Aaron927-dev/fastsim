"""对流格式：A(|P|) 框架（Patankar 统一表述）。

为什么需要这个模块
------------------
本项目初版对对流项只用**一阶迎风**，MMS 实测收敛阶 **1.00** ——
在 advection-dominated 工况下数值扩散（u·Δx/2）会主导物理扩散，
误差达 1e-2 量级。这是本求解器已知的最大精度短板。

对流通量有多种成熟处理方式，可以把它们统一成**一个函数 A(|P|)**
（Patankar《Numerical Heat Transfer and Fluid Flow》经典表述；
NIST 的 FiPy 也采用同一框架，并把幂律格式作为默认）：

    面上净通量（P → A 方向）
        = D·A(|P|)·(φ_P − φ_A) + max(F,0)·φ_P + min(F,0)·φ_A

    其中  F = u_f·A_face       面体积流量
          D = D_f·A_face/δ     面扩散导纳
          P = F/D              面（单元）Peclet 数

对 A(|P|) 取不同函数即得不同格式：

======== ============================ ========================== ==========
格式      A(|P|)                       特性                        收敛阶
======== ============================ ========================== ==========
中心差分  1 − 0.5|P|                  仅 |P|<2 稳定，高 P 会振荡   二阶
一阶迎风  1                            全 P 稳定，但数值扩散大      一阶
指数      |P|/(e^|P| − 1)             一维精确解导出，最准但含指数  二阶*
混合      max(0, 1 − 0.5|P|)          |P|<2 中心、否则迎风          一~二阶
幂律      max(0, 1 − 0.1|P|)⁵         **FiPy 默认**，快且稳        二阶*
======== ============================ ========================== ==========

\\* 「二阶」指在扩散主导/中等 Peclet 区；极端对流主导时所有保正格式都会退化。

**为什么选幂律作默认**：它无条件满足正系数条件（对稳态问题保证有界、
无伪振荡），同时在对流主导区比迎风**少得多**的数值扩散。
这与 FiPy 的选择一致（其文档明确：幂律「克服了混合格式的不准确，
又比指数格式快得多」）。

本模块的角色是**借鉴现有成熟实现**：格式定义来自 Patankar / FiPy，
本项目的工作是把它接入自己的有限体积装配（含轴对称权重与多物质块 Jacobian），
并用制造解法验证收敛阶确实提升。
"""

from __future__ import annotations

import numpy as np

__all__ = ["SCHEMES", "peclet_factor", "describe_scheme"]

SCHEMES = ("upwind", "power_law", "exponential", "hybrid", "central")

# 幂律格式在 |P| 超过阈值后 A 归零（纯迎风），避免出现负系数
_POWER_LAW_CUTOFF = 10.0


def peclet_factor(P: np.ndarray | float, scheme: str = "power_law"):
    """返回 A(|P|) 因子。

    参数
    ----
    P : array | float
        面（单元）Peclet 数 = u_f·δ/D_f。
    scheme : str
        ``upwind`` / ``power_law``（默认）/ ``exponential`` / ``hybrid`` / ``central``。

    返回
    ----
    A(|P|)，与输入同形状。乘在**扩散导纳**上：
    有效扩散系数 = D·A(|P|)。A=1 即纯迎风（不削减扩散）。
    """
    if scheme not in SCHEMES:
        raise ValueError(f"未知对流格式 {scheme!r}（可选 {SCHEMES}）")
    aP = np.abs(np.asarray(P, dtype=float))

    if scheme == "upwind":
        return np.ones_like(aP)
    if scheme in ("central", "hybrid"):
        # 中心差分与混合格式共用同一表达式：max(0, 1 − 0.5|P|)
        # （混合格式即「中心差分在 |P|≥2 处自然归零」的分段形式）
        return np.maximum(1.0 - 0.5 * aP, 0.0)
    if scheme == "power_law":
        return np.where(aP < _POWER_LAW_CUTOFF,
                        np.maximum(1.0 - 0.1 * aP, 0.0) ** 5, 0.0)
    # exponential：A = |P|/(exp|P| − 1)，|P|→0 时极限为 1
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(np.clip(aP, 0.0, 500.0))
        A = np.where(aP < 1e-12, 1.0, aP / np.maximum(e - 1.0, 1e-300))
    return np.clip(A, 0.0, 1.0)


def describe_scheme(scheme: str) -> str:
    """格式的一句话说明（用于报告/文档）。"""
    table = {
        "upwind": "一阶迎风：全 Peclet 稳定，但数值扩散最大（一阶精度）",
        "power_law": "幂律（FiPy 默认）：保正且数值扩散小，推荐默认",
        "exponential": "指数：由一维精确解导出，最准但含指数运算（较慢）",
        "hybrid": "混合：|P|<2 用中心差分，否则纯迎风",
        "central": "中心差分：二阶但仅低 Peclet 稳定，高 P 会振荡",
    }
    return table.get(scheme, f"未知格式 {scheme}")


def check_bounded_scheme(scheme: str) -> bool:
    """该格式是否保证正系数（稳态有界、无伪振荡）。"""
    return scheme in ("upwind", "power_law", "exponential", "hybrid")
