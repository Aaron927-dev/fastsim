"""环境高级氧化（AOP）核心反应库：臭氧链式分解 + 经典芬顿。

定位
----
对应 COMSOL 化学反应工程模块中「输入反应式 → 自动生成动力学」的能力，
但以**环境 AOP 的标准机理**预置成库：使用者一行代码拿到带文献速率常数的
`ReactionNetwork`，可直接接 0D 反应器（`scipy` 积分）或
`ADR1D/ADR2D` 反应-传质求解器。

机理与速率常数
--------------
臭氧链：Staehelin–Hoigné（1985）体系的简化 9 步链
（快速中间体 O₃·⁻/HO₃·/HO₄· 已按准稳态合并，pH>5 时超氧按 O₂·⁻ 计）。
芬顿：经典 Haber–Weiss 循环 8 步（酸性 pH 2.8–3.5 适用，超氧按 HO₂· 计）。

⚠ 速率常数为 25 °C 经典文献值（离子强度/温度效应未校正）。
量级可信（±30% 内的文献分散），但**论文级复现请核对原始文献**——
每条反应的出处标注在 `Mechanism.sources`。

建模假设（务必读，影响适用域）
------------------------------
1. **pH 由外部缓冲固定**：H⁺/OH⁻ 不作为 ODE 物种，其浓度以准一级速率
   并入引发步骤（k₁_eff = 70·10^(pH−14)）。近中性体系有碳酸盐缓冲时成立。
2. **快速中间体合并**：O₃·⁻ + H⁺ → HO₃· → ·OH + O₂ 合并进
   O₃ + O₂·⁻ 一步；HO₄· 合并进 ·OH + O₃（pH>5 下 HO₂· 即刻去质子化）。
3. 质子不显式守恒（缓冲假设的推论）。
4. 水的活度并入速率常数（稀溶液）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .network import ReactionNetwork

__all__ = [
    "Mechanism",
    "ozone_chain",
    "fenton_classic",
    "AOM_MG_PER_M",
    "mgL_to_M",
    "M_to_mgL",
    "k_second_order_to_SI",
    "k_second_order_from_SI",
]

# 臭氧分子量（g/mol），mg/L ↔ mol/L 换算用
AOM_MG_PER_M = 48.0


# ── 速率常数的单位换算（重要，易错）─────────────────────────
# 本机理库中的速率常数是**文献惯例的 M 基**（一级 1/s，二级 M⁻¹s⁻¹）。
# 但 `ADR1D`/`ADR2D` 的守恒式要求浓度用 **SI 的 mol/m³**：
#     浓度[mol/m³] × 体积流量[m³/s] = 摩尔流量[mol/s]
# 才能与法拉第通量（mol·m⁻²·s⁻¹）在同一量纲上守恒。
#
# 换算关系：1 M = 1000 mol/m³
#   一级：k[1/s] 不变
#   二级：R[M/s] = k[M⁻¹s⁻¹]·c₁[M]·c₂[M]
#         R'[mol/m³/s] = 1000·R,  c' = 1000·c
#         ⇒ k'[m³·mol⁻¹·s⁻¹] = k[M⁻¹s⁻¹] / 1000

def k_second_order_to_SI(k_M: float) -> float:
    """二级速率常数 M⁻¹s⁻¹ → m³·mol⁻¹·s⁻¹（供 ADR1D/ADR2D 使用）。"""
    return k_M / 1000.0


def k_second_order_from_SI(k_SI: float) -> float:
    """二级速率常数 m³·mol⁻¹·s⁻¹ → M⁻¹s⁻¹（便于与文献对比）。"""
    return k_SI * 1000.0


@dataclass
class Mechanism:
    """预置机理：反应网络 + 假设清单 + 速率常数出处。"""

    network: ReactionNetwork
    assumptions: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    name: str = ""

    def initial(self, **conc: float) -> np.ndarray:
        """按物质名给初值（转发到 ReactionNetwork.initial）。"""
        return self.network.initial(**conc)


def ozone_chain(pH: float = 7.0) -> Mechanism:
    """臭氧自分解链式机理（pH>5 适用；pH 由外部缓冲固定）。

    简化 9 步链（快速中间体已合并），速率常数 25 °C：

    ======  =============================================  ==========  ==================
    序号   反应                                          k           备注
    ======  =============================================  ==========  ==================
    (0a)   H₂O₂ + OH⁻ → HO₂⁻                              450 M⁻¹s⁻¹  以 [OH⁻] 并入后为准一级
    (0b)   HO₂⁻ → H₂O₂ + OH⁻                              1.8 s⁻¹     与 pKa=11.6 自洽
    (1)    O₃ + OH⁻ → HO₂⁻ + O₂                           70 M⁻¹s⁻¹   链引发
    (2)    O₃ + HO₂⁻ → ·OH + O₂·⁻ + O₂                    2.8e6       引发主通道
    (3)    O₃ + O₂·⁻ → ·OH + 2 O₂                         1.6e9       链传播（合并 O₃·⁻）
    (4)    ·OH + O₃ → O₂·⁻ + O₂                           1.1e8       链传播（合并 HO₄·）
    (5)    ·OH + O₂·⁻ → OH⁻ + O₂                          1.0e10      淬灭
    (6)    ·OH + ·OH → H₂O₂                               5.5e9       歧化
    (7)    ·OH + H₂O₂ → O₂·⁻ + H₂O                        2.7e7       H₂O₂ 淬灭
    (8)    ·OH + pCBA → 产物                               5.0e9       探针（pCBA 常用量）
    ======  =============================================  ==========  ==================

    参数
    ----
    pH : float
        体系 pH（>5 适用；决定引发速率与 HO₂⁻ 分配）。
    """
    if pH <= 5.0:
        raise ValueError(
            f"ozone_chain 的合并假设仅适用于 pH>5（收到 pH={pH}）；"
            "酸性体系请直接用 ReactionNetwork 自建原始 SH85 全链"
        )
    c_oh = 10.0 ** (pH - 14.0)

    net = ReactionNetwork()
    k = {
        "diss": 450.0 * c_oh,     # H2O2 + OH⁻ → HO₂⁻（OH⁻ 并入）
        "prot": 1.8,              # HO₂⁻ → H₂O₂ + OH⁻（与 pKa 11.6 自洽）
        "init": 70.0 * c_oh,      # O₃ + OH⁻ → HO₂⁻ + O₂（引发，准一级）
        "oh_ho2": 2.8e6,          # O₃ + HO₂⁻
        "o3_sup": 1.6e9,          # O₃ + O₂·⁻
        "oh_o3": 1.1e8,           # ·OH + O₃
        "oh_sup": 1.0e10,         # ·OH + O₂·⁻
        "oh_oh": 5.5e9,           # ·OH 歧化
        "oh_h2o2": 2.7e7,         # ·OH + H₂O₂
        "oh_pcba": 5.0e9,         # ·OH + pCBA
    }
    net.add("H2O2 -> HO2m", k["diss"])
    net.add("HO2m -> H2O2", k["prot"])
    net.add("O3 -> HO2m + O2", k["init"])
    net.add("O3 + HO2m -> OHr + O2r + O2", k["oh_ho2"])
    net.add("O3 + O2r -> OHr + O2 + O2", k["o3_sup"])
    net.add("OHr + O3 -> O2r + O2", k["oh_o3"])
    net.add("OHr + O2r -> O2", k["oh_sup"])
    net.add("OHr + OHr -> H2O2", k["oh_oh"])
    net.add("OHr + H2O2 -> O2r", k["oh_h2o2"])
    net.add("OHr + pCBA -> PpCBA", k["oh_pcba"])

    src = {
        "H2O2→HO2m / HO2m→H2O2": "H₂O₂ 解离 pKa=11.6，正逆速率自洽（k_r/k_f=Ka/Kw）",
        "O₃+OH⁻=70": "Staehelin & Hoigné 1985",
        "O₃+HO₂⁻=2.8e6": "Staehelin & Hoigné 1985",
        "O₃+O₂·⁻=1.6e9": "Bühler, Staehelin & Hoigné 1984（O₃·⁻ 快速合并）",
        "·OH+O₃=1.1e8": "文献有 1.1e8–3e8 分散（合并 HO₄·）；取保守低端",
        "·OH 淬灭组": "HTU/HO 脉解放射经典值（Buxton et al. 1988 汇编）",
        "·OH+pCBA=5e9": "Elovitz & von Gunten 1999（·OH 探针标准值）",
    }
    assumptions = [
        f"pH={pH} 由外部缓冲固定，OH⁻ 并入准一级速率（k_init={k['init']:.2e} s⁻¹）",
        "O₃·⁻/HO₃·/HO₄· 按准稳态合并（pH>5 时 HO₂· 即刻去质子化为 O₂·⁻）",
        "质子不显式守恒；碳酸盐淬灭未含（含 HCO₃⁻ 体系需自行加 ·OH+HCO₃⁻=8.5e6）",
        "速率常数为 25 °C 经典值，离子强度未校正",
    ]
    return Mechanism(network=net, assumptions=assumptions, sources=src,
                     name=f"臭氧自分解链（pH={pH}）")


def fenton_classic() -> Mechanism:
    """经典芬顿机理（酸性 pH 2.8–3.5 适用；超氧按 HO₂· 计）。

    简化 8 步循环（速率常数 25 °C）：

    ======  ========================================  ===========  ==================
    序号   反应                                     k            备注
    ======  ========================================  ===========  ==================
    (1)    Fe²⁺ + H₂O₂ → Fe³⁺ + ·OH + OH⁻           63 M⁻¹s⁻¹    中心反应（63–76 分散）
    (2)    Fe³⁺ + H₂O₂ → Fe²⁺ + HO₂· + H⁺           0.01         高铁再生（慢步）
    (3)    Fe²⁺ + ·OH → Fe³⁺ + OH⁻                  3.2e8        ·OH 淬灭（亚铁牺牲）
    (4)    ·OH + H₂O₂ → HO₂· + H₂O                  2.7e7        H₂O₂ 淬灭
    (5)    ·OH + HO₂· → O₂ + H₂O                    7.1e9        自由基互灭
    (6)    HO₂· + HO₂· → H₂O₂ + O₂                  8.3e5        二聚歧化
    (7)    Fe³⁺ + HO₂· → Fe²⁺ + O₂ + H⁺             2e4          高铁再生（值不确定度大）
    (8)    ·OH + phenol → 产物                       6.8e9        探针（苯酚）
    ======  ========================================  ===========  ==================
    """
    net = ReactionNetwork()
    k = {
        "fe2_h2o2": 63.0,
        "fe3_h2o2": 0.01,
        "fe2_oh": 3.2e8,
        "oh_h2o2": 2.7e7,
        "oh_ho2": 7.1e9,
        "ho2_ho2": 8.3e5,
        "fe3_ho2": 2.0e4,
        "oh_phenol": 6.8e9,
    }
    net.add("Fe2 + H2O2 -> Fe3 + OHr", k["fe2_h2o2"])
    net.add("Fe3 + H2O2 -> Fe2 + HO2r", k["fe3_h2o2"])
    net.add("Fe2 + OHr -> Fe3", k["fe2_oh"])
    net.add("OHr + H2O2 -> HO2r", k["oh_h2o2"])
    net.add("OHr + HO2r -> O2", k["oh_ho2"])
    net.add("HO2r + HO2r -> H2O2 + O2", k["ho2_ho2"])
    net.add("Fe3 + HO2r -> Fe2 + O2", k["fe3_ho2"])
    net.add("OHr + phenol -> Pphenol", k["oh_phenol"])

    src = {
        "Fe²⁺+H₂O₂=63": "Walling 1975 / Kremer 2003（文献 63–76 分散，取常用值）",
        "Fe³⁺+H₂O₂=0.01": "Walling 1975（0.002–0.01 分散）",
        "Fe³⁺+HO₂·=2e4": "文献分散大（1.2e3–1e6），对总速率影响二阶",
        "·OH 组": "Buxton et al. 1988 汇编",
        "·OH+phenol=6.8e9": "Buxton et al. 1988",
    }
    assumptions = [
        "酸性体系（pH 2.8–3.5），超氧按 HO₂· 计（pH<4.8 时占比 >94%）",
        "OH⁻ 并入速率常数；铁氢氧化物沉淀/络合未建模",
        "速率常数为 25 °C 经典值",
    ]
    return Mechanism(network=net, assumptions=assumptions, sources=src,
                     name="经典芬顿（酸性）")


# ── 便捷换算 ─────────────────────────────────────────────
def mgL_to_M(mg_per_L: float, molar_mass: float = 48.0) -> float:
    """mg/L → mol/L（默认臭氧 48 g/mol）。"""
    return mg_per_L / 1e3 / molar_mass


def M_to_mgL(conc_M: float, molar_mass: float = 48.0) -> float:
    """mol/L → mg/L。"""
    return conc_M * 1e3 * molar_mass
