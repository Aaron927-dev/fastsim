"""反应网络内核：化学计量解析 + 质量作用定律速率 + 解析 Jacobian。

设计对齐 COMSOL 化学反应工程模块的「输入反应式 → 自动生成反应动力学」能力，
但目标是最小依赖、可批量调用（COMSOL 每次求解都要重建模型，这里只是一次矩阵运算）。

Jacobian 提供闭式解，是刚性积分器与牛顿稳态求解能收敛的关键
（COMSOL 内部同样依赖解析/数值 Jacobian；有限差分 Jacobian 在 38 基元网络下代价过高）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["parse_equation", "Reaction", "ReactionNetwork"]


def parse_equation(equation: str) -> tuple[dict[str, float], dict[str, float]]:
    """解析 "2 A + B -> C + D" 形式的反应式，返回 (反应物, 产物) 化学计量字典。

    支持分隔符 -> 与 →；系数可写 "2 A" 或 "2A"；空格自由。
    """
    text = equation.strip()
    if not text:
        raise ValueError("空反应式")

    parts = re.split(r"->|→", text)
    if len(parts) != 2:
        raise ValueError(f"反应式必须恰好含一个箭头: {equation!r}")

    def parse_side(side: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            m = re.match(r"^(\d+(?:\.\d+)?)\s*(.+)$", term)
            if m:
                nu, code = float(m.group(1)), m.group(2).strip()
            else:
                nu, code = 1.0, term
            out[code] = out.get(code, 0.0) + nu
        return out

    return parse_side(parts[0]), parse_side(parts[1])


@dataclass
class Reaction:
    """单条基元反应（质量作用定律）。"""

    reactants: dict[str, float]
    products: dict[str, float]
    k: float
    name: str = ""

    @property
    def order(self) -> float:
        """总反应级数（质量作用定律下等于反应物化学计量数之和）。"""
        return float(sum(self.reactants.values()))

    def rate(self, conc: np.ndarray, index: dict[str, int]) -> float:
        r = self.k
        for code, nu in self.reactants.items():
            r *= conc[index[code]] ** nu
        return r


@dataclass
class ReactionNetwork:
    """一组基元反应 + 物质列表，提供 dC/dt 与解析 Jacobian。

    用 ``add()`` 依次加入反应；``rates()`` 返回净生成速率；
    ``jacobian()`` 返回解析 Jacobian（d(dC/dt)/dC）。
    """

    species: list[str] = field(default_factory=list)
    reactions: list[Reaction] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _nu_net: np.ndarray | None = field(default=None, init=False, repr=False)

    # ── 构建 ────────────────────────────────────────────────
    def add_species(self, code: str) -> int:
        if code not in self._index:
            self._index[code] = len(self.species)
            self.species.append(code)
            self._nu_net = None
        return self._index[code]

    def add(self, equation: str, k: float, name: str = "") -> Reaction:
        """加入一条反应。物质若不存在则自动登记（对齐 COMSOL 的自动物质登记）。"""
        reactants, products = parse_equation(equation)
        for code in list(reactants) + list(products):
            self.add_species(code)
        rxn = Reaction(reactants=reactants, products=products, k=float(k), name=name or equation)
        self.reactions.append(rxn)
        self._nu_net = None
        return rxn

    def set_k(self, index: int, k: float) -> None:
        self.reactions[index].k = float(k)

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    @property
    def nu_net(self) -> np.ndarray:
        """净化学计量矩阵，形状 (n_species, n_reactions)。"""
        if self._nu_net is None:
            nu = np.zeros((self.n_species, self.n_reactions))
            for j, rxn in enumerate(self.reactions):
                for code, s in rxn.products.items():
                    nu[self._index[code], j] += s
                for code, s in rxn.reactants.items():
                    nu[self._index[code], j] -= s
            self._nu_net = nu
        return self._nu_net

    # ── 动力学 ──────────────────────────────────────────────
    def rates(self, conc: np.ndarray) -> np.ndarray:
        """各反应的速率向量 r_j（质量作用定律）。"""
        r = np.empty(self.n_reactions)
        for j, rxn in enumerate(self.reactions):
            r[j] = rxn.rate(conc, self._index)
        return r

    def __call__(self, t: float, conc: np.ndarray) -> np.ndarray:
        """dC/dt，签名兼容 scipy.integrate.solve_ivp。"""
        return self.nu_net @ self.rates(conc)

    # ── 批量接口（二维/三维场求解用，避免逐单元 Python 循环）──────
    def rates_batch(self, conc: np.ndarray) -> np.ndarray:
        """一次算 N 个单元的速率。conc 形状 (N, n_species) → (N, n_reactions)。"""
        c = np.atleast_2d(np.asarray(conc, dtype=float))
        R = np.empty((c.shape[0], self.n_reactions))
        for j, rxn in enumerate(self.reactions):
            r = np.full(c.shape[0], rxn.k)
            for code, nu in rxn.reactants.items():
                r = r * c[:, self._index[code]] ** nu
            R[:, j] = r
        return R

    def source_batch(self, conc: np.ndarray) -> np.ndarray:
        """批量净生成速率 (N, n_species)。"""
        return self.rates_batch(conc) @ self.nu_net.T

    def jacobian_batch(self, conc: np.ndarray) -> np.ndarray:
        """批量解析 Jacobian，形状 (N, n_species, n_species)。"""
        c = np.atleast_2d(np.asarray(conc, dtype=float))
        n_s = self.n_species
        J = np.zeros((c.shape[0], n_s, n_s))
        for j, rxn in enumerate(self.reactions):
            idx = [(self._index[code], nu) for code, nu in rxn.reactants.items()]
            for m, nu_m in idx:
                if nu_m == 0.0:
                    continue
                d = rxn.k * nu_m * np.where(
                    c[:, m] > 0.0, c[:, m] ** (nu_m - 1.0), 0.0
                )
                for i, nu_i in idx:
                    if i != m:
                        d = d * c[:, i] ** nu_i
                J[:, :, m] += self.nu_net[:, j][None, :] * d[:, None]
        return J

    def jacobian(self, t: float, conc: np.ndarray) -> np.ndarray:
        """解析 Jacobian d(dC/dt)/dC，形状 (n_species, n_species)。

        对每条反应 j：d r_j / d C_m = k_j · ν_mj · C_m^(ν_mj − 1) · Π_{i≠m} C_i^(ν_ij)
        （幂为 0 的项不参与连乘，避免 0^0 与除零）。
        """
        n = self.n_species
        J = np.zeros((n, n))
        for j, rxn in enumerate(self.reactions):
            k = rxn.k
            idx = [(self._index[c], nu) for c, nu in rxn.reactants.items()]
            # 逐物质求偏导
            for m, nu_m in idx:
                if nu_m == 0.0:
                    continue
                if conc[m] == 0.0 and nu_m < 1.0:
                    continue  # 非整数级且浓度为零 → 导数奇异，跳过（物理上该反应已停）
                d = k * nu_m * conc[m] ** (nu_m - 1.0)
                for i, nu_i in idx:
                    if i != m:
                        d *= conc[i] ** nu_i
                J[:, m] += self.nu_net[:, j] * d
        return J

    # ── 便捷构造 ────────────────────────────────────────────
    def initial(self, **conc: float) -> np.ndarray:
        """按物质名给初值，未指定的取 0。"""
        c = np.zeros(self.n_species)
        for code, v in conc.items():
            if code not in self._index:
                raise KeyError(f"未知物质 {code!r}；现有：{self.species}")
            c[self._index[code]] = v
        return c

    def names(self) -> list[str]:
        return list(self.species)
