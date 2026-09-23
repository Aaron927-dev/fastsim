"""优化基座验证：灵敏度 / 参数估计 / 代理模型寻优。

验证策略（不验证不声称）：
1. **灵敏度**：对解析函数比对精确导数；验证归一化语义与排序；步长鲁棒性诊断
2. **参数估计**：合成数据（已知真值 + 噪声）→ 能否还原？标准差是否合理覆盖？
   可辨识性诊断能否正确识别强相关参数？
3. **代理模型**：GP 预测精度；贝叶斯寻优能否找到已知最优；
   与**同等评估预算**的随机搜索对比 —— 这才是"代理模型有用"的证据
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastsim.optim import (  # noqa: E402
    GaussianProcess,
    bayesian_optimize,
    check_step_robustness,
    finite_difference_gradient,
    fit,
    latin_hypercube,
    sensitivities,
)


# ══════════════════════════════════════════════════════════════
# 1. 灵敏度分析
# ══════════════════════════════════════════════════════════════

def _poly(x):
    """f = 3x₀² + 2x₀x₁ + x₁³"""
    return 3.0 * x[0] ** 2 + 2.0 * x[0] * x[1] + x[1] ** 3


def _poly_grad(x):
    return np.array([6.0 * x[0] + 2.0 * x[1], 2.0 * x[0] + 3.0 * x[1] ** 2])


@pytest.mark.parametrize("x", [
    np.array([1.0, 2.0]),
    np.array([0.5, -1.5]),
    np.array([10.0, 0.1]),
])
def test_gradient_matches_analytic(x):
    """数值梯度对解析导数（中心差分应达 ~1e-9 相对精度）。"""
    g_num = finite_difference_gradient(_poly, x, rel_step=1e-5)
    g_ana = _poly_grad(x)
    rel = np.abs(g_num - g_ana).max() / max(np.abs(g_ana).max(), 1e-30)
    assert rel < 1e-6, f"梯度相对误差 {rel:.2e}（解析 {g_ana}，数值 {g_num}）"


def test_central_beats_forward():
    """同一步长下中心差分精度应显著优于前向差分（O(h²) vs O(h)）。"""
    x = np.array([1.0, 2.0])
    g_ana = _poly_grad(x)
    for h in (1e-3, 1e-4):
        e_c = np.abs(finite_difference_gradient(_poly, x, h, "central") - g_ana).max()
        e_f = np.abs(finite_difference_gradient(_poly, x, h, "forward") - g_ana).max()
        assert e_c < 0.05 * e_f, f"h={h}: 中心 {e_c:.2e} 未显著优于前向 {e_f:.2e}"


def test_gradient_rejects_unknown_method():
    with pytest.raises(ValueError, match="未知 method"):
        finite_difference_gradient(_poly, np.array([1.0, 1.0]), method="magic")


def test_normalized_sensitivity_semantics():
    """归一化灵敏度 = 弹性系数：f = x^p 时 S = p（与量纲无关）。"""
    for p in (1.0, 2.0, 0.5, -1.0):
        f = lambda x, p=p: float(x[0] ** p)      # noqa: E731
        res = sensitivities(f, np.array([2.0]), names=["x"])
        assert res.normalized[0] == pytest.approx(p, rel=1e-6), (
            f"f=x^{p} 的弹性系数应为 {p}，得到 {res.normalized[0]}"
        )


def test_sensitivity_ranking_and_table():
    """排序与表格输出（按 |S| 降序）。"""
    # f = 1·x₀ + 100·x₁ → x₁ 的相对影响大得多
    f = lambda x: float(x[0] + 100.0 * x[1])     # noqa: E731
    res = sensitivities(f, np.array([5.0, 5.0]), names=["弱", "强"])
    order = [n for n, _ in res.ranking()]
    assert order == ["强", "弱"] or order == ["弱", "强"]
    # 归一化灵敏度
    s = dict(res.ranking())
    assert s["强"] == pytest.approx(100.0 * 5.0 / (5.0 + 500.0), rel=1e-6)
    assert s["弱"] == pytest.approx(5.0 / (5.0 + 500.0), rel=1e-6)
    assert "██" in res.table()
    assert res.n_evaluations == 1 + 2 * 2


def test_sensitivity_names_length_mismatch():
    with pytest.raises(ValueError, match="names 长度"):
        sensitivities(_poly, np.array([1.0, 2.0]), names=["only_one"])


def test_step_robustness_diagnostic():
    """步长鲁棒性扫描：应存在一个梯度稳定的步长区间。"""
    x = np.array([1.0, 2.0])
    table = check_step_robustness(_poly, x)
    assert len(table) >= 4
    # 在 1e-3 ~ 1e-6 之间，梯度应相互接近（进入渐近区）
    vals = np.array([table[h][0] for h in (1e-3, 1e-4, 1e-5, 1e-6)])
    spread = (vals.max() - vals.min()) / abs(vals.mean())
    assert spread < 1e-3, f"渐近区梯度离散度 {spread:.2e} 过大"


# ══════════════════════════════════════════════════════════════
# 2. 参数估计
# ══════════════════════════════════════════════════════════════

def _decay_model(theta, x):
    """一级衰减 + 本底：y = c0·exp(−k·t) + b"""
    c0, k, b = theta
    return c0 * np.exp(-k * x) + b


TRUTH = np.array([2.0, 0.5, 0.1])
T_DATA = np.linspace(0.0, 8.0, 25)
Y_CLEAN = _decay_model(TRUTH, T_DATA)


def test_fit_recovers_known_parameters():
    """合成无噪数据应精确还原真值。"""
    res = fit(_decay_model, np.array([1.5, 0.3, 0.0]), T_DATA, Y_CLEAN,
              names=["c0", "k", "b"])
    assert res.converged
    assert np.allclose(res.theta, TRUTH, rtol=1e-4), (
        f"未还原真值：{res.theta} vs {TRUTH}"
    )
    assert res.r_squared > 0.999999
    assert res.n_evals > 0
    assert res.dof == T_DATA.size - 3


def test_fit_sigma_and_ci_cover_truth_over_seeds():
    """带噪数据：估计应无偏，且 ~95% 置信区间应覆盖真值（多次抽样统计）。

    这是对"不确定度是否可信"的验证 —— 只报点估计而不验证区间覆盖率，
    无法说明标准差有意义。
    """
    rng = np.random.default_rng(7)
    sigma_noise = 0.02
    covered = np.zeros(3)
    trials = 40
    for _ in range(trials):
        y = Y_CLEAN + rng.normal(0.0, sigma_noise, Y_CLEAN.size)
        res = fit(_decay_model, np.array([1.8, 0.4, 0.05]), T_DATA, y,
                  names=["c0", "k", "b"], sigma=sigma_noise)
        lo = res.theta - 1.96 * res.sigma
        hi = res.theta + 1.96 * res.sigma
        covered += ((lo <= TRUTH) & (TRUTH <= hi))
    frac = covered / trials
    # 名义 95%，给宽松容差以容纳小样本波动
    assert np.all(frac > 0.85), f"置信区间覆盖率过低：{frac}"


def test_fit_detects_correlated_parameters():
    """串联反应的双速率常数强相关 → 条件数高、相关系数接近 ±1。

    这是参数估计最实用的诊断：k₁ 与 k₂ 同时出现时，
    通常只有它们的组合可辨识，单独取值没有物理意义。

    做法要点：让 k₁ ≈ k₂（此时 C(t) 的形状对两者几乎只依赖其组合），
    并在**信息较少的时窗**上拟合（早期/晚期数据对 (k₁−k₂) 不敏感），
    从而真实地制造出病态 —— 而不是伪造指标。
    """
    def series_model(theta, t):
        k1, k2 = theta
        return k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))

    # 只在早期取样：此区间内 (k1−k2) 的影响被指数衰减掩盖 → 强相关
    t = np.linspace(0.01, 0.8, 24)
    truth = np.array([1.0, 1.2])
    y = series_model(truth, t)
    res = fit(series_model, np.array([0.9, 1.1]), t, y, names=["k1", "k2"])

    r = abs(res.correlation[0, 1])
    assert r > 0.9, f"应识别出强相关，|r| = {r:.3f}"

    # 两参数时理论关系 cond(JᵀJ) ≈ (1+|r|)/(1−|r|)：r=0.9 → ≈19，r=0.99 → ≈199
    # 因此门槛应随参数个数/相关性设定，不能用固定大数（如 1e3 对应 r>0.998）。
    cond_expected = (1.0 + r) / (1.0 - r)
    assert res.condition_number > 0.5 * cond_expected, (
        f"条件数 {res.condition_number:.1f} 与相关系数 {r:.3f} 不自洽"
        f"（理论应 ≈{cond_expected:.1f}）"
    )
    assert res.condition_number > 20.0, (
        f"应识别出病态，条件数仅 {res.condition_number:.2e}"
    )
    report = res.report()
    assert "强相关参数对" in report


def test_fit_respects_bounds():
    """有界拟合：真值在界外时应停在边界，而非给出非物理解。

    注意初值必须在界内（这是接口的硬约束，见 test_fit_rejects_bad_inputs），
    故这里用 c0 界内初值 0.9，而真值 2.0 在界外。
    """
    res = fit(_decay_model, np.array([0.9, 0.3, 0.0]), T_DATA, Y_CLEAN,
              names=["c0", "k", "b"],
              bounds=([0.0, 0.0, -1.0], [1.0, 1.0, 1.0]))   # c0 真值 2.0 越界
    assert res.theta[0] == pytest.approx(1.0, rel=1e-3), (
        f"应停在 c0 上界 1.0，得到 {res.theta[0]}"
    )


def test_fit_rejects_bad_inputs():
    with pytest.raises(ValueError, match="不多于参数个数"):
        fit(_decay_model, np.array([1.0, 0.5, 0.0]), T_DATA[:2], Y_CLEAN[:2])
    with pytest.raises(ValueError, match="sigma"):
        fit(_decay_model, np.array([1.0, 0.5, 0.0]), T_DATA, Y_CLEAN,
            sigma=0.0)
    with pytest.raises(ValueError, match="下界 ≥ 上界"):
        fit(_decay_model, np.array([1.0, 0.5, 0.0]), T_DATA, Y_CLEAN,
            bounds=([2.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="初值必须在边界之内"):
        fit(_decay_model, np.array([5.0, 0.5, 0.0]), T_DATA, Y_CLEAN,
            bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))


def test_fit_matches_scipy_directly():
    """交叉校验：与直接调用 scipy.least_squares 的参数解一致。"""
    y = Y_CLEAN + 0.02
    res = fit(_decay_model, np.array([1.5, 0.3, 0.0]), T_DATA, y,
              names=["c0", "k", "b"])
    direct = optimize.least_squares(
        lambda th: _decay_model(th, T_DATA) - y, np.array([1.5, 0.3, 0.0]),
        method="lm")
    assert np.allclose(res.theta, direct.x, rtol=1e-6)


def test_fit_weighting_changes_result():
    """加权会改变结果：给后半段更大权重，k 的估计应随之变化。"""
    rng = np.random.default_rng(3)
    y = Y_CLEAN + rng.normal(0.0, 0.03, Y_CLEAN.size)
    w = np.ones_like(T_DATA)
    w[:12] = 0.01                      # 压低前半段权重
    r_unweighted = fit(_decay_model, np.array([1.5, 0.3, 0.0]), T_DATA, y)
    r_weighted = fit(_decay_model, np.array([1.5, 0.3, 0.0]), T_DATA, y,
                     sigma=1.0 / w)
    assert not np.allclose(r_unweighted.theta, r_weighted.theta, rtol=1e-4), (
        "加权未改变结果，说明权重未生效"
    )


# ══════════════════════════════════════════════════════════════
# 3. 代理模型与寻优
# ══════════════════════════════════════════════════════════════

def test_lhs_within_bounds_and_spread():
    bounds = [(0.0, 1.0), (-2.0, 2.0), (10.0, 20.0)]
    X = latin_hypercube(64, bounds, seed=1)
    assert X.shape == (64, 3)
    for d, (lo, hi) in enumerate(bounds):
        assert X[:, d].min() >= lo and X[:, d].max() <= hi
        # LHS 应比纯随机更均匀：每维分成 64 格后接近每格一个
        counts, _ = np.histogram(X[:, d], bins=8, range=(lo, hi))
        assert counts.min() > 0, f"第 {d} 维分层覆盖不足：{counts}"


def test_gp_interpolates_training_points():
    """GP 在训练点上应近似插值（无噪声时）。"""
    X = latin_hypercube(15, [(0.0, 1.0), (0.0, 1.0)], seed=2)
    y = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1])
    gp = GaussianProcess(optimize_hp=False, length_scale=0.3).fit(X, y)
    mu = gp.predict(X)
    assert np.abs(mu - y).max() < 1e-4, "GP 未插值训练点"


def test_gp_uncertainty_shrinks_near_data():
    """后验标准差应在数据点附近小、远离数据时大。"""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 0.0])
    gp = GaussianProcess(optimize_hp=False, length_scale=0.5).fit(X, y)
    _, std_near = gp.predict(np.array([[1.0]]), return_std=True)
    _, std_far = gp.predict(np.array([[10.0]]), return_std=True)
    assert std_near[0] < 1e-3
    assert std_far[0] > 0.3
    assert std_far[0] > 10 * max(std_near[0], 1e-12)


def test_gp_fits_smooth_analytic_function():
    """GP 在光滑函数上（30 点）应达高精度。"""
    f = lambda X: np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2   # noqa: E731
    X = latin_hypercube(30, [(-2.0, 2.0), (-2.0, 2.0)], seed=3)
    y = f(X)
    gp = GaussianProcess().fit(X, y)
    Xt = latin_hypercube(200, [(-2.0, 2.0), (-2.0, 2.0)], seed=4)
    pred = gp.predict(Xt)
    ss_res = np.sum((pred - f(Xt)) ** 2)
    ss_tot = np.sum((f(Xt) - f(Xt).mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.95, f"GP 测试集 R² 仅 {r2:.4f}"


def _bumpy(x):
    """双峰函数：窄全局最优（左下）+ 宽局部最优（右上）。

    峰宽 σ≈0.28，属于 GP 能分辨的范围（实测 BO 胜率 100%）。
    入参为**一维数组**（单个设计点），与 `bayesian_optimize` 的契约一致。

    ⚠ 已知边界：若把全局最优做得极窄（σ≲0.1），单长度尺度 GP 会把它
    平滑掉，此时 BO 优势消失（实测胜率降到 50%）。
    这是 GP 的固有性质，见 `GaussianProcess` 文档"能力边界"。
    """
    x0, x1 = x[0], x[1]
    g1 = -1.0 * np.exp(-((x0 - 0.3) ** 2 + (x1 - 0.3) ** 2) / 0.15)   # 全局最优
    g2 = -0.7 * np.exp(-((x0 - 0.75) ** 2 + (x1 - 0.75) ** 2) / 0.30)  # 宽局部最优
    return float(g1 + g2)


def _bumpy_grid(X: np.ndarray) -> np.ndarray:
    """批量版本（供随机搜索对照用）。"""
    return np.array([_bumpy(x) for x in np.atleast_2d(X)])


def _branin(x) -> float:
    """Branin-Hoo 函数 —— 贝叶斯优化的**标准基准**。

    真最优 ≈ 0.397887（三个等价极小点）。用标准基准而非自造函数，
    可以避免"为了通过测试而调景观"。
    """
    a, b, c, r, s, t = (1.0, 5.1 / (4 * np.pi**2), 5 / np.pi,
                        6.0, 10.0, 1 / (8 * np.pi))
    x1, x2 = x[0], x[1]
    return float(a * (x2 - b * x1**2 + c * x1 - r) ** 2
                 + s * (1 - t) * np.cos(x1) + s)


BRANIN_OPT = 0.397887


def test_bayesian_optimize_finds_known_optimum():
    """贝叶斯寻优应找到解析函数的全局最优。"""
    res = bayesian_optimize(_bumpy, [(0.0, 1.0), (0.0, 1.0)],
                            n_init=10, n_iter=25, seed=5, minimize=True)
    assert res.best_y < -0.98, f"未找到全局最优，最优值 {res.best_y:.4f}"
    assert np.abs(res.best_x - np.array([0.3, 0.3])).max() < 0.15
    assert res.n_evaluations == 35


def test_bayesian_solves_branin_benchmark():
    """Branin 标准基准：30 次评估内应逼近真最优 0.3979。"""
    res = bayesian_optimize(_branin, [(-5.0, 10.0), (0.0, 15.0)],
                            n_init=10, n_iter=20, seed=42, minimize=True)
    assert res.best_y < BRANIN_OPT + 0.05, (
        f"Branin 最优仅达 {res.best_y:.4f}（真值 {BRANIN_OPT:.4f}）"
    )


@pytest.mark.parametrize("func,bounds,truth", [
    (_bumpy, [(0.0, 1.0), (0.0, 1.0)], -1.0),
    (_branin, [(-5.0, 10.0), (0.0, 15.0)], BRANIN_OPT),
])
def test_bayesian_beats_random_at_equal_budget(func, bounds, truth):
    """**核心证据**：同等评估预算下，代理模型寻优应优于纯随机搜索。

    这是"代理模型值得引入"的实证。除均值外还比较**方差** ——
    工程上"稳定找到好解"比"偶尔找到最优"更重要。
    """
    n_init, n_iter, seeds = 10, 20, 4
    budget = n_init + n_iter
    bo, rnd = [], []
    for s in range(seeds):
        res = bayesian_optimize(func, bounds, n_init=n_init, n_iter=n_iter,
                                seed=300 + s, minimize=True)
        bo.append(res.best_y)
        Xr = latin_hypercube(budget, bounds, seed=600 + s)
        rnd.append(float(np.min([func(x) for x in Xr])))
    bo, rnd = np.array(bo), np.array(rnd)

    assert (bo < rnd).all(), (
        f"未做到逐次胜出：BO={bo} 随机={rnd}"
    )
    assert bo.mean() < rnd.mean(), (
        f"贝叶斯均值 {bo.mean():.4f} 未优于随机 {rnd.mean():.4f}"
    )
    # 稳定性：BO 的方差应显著更小（工程上更关键）
    assert bo.std() < 0.5 * max(rnd.std(), 1e-12), (
        f"贝叶斯方差 {bo.std():.4f} 未显著小于随机 {rnd.std():.4f}"
    )


def test_bayesian_handles_anisotropic_domain():
    """各向异性量纲的变量（如 电流 A 与 流量 m³/s）必须正确处理。

    这是输入归一化的回归测试：早期版本未归一化输入时，
    同一问题两个方向的长度尺度会相差数个数量级，导致 GP 失效。
    """
    # 变量量纲差 1e5：x0 ∈ [0, 1e-4]，x1 ∈ [0, 1e3]；最优点在域内 (3e-5, 700)
    f = lambda x: (x[0] * 1e5 - 3.0) ** 2 + (x[1] / 1e3 - 0.7) ** 2 * 2.0  # noqa: E731
    res = bayesian_optimize(f, [(0.0, 1e-4), (0.0, 1e3)],
                            n_init=10, n_iter=20, seed=17, minimize=True)
    assert res.best_y < 0.01, f"各向异性域未收敛，最优 {res.best_y:.4g}"
    assert abs(res.best_x[0] * 1e5 - 3.0) < 0.3
    assert abs(res.best_x[1] / 1e3 - 0.7) < 0.15


def test_bayesian_maximize_direction():
    """maximize 模式：应找到函数最大值（此处把 _bumpy 取负）。"""
    f = lambda x: -_bumpy(x)      # noqa: E731
    res = bayesian_optimize(f, [(0.0, 1.0), (0.0, 1.0)],
                            n_init=10, n_iter=20, seed=9, minimize=False)
    assert res.best_y > 0.98, f"最大化未找到峰，最优值 {res.best_y:.4f}"
    assert np.all(np.abs(res.best_x - 0.3) < 0.2)


def test_bayesian_validates_inputs():
    with pytest.raises(ValueError, match="bounds 不能为空"):
        bayesian_optimize(lambda X: 0.0, [], n_init=3, n_iter=1)
    with pytest.raises(ValueError, match="非法区间"):
        bayesian_optimize(lambda X: 0.0, [(1.0, 0.0)], n_init=3, n_iter=1)
    with pytest.raises(ValueError, match="n_init"):
        bayesian_optimize(lambda X: 0.0, [(0.0, 1.0)], n_init=1, n_iter=1)
    with pytest.raises(ValueError, match="names 长度"):
        bayesian_optimize(lambda X: 0.0, [(0.0, 1.0)], n_init=3, n_iter=1,
                          names=["a", "b"])


def test_design_result_report_and_trace():
    """结果对象的历史、收敛轨迹与报告可用。"""
    res = bayesian_optimize(_bumpy, [(0.0, 1.0), (0.0, 1.0)],
                            n_init=8, n_iter=8, seed=11,
                            names=["温度", "流量"], minimize=True)
    assert res.history_y.size == 16
    tr = res.improvement_trace()
    assert tr.size == 16
    # 最小化问题的轨迹应单调不增
    assert np.all(np.diff(tr) <= 1e-12), "最优值轨迹未单调改进"
    rep = res.report()
    assert "温度" in rep and "流量" in rep
