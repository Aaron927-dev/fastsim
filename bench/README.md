# bench 索引

本目录混有两类脚本，**请勿混淆**。

## A. 验证套件（可作为证据引用）

| 脚本 | 用途 | 是否需要 COMSOL |
|---|---|---|
| `mms_convergence.py` | 二维制造解法收敛阶（扩散/对流各格式） | 否 |
| `fastsim_radial_convergence.py` | 环隙径向离散误差：实测 vs 闭式预测 `R_FV/R_exact` | 否 |
| `timing_comparison.py` | fastsim vs COMSOL 同题耗时对比（口径见文件头） | 是 |
| `comsol_reference.py` + `comsol_reference.npz` | 由 COMSOL 官方应用库模型 `transport_and_adsorption.mph` 改造生成的 ADR 参考解 | 生成时需 |
| `cross_validate.py` | 三方对标：COMSOL / fastsim / 解析解（一维 ADR） | 否（用 .npz） |
| `resolution_check.py` | 证明 COMSOL 的 0.98% 偏差源于分辨率 | 否 |
| `bench_adr_vs_comsol.py` | 同上，早期版本 | 否 |
| `comsol_mesh_study.py` | COMSOL 侧网格研究 | 是 |
| `validate_vs_naturecomms.py` | 对 *Nat. Commun.* 16, 7175 (2025) 的顶刊案例验证 | 否 |
| **`comsol_scd_benchmark.py`** | **笛卡尔平行板 SCD 基准 + 网格收敛扫描** | 是 |
| **`threeway_annulus.py`** | **环隙 SCD 三方同题比对（COMSOL × fastsim × 解析解）** | 是 |

## B. 已被取代的探索脚本（**未随仓库发布**）

这些脚本建立在**错误的边界编号假设**上（把电极放在 4 与 2，实际应放 1 与 4），
因此它们得到的场型是「角部邻边电极」的正常现象，却被误读为 COMSOL 有根本性问题。
真值、测定方法与教训见 `docs/COMSOL脚本化踩坑记录.md` 第六节。
它们已被 `.gitignore` 排除，此处仅登记名字与失败原因，以免重蹈。

| 脚本 | 为何被取代 |
|---|---|
| `cartesian_scd_check.py` | 用错误的电极边 (4,2)/(1,3) 做隔离测试 |
| `scd_polygon_check.py` | 改用 Polygon 试图绕开编号问题（绕不开，问题不在几何类型） |
| `scd_ec_bisect.py` | 二分定位脚本，含已被否证的接口名猜测 |
| `scd_boundary_truth.py` | 用「2% 带宽条带」判读边界 —— **这个方法本身会同时覆盖两条边，是错误结论的直接来源** |
| `scd_parallel_plate.py` | 已含正确映射，但判据用了无效的一次分布靶（i0 太小，过电位达 0.58 V） |
| `scd_render_png.py` | PNG 导出失败（`m.result().plotGroup` 在 Java 客户端不存在） |
| `scd_field_map.py` | ASCII 云图诊断工具 —— **正是它破了案**（见下方 `comsol_field_map.py`） |

**ASCII 云图是诊断 COMSOL 场型的有效手段**：把 φ 按 (x,y) 分箱打印成文本云图，
一眼就能看出电流往哪走，不需要任何边界编号假设。需要时按上表最后一行的方法
重建（只用最简的 `ec`/`ConductiveMedia` 界面，与 SCD 的复杂节点无关）。

## 判读铁律（本轮教训）

1. **不要用「带宽条带」统计场量判读边界位置** —— 条带跨越两条边的邻域，
   会把同一编号推断成两条不同的边。用 `|x_n| ≤ 1e-9` 这类**精确边界节点**判据。
2. **不要用「标准矩形编号」假设**（1=底 2=右 3=顶 4=左）。实测编号是
   **1=x=0/r=r_inner、2=y=0/z=0、3=y=L/z=L、4=x=gap/r=r_outer**（1 与 2 互换）。
3. 判据靶子要与物理前提匹配：**i0 很小时过电位可达 0.58 V，一次分布（线性）靶无效**，
   必须用含欧姆降 + Butler-Volmer 的精确解。
4. 两个独立实现**同向偏高**时，先怀疑共有的离散误差，而不是物理模型差异。
