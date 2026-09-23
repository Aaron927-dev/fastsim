# COMSOL 脚本化（mph + Java API）踩坑记录

> 环境：COMSOL Multiphysics 6.0.318（`D:\COMSOL\COMSOL60\`）+ `mph` 1.3.1 + Python 3.12
> 授权：Electrochemistry / Chemical Reaction Engineering / Optimization / CFD /
> Microfluidics / Fuel Cell & Electrolyzer 等 47 个模块全部可用
> 会话日期：2026-09-18

这份记录的目的不是"吐槽"，而是**量化"依赖 COMSOL"的隐性成本**：
下面每一条都是为了让一个**一维线性对流-扩散-反应问题**跑起来而付出的真实代价。
自研求解器没有这些成本 —— 这是它除了速度之外的第二重价值。

---

## 一、已验证可用的部分（可复用）

### 1. 环境连通
```python
import os
os.environ["COMSOL_ROOT"] = r"D:\COMSOL\COMSOL60"
os.environ["PATH"] = r"D:\COMSOL\COMSOL60\Multiphysics\bin\win64" + os.pathsep + os.environ["PATH"]
import mph
client = mph.start(cores=2)      # 冷启动 13–17 s
```
- `detect_comsol()` 在 `D:\COMSOL\` 下按目录名含 "COMSOL" 且存在
  `Multiphysics\bin\win64\comsol.exe` 判定 → 实际命中 `COMSOL60`
- 授权模块列表：`client.modules()`（返回 47 项）

### 2. model 与组件的正确获取方式
```python
model = client.create("name")     # mph.Model
m = model.java                    # com.comsol.model.Model
comp = m.component().create("comp1", True)
```
⚠️ `client.java.model(name)` **不可用** —— `ModelUtil.model(String)` 的签名不接受
Python 传参（报 "No matching overloads"）。必须走 `client.create(...).java`。

### 3. 1D 几何（Interval）
```python
g = comp.geom().create("geom1", 1)
g.create("i1", "Interval").set("coord", [0.0, L])    # ★ 属性名是 coord，不是 min/max
g.run()
```

### 4. 稀物质传递接口
```python
p = comp.physics().create("tds", "DilutedSpecies", "geom1")   # ★ 类型名 DilutedSpecies
```
可用节点（1D）：

| 节点 | 类型 | 维度 | 关键属性 |
|---|---|---|---|
| `cdm1` | ConvectionDiffusionMigration | 域 | `u_src`（`"userdef"`/`"fromCommonDef"`）、`u`（**长度必须为 3**）、`D_c`、`DiffusionCoefficientSource`（`"mat"`/`"chem"`）|
| `reac1` | Reactions | 域 | `R_c`（如 `"-kr*c"`）|
| `init1` | init | 域 | `initc` |
| `in1` | Inflow | 边界 | `c0`、`BoundaryConditionType` |
| `out1` | Outflow | 边界 | — |
| `cL1` | Concentration | 边界 | `c0`（**注意不是 `c0_c`**）|
| `nflx1` | NoFlux（默认）| 边界 | 后加边界条件时其选择域会**自动收缩**（实测 [1,2]→[2]→[]）|

⚠️ **速度矢量必须给 3 个分量**，即便模型是 1D：
`cdm.set("u", ["0.01[m/s]", "0", "0"])`；给 1 个报 "A vector of length X expected#3"。

### 5. 边界条件语义（实测判定，非文档推断）
COMSOL 的 **`Inflow` 节点默认是 Dirichlet 口径**，不是化工标准的 Danckwerts：
- `BoundaryConditionType` 报告 `ConcentrationConstraint`
- 与 Dirichlet 解析解偏差 **0.0000%**，与 Danckwerts 解析解偏差 **2.38%**

⇒ 对标时必须声明入口口径，否则两边解不可比（低 Pe 下差异可达 60%+）。

---

## 二、尚未打通的部分（阻塞点）

### 现象
1D tds 稳态问题（纯扩散、两端 Dirichlet、或含对流+反应）经脚本求解后，
**返回的"解"恒等于初始值**（`initc`）：

| 实验 | 设置 | 返回 |
|---|---|---|
| 纯扩散，两端 Dirichlet 1 / 0，initc=0 | 解析应线性 1→0 | c ≡ 0 |
| 同上，initc=5 | 稳态解与初值无关 | c ≡ 5 |
| 对流+反应，Inflow+Outflow，initc=0 | 解析应 1→0.386 | c ≡ 0 |
| 同上，initc=1 | 同上 | c ≡ 1 |

数据集自洽（`dset1` → `sol1`），但 `sol1` 里存的就是初值。

### 已排除的原因
- ✅ 物理场未激活 —— `st1.activate` 读回 `['tds','on','frame:spatial1','on',...]`
- ✅ 域选择为空 —— `physics.selection()` = `[1]`，`cdm1.selection()` = `[1]`
- ✅ 默认 NoFlux 覆盖了新边界条件 —— 实测自动收缩，覆盖链正常
- ✅ 参数未解析 —— `cin`/`kr`/`Dc`/`u0` 均能读回并在表达式中生效（`tds.u` 求值得 0.01）
- ✅ 网格问题 —— 手动建 mesh 与让求解器自建，结果相同
- ✅ 求解器序列未生成 —— `createAutoSequence("std1")` 生成 `st1/v1/s1` 三步，齐全
- ✅ 求解未执行 —— `runAll()` 正常返回，无异常

### 主要剩余假设（未验证）
**该 API 路径下 `tds` 接口可能没有实际生效的自由度（species 未建立）。**
GUI 中添加「稀物质传递」会自动带一个名为 `c` 的物质；脚本创建时该物质是否建立未见保证，
且 `Species` 属性在本版本 API 中没有可读接口（`getStringArray("Species")` 不可用）。
若自由度为空，方程系统退化，解即初值 —— 与全部观测一致。

### 另一条致命弯路（记录以免重蹈）
```python
m.study("std1").createAutoSequences("tds")   # ★ 不产生任何研究步（静默失败）
```
该方法**不报错但不产生步骤**，导致"求解成功"却什么都没算 —— 表现为解恒为初值。
正确写法是在**求解器**上生成并传入**研究标签**：
```python
m.sol().create("sol1")
m.sol("sol1").createAutoSequence("std1")     # 生成 st1 / v1 / s1
m.sol("sol1").runAll()
```

---

## 五、后续补充：从零建模型的真正根因（2026-09-19 查明）

此前「从零建 tds/SCD 模型 → 求解成功但解恒为初值」的困惑，**根因已查明**。

### ★ 根因：物性默认来源是「材料」，而模型里没有材料

```
未定义"电解质 1"所需的材料属性"sigmal"
```

COMSOL 的物理场物性（电导率 `sigmal`、扩散系数 `D_c` 等）默认来源是
**Material 节点**。用脚本从零建模型时**不会自动创建材料**，
于是物性未定义 → 方程系统退化 → 求解器只能返回初值（不报错！）。

**正确做法**：把来源显式改为用户自定义，再给值：

```python
ice.set("sigmal_mat", "userdef")     # SCD 电解液电导率
ice.set("sigmal", "kappa")

cdm.set("D_c_mat", "userdef")        # tds 扩散系数（注意是 D_c_mat）
cdm.set("D_c", "D")
```

这条修正后，从零建的 SCD 模型**立即求解出非平凡解**
（φ ∈ [−0.994, −0.515] V）。

> 教训：COMSOL 的「静默失败」不限于 API 属性名 —— **方程系统退化也不报错**，
> 只给一个平凡解。这类失败必须靠「解是否非平凡」来判据，不能只看有没有抛异常。

### ★ 默认 Insulation 覆盖全部边界，且不会自动收缩

```python
print(list(cd.feature("ins1").selection().entities()))
# 建立电极边界条件后仍然是 [1, 2, 3, 4] —— 没有自动排除电极边界
```

后果：电极被一并绝缘 → 解恒为 0。而且该默认节点：

- `ins1.selection().set([1, 3])` → **`Selection is not editable`**（选择类型是「全部边界」）
- `cd.feature().remove("ins1")` → **`Object cannot be removed`**（默认强制节点）

实测还发现：单独用 `ElectricPotential` 定电位会被默认 Insulation 压过（解恒为 0）；
但加上 `ElectrodeSurface` 后能产生非平凡解。**这两者的优先级关系尚未查清**，
是环形 SCD 对标未完成的直接原因。

### tds 属性名与创建维度（供参考）

| 项 | 正确写法 | 备注 |
|---|---|---|
| 扩散系数来源 | `D_c_mat = "userdef"` | 还有 `DiffusionCoefficientSource`（"mat"/"chem"） |
| 速度 | `u = [ux, uy, uz]` | **必须 3 分量**，1D 也要 |
| 入口浓度 | `c0`（Concentration 节点） | 不是 `c0_c` |
| 初始值 | `initc` | — |
| 特征创建维度 | `create(tag, type, edim)` | **edim 是单元维度**：2D 域=2、边界=1；1D 域=1、边界=0 |
| 网格单元数 | `mesh.getNumElem("tri")` | 2D 自由三角形；"domain" 会报错 |
| 时间列表 | `range(start, step, stop)` | **中间是步长** |
| 电极动力学 | `er1` 子节点：`i0`/`alphaa`/`alphac`/`Eeq` | **无下划线**；`ElectrodeKinetics` 默认 `LinButlerVolmer` |
| 电极电位 | `ElectricPotential.phisbnd` | 不是 `V0`/`phisext` |

### 几何长度单位（重要）

应用库模型的几何常带长度单位。例如
`transport_and_adsorption.mph` 的 `Rectangle(pos=[0,-0.1], size=[0.1,0.3])`
**单位是 mm**，实际尺寸是 1e-4 × 3e-4 m，不是 0.1 × 0.3 m。
对接参数前必须先确认单位，否则差 1000 倍。

---

## 六、SCD（二次电流分布）专项：电极边界语义（2026-09-22 补充）

尝试自建环隙 SCD 基准做电化学对标，卡在电极语义上。
以下是把问题缩小到最小的实测记录 —— **这些 API 事实本身是可复用的**。

### ★ 电极电位的正确设置方式

对照官方可用模型 `Electrochemical_Engineering/wire_electrode.mph` 得出：

```python
es = cd.feature().create("es1", "ElectrodeSurface", 1)
es.set("BoundaryCondition", "ElectricPotential")   # ★ 先设这个
es.set("phisext0", "Vcell")                        # ★ 电位属性
```

- `phisext0` 是电极的外加电位；**只有在 `BoundaryCondition="ElectricPotential"`
  之后才会出现在属性列表里** —— 这就是此前反复找不到电位属性名的原因
- 此前用另建的 `ElectricPotential` 边界节点设电位，**会被默认 Insulation 压过
  而得到平凡解 φ≡0**（实测）
- 官方模型中：阴极 `phisext0` 不设（默认 0），阳极设为 `Ecell`

### 电极动力学（子节点）

```python
er = es.feature("er1")            # ElectrodeReaction
er.set("ElectrodeKinetics", "ButlerVolmer")   # 默认是 LinButlerVolmer（线性化）
er.set("i0Type", "userdef"); er.set("i0", "i0")
er.set("alphaa", "0.5"); er.set("alphac", "0.5")     # 注意：无下划线
er.set("Eeq_mat", "userdef"); er.set("Eeq_ref", "0[V]"); er.set("Eeq", "0[V]")
```

### ★ entitydim 必须传字符串（Java 重载歧义）

```python
sel.set("entitydim", "1")     # ✓
sel.set("entitydim", 1)       # ✗ Ambiguous overloads:
                              #   set(String,boolean) vs set(String,int)
```

### Box 选择按「包围盒」匹配，无法精确定位单条边

```python
sel = comp.selection().create("b1", "Box")   # 必须建在**组件**上，不是几何上
sel.set("entitydim", "1")
sel.set("xmin", ...); sel.set("xmax", ...)
```

实测：用 `x∈[R_A±1e-6]` 只想选内壁那一条边，却命中 **[1,2,3]** 三条 ——
因为上下两条边的包围盒横跨整个 x 范围，与窄框重叠。故**不能用窄 Box 选单条边**。

### ★ 已解决：边界编号 → 几何位置的映射（2026-09-22 查明，前一版结论有误）

**前一版记录的「无法确定编号 1–4 各对应哪条边」是错的 —— 映射是确定且可复现的。**

真值（`Rectangle(pos=[0,0], size=[gap, L])`，笛卡尔与轴对称均适用）：

| 编号 | 笛卡尔 (x,y) | 轴对称 (r,z) | 长度 |
|---|---|---|---|
| 1 | x=0（左） | r=r_inner（内壁） | 长边 |
| 2 | y=0（下） | z=0（入口端） | 短边 |
| 3 | y=L（上） | z=L（出口端） | 短边 |
| 4 | x=gap（右） | r=r_outer（外壁） | 长边 |

**与「标准矩形编号 1=底 2=右 3=顶 4=左」相比，1 与 2 是互换的** —— 这就是
此前所有尝试都不自洽的**唯一**原因。平行板/环隙的两个电极边是 **1 与 4**，
不是我一直用的 4 与 2。

**如何测出来的（可复用的方法）**，两步：

1. **先用最简界面做一次「几何真值」实验**：`ec`（Java 类型名 `ConductiveMedia`，
   注意**不是** `ElectricCurrents`）只有 `Ground` / `ElectricPotential` 两种边界节点，
   无动力学、无子节点、无材料来源坑。一条边接地、一条边给电位，求解后
   **看哪条边的 φ 恒为 0、哪条恒为 V** —— 即可读出映射。
2. **判定电极位置必须用「精确落在边界上的网格节点」**，不能用带宽条带。
   判据：`x_n <= 1e-9` / `y_n >= L-1e-9` 等。早先用「2% 带宽条带」统计 φ，
   条带会同时覆盖两条边的邻域，导致同一编号被推断成两条不同的边 ——
   这是前一版得出错误结论的直接原因。

**为什么之前那张「电流沿轴流」的云图会误导**：电极设在 (4,2) 时，4=右、2=下
是**相邻的两条边**，电流只能沿角部斜着走，于是出现「全域恒为 2 V、
压降挤在一条 0.006 m 薄层里」的怪场型。它看起来像轴向电流，实际是
**角部邻边电极**的正常现象。同理 (1,3) 是另一组邻边，给出镜像结果。

**同时被否证的假设**：「2D 轴对称要求计算域触及轴线 r=0」——错。
`r∈[0.005,0.01]` 的环隙在轴对称下正常求解，`r=r_inner` 边 φ 恒为 2.000000、
`r=r_outer` 边恒为 0.000000，跨度 0。

### 结论（更正版）

**SCD 基准已在 COMSOL 中建成并通过验证**，见 `bench/comsol_scd_benchmark.py`
与 `bench/threeway_annulus.py`：

| 算例 | 解析解 | COMSOL | COMSOL 偏差 |
|---|---|---|---|
| 笛卡尔平行板（一次分布） | 线性 φ(x) | 斜率 −204.96 vs −200 | 0.89% |
| 笛卡尔平行板（BV, i0=1e-3/1e-1/1） | 一维欧姆+BV | — | 3.08% / 2.34% / 2.84% |
| 环隙一次分布 | I′=18.129441 A/m | 18.164839 | +0.20% |
| 环隙二次分布 | I′=6.982952 A/m | 7.117602 | +1.93% |

**并且实测出 COMSOL 自身的数值误差是分辨率限制的**（笛卡尔平行板网格收敛扫描）：

| 每 gap 单元数 | RMS vs 解析 | y 向跨度（应=0） |
|---|---|---|
| 16 | 1.851e−2 | 1.271e−3 |
| 32 | 1.775e−2 | 7.580e−4 |
| 64 | 6.480e−3 | 3.279e−4 |

y 向跨度按二阶收敛（比值 1.68、2.31）。**即在最平凡的平行板问题上，
COMSOL 自身也带 0.3–0.9% 数值误差 —— 这直接标定了「达到 COMSOL 90% 精度」
这个目标的天花板：COMSOL 不是精确参照物。**

### 仍未解决 / 已知限制

- `i0 ≳ 10 A/m²` 时 COMSOL 牛顿法不收敛（「没有返回所有参数步长」），
  需延拓/斜坡。i0=1 可用。
- 边界数据集取坐标（`dataset().create(tag,"Edge")` + `Eval`）在本版本仍报错，
  故第 1 步的「最简界面反推法」是目前唯一的映射测定手段。
- `m.result().plotGroup` 在 Java 客户端不存在（`ResultsClient` 无此属性），
  导出结果图需另找路径。

---

## 三、成本量化（这是本记录的核心价值）

要让一个**一维线性问题**跑通，本会话付出的真实代价：

| 项目 | 次数/耗时 |
|---|---|
| COMSOL 冷启动 | 13–17 s / 次（每次对比都要重来） |
| API 属性名试错（coord / c0 / u_src / DiffusionCoefficientSource / 类型名…） | 12+ 轮探测脚本 |
| 静默失败的坑 | 3 处（`createAutoSequences` 无步骤、`getReal()` 只返回首点、`u` 长度校验） |
| 到达"模型能建+能求解+能取值" | 约 2 小时 |
| 最终状态 | **仍未拿到正确的解**（卡在自由度/物种建立） |

对比：同一个 1D 对流-扩散-反应问题，`fastsim` 从零实现 + 27 项解析解验证
（含解析 Jacobian 对有限差分、平推流/全混流极限、质量守恒）用了**同一时段**
且**已全部通过**，单次求解 **16–20 ms**。

> **结论**：对本类问题，「自研求解器」不只是快，**开发和调试的总成本也更低** ——
> 因为方程、离散、边界条件都在自己的代码里，出错时能读能改；
> 而 COMSOL 的脚本化是"黑箱 + 静默失败"，一个属性名猜错就在错误的方向上耗掉大量时间。

---

## 四、后续若要继续打通 COMSOL 对标，建议的下一步

1. **先用 GUI 保存一个能跑通的 1D tds 模型（.mph）**，再用
   `m = m.physics()` 逐节点 dump 属性，与脚本建出的模型做 `diff` —— 这是最快的定位方式。
2. 或改用 `mph` 官方示例模型作为模板（`client.load(路径)`），只改参数，不改结构。
3. 或绕开 tds，用 **系数型偏微分方程（PDE, Coefficient Form）** 接口直接写
   对流-扩散-反应方程 —— 自由度明确、属性名少、异常行为也少，更适合做数值对标。
4. 对标的意义仅在于**独立交叉验证**；`fastsim` 已对**解析解**完成验证（更强基准）。
   COMSOL 交叉验证可以延后到真正需要"复杂几何高保真"时再做。
