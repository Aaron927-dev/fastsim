"""COMSOL 环境配置 —— 集中一处，供 bench/ 下所有需要 COMSOL 的脚本复用。

为什么不写死安装路径
    "D:\\COMSOL\\COMSOL60" 只在本机成立，别人克隆后必然跑不起来。
    这里按优先级查找，并把结果写回环境变量（mph 依赖 COMSOL_ROOT / PATH）。

查找顺序
    1. 环境变量 COMSOL_ROOT（显式指定，最优先）
    2. 常见安装位置：各盘符 × COMSOL60 / COMSOL61 / COMSOL62 / COMSOL6x
    3. PATH 上的 `comsol` 可执行文件反推（Linux/macOS 常见）
    4. mph 自带的探测（mph.discovery，若可用）

用法
    from _comsol_env import setup_comsol
    root = setup_comsol()          # 返回 Path；找不到则抛 RuntimeError

设置方法（找不到时）
    项目内所有需要 COMSOL 的脚本都可以用环境变量覆盖：

        Windows (PowerShell)   $env:COMSOL_ROOT = "D:\\COMSOL\\COMSOL60"
        Windows (cmd)          set COMSOL_ROOT=D:\\COMSOL\\COMSOL60
        Git Bash / Linux / mac  export COMSOL_ROOT=/opt/comsol/COMSOL60
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# 版本目录名按新→旧尝试：新版本先命中
_VERSION_DIRS = ("COMSOL62", "COMSOL61", "COMSOL60", "COMSOL6", "COMSOL")
# 可执行文件相对安装根的路径（用于验证候选目录确实是安装根）
_BIN_SUBPATHS = (
    Path("Multiphysics") / "bin" / "win64",
    Path("bin"),
    Path("Multiphysics") / "bin" / "glnxa64",
    Path("Multiphysics") / "bin" / "maci64",
)
# 常见安装盘符（仅 Windows 有意义；不存在时 is_dir() 会直接跳过）
_DRIVES = ("C:", "D:", "E:", "F:")


def _is_comsol_root(p: Path) -> bool:
    """候选目录是否是 COMSOL 安装根：含 bin 目录且能找到 comsol 可执行文件。"""
    if not p.is_dir():
        return False
    exe_names = ("comsol.exe", "comsol")
    return any((p / sub / n).is_file() for sub in _BIN_SUBPATHS for n in exe_names)


def _bin_dir(root: Path) -> Path | None:
    for sub in _BIN_SUBPATHS:
        if (root / sub).is_dir():
            return root / sub
    return None


def find_comsol_root() -> Path | None:
    """按优先级查找 COMSOL 安装根，找不到返回 None。"""
    env = os.environ.get("COMSOL_ROOT")
    if env:
        p = Path(env)
        if _is_comsol_root(p):
            return p
        # 显式指定但无效时明确报错，而不是悄悄回退到别的版本
        raise RuntimeError(
            f"环境变量 COMSOL_ROOT 指向 {p}，但该目录下找不到 COMSOL 可执行文件。\n"
            f"  期望存在：{p / 'Multiphysics' / 'bin' / 'win64' / 'comsol.exe'} 之类"
        )

    # PATH 上直接有 comsol（Linux/macOS 常见）
    exe = shutil.which("comsol")
    if exe:
        # .../COMSOL60/bin/comsol → 上溯两级即为安装根
        cand = Path(exe).resolve().parent.parent
        if _is_comsol_root(cand):
            return cand

    # 扫常见位置
    #
    # ⚠ 必须用 Path(f"{d}/") 而不是 Path(d)：后者在 Windows 上得到无锚点的
    #   驱动器相对路径，Path("D:")/"COMSOL" 会拼成 "D:COMSOL"（缺分隔符），
    #   解析为「D 盘当前目录下的 COMSOL」，而不是 "D:\COMSOL" —— 会静默漏掉
    #   真实安装。加一个 "/" 才让它带上锚点。
    for drive in _DRIVES:
        base = Path(f"{drive}/")
        for ver in _VERSION_DIRS:
            for cand in (base / "COMSOL" / ver,   # D:\COMSOL\COMSOL60（常见）
                         base / ver):             # D:\COMSOL60
                if _is_comsol_root(cand):
                    return cand
    return None


def setup_comsol(verbose: bool = True) -> Path:
    """定位 COMSOL 并写入环境变量。返回安装根 Path。

    写入的两项：
        COMSOL_ROOT —— mph / COMSOL 自身读取
        PATH        —— 追加 bin 目录，使 `comsol` 可被直接调用
    """
    root = find_comsol_root()
    if root is None:
        msg = (
            "找不到 COMSOL 安装。请设置环境变量 COMSOL_ROOT 指向安装根目录，例如：\n"
            '    Windows (PowerShell)   $env:COMSOL_ROOT = "D:\\COMSOL\\COMSOL60"\n'
            "    Git Bash / Linux / mac  export COMSOL_ROOT=/opt/comsol/COMSOL60\n"
            "已尝试的常见位置：" + "、".join(f"{d}/COMSOL/{v}" for d in _DRIVES
                                            for v in _VERSION_DIRS[:3])
        )
        raise RuntimeError(msg)

    os.environ["COMSOL_ROOT"] = str(root)
    bd = _bin_dir(root)
    if bd is not None and str(bd) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(bd) + os.pathsep + os.environ.get("PATH", "")

    if verbose:
        print(f"[comsol-env] COMSOL_ROOT = {root}", file=sys.stderr)
        if bd is not None:
            print(f"[comsol-env] bin        = {bd}", file=sys.stderr)
    return root


def comsol_applications_dir(root: Path | None = None) -> Path:
    """COMSOL 应用库目录（.mph 官方模型所在处）。"""
    root = root or Path(os.environ["COMSOL_ROOT"])
    return root / "Multiphysics" / "applications"


if __name__ == "__main__":  # 自检：打印实际定位结果
    r = setup_comsol()
    print(f"OK  COMSOL 安装根 = {r}")
    print(f"    应用库目录     = {comsol_applications_dir(r)}")
