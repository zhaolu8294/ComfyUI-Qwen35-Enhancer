# -*- coding: utf-8 -*-
"""把本工作区（h3_workflow）的开发脚本同步备份到节点仓库的 devtools/ 目录。

用途：GitHub 仓库作为代码备份渠道。节点代码本来就在仓库里，但
测试 / 基准 / 诊断 / 生成脚本一直只存在于 WorkBuddy 工作区，
工作区一旦丢失就全没了 —— 这个脚本把它们一并纳入备份。

可重复执行：以后新增测试再跑一次即可，只覆盖有变化的文件。

排除项（有意不备份）：
  __pycache__/            编译缓存，无价值
  *.pyc                   同上
  *.bak_attention         4 个示例工作流的旧快照，与仓库 examples/ 重复
  returns                 早前 selftest 残留的垃圾文件

后处理：把脚本里硬编码的 WF（工作区绝对路径）改成「脚本自身所在目录」，
这样脚本搬到 devtools/ 后仍能定位同级的工作流 JSON。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe sync_devtools_to_repo.py
    （加 --dry-run 只看会发生什么，不写盘）
"""
import argparse
import os
import shutil
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.path.join(
    r"E:\AI\ComfyUI-aki-v3", "ComfyUI", "custom_nodes", "ComfyUI-Qwen35-Enhancer"
)
DST = os.path.join(NODE_DIR, "devtools")

EXCLUDE_DIRS = {"__pycache__"}
# `_fla_probe*` 是探 fla 各版本兼容性时把 wheel 解包到工作区的目录，
# 里面是**第三方包本体**（fla 0.5.2 / 0.2.2 / 0.3.2 / 0.4.2 四份 site-packages，
# 合计约 7MB、700+ 文件），不是我们写的脚本，不该进仓库。
# 按前缀排除（那些目录名带版本号后缀，逐个列举容易漏）。
EXCLUDE_DIR_PREFIXES = ("_fla_probe",)
# _pyspy_native.json 是 py-spy --native 的原始采样 dump，一份 3.2MB。
# 结论不对（采样偏线程）但它仍是「为什么读错了」的证据，重跑命令就在报告里，
# 没必要把 3MB 原始样本塞进仓库；同名的 _pyspy_dump1.txt（几 KB 的汇总）照常同步。
EXCLUDE_FILES = {"returns", "_pyspy_native.json"}
EXCLUDE_SUFFIXES = (".pyc", ".bak_attention")

# 需要改写 WF 定义的文件（**白名单**，不要放宽）。
# 用黑名单不安全：本脚本自身也含下面那段字面量，被改写会把常量本身写坏。
PATCH_TARGETS = {"verify_h3_workflows.py", "update_existing_enhancers.py"}

# 同步时改名的文件：工作区名 -> 仓库名
# README.md 讲的是 H3 视频工作流（不是 devtools 索引），而仓库里那个名字
# 要留给 devtools 的索引用；改成与同目录 README_prompt_enhancer.md /
# README_qwen35_enhancer.md 一致的命名。
RENAME = {"README.md": "README_h3_video.md"}

# 工作区绝对路径 -> 脚本所在目录
WF_OLD = r'WF = r"C:\Users\ADMIN\WorkBuddy\2026-10-06-18-56-50\h3_workflow"'
WF_NEW = (
    "# 原为工作区绝对路径；备份到仓库后改为「脚本自身所在目录」，\n"
    "# 因为工作流 JSON 与脚本同级存放。\n"
    "WF = os.path.dirname(os.path.abspath(__file__))"
)


def wanted(name: str) -> bool:
    if name in EXCLUDE_FILES:
        return False
    return not name.endswith(EXCLUDE_SUFFIXES)


def collect():
    """返回 [(相对路径, 绝对源路径)]，按相对路径排序保证可复现。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames
                       if d not in EXCLUDE_DIRS
                       and not d.startswith(EXCLUDE_DIR_PREFIXES)]
        for fn in filenames:
            if not wanted(fn):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, SRC)
            out.append((rel.replace("\\", "/"), full))
    return sorted(out)


def patch_wf(name: str, text: str) -> tuple:
    """把硬编码 WF 换成基于 __file__ 的写法。返回 (新文本, 是否改动, 是否告警)。"""
    if name not in PATCH_TARGETS:
        return text, False, False
    if WF_OLD not in text:
        return text, False, False
    if "\nimport os" not in text and not text.startswith("import os"):
        return text, False, True          # 没 import os，不敢乱改
    return text.replace(WF_OLD, WF_NEW), True, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(NODE_DIR):
        sys.exit(f"[错误] 找不到节点目录：{NODE_DIR}")

    items = collect()
    print(f"源目录 : {SRC}")
    print(f"目标   : {DST}")
    print(f"候选   : {len(items)} 个文件")
    print()

    added, updated, same, patched, warned = [], [], [], [], []

    for rel, full in items:
        rel_out = RENAME.get(rel, rel)
        dst = os.path.join(DST, rel_out.replace("/", os.sep))
        raw = open(full, "rb").read()

        if rel.endswith(".py"):
            text = raw.decode("utf-8")
            new_text, changed, warn = patch_wf(rel, text)
            if warn:
                warned.append(rel)
            if changed:
                raw = new_text.encode("utf-8")
                patched.append(rel)

        old = open(dst, "rb").read() if os.path.exists(dst) else None
        if old == raw:
            same.append(rel_out)
            continue

        if not args.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(raw)

        (updated if old is not None else added).append(rel_out)

    def show(title, names):
        if names:
            print(f"{title}（{len(names)}）:")
            for n in names:
                print("   ", n)
            print()

    verb = "将执行" if args.dry_run else "已执行"
    show(f"{verb} 新增", added)
    show(f"{verb} 更新", updated)
    print(f"未变化: {len(same)} 个")
    print()
    show("已改写 WF 路径", patched)
    if warned:
        show("!! 含 WF 硬编码但缺 import os，未改动", warned)

    if not args.dry_run:
        print(f"备份完成 -> {DST}")
        print("提示：仓库里的 devtools/ 是副本，请勿直接编辑；改工作区后重跑本脚本。")


if __name__ == "__main__":
    main()
