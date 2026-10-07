# -*- coding: utf-8 -*-
"""前端扩展 web/qwen35_backend_ui.js 的离线行为测试。

为什么值得单独测：这段 JS 干的事是「连上 GGUF 后端 -> 隐藏 HF 模型选择控件」，
看起来无害，但它碰的是**本插件最怕出错的地方** —— node.widgets 数组。
ComfyUI 序列化 widgets_values 就是按这个数组的顺序逐个取值，一旦我们把它
改成少一项，旧工作流载入后所有控件的值会整体错位一格，而且**不报任何错**。
所以这里把「数组长度 / 顺序 / 值的原样性」当成硬断言。

不需要浏览器：用 node 跑一段 mock，把 import 换成注入的假 app，
然后直接调用扩展注册上来的钩子。

用法（用 ComfyUI 自带解释器或托管解释器都能跑）：
    python test_qwen35_webui.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JS_NAME = "qwen35_backend_ui.js"
PLUGIN_DIR = os.environ.get(
    "QWEN35_PLUGIN_DIR",
    r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer",
)
# 优先插件目录里那份；仓库的 devtools/ 里只有这个测试脚本，没有 web/，
# 所以从 devtools 跑时也是去读真插件里的 js —— 这正是想要的行为。
CANDIDATES = [
    os.path.join(PLUGIN_DIR, "web", JS_NAME),
    os.path.join(HERE, "web", JS_NAME),
]
NODE_EXE = r"C:\Users\ADMIN\.workbuddy\binaries\node\versions\22.22.2-6\node.exe"

passed = []
failed = []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail and not ok else ""))


def find_js():
    for p in CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


HARNESS = r"""
// ---- 假的 app，只收下扩展对象 ----
globalThis.__EXT = null;
globalThis.__DIRTY = 0;
const app = {
  registerExtension: (e) => { globalThis.__EXT = e; },
  graph: { setDirtyCanvas: () => { globalThis.__DIRTY++; } },
};

// 浏览器里才有；测试里同步执行即可（被测代码只用它延后一次刷新，
// 同步跑反而让断言更确定）
globalThis.requestAnimationFrame = (fn) => {
  try { fn(); } catch (e) { /* 忽略 */ }
};

// ---- 被测代码（import 行已被替换成上面那份 app）----
__CODE__

// ---- 测试驱动 ----
const ext = globalThis.__EXT;
const results = [];
function rec(name, ok, detail) { results.push({ name, ok, detail: detail || "" }); }

if (!ext) { console.log(JSON.stringify([{ name: "扩展成功注册", ok: false }])); process.exit(0); }
rec("扩展对象有 name", typeof ext.name === "string" && ext.name.length > 0, ext && ext.name);

// 造一个假的 nodeType（与 ComfyUI 调用方式一致：传构造函数）
function FakeNode() {}
ext.beforeRegisterNodeDef(FakeNode, { name: "Qwen35PromptEnhancer" });
rec("目标节点被挂钩子", typeof FakeNode.prototype.onConnectionsChange === "function"
    && typeof FakeNode.prototype.onConfigure === "function");

// 另一个节点不应该被挂钩子
function OtherNode() {}
ext.beforeRegisterNodeDef(OtherNode, { name: "SomeOtherNode" });
rec("非目标节点不被挂钩子", typeof OtherNode.prototype.onConnectionsChange === "undefined");

function makeNode(cls) {
  const node = {
    comfyClass: cls,
    widgets: [
      { name: "model_name", type: "combo", value: "Qwen3.5-9B  [X]", computeSize: undefined },
      { name: "custom_model_path", type: "text", value: "", computeSize: undefined },
      { name: "temperature", type: "number", value: 0.4, computeSize: undefined },
      { name: "bilingual", type: "combo", value: "off", computeSize: undefined },
    ],
    inputs: [{ name: "image", link: null }, { name: "backend", link: null }],
    size: [320, 240],
    dirty: 0,
    computeSize() {
      // 模拟 litegraph：折叠掉的控件不贡献高度
      let h = 30;
      for (const w of this.widgets) {
        const cs = typeof w.computeSize === "function" ? w.computeSize() : [200, 24];
        h += Math.max(cs[1], 0);
      }
      return [320, h];
    },
    setSize(s) { this.size = s; },
    setDirtyCanvas() { this.dirty++; },
  };
  // 真实 ComfyUI 里节点高度就是 computeSize 的结果，mock 必须自洽，
  // 否则「还原后高度回到初始」这条断言会因为 mock 自身矛盾而假失败
  node.size = node.computeSize();
  node.__proto__ = FakeNode.prototype;   // 让 node 拿到钩子方法
  return node;
}

const widgetNames = (n) => n.widgets.map((w) => w.name).join(",");
const widgetValues = (n) => JSON.stringify(n.widgets.map((w) => w.value));
const collapsed = (n, name) => {
  const w = n.widgets.find((x) => x.name === name);
  return !!(w && w.__q35Collapsed);
};
const computeOf = (n, name) => {
  const w = n.widgets.find((x) => x.name === name);
  return w && typeof w.computeSize === "function" ? JSON.stringify(w.computeSize()) : "none";
};

// ---- 1) 没连线：什么都不该藏 ----
const n1 = makeNode("Qwen35PromptEnhancer");
ext.nodeCreated(n1);
rec("未连线时 model_name 不折叠", collapsed(n1, "model_name") === false);
rec("未连线时 custom_model_path 不折叠", collapsed(n1, "custom_model_path") === false);
const h1 = n1.size[1];

// ---- 2) 连上 backend：两个 HF 控件折叠，其它不动 ----
const orderBefore = widgetNames(n1);
const valuesBefore = widgetValues(n1);
n1.inputs[1].link = 42;
n1.onConnectionsChange(1, 1, true, { origin_id: 7 });
rec("连线后 model_name 被折叠", collapsed(n1, "model_name") === true);
rec("连线后 custom_model_path 被折叠", collapsed(n1, "custom_model_path") === true);
rec("连线后 temperature 不受影响", collapsed(n1, "temperature") === false);
rec("连线后 bilingual 不受影响", collapsed(n1, "bilingual") === false);
rec("折叠后 model_name 的 computeSize 返回 [0,-4]", computeOf(n1, "model_name") === "[0,-4]");
rec("折叠后节点变矮", n1.size[1] < h1, `${h1} -> ${n1.size[1]}`);

// ---- 3) 最关键：数组长度 / 顺序 / 值一个都不能变 ----
rec("widgets 数量不变", n1.widgets.length === 4, String(n1.widgets.length));
rec("widgets 顺序不变", widgetNames(n1) === orderBefore, widgetNames(n1));
rec("widgets 值不变", widgetValues(n1) === valuesBefore);
rec("widget.type 未被改写", n1.widgets[0].type === "combo" && n1.widgets[1].type === "text");

// ---- 4) 再调一次：幂等，不能反复缩高 ----
const h2 = n1.size[1];
n1.onConnectionsChange(1, 1, true, { origin_id: 7 });
rec("重复触发是幂等的（高度不再变）", n1.size[1] === h2, `${h2} -> ${n1.size[1]}`);
rec("重复触发后仍然只折叠两个", n1.widgets.filter((w) => w.__q35Collapsed).length === 2);

// ---- 5) 断开：还原 ----
n1.inputs[1].link = undefined;
n1.onConnectionsChange(1, 1, false, null);
rec("断开后 model_name 还原", collapsed(n1, "model_name") === false);
rec("断开后 computeSize 恢复原样", computeOf(n1, "model_name") === "none", computeOf(n1, "model_name"));
rec("断开后高度回到初始", n1.size[1] === h1, `${n1.size[1]} vs ${h1}`);
rec("断开后 widgets 仍然完好", n1.widgets.length === 4 && widgetNames(n1) === orderBefore);

// ---- 6) 载入旧工作流（onConfigure 路径）----
const n2 = makeNode("Qwen35BatchImageTagger");
n2.inputs[1].link = 99;             // 已经有连线
n2.onConfigure({ id: 3, widgets_values: [] });
rec("onConfigure 立即处理了连线态", collapsed(n2, "model_name") === true);

// ---- 7) 打标节点也在覆盖范围内 ----
const n3 = makeNode("Qwen35BatchImageTagger");
rec("打标节点也受管", typeof n3.onConfigure === "function");

// ---- 8) input 缺失时不炸 ----
const n4 = makeNode("Qwen35PromptEnhancer");
n4.inputs = [];
let threw = false;
try { n4.onConnectionsChange(0, 0, false); } catch (e) { threw = true; }
rec("没有 backend 输入时不抛异常", threw === false);

console.log(JSON.stringify(results));
"""


def main():
    js_path = find_js()
    print("=" * 72)
    print(f"被测文件: {js_path}")
    if not js_path:
        print("  找不到 qwen35_backend_ui.js —— 跳过（在仓库里跑时属正常）")
        return 0

    with open(js_path, "r", encoding="utf-8") as f:
        code = f.read()

    # 把 ES module 的 import 行去掉（app 已由 harness 提供）
    code = re.sub(r'^\s*import\s+.*?;\s*$', '', code, flags=re.MULTILINE)

    harness = HARNESS.replace("__CODE__", code)
    tmp = os.path.join(HERE, "_tmp_webui_harness.mjs")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(harness)

    try:
        node = NODE_EXE if os.path.isfile(NODE_EXE) else "node"
        proc = subprocess.run([node, tmp], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60)
        if proc.returncode != 0:
            print("  node 退出码", proc.returncode)
            print(proc.stderr[-2000:])
            return 1
        # 最后一行是结果 JSON（前面可能还有 console.log 的输出）
        line = [l for l in proc.stdout.splitlines() if l.strip().startswith("[")]
        if not line:
            print("  没拿到结果 JSON。stdout:")
            print(proc.stdout[-2000:])
            return 1
        import json as _json
        for r in _json.loads(line[-1]):
            check(r["name"], r["ok"], r.get("detail", ""))
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    print()
    print(f"通过 {len(passed)} / {len(passed) + len(failed)}")
    if failed:
        print("失败项：")
        for f_ in failed:
            print("   -", f_)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
