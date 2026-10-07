/**
 * GGUF 后端连线时，隐藏「原（HF）模型选择」相关控件。
 *
 * 为什么要有这个：扩写节点和批量打标节点上的 `model_name` / `custom_model_path`
 * 只对 transformers（HF）那条路有意义 —— 一旦把「GGUF 后端（llama.cpp）」节点连到
 * `backend` 输入上，推理就交给独立的 llama-server 进程了，这两个控件完全不生效，
 * 留在画布上只会让人误以为「选了 HF 模型但又走 GGUF，到底听谁的」。
 *
 * ★ 关键约束（本项目最容易踩的坑）：**只折叠视觉高度，绝不改 widget.type、
 *   绝不把 widget 移出 node.widgets 数组**。
 *   ComfyUI 序列化 `widgets_values` 是按 node.widgets 数组的顺序逐个取值，
 *   一旦少一项，整条值序列就错位（而且 ComfyUI 不会报错，值会悄悄错配到
 *   别的参数上）。用 computeSize 返回 [0, -4] 折叠高度是社区通行做法，
 *   数组和值的顺序都不动，所以旧工作流打开后行为完全一致。
 */

import { app } from "../../scripts/app.js";

// 受影响的节点（class_type）
const TARGET_NODES = new Set(["Qwen35PromptEnhancer", "Qwen35BatchImageTagger"]);

// 只对 HF 路径有意义的控件。想只隐藏其中一个，从数组里删掉即可。
const HF_ONLY_WIDGETS = ["model_name", "custom_model_path"];

// 连线输入的名字（不占 widgets_values 槽位）
const BACKEND_INPUT = "backend";

const TAG = "[Qwen35]";

/** 后端那根线连上了没有 */
function isBackendConnected(node) {
  const inputs = node?.inputs;
  if (!Array.isArray(inputs)) return false;
  const idx = inputs.findIndex((i) => i && i.name === BACKEND_INPUT);
  if (idx < 0) return false;
  const slot = inputs[idx];
  // 新版是 slot.link（数字 id），老版可能挂在 slot.link 为 null
  return slot.link != null;
}

/** 折叠 / 还原一个控件（仅在 node.widgets 里就地标记，不动数组） */
function setWidgetCollapsed(widget, collapsed) {
  if (!widget) return false;
  if (collapsed) {
    if (widget.__q35Collapsed) return false;
    widget.__q35Collapsed = true;
    widget.__q35OrigComputeSize = widget.computeSize;
    // [0, -4]：宽度 0、比标准行高再矮一点点 —— 连行间距都不占
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
    return true;
  }
  if (!widget.__q35Collapsed) return false;
  widget.__q35Collapsed = false;
  widget.hidden = false;
  if (typeof widget.__q35OrigComputeSize === "function") {
    widget.computeSize = widget.__q35OrigComputeSize;
  } else {
    // 原来就没有自定义 computeSize -> 删掉我们加的这个，回到默认行为
    delete widget.computeSize;
  }
  return true;
}

/** 按当前连线状态统一刷一遍 */
function applyVisibility(node) {
  if (!node || !Array.isArray(node.widgets)) return;
  const connected = isBackendConnected(node);
  let changed = false;
  for (const w of node.widgets) {
    if (!w || !HF_ONLY_WIDGETS.includes(w.name)) continue;
    if (setWidgetCollapsed(w, connected)) changed = true;
  }
  if (!changed) return;

  // 折叠后节点高度要跟着缩一下，否则原来占的位置会留一块空白
  try {
    const computed = node.computeSize?.();
    const h = Array.isArray(computed) ? computed[1] : null;
    if (typeof h === "number" && h > 0) {
      node.setSize([node.size[0], h]);
    }
  } catch (e) {
    /* 尺寸算不出来也无所谓，控件本身已经藏了 */
  }
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "Qwen35.HideHFModelPickerOnGGUF",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODES.has(nodeData?.name)) return;

    // ---- 连线 / 断线 ----
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange?.apply(this, arguments);
      applyVisibility(this);
      return r;
    };

    // ---- 载入已有工作流 ----
    // onConfigure 时 widget 与 input 的连接信息未必都已经就位，
    // 所以延一帧再算一次；顺便补一次立即计算，覆盖大部分情况。
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      applyVisibility(this);
      requestAnimationFrame(() => applyVisibility(this));
      return r;
    };
  },

  // ---- 从菜单新建节点 ----
  nodeCreated(node) {
    if (!TARGET_NODES.has(node?.comfyClass)) return;
    applyVisibility(node);
    requestAnimationFrame(() => applyVisibility(node));
  },
});

console.log(`${TAG} 前端扩展已加载：GGUF 后端连线时会隐藏 HF 模型选择控件`);
