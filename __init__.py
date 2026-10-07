# -*- coding: utf-8 -*-
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 前端扩展目录：连上 GGUF 后端时自动隐藏「原（HF）模型选择」控件。
# 见 web/qwen35_backend_ui.js 头部的说明 —— 只折叠高度，不动 widgets_values
# 的顺序，所以旧工作流行为完全一致。
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
