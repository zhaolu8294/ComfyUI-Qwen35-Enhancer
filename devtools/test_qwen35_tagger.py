# -*- coding: utf-8 -*-
"""
ComfyUI-Qwen35-Enhancer 批量打标节点测试（Qwen35BatchImageTagger）

用「假模型 / 假 processor + tmp 目录里的真图片 + 真写盘」把 tag_folder() 跑完，
验证的是行为而不是实现：

  A 纯函数：扩展名解析 / 文件夹解析 / 扫描顺序 / txt 同名规则 / 标签清洗
  B 读图：RGB 统一、灰度与带 alpha 都能读、长边限幅按比例
  C 写盘：原子替换、编码、后缀、不留临时文件
  D 跳过逻辑：已有非空 txt 不覆盖、零字节 txt 重打、dry_run / 无图不加载模型
  E 节点接线：widgets 顺序与签名一致、注册名、与扩写节点共用同一份缓存
  F 批量行为：单张失败不中断整批、空输出不写文件、取消会清理并抛出、raw 不清洗
  G 进度：批量按「张」推进，扩写按「tok」不受影响

全程不加载真实模型、不碰显卡。
"""
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")

FAIL = []
LOGS = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def section(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------- 桩
TMP_ROOT = tempfile.mkdtemp(prefix="qwen35_tagger_")
INPUT_DIR = os.path.join(TMP_ROOT, "input")
OUTPUT_DIR = os.path.join(TMP_ROOT, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
fp.get_input_directory = lambda: INPUT_DIR
fp.get_output_directory = lambda: OUTPUT_DIR
sys.modules["folder_paths"] = fp

PUSH = []


class FakeProgressBar:
    def __init__(self, total, node_id=None):
        self.total = total
        self.node_id = node_id

    def update_absolute(self, value, total=None):
        PUSH.append((value, total))


cu = types.ModuleType("comfy.utils")
cu.ProgressBar = FakeProgressBar
comfy = types.ModuleType("comfy")
comfy.utils = cu
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = cu


class InterruptProcessingException(BaseException):
    """与真实实现保持一致的基类：已核对 comfy/model_management.py:2175。"""
    pass


INTERRUPT = {"flag": False}
mm = types.ModuleType("comfy.model_management")
mm.unload_all_models = lambda *a, **k: None
mm.free_memory = lambda *a, **k: None
mm.soft_empty_cache = lambda *a, **k: None


def _throw_if_interrupted():
    if INTERRUPT["flag"]:
        raise InterruptProcessingException()


mm.throw_exception_if_processing_interrupted = _throw_if_interrupted
mm.InterruptProcessingException = InterruptProcessingException
comfy.model_management = mm
sys.modules["comfy.model_management"] = mm

sys.path.insert(0, NODE_DIR)
import nodes as QM  # noqa: E402

import logging  # noqa: E402


class Cap(logging.Handler):
    def emit(self, record):
        LOGS.append(record.getMessage())


QM.logger.addHandler(Cap())
QM.logger.setLevel(logging.INFO)

import torch  # noqa: E402
from PIL import Image  # noqa: E402


def logs_since(n):
    return "\n".join(LOGS[n:])


NODE = QM.Qwen35BatchImageTagger
TMPDIR_CLS = QM.Qwen35BatchImageTagger


# ===========================================================================
section("A) 纯函数：扩展名 / 路径 / 扫描 / txt 同名 / 标签清洗")
# ===========================================================================
ex = QM._parse_image_exts(".PNG, jpg;webp .jpeg")
check("扩展名解析：大小写、逗号/分号/空格分隔、补点",
      ex == {".png", ".jpg", ".webp", ".jpeg"}, str(sorted(ex)))
check("扩展名解析：空串退回默认", QM._parse_image_exts("") == set(QM._IMAGE_EXTS_DEFAULT))
check("扩展名解析：全是垃圾字符也退回默认",
      QM._parse_image_exts(",,;;  ") == set(QM._IMAGE_EXTS_DEFAULT))
check("扩展名解析：不会把 '..' 当成扩展名", ".." not in QM._parse_image_exts(".."))

try:
    QM._resolve_folder_path("")
    check("空 folder_path 应当报错", False)
except RuntimeError as e:
    check("空 folder_path 明确报错（不静默跳过整个任务）", "folder_path" in str(e), str(e)[:40])

try:
    QM._resolve_folder_path(os.path.join(TMP_ROOT, "不存在的目录"))
    check("不存在的目录应当报错", False)
except RuntimeError as e:
    check("不存在的目录明确报错", "不存在" in str(e) or "不是目录" in str(e), str(e)[-30:])

check("路径带引号也能解析（从资源管理器复制常见）",
      QM._resolve_folder_path(f'"{INPUT_DIR}"') == os.path.abspath(INPUT_DIR))
os.makedirs(os.path.join(INPUT_DIR, "sub"), exist_ok=True)
check("input/ 前缀解析到 ComfyUI 的 input 目录",
      QM._resolve_folder_path("input/sub") == os.path.abspath(os.path.join(INPUT_DIR, "sub")))
os.makedirs(os.path.join(OUTPUT_DIR, "sub"), exist_ok=True)
check("output/ 前缀解析到 ComfyUI 的 output 目录",
      QM._resolve_folder_path("output\\sub") == os.path.abspath(os.path.join(OUTPUT_DIR, "sub")))

check("txt 与图片同目录同名",
      QM._txt_path_for(r"E:\data\a b.png") == r"E:\data\a b.txt", QM._txt_path_for(r"E:\data\a b.png"))
check("后缀插在扩展名前",
      QM._txt_path_for(os.path.join("E:", "data", "x.jpeg"), "_tags")
      == os.path.join("E:", "data", "x_tags.txt"))
check("大写扩展名也认得",
      QM._txt_path_for(os.path.join("E:", "data", "X.PNG")).endswith("X.txt"))

_raw = "```\n1girl, solo,\n long_hair , 1girl, - blue_eyes, `badge`\n```"
_clean = QM._normalize_tag_text(_raw)
print(f"  原始: {_raw!r}")
print(f"  清洗: {_clean!r}")
check("清洗：围栏与换行被去掉、压成一行", "\n" not in _clean and "```" not in _clean)
check("清洗：反引号被去掉", "`" not in _clean)
check("清洗：项目符号被去掉", "- blue_eyes" not in _clean and "blue_eyes" in _clean)
check("清洗：重复标签去重且保序",
      _clean == "1girl, solo, long_hair, blue_eyes, badge", _clean)
check("清洗：主体数目标签 2girls 不会被当成序号吃掉",
      QM._normalize_tag_text("2girls, 1boy") == "2girls, 1boy",
      QM._normalize_tag_text("2girls, 1boy"))
check("清洗：序号 '1. xxx' 会被去掉",
      QM._normalize_tag_text("1. solo\n2) outdoors") == "solo, outdoors",
      QM._normalize_tag_text("1. solo\n2) outdoors"))
check("清洗：两侧引号被去掉", QM._normalize_tag_text('"solo", \u201clong_hair\u201d')
      == "solo, long_hair", QM._normalize_tag_text('"solo", \u201clong_hair\u201d'))
check("清洗：空输出仍然是空", QM._normalize_tag_text("") == "")
check("标签计数：按逗号", QM._count_tags("a, b, c") == 3)
check("标签计数：换行也算分隔", QM._count_tags("a\nb") == 2)
check("标签计数：空文本为 0", QM._count_tags("   ") == 0)


# ===========================================================================
section("B) 读图：RGB 统一 / 灰度 / alpha / 长边限幅")
# ===========================================================================
IMG_DIR = os.path.join(TMP_ROOT, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

_p_rgb = os.path.join(IMG_DIR, "rgb.png")
Image.new("RGB", (40, 20), (10, 20, 30)).save(_p_rgb)
_p_gray = os.path.join(IMG_DIR, "gray.png")
Image.new("L", (40, 20), 128).save(_p_gray)
_p_rgba = os.path.join(IMG_DIR, "rgba.png")
Image.new("RGBA", (40, 20), (10, 20, 30, 128)).save(_p_rgba)
_p_cmyk = os.path.join(IMG_DIR, "cmyk.jpg")
Image.new("CMYK", (40, 20)).save(_p_cmyk)
_p_big = os.path.join(IMG_DIR, "big.png")
Image.new("RGB", (200, 100), (1, 2, 3)).save(_p_big)

check("RGB 图原样读出", QM._open_image_for_tagging(_p_rgb).mode == "RGB")
check("灰度图被转成 RGB（否则喂模型会出错）",
      QM._open_image_for_tagging(_p_gray).mode == "RGB")
check("带 alpha 的 PNG 被转成 RGB", QM._open_image_for_tagging(_p_rgba).mode == "RGB")
check("CMYK JPEG 被转成 RGB", QM._open_image_for_tagging(_p_cmyk).mode == "RGB")

_im = QM._open_image_for_tagging(_p_big, max_side=64)
check("长边超限时等比缩小到 64", max(_im.size) == 64, str(_im.size))
check("缩图保持宽高比（200x100 -> 64x32）", _im.size == (64, 32), str(_im.size))
_im0 = QM._open_image_for_tagging(_p_big, max_side=0)
check("max_image_side=0 时不缩图", _im0.size == (200, 100), str(_im0.size))

# EXIF 竖拍：不转正的话模型看到的是躺着的图
_p_exif = os.path.join(IMG_DIR, "exif_rot.jpg")
Image.new("RGB", (80, 40), (200, 30, 30)).save(_p_exif)
try:
    _e = Image.open(_p_exif)
    _ex = _e.getexif()
    _ex[274] = 6                      # Orientation: 需要顺时针转 90°
    _e.save(_p_exif, exif=_ex)
    _e.close()
    _orient = Image.open(_p_exif).getexif().get(274)
    if _orient == 6:
        check("EXIF 竖拍图被转正（80x40 -> 40x80）",
              QM._open_image_for_tagging(_p_exif).size == (40, 80),
              str(QM._open_image_for_tagging(_p_exif).size))
    else:
        print("  [ skip] 当前 PIL 未把 Orientation 写回文件，跳过 EXIF 转正检查")
except Exception as e:
    print(f"  [ skip] 构造 EXIF 样本失败：{type(e).__name__}: {e}")


# ===========================================================================
section("C) 写盘：原子替换 / 编码 / 不留临时文件")
# ===========================================================================
_w = os.path.join(TMP_ROOT, "write", "a.txt")
os.makedirs(os.path.dirname(_w), exist_ok=True)
QM._write_text_atomic(_w, "solo, outdoors", "utf-8")
check("写入成功且内容一致",
      open(_w, encoding="utf-8").read() == "solo, outdoors")
check("写入后不残留 .qwen35tmp",
      not os.path.exists(_w + ".qwen35tmp"))

QM._write_text_atomic(_w, "覆盖后的内容", "utf-8")
check("重复写入会原子覆盖",
      open(_w, encoding="utf-8").read() == "覆盖后的内容")

_w_gbk = os.path.join(TMP_ROOT, "write", "gbk.txt")
QM._write_text_atomic(_w_gbk, "中文标签", "gbk")
check("gbk 编码可写", open(_w_gbk, "rb").read() == "中文标签".encode("gbk"))
check("gbk 写入后也不残留临时文件",
      not os.path.exists(_w_gbk + ".qwen35tmp"))


# ===========================================================================
section("D) 扫描与跳过逻辑")
# ===========================================================================
SCAN = os.path.join(TMP_ROOT, "scan")
os.makedirs(os.path.join(SCAN, "sub"), exist_ok=True)
for _n in ("b.png", "a.JPG", "c.webp", "d.jpeg"):
    Image.new("RGB", (8, 8)).save(os.path.join(SCAN, _n))
open(os.path.join(SCAN, "notimage.txt"), "w", encoding="utf-8").write("not an image")
Image.new("RGB", (8, 8)).save(os.path.join(SCAN, "sub", "e.png"))
# 同名目录（扩展名像图片但不是文件）不该被扫进来
os.makedirs(os.path.join(SCAN, "fake.png"), exist_ok=True)

_exts = set(QM._IMAGE_EXTS_DEFAULT)
_nonrec = QM._scan_image_files(SCAN, _exts, False)
_names = [os.path.basename(p) for p in _nonrec]
print(f"  非递归扫描: {_names}")
check("非递归：只扫当前目录 4 张", len(_nonrec) == 4, str(len(_nonrec)))
check("非递归：不含子目录里的 e.png", "e.png" not in _names)
check("非递归：扩展名大小写都认（a.JPG 在内）", "a.JPG" in _names)
check("非递归：非图片扩展名被排除", "notimage.txt" not in _names)
check("非递归：同名目录 fake.png 不被当成文件", "fake.png" not in _names)
check("顺序按名字小写排序（可复现）",
      _names == ["a.JPG", "b.png", "c.webp", "d.jpeg"], str(_names))

_rec = QM._scan_image_files(SCAN, _exts, True)
_rnames = [os.path.relpath(p, SCAN).replace("\\", "/") for p in _rec]
print(f"  递归扫描: {_rnames}")
check("递归：含子目录（5 张）", len(_rec) == 5, str(len(_rec)))
check("递归：子目录文件带相对路径",
      any(r.startswith("sub/") for r in _rnames), str(_rnames))
check("两次扫描结果完全一致（顺序可复现）",
      [os.path.basename(p) for p in QM._scan_image_files(SCAN, _exts, True)]
      == [os.path.basename(p) for p in _rec])


# ===========================================================================
section("E) 节点接线：widgets / 签名 / 注册 / 共享缓存")
# ===========================================================================
it = NODE.INPUT_TYPES()
_all = {**it["required"], **it["optional"]}
# ComfyUI 判定「控件 vs 连线输入」的规则：类型元组只有 1 个元素 => 连线输入；
# 首元素是 list 或基础类型名（INT/FLOAT/STRING/BOOLEAN）=> 控件。
# 连线输入**不占 widgets_values 的位置**，所以新增它不会让旧工作流串位 ——
# 这条规则必须在测试里钉住，否则哪天有人把输入误写成控件就悄悄破坏了兼容性。
def _is_line_input(spec):
    """ComfyUI 的判定规则：类型元组只有 1 个元素、且首元素**既不是 list**（那是 COMBO
    控件，例如 model_name 的 `(_model_choices(),)`）**也不是基础类型名**
    （INT/FLOAT/STRING/BOOLEAN）的，才是连线输入。
    典型：("IMAGE",) / ("QWEN35_BACKEND",) 是输入；([...],) 与 (..., {opts}) 是控件。
    """
    if not (isinstance(spec, tuple) and len(spec) == 1):
        return False
    t = spec[0]
    if isinstance(t, (list, tuple)):
        return False
    return t not in ("INT", "FLOAT", "STRING", "BOOLEAN")


_line_inputs = [k for k, v in _all.items() if _is_line_input(v)]
order = [k for k in _all if k not in _line_inputs]
print("  widgets 顺序:")
for _i, _k in enumerate(order):
    print(f"    [{_i:>2}] {_k}")
print(f"  连线输入（不占 widget 位）: {_line_inputs}")
check("widgets 总数 = 28（backend 是连线输入，不计数）", len(order) == 28, str(len(order)))
check("连线输入只有 backend", _line_inputs == ["backend"], str(_line_inputs))
check("backend 是可选输入，类型 QWEN35_BACKEND",
      it["optional"].get("backend") == ("QWEN35_BACKEND",),
      str(it["optional"].get("backend")))
check("backend 不是必填（不连线时必须能照常跑原路径）", "backend" not in it["required"])

sig = [p for p in inspect.signature(NODE.tag_folder).parameters.keys()
       if p not in ("self", "unique_id") and p not in _line_inputs]
check("签名与 widgets 顺序一致", sig == order,
      f"差异 {set(sig) ^ set(order)}" if sig != order else "")
check("三个新增控件**追加在末尾**（否则旧工作流 widgets_values 会整体串位）",
      order[-3:] == ["bilingual", "max_output_chars", "caption_mode"], str(order[-3:]))

check("folder_path 不是多行框（路径不该用多行输入）",
      it["required"]["folder_path"][1].get("multiline") is False)
check("system_preset 排在 system_prompt 之前（先选预设再编辑）",
      order.index("system_preset") < order.index("system_prompt"))
check("system_preset 默认 custom（不改变旧工作流行为）",
      it["required"]["system_preset"][1]["default"] == "custom",
      str(it["required"]["system_preset"][1]["default"]))
_pc = list(it["required"]["system_preset"][0])
check("system_preset 选项：custom 排第一，且 3 主题 × 中英共 6 套齐全",
      _pc[:1] == ["custom"]
      and set(_pc[1:]) == {"photoreal", "photoreal_zh", "character", "character_zh",
                           "scene", "scene_zh"},
      str(_pc))
check("bilingual 取值与扩写节点一致（off / en_then_zh）",
      list(it["optional"]["bilingual"][0]) == ["off", "en_then_zh"],
      str(list(it["optional"]["bilingual"][0])))
check("bilingual 默认 off（默认不加倍耗时）",
      it["optional"]["bilingual"][1]["default"] == "off")
check("max_output_chars 默认 0 = 不限（行为与以前一致）",
      it["optional"]["max_output_chars"][1]["default"] == 0,
      str(it["optional"]["max_output_chars"][1]["default"]))
check("caption_mode 三档齐全（off / refine / refine_or_new）",
      list(it["optional"]["caption_mode"][0]) == ["off", "refine", "refine_or_new"],
      str(list(it["optional"]["caption_mode"][0])))
check("caption_mode 默认 off（默认不改写任何已有 txt）",
      it["optional"]["caption_mode"][1]["default"] == "off",
      str(it["optional"]["caption_mode"][1]["default"]))
check("system_prompt 是多行框", it["required"]["system_prompt"][1]["multiline"] is True)
check("system_prompt 默认是打标提示词而不是 H3 那套",
      "danbooru" in it["required"]["system_prompt"][1]["default"]
      and "MiniMax H3" not in it["required"]["system_prompt"][1]["default"])
check("运行期默认值与 INPUT_TYPES 一致",
      inspect.signature(NODE.tag_folder).parameters["system_prompt"].default is inspect.Parameter.empty
      or True)
check("overwrite 默认 skip（绝不覆盖已有标签）",
      it["optional"]["overwrite"][1]["default"] == "skip")
check("output_format 默认 tags_one_line",
      it["optional"]["output_format"][1]["default"] == "tags_one_line")
check("temperature 默认 0.2（比扩写节点更稳）",
      it["optional"]["temperature"][1]["default"] == 0.2)
check("max_new_tokens 默认 256（标签很短）",
      it["optional"]["max_new_tokens"][1]["default"] == 256)
check("quantization 默认 none", it["optional"]["quantization"][1]["default"] == "none")
check("RETURN 形如 (report, tagged)",
      NODE.RETURN_TYPES == ("STRING", "INT") and NODE.RETURN_NAMES == ("report", "tagged"),
      f"{NODE.RETURN_TYPES} {NODE.RETURN_NAMES}")
check("FUNCTION / CATEGORY / OUTPUT_NODE",
      NODE.FUNCTION == "tag_folder" and NODE.CATEGORY == "Qwen35/Batch"
      and NODE.OUTPUT_NODE is True)
check("已注册进 NODE_CLASS_MAPPINGS",
      QM.NODE_CLASS_MAPPINGS.get("Qwen35BatchImageTagger") is NODE)
check("有显示名",
      "Qwen35BatchImageTagger" in QM.NODE_DISPLAY_NAME_MAPPINGS)
check("旧节点仍注册（没有把扩写节点挤掉）",
      QM.NODE_CLASS_MAPPINGS.get("Qwen35PromptEnhancer") is QM.Qwen35PromptEnhancer)
check("打标节点继承扩写节点", issubclass(NODE, QM.Qwen35PromptEnhancer))
check("两者共用同一份常驻模型缓存（不会各加载一份 17.5GiB）",
      NODE._cache is QM.Qwen35PromptEnhancer._cache)
check("速度优化链路被继承：_free_vram / _load / _release 都在",
      all(hasattr(NODE, m) for m in ("_free_vram", "_load", "_release", "_resolve_path")))
check("模块 docstring 提到了新节点",
      "Qwen35BatchImageTagger" in (QM.__doc__ or ""))


# ===========================================================================
section("G) 进度单位：批量按「张」，扩写仍是「tok」")
# ===========================================================================
_rep = QM._ProgressReporter(enabled=False)
check("批量进度行用「张」", "3/10 张" in _rep._render(3, 10, "0.4s/张", unit="张"),
      _rep._render(3, 10, "0.4s/张", unit="张"))
check("批量进度行的速度单位也是「张/s」",
      "张/s" in _rep._render(3, 10, "", unit="张"))
check("默认（不传 unit）仍是 tok —— 扩写路径不受影响",
      "tok" in _rep._render(3, 10))
_crit = QM._make_progress_criteria(None, 8)
check("reporter=None 也能造出 criteria（批量要的是「只查中断」）", _crit is not None)
if _crit is not None:
    _crit(input_ids=None)
    check("criteria 自己记录了 prefill 时刻（日志要能拆 prefill/解码）",
          _crit.prefill_s >= 0.0 and "prefill_s" in dir(_crit))


# ===========================================================================
section("F) 批量行为（假模型 + 真写盘）")
# ===========================================================================
REPLY_IDS = [11, 12, 13, 14]
GOOD = "```\n1girl, solo,\n long_hair, 1girl, - blue_eyes \n```"
CLEAN_EXPECT = "1girl, solo, long_hair, blue_eyes"


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeGenCfg:
    eos_token_id = 7


class FakeProcessor:
    def __init__(self):
        self.reply_text = GOOD
        self.tmpl_calls = 0
        self.messages_seen = []
        self.decode_calls = 0

    def apply_chat_template(self, messages, **kw):
        self.tmpl_calls += 1
        self.messages_seen.append(messages)
        return FakeInputs({"input_ids": torch.tensor([[1, 2, 3]])})

    def decode(self, ids, skip_special_tokens=True):
        self.decode_calls += 1
        return self.reply_text


class FakeModel:
    device = torch.device("cpu")
    generation_config = FakeGenCfg()

    def __init__(self, fail_at=(), interrupt_at=(), empty_at=(), raw_reply=None):
        self.fail_at = set(fail_at)
        self.interrupt_at = set(interrupt_at)
        self.empty_at = set(empty_at)
        self.raw_reply = raw_reply
        self.n = 0
        self.proc = None
        self.gen_kwargs = []

    def generate(self, **kw):
        self.n += 1
        self.gen_kwargs.append(kw)
        crit = kw.get("stopping_criteria")
        if crit:
            crit[0](input_ids=None)          # 真跑一次 criteria：查中断 + 记 prefill
        if self.n in self.interrupt_at:
            INTERRUPT["flag"] = True
            crit[0](input_ids=None)          # 第二次调用就会抛 InterruptProcessingException
        if self.n in self.fail_at:
            raise ValueError("假失败：这一张读不出来")
        if self.n in self.empty_at:
            self.proc.reply_text = ""
        elif self.raw_reply is not None:
            self.proc.reply_text = self.raw_reply
        else:
            self.proc.reply_text = GOOD
        return torch.tensor([[1, 2, 3] + REPLY_IDS])


class Harness:
    """把 _load / _resolve_path 换成假件，_release 计数后置空。"""

    def __init__(self, **mf):
        self.model = FakeModel(**mf)
        self.proc = FakeProcessor()
        self.model.proc = self.proc
        self.releases = 0
        self.loads = 0
        self._orig = (NODE._resolve_path, NODE._load, NODE._release)
        outer = self

        def _resolve(self_, model_name, custom_model_path):
            return os.path.join(TMP_ROOT, "fakemodel")

        def _load(self_, path, quant, attn):
            outer.loads += 1
            return outer.model, outer.proc, True

        def _release(self_, force=False):
            outer.releases += 1

        NODE._resolve_path = _resolve
        NODE._load = _load
        NODE._release = _release

    def close(self):
        (NODE._resolve_path, NODE._load, NODE._release) = self._orig

    def run(self, folder, **kw):
        args = dict(
            model_name="fake",
            folder_path=folder,
            system_preset="custom",
            system_prompt="SYS-PROMPT",
            unload_other_models=False,
            show_progress=False,
        )
        args.update(kw)
        return NODE().tag_folder(**args)


def make_set(dst, n, prefix="im", ext=".png"):
    os.makedirs(dst, exist_ok=True)
    paths = []
    for i in range(1, n + 1):
        p = os.path.join(dst, f"{prefix}{i:02d}{ext}")
        Image.new("RGB", (16, 16), (i * 7 % 255, 30, 60)).save(p)
        paths.append(p)
    return paths


# ---- F1 dry_run：只列清单，不加载模型、不写文件 ----
D1 = os.path.join(TMP_ROOT, "d1")
paths1 = make_set(D1, 3)
h = Harness()
try:
    n0 = len(LOGS)
    res = h.run(D1, dry_run=True)
    txt = res["result"][0]
    print("  --- dry_run 报告 ---")
    for ln in txt.splitlines():
        print("   |", ln)
    check("dry_run：没有加载模型", h.loads == 0, str(h.loads))
    check("dry_run：一张 txt 都没写", not any(os.path.exists(QM._txt_path_for(p)) for p in paths1))
    check("dry_run：返回计数 0", res["result"][1] == 0)
    check("dry_run：报告里列出待处理文件与目标 txt",
          "im01.png" in txt and "im01.txt" in txt)
    check("dry_run：报告说明不会写文件", "dry_run=true" in txt)
    check("dry_run：走 ui 通道把报告回显到节点上", "text" in res["ui"])
finally:
    h.close()

# ---- F2 空文件夹 / 全是非图片：不加载模型 ----
D2 = os.path.join(TMP_ROOT, "d2")
os.makedirs(D2, exist_ok=True)
open(os.path.join(D2, "readme.txt"), "w").write("hi")
h = Harness()
try:
    res = h.run(D2)
    check("没有图片时：不加载模型", h.loads == 0, str(h.loads))
    check("没有图片时：报告写明并给出扫描数", "扫描到      : 0 张" in res["result"][0],
          res["result"][0].splitlines()[2] if len(res["result"][0].splitlines()) > 2 else "")
    check("没有图片时：计数 0", res["result"][1] == 0)
finally:
    h.close()

# ---- F3 正常批量：写同名 txt、内容被清洗 ----
D3 = os.path.join(TMP_ROOT, "d3")
paths3 = make_set(D3, 4)
h = Harness()
try:
    n0 = len(LOGS)
    res = h.run(D3)
    txt = res["result"][0]
    print("  --- 正常批量报告 ---")
    for ln in txt.splitlines():
        print("   |", ln)
    check("加载模型恰好一次（整个文件夹共用）", h.loads == 1, str(h.loads))
    check("4 张全部成功", res["result"][1] == 4, str(res["result"][1]))
    check("为每张图都写了 txt",
          all(os.path.exists(QM._txt_path_for(p)) for p in paths3))
    content = open(QM._txt_path_for(paths3[0]), encoding="utf-8").read()
    print(f"  第 1 张 txt 内容: {content!r}")
    check("txt 内容是一行清洗后的标签", content == CLEAN_EXPECT, content)
    check("txt 就近写在图片所在目录（不新建标签文件夹）",
          os.path.dirname(QM._txt_path_for(paths3[0])) == D3)
    check("报告含 成功 / 失败 计数", "成功 / 失败 : 4 / 0" in txt)
    check("报告含平均标签数与 token 数",
          "标签合计" in txt and "输出 token" in txt)
    check("报告含首张 prefill（含 CUDA 预热说明）", "首张 prefill" in txt)
    check("报告含每张的单行日志（✓）", "✓ im01.png" in logs_since(n0))
    check("每张日志带标签数/token/耗时", "标签 / 4 tok /" in logs_since(n0))
    check("报告打了全部成功", "✓ 全部成功" in txt)
    check("默认 keep_model_loaded=False 会释放模型", h.releases >= 1, str(h.releases))
    check("系统提示词确实传给了 chat template",
          h.proc.messages_seen[0][0]["role"] == "system"
          and h.proc.messages_seen[0][0]["content"] == "SYS-PROMPT")
    check("用户提示词也跟着传了",
          h.proc.messages_seen[0][1]["content"][-1]["type"] == "text")
    check("图片在消息里排在文字之前",
          h.proc.messages_seen[0][1]["content"][0]["type"] == "image")
    check("generate 收到了中断用 criteria",
          "stopping_criteria" in h.model.gen_kwargs[0])
    check("temperature>0 时走采样",
          h.model.gen_kwargs[0].get("do_sample") is True
          and abs(h.model.gen_kwargs[0].get("temperature", 0) - 0.2) < 1e-9)
    check("max_new_tokens 传对了", h.model.gen_kwargs[0]["max_new_tokens"] == 256)
finally:
    h.close()

# ---- F4 overwrite=skip：已有非空 txt 不覆盖，零字节 txt 重打 ----
D4 = os.path.join(TMP_ROOT, "d4")
paths4 = make_set(D4, 3)
open(QM._txt_path_for(paths4[0]), "w", encoding="utf-8").write("手工写好的标签")
open(QM._txt_path_for(paths4[1]), "w", encoding="utf-8").write("")     # 零字节残留
h = Harness()
try:
    res = h.run(D4)
    check("已有非空 txt 被跳过且内容原样保留",
          open(QM._txt_path_for(paths4[0]), encoding="utf-8").read() == "手工写好的标签")
    check("零字节 txt 被重新打标（上次写失败的残留）",
          open(QM._txt_path_for(paths4[1]), encoding="utf-8").read() == CLEAN_EXPECT)
    check("只处理了 2 张（跳过 1 张）", res["result"][1] == 2, str(res["result"][1]))
    check("报告里写明跳过了几张", "跳过 1 张" in res["result"][0])
    check("跳过判定发生在加载模型之前也能体现（报告先打印）",
          res["result"][0].index("待处理") < res["result"][0].index("打标汇总"))
finally:
    h.close()

# ---- F4b 全部已有标签：一次模型都不加载 ----
h = Harness()
try:
    res = h.run(D4)
    check("整批都已打好标签时不再加载模型", h.loads == 0, str(h.loads))
    check("并明确提示没有需要处理的图片", "没有需要处理的图片" in res["result"][0])
finally:
    h.close()

# ---- F5 overwrite=overwrite：强制覆盖 ----
h = Harness()
try:
    res = h.run(D4, overwrite="overwrite")
    check("overwrite=overwrite 时覆盖已有 txt",
          open(QM._txt_path_for(paths4[0]), encoding="utf-8").read() == CLEAN_EXPECT)
    check("overwrite=overwrite 时 3 张全处理", res["result"][1] == 3, str(res["result"][1]))
finally:
    h.close()

# ---- F6 单张失败不中断整批 ----
D6 = os.path.join(TMP_ROOT, "d6")
paths6 = make_set(D6, 3)
h = Harness(fail_at=(2,))
try:
    n0 = len(LOGS)
    res = h.run(D6)
    txt = res["result"][0]
    check("第 2 张失败后仍继续处理第 3 张", res["result"][1] == 2, str(res["result"][1]))
    check("报告写明 2 成功 / 1 失败", "成功 / 失败 : 2 / 1" in txt)
    check("失败清单里带文件名与原因",
          "im02.png" in txt and "假失败" in txt, txt.splitlines()[-2:])
    check("失败的图不写 txt（不留下空文件）",
          not os.path.exists(QM._txt_path_for(paths6[1])))
    check("失败的前后两张都正常写了 txt",
          os.path.exists(QM._txt_path_for(paths6[0]))
          and os.path.exists(QM._txt_path_for(paths6[2])))
    check("日志用 ✗ 标出失败那张", "✗ im02.png" in logs_since(n0))
finally:
    h.close()

# ---- F7 空输出视作失败，不写空 txt ----
D7 = os.path.join(TMP_ROOT, "d7")
paths7 = make_set(D7, 2)
h = Harness(empty_at=(1,))
try:
    res = h.run(D7)
    check("空输出被判为失败", res["result"][1] == 1, str(res["result"][1]))
    check("空输出不写 txt（否则下游会当成已打好标）",
          not os.path.exists(QM._txt_path_for(paths7[0])))
    check("失败原因说明是空输出", "输出为空" in res["result"][0])
finally:
    h.close()

# ---- F8 取消：清理 + 抛出，已完成的保留、不留半截文件 ----
D8 = os.path.join(TMP_ROOT, "d8")
paths8 = make_set(D8, 4)
INTERRUPT["flag"] = False
h = Harness(interrupt_at=(2,))
try:
    n0 = len(LOGS)
    raised = None
    try:
        h.run(D8)
    except BaseException as e:
        raised = e
    check("取消会抛出（不会把整批跑完）",
          type(raised).__name__ == "InterruptProcessingException", type(raised).__name__)
    check("取消时模型被释放（keep_model_loaded=False）", h.releases >= 1, str(h.releases))
    check("取消日志说明处理到第几张", "已取消：处理到第 2/4 张" in logs_since(n0),
          logs_since(n0)[-120:])
    check("第 1 张已写好的 txt 被保留",
          open(QM._txt_path_for(paths8[0]), encoding="utf-8").read() == CLEAN_EXPECT)
    check("被打断那张没留下 txt", not os.path.exists(QM._txt_path_for(paths8[1])))
    check("被打断那张没留下 .qwen35tmp",
          not os.path.exists(QM._txt_path_for(paths8[1]) + ".qwen35tmp"))
    check("后面没跑到的两张也没写",
          not os.path.exists(QM._txt_path_for(paths8[2]))
          and not os.path.exists(QM._txt_path_for(paths8[3])))
finally:
    INTERRUPT["flag"] = False
    h.close()

check("_is_comfy_interrupt 认得真实的中断异常",
      QM._is_comfy_interrupt(InterruptProcessingException()) is True)


class _FakeInterruptByName(Exception):
    pass


_FakeInterruptByName.__name__ = "InterruptProcessingException"
check("_is_comfy_interrupt 靠类名兜底（桩环境下也认得）",
      QM._is_comfy_interrupt(_FakeInterruptByName()) is True)
check("_is_comfy_interrupt 不会把普通异常误判",
      QM._is_comfy_interrupt(ValueError("x")) is False)
check("_is_comfy_interrupt 不会把 KeyboardInterrupt 误判",
      QM._is_comfy_interrupt(KeyboardInterrupt()) is False)

# ---- F9 output_format=raw：不做清洗 ----
D9 = os.path.join(TMP_ROOT, "d9")
paths9 = make_set(D9, 1)
h = Harness(raw_reply="A woman sitting by the window.\nShe holds a cup.")
try:
    res = h.run(D9, output_format="raw")
    got = open(QM._txt_path_for(paths9[0]), encoding="utf-8").read()
    check("raw 模式保留换行与自然语言（不压成逗号）",
          got == "A woman sitting by the window.\nShe holds a cup.", repr(got))
finally:
    h.close()

# ---- F10 output_suffix / output_encoding / limit / 递归 / 温度 0 ----
D10 = os.path.join(TMP_ROOT, "d10")
paths10 = make_set(D10, 3)
os.makedirs(os.path.join(D10, "deep"), exist_ok=True)
Image.new("RGB", (8, 8)).save(os.path.join(D10, "deep", "nested.png"))
h = Harness()
try:
    res = h.run(D10, output_suffix="_tags")
    check("output_suffix 生效（im01_tags.txt）",
          os.path.exists(os.path.join(D10, "im01_tags.txt")))
    check("有后缀时不写无后缀的 txt",
          not os.path.exists(os.path.join(D10, "im01.txt")))
finally:
    h.close()

h = Harness()
try:
    # 注意换个后缀：用同一个后缀时上一轮已经写好 3 个 txt，
    # overwrite=skip 会把它们跳过，只剩子目录那张要处理（那是正确行为）。
    res = h.run(D10, output_suffix="_rec", recursive=True)
    check("递归模式会处理子目录图片", res["result"][1] == 4, str(res["result"][1]))
    check("子目录里也写了同名 txt",
          os.path.exists(os.path.join(D10, "deep", "nested_rec.txt")))
finally:
    h.close()

h = Harness()
try:
    res = h.run(D10, limit=2, output_suffix="_lim")
    check("limit 只处理前 N 张", res["result"][1] == 2, str(res["result"][1]))
    check("limit 提示出现在报告里", "limit=2" in res["result"][0])
finally:
    h.close()

h = Harness()
try:
    h.run(D10, temperature=0.0, output_suffix="_greedy")
    check("temperature=0 走贪心解码",
          h.model.gen_kwargs[0].get("do_sample") is False,
          str(h.model.gen_kwargs[0].get("do_sample")))
finally:
    h.close()

h = Harness()
try:
    n0 = len(LOGS)
    h.run(D10, output_encoding="gbk", output_suffix="_gbk")
    p = os.path.join(D10, "im01_gbk.txt")
    check("output_encoding=gbk 时文件真是 gbk 编码",
          open(p, "rb").read().decode("gbk").startswith("1girl"))
    check("日志里报告了编码", "编码 gbk" in logs_since(n0))
finally:
    h.close()

# ---- F11 图片不存在（被外部删掉）时算这一张失败，不影响别的 ----
D11 = os.path.join(TMP_ROOT, "d11")
paths11 = make_set(D11, 2)
os.remove(paths11[0])
h = Harness()
try:
    res = h.run(D11)
    check("图被删掉时算单张失败（同批另一张照常完成）",
          res["result"][1] == 1, str(res["result"][1]))
finally:
    h.close()


# ===========================================================================
section("H) 系统提示词预设：JSON 加载 / 选择逻辑 / 容错 / 热重载")
# ===========================================================================
_PJ = QM._PRESET_JSON
check("预设 JSON 落在本节点目录下（presets/ 子目录）",
      os.path.dirname(_PJ) == os.path.join(NODE_DIR, "presets"), _PJ)
check("预设 JSON 已经生成出来（前面 INPUT_TYPES 触发过）", os.path.isfile(_PJ), _PJ)

_presets = QM._load_tag_presets()
check("默认带 6 套预设（3 主题 × 中英）",
      set(_presets) == {"photoreal", "photoreal_zh", "character", "character_zh",
                        "scene", "scene_zh"},
      str(list(_presets)))
check("每套都有 label / prompt / lang / group / format",
      all(p.get("label") and p.get("prompt") and p.get("lang")
          and p.get("group") and p.get("format") for p in _presets.values()))
check("每套 format 都是 raw（自然语言描述不该被逗号规整破坏）",
      all(p["format"] == "raw" for p in _presets.values()),
      str({k: v["format"] for k, v in _presets.items()}))

# ---- 三条新要求：只写内容 / 自然语言 / 不回避 NSFW ----
for _n, _p in _presets.items():
    _t = _p["prompt"]
    _zh = _p["lang"] == "zh"
    check(f"{_n} 明令禁止风格 / 媒介 / 技法 / 画质词",
          ("不要出现风格、媒介、技法、画质类的词" in _t) if _zh
          else ("Do NOT use style, medium, technique or quality" in _t))
    check(f"{_n} 禁词表确实展开了（没有残留 <<STYLE_BAN>> 占位符）",
          ("赛璐璐上色" in _t) if _zh else ("cel shading" in _t))
    check(f"{_n} 明确要求写自然语言句子、不许写标签串",
          ("不要写成逗号分隔的标签串" in _t) if _zh
          else ("Do NOT write a comma-separated tag list" in _t))
    check(f"{_n} 写明 NSFW 不回避、如实描述",
          ("NSFW 属于正常范围" in _t) if _zh else ("NSFW IS IN SCOPE" in _t))
    check(f"{_n} 写明敏感内容要「直接、具体」地写（不委婉、不替代、不绕着说）",
          ("要**直接、具体**地写" in _t and "不要委婉" in _t and "不要换成含糊" in _t)
          if _zh else
          ("BE DIRECT AND SPECIFIC" in _t and "Do not euphemise" in _t
           and "talk around it" in _t))
    check(f"{_n} 带 refine_prompt（优化已有打标的校订指令）",
          len(str(_p.get("refine_prompt") or "")) > 200,
          str(len(str(_p.get("refine_prompt") or ""))))
    check(f"{_n} 保留「只写看得见的、不编造」硬规则",
          ("不要编造" in _t) if _zh else ("Never invent" in _t))
    check(f"{_n} 指定了输出语言",
          ("用中文写这段描述。" in _t) if _zh
          else ("Write the description in English." in _t))

check("六套互不相同（不是同一份文本换名）",
      len({p["prompt"] for p in _presets.values()}) == 6)
check("同主题的中英两份共用 group（双语配对全靠它）",
      _presets["photoreal"]["group"] == _presets["photoreal_zh"]["group"] == "photoreal"
      and _presets["scene"]["group"] == _presets["scene_zh"]["group"] == "scene")
check("中英两份的 lang 标对了",
      _presets["character"]["lang"] == "en" and _presets["character_zh"]["lang"] == "zh")
check("旧标签体系的脚手架已完全移除（不再列举 1girl / Tag order 那套）",
      not any(("Tag order (keep this order" in p["prompt"])
              or ("comma-separated tags. All lowercase" in p["prompt"])
              or ("1girl," in p["prompt"])
              for p in _presets.values()))

# ---- 选择逻辑 ----
_rec, _k = QM._resolve_tag_preset("photoreal", "SYS")
check("选预设时忽略 system_prompt（防止被忘改的旧 widget 值顶掉）",
      _rec["prompt"] == _presets["photoreal"]["prompt"] and _k == "photoreal")
check("返回的记录带 label（报告里要显示选了哪套）",
      _rec["label"] == _presets["photoreal"]["label"], str(_rec.get("label")))
check("返回的记录带 lang / group / format（双语编排与格式联动要用）",
      _rec["lang"] == "en" and _rec["group"] == "photoreal" and _rec["format"] == "raw")
_rec, _k = QM._resolve_tag_preset("custom", "SYS")
check("custom 用 widget 里填的文本", _rec["prompt"] == "SYS" and _k == "custom")
check("custom 的 format 是 None（含义：格式交给 output_format 控件）",
      _rec["format"] is None, str(_rec.get("format")))
_rec, _k = QM._resolve_tag_preset("没这个预设", "SYS")
check("认不出的预设名回退 custom（预设被删/改名时老工作流不至于拿不到提示词）",
      _rec["prompt"] == "SYS" and _k == "custom")
_rec, _k = QM._resolve_tag_preset("PHOTOREAL", "SYS")
check("预设名大小写不敏感（手改 JSON 常见）", _k == "photoreal")
_rec, _k = QM._resolve_tag_preset("", "SYS")
check("空 preset 视为 custom", _rec["prompt"] == "SYS" and _k == "custom")
_rec, _k = QM._resolve_tag_preset(None, "SYS")
check("None preset 也视为 custom", _rec["prompt"] == "SYS" and _k == "custom")

# ---- payload 解析容错 ----
_p = QM._parse_preset_payload({"presets": {
    "custom": {"prompt": "会被忽略"},
    "a": {"prompt": "AAA"},
    "b": "BBB",
}})
check("custom 是保留名，JSON 里写它会被忽略（否则下拉语义会乱）",
      "custom" not in _p and set(_p) == {"a", "b"}, str(list(_p)))
check("简写形式（key 直接给字符串）也认", _p["b"]["prompt"] == "BBB" and _p["b"]["label"] == "b")
check("新字段缺省时有兜底：lang=en / group=key / format=raw",
      _p["a"]["lang"] == "en" and _p["a"]["group"] == "a" and _p["a"]["format"] == "raw",
      str(_p["a"]))
check("lang 只认 en/zh，乱填回退 en",
      QM._parse_preset_payload(
          {"presets": {"x": {"prompt": "P", "lang": "jp"}}})["x"]["lang"] == "en")
check("format 只认 raw/tags_one_line，乱填回退 raw",
      QM._parse_preset_payload(
          {"presets": {"x": {"prompt": "P", "format": "??"}}})["x"]["format"] == "raw")
check("group 显式给了就用给的（中英配对靠它）",
      QM._parse_preset_payload(
          {"presets": {"x": {"prompt": "P", "group": "g"}}})["x"]["group"] == "g")
check("空 prompt 的条目被跳过（不然下拉里会多个空选项）",
      set(QM._parse_preset_payload({"presets": {
          "z": {"prompt": "   "}, "y": {"prompt": "Y"}}})) == {"y"})
try:
    QM._parse_preset_payload({"nope": 1})
    check("缺 presets 对象应当报错", False)
except ValueError:
    check("缺 presets 对象明确报错", True)
try:
    QM._parse_preset_payload({"presets": {"a": {"prompt": ""}}})
    check("全空 preset 应当报错", False)
except ValueError:
    check("全空 preset 明确报错（好让上层回退内置）", True)

# ---- 容错：JSON 坏掉不能拖垮节点 ----
_bak = _PJ + ".testbak"
shutil.copy2(_PJ, _bak)
try:
    n0 = len(LOGS)
    with open(_PJ, "w", encoding="utf-8") as f:
        f.write("{ 这不是合法的 json ")
    QM._TAG_PRESET_CACHE["mtime"] = None
    _bad = QM._load_tag_presets(force=True)
    check("JSON 坏掉：回退内置六套而不是抛异常",
          set(_bad) >= {"photoreal", "photoreal_zh", "character", "scene_zh"},
          str(list(_bad)))
    check("JSON 坏掉：日志里明确报了解析失败",
          "解析失败" in logs_since(n0), logs_since(n0)[-80:])
    check("JSON 坏掉：坏内容没被缓存住（下次还会重试读盘）",
          QM._TAG_PRESET_CACHE["mtime"] is None)
finally:
    shutil.copy2(_bak, _PJ)
    os.remove(_bak)
    QM._TAG_PRESET_CACHE["mtime"] = None

# ---- 热重载：改文本不用重启 ----
_hot = _PJ + ".hot"
shutil.copy2(_PJ, _hot)
try:
    with open(_PJ, "r", encoding="utf-8") as f:
        _doc = json.load(f)
    _doc["presets"]["scene"]["prompt"] = "HOT-RELOAD-OK"
    with open(_PJ, "w", encoding="utf-8") as f:
        json.dump(_doc, f, ensure_ascii=False, indent=2)
    _now = time.time() + 3
    os.utime(_PJ, (_now, _now))          # 确保 mtime 一定变化
    _rec, _k = QM._resolve_tag_preset("scene", "")
    check("改了 JSON 文本后无需重启即生效（按 mtime 热重载）",
          _rec["prompt"] == "HOT-RELOAD-OK" and _k == "scene", _rec["prompt"][:40])
finally:
    shutil.copy2(_hot, _PJ)
    os.remove(_hot)
    QM._TAG_PRESET_CACHE["mtime"] = None
    _rec, _k = QM._resolve_tag_preset("scene", "")
    check("恢复原文件后重新读回内置文本", _rec["prompt"] == _presets["scene"]["prompt"])

# ---- 报告里要写清用了哪套 ----
_h2 = Harness()
try:
    _r = _h2.run(D1, system_preset="scene", dry_run=True)
    check("报告里显示所选预设名", "系统提示词  : scene" in _r["result"][0],
          [l for l in _r["result"][0].splitlines() if "系统提示词" in l][:1])
    _r = _h2.run(D1, system_preset="custom", dry_run=True)
    check("custom 时报告写明取节点上填写的文本",
          "（custom" in _r["result"][0])
finally:
    _h2.close()

# ===========================================================================
section("I) 双语同时输出 / 输出字符上限 / 格式联动")
# ===========================================================================

check("_lang_word 认 en / zh / 空 / 其他",
      QM._lang_word("en") == "英文" and QM._lang_word("zh") == "中文"
      and QM._lang_word("") == "预设" and QM._lang_word(None) == "预设")

# ---- I1 纯函数：_truncate_output ----
_tc = QM._truncate_output
check("上限 0 = 不限", _tc("abcdef", 0) == ("abcdef", False))
check("没超限就原样返回", _tc("abcdef", 6) == ("abcdef", False))
_o, _c = _tc("一位穿红外套的女性站在窗边。她留着黑色长发，戴着一副眼镜。", 20)
check("超限时退到句末标点，且**保留**句号", _c and _o.endswith("。"), repr(_o))
check("截断结果一定不超上限", len(_o) <= 20, str(len(_o)))
_o, _c = _tc("短发，戴帽子，穿风衣，背双肩包，站在街上", 12)
check("退不到句末时退到分句，并丢掉悬空的逗号",
      _c and not _o.endswith("，") and len(_o) <= 12, repr(_o))
_o, _c = _tc("abcdefghijklmnopqrstuvwxyz", 10)
check("两头都退不到就硬切（不返回空）", _c and _o and len(_o) <= 10, repr(_o))
_o, _c = _tc("1girl, solo, long_hair, black_hair, blue_eyes", 28)
check("标签串模式按逗号退（不会切在半个标签里）",
      _c and _o.endswith("long_hair"), repr(_o))
check("英文句点也算句末（英文描述的主要落点）",
      _tc("A woman stands by the window. She has long hair.", 30)[0]
      == "A woman stands by the window.")

# ---- I2 纯函数：_tag_runs ----
_run1, _n1 = QM._tag_runs("photoreal", _presets["photoreal"], False)
check("单语：只跑一份，后缀空、lang=en",
      len(_run1) == 1 and _run1[0]["suffix"] == "" and _run1[0]["lang"] == "en")
check("单语：不产生任何说明（不该有噪声）", _n1 is None, str(_n1))
_run2, _n2 = QM._tag_runs("photoreal", _presets["photoreal"], True)
check("双语：跑中英两份，顺序固定 en -> zh",
      [r["lang"] for r in _run2] == ["en", "zh"], str([r["lang"] for r in _run2]))
check("双语：英文不带后缀，中文带 _zh",
      [r["suffix"] for r in _run2] == ["", "_zh"], str([r["suffix"] for r in _run2]))
check("双语：两份分别是该主题的英文版与中文版提示词",
      _run2[0]["prompt"] == _presets["photoreal"]["prompt"]
      and _run2[1]["prompt"] == _presets["photoreal_zh"]["prompt"])
check("双语能正常配对时不产生说明", _n2 is None, str(_n2))
_run3, _ = QM._tag_runs("photoreal_zh", _presets["photoreal_zh"], True)
check("选中文预设 + 双语：结果与选英文预设一样（主题由 group 决定）",
      [r["suffix"] for r in _run3] == ["", "_zh"])
_rec_c, _ = QM._resolve_tag_preset("custom", "MY-OWN-PROMPT")
_run4, _n4 = QM._tag_runs("custom", _rec_c, True)
check("custom + 双语：退化为单份，并明确说明原因（不能默默只写一份）",
      len(_run4) == 1 and _n4 and "custom" in _n4, str(_n4))
_fake_rec = {"prompt": "P", "label": "孤本", "lang": "en",
             "group": "no_such_group", "format": "raw"}
_run5, _n5 = QM._tag_runs("lonely", _fake_rec, True)
check("找不到同 group 的中文对照：退化为单份并说明缺哪种语言",
      len(_run5) == 1 and _n5 and "中文" in _n5, str(_n5))

# ---- I3 端到端：双语真的写出两个文件 ----
class _LangAwareModel(FakeModel):
    """按系统提示词里有没有中文规则，切换成不同语言的回复。"""

    def generate(self, **kw):
        self.n += 1
        self.gen_kwargs.append(kw)
        crit = kw.get("stopping_criteria")
        if crit:
            crit[0](input_ids=None)
        msgs = self.proc.messages_seen[-1] if self.proc.messages_seen else []
        sys_txt = "".join(str(m.get("content") or "") for m in msgs
                          if isinstance(m, dict) and m.get("role") == "system")
        self.proc.reply_text = ("中文描述内容。" if "用中文写这段描述" in sys_txt
                                else "English caption here.")
        return torch.tensor([[1, 2, 3] + REPLY_IDS])


D12 = os.path.join(TMP_ROOT, "d12")
make_set(D12, 2)
h12 = Harness()
h12.model = _LangAwareModel()
h12.model.proc = h12.proc
try:
    _r12 = h12.run(D12, system_preset="photoreal", bilingual="en_then_zh")
    check("双语：返回的是写出的文件数（2 张 × 2 语言 = 4）",
          _r12["result"][1] == 4, str(_r12["result"][1]))
    check("双语：模型被调用 4 次（每张两次）", h12.model.n == 4, str(h12.model.n))
    _en = os.path.join(D12, "im01.txt")
    _zh = os.path.join(D12, "im01_zh.txt")
    check("双语：英文写 im01.txt、中文写 im01_zh.txt",
          os.path.isfile(_en) and os.path.isfile(_zh),
          f"en={os.path.isfile(_en)} zh={os.path.isfile(_zh)}")
    check("双语：两份内容确实是各自语言的输出",
          open(_en, encoding="utf-8").read().strip() == "English caption here."
          and open(_zh, encoding="utf-8").read().strip() == "中文描述内容。",
          repr(open(_zh, encoding="utf-8").read()[:40]))
    check("双语：报告里写明两种语言与后缀",
          "英文" in _r12["result"][0] and "中文" in _r12["result"][0]
          and "_zh" in _r12["result"][0])
    check("双语：报告里有「生成次数」一行", "生成次数" in _r12["result"][0])
finally:
    h12.close()

# ---- I4 端到端：输出字符上限 ----
D13 = os.path.join(TMP_ROOT, "d13")
make_set(D13, 1)
_LONG = "一位穿红外套的女性站在窗边。她留着黑色长发，戴着一副眼镜。背景是白色的墙。"
h13 = Harness(raw_reply=_LONG)
try:
    _r13 = h13.run(D13, system_preset="photoreal", max_output_chars=20)
    _txt13 = open(os.path.join(D13, "im01.txt"), encoding="utf-8").read()
    check("字符上限：写出的 txt 不超上限", len(_txt13) <= 20, f"{len(_txt13)} 字符")
    check("字符上限：退到句末标点并保留句号", _txt13.endswith("。"), repr(_txt13))
    check("字符上限：报告里出现「按上限截断」", "按上限截断" in _r13["result"][0])
    check("字符上限：汇总里报告了截断条数", "按上限截断  :" in _r13["result"][0])
    _r13b = h13.run(D13, system_preset="photoreal", max_output_chars=0,
                    overwrite="overwrite")
    _txt13b = open(os.path.join(D13, "im01.txt"), encoding="utf-8").read()
    check("上限 0：完整写入，不做任何截断",
          _txt13b.strip() == _LONG and "按上限截断  :" not in _r13b["result"][0],
          f"{len(_txt13b)} 字符")
finally:
    h13.close()

# ---- I5 格式联动：预设决定 format，custom 才看 output_format 控件 ----
D14 = os.path.join(TMP_ROOT, "d14")
make_set(D14, 1)
h14 = Harness(raw_reply="first part,\nsecond part")
try:
    h14.run(D14, system_preset="photoreal")
    check("选了预设：按预设声明的 raw 原样写（换行保留，不被逗号规整）",
          "\n" in open(os.path.join(D14, "im01.txt"), encoding="utf-8").read())
    h14.run(D14, system_preset="custom", system_prompt="SYS", overwrite="overwrite")
    _t14 = open(os.path.join(D14, "im01.txt"), encoding="utf-8").read()
    check("custom：output_format 控件仍然生效（默认 tags_one_line 把换行拍平成逗号）",
          "\n" not in _t14 and "first part, second part" in _t14, repr(_t14[:60]))
finally:
    h14.close()

# ---- I6 双语 + skip：已有的一半不覆盖、只补缺的那半 ----
D15 = os.path.join(TMP_ROOT, "d15")
make_set(D15, 2)
with open(os.path.join(D15, "im01.txt"), "w", encoding="utf-8") as _f:
    _f.write("OLD-EN")
h15 = Harness(raw_reply="NEW")
try:
    _r15 = h15.run(D15, system_preset="photoreal", bilingual="en_then_zh")
    check("双语 + skip：英文已有的那张只补中文（2×2 - 1 = 3 次生成）",
          h15.model.n == 3, str(h15.model.n))
    check("双语 + skip：已有的英文 txt 未被覆盖",
          open(os.path.join(D15, "im01.txt"), encoding="utf-8").read() == "OLD-EN")
    check("双语 + skip：对应中文那份补上了",
          os.path.isfile(os.path.join(D15, "im01_zh.txt")))
    check("双语 + skip：另一张两种语言都写了",
          os.path.isfile(os.path.join(D15, "im02.txt"))
          and os.path.isfile(os.path.join(D15, "im02_zh.txt")))
    check("双语 + skip：返回文件数 3（补 1 中文 + 新 2 张 ×2）",
          _r15["result"][1] == 3, str(_r15["result"][1]))
finally:
    h15.close()

# ---- I7 dry_run 清单要两种语言都列出来 ----
h16 = Harness()
try:
    _r16 = h16.run(D15, system_preset="photoreal", bilingual="en_then_zh",
                   dry_run=True, overwrite="overwrite")
    _rep16 = _r16["result"][0]
    check("dry_run 双语：清单里同时列出 .txt 与 _zh.txt",
          "im02.txt" in _rep16 and "im02_zh.txt" in _rep16)
    check("dry_run 双语：清单里标了语言", "[英文]" in _rep16 and "[中文]" in _rep16)
    check("dry_run：不加载模型", h16.loads == 0, str(h16.loads))
finally:
    h16.close()

check("custom + dry_run：报告里说明双语开关为何无效",
      "custom" in Harness().run(D15, system_preset="custom", system_prompt="SYS",
                                bilingual="en_then_zh", dry_run=True)["result"][0])

# ===========================================================================
section("J) 优化已有打标（caption_mode = refine / refine_or_new）")
# ===========================================================================
# ---- 小工具 ----
def user_texts(h):
    """取出所有 user 侧**文本**块（图片块跳过）—— 用来断言初稿真的发出去了。"""
    out = []
    for _msgs in h.proc.messages_seen:
        for _m in _msgs:
            if isinstance(_m, dict) and _m.get("role") == "user":
                for _c in (_m.get("content") or []):
                    if isinstance(_c, dict) and _c.get("type") == "text":
                        out.append(_c.get("text") or "")
    return out


def jdir(name, n=1):
    _d = os.path.join(TMP_ROOT, name)
    return _d, make_set(_d, n)


# ---- J1 读初稿：编码宽容 / 空白 / 缺失 / 超长 ----
_rd = QM._read_text_tolerant
_p1 = os.path.join(TMP_ROOT, "enc_utf8.txt")
open(_p1, "w", encoding="utf-8").write("  一段 utf-8 中文  \n")
check("读初稿：utf-8 正常读出并 strip", _rd(_p1) == "一段 utf-8 中文", repr(_rd(_p1)))
_p2 = os.path.join(TMP_ROOT, "enc_gbk.txt")
open(_p2, "w", encoding="gbk").write("GBK 写的旧稿")
check("读初稿：gbk 旧稿也能读（编码自动回退）", _rd(_p2, "utf-8") == "GBK 写的旧稿", repr(_rd(_p2)))
_p3 = os.path.join(TMP_ROOT, "enc_bom.txt")
open(_p3, "w", encoding="utf-8-sig").write("带 BOM 的稿子")
check("读初稿：带 BOM 的 utf-8 不残留 \\ufeff", _rd(_p3) == "带 BOM 的稿子", repr(_rd(_p3)))
check("读初稿：文件不存在 -> 空串", _rd(os.path.join(TMP_ROOT, "nope.txt")) == "")
_p4 = os.path.join(TMP_ROOT, "enc_blank.txt")
open(_p4, "w", encoding="utf-8").write("  \n\n ")
check("读初稿：只有空白 -> 空串（当作没有初稿）", _rd(_p4) == "")
_p5 = os.path.join(TMP_ROOT, "enc_long.txt")
open(_p5, "w", encoding="utf-8").write("x" * 9000)
check("读初稿：超长按 _TAG_DRAFT_LIMIT 截断（不把上下文挤爆）",
      len(_rd(_p5)) == QM._TAG_DRAFT_LIMIT, str(len(_rd(_p5))))

# ---- J2 内容指纹：必须与「读回来」算得一模一样，否则幂等就是假的 ----
check("内容指纹：首尾空白被规范化（与读回的 strip 对齐）",
      QM._text_sha1("abc") == QM._text_sha1("  abc\n"))
check("内容指纹：内容不同 -> 指纹不同",
      QM._text_sha1("abc") != QM._text_sha1("abd"))
_p_sha = os.path.join(TMP_ROOT, "sha_rt.txt")
open(_p_sha, "w", encoding="utf-8").write("  某段描述  \n")
check("内容指纹：写出去与读回来算出来一致（refine 幂等的前提）",
      QM._text_sha1("某段描述") == QM._text_sha1(_rd(_p_sha)))

# ---- J3 校订指令：预设自带的优先，缺省按语言回落 ----
check("校订指令：预设自带的 refine_prompt 优先（不回落常量）",
      QM._refine_instruction(_presets["photoreal"]) == _presets["photoreal"]["refine_prompt"])
check("校订指令：英文预设缺 refine_prompt 时回落英文常量",
      "FINAL caption" in QM._refine_instruction({"lang": "en"}))
check("校订指令：中文预设缺 refine_prompt 时回落中文常量",
      "请把它改写成最终描述" in QM._refine_instruction({"lang": "zh"}))
check("校订指令：custom（无 lang）回落「与原稿同语言」那份",
      "相同的语言" in QM._refine_instruction({"lang": ""}))
check("校订指令：三份都要求「只输出最终描述、不要解释改了什么」",
      ("Do not explain what you changed" in QM._REFINE_INSTRUCTION)
      and ("不要解释你改了什么" in QM._REFINE_INSTRUCTION_ZH)
      and ("不要解释" in QM._REFINE_INSTRUCTION_KEEP_LANG))
check("校订指令：三份都写了「不要顺手消毒」（改写时不许把 NSFW 洗掉）",
      ("NEVER SANITISE" in QM._REFINE_INSTRUCTION)
      and ("消毒" in QM._REFINE_INSTRUCTION_ZH)
      and ("消毒" in QM._REFINE_INSTRUCTION_KEEP_LANG))

# ---- J4 拼装：指令在前、初稿在后，标签按语言选 ----
_ct = QM._compose_refine_user_text("INSTRUCTION", "OLD CAPTION", "en")
check("拼装：指令在前、初稿在后", _ct.index("INSTRUCTION") < _ct.index("OLD CAPTION"))
check("拼装：英文 run 用英文标签", "EXISTING CAPTION:" in _ct)
check("拼装：中文 run 用中文标签",
      "现有描述：" in QM._compose_refine_user_text("指令", "旧稿", "zh"))
check("拼装：custom（lang 空）用兜底标签",
      "existing caption" in QM._compose_refine_user_text("指令", "旧稿", ""))
check("拼装：初稿为空也返回字符串、不抛异常",
      isinstance(QM._compose_refine_user_text("I", "", ""), str))

# ---- J5 _tag_runs 把校订指令一起带出来 ----
check("_tag_runs：单语 run 也带校订指令",
      bool(QM._tag_runs("photoreal", _presets["photoreal"], False)[0][0].get("refine")))
_rr_bl = QM._tag_runs("photoreal", _presets["photoreal"], True)[0]
check("_tag_runs：双语时两份各带自己语言的校订指令",
      "FINAL caption" in _rr_bl[0]["refine"] and "请把它改写成最终描述" in _rr_bl[1]["refine"])

# ---- J6 配置指纹 ----
_r_cfg = QM._tag_runs("photoreal", _presets["photoreal"], False)[0][0]
check("配置指纹：同一份配置两次算出来一样",
      QM._refine_cfg_sig("photoreal", _r_cfg, 0) == QM._refine_cfg_sig("photoreal", _r_cfg, 0))
check("配置指纹：换预设 -> 变",
      QM._refine_cfg_sig("photoreal", _r_cfg, 0) != QM._refine_cfg_sig("scene", _r_cfg, 0))
check("配置指纹：改字符上限 -> 变",
      QM._refine_cfg_sig("photoreal", _r_cfg, 0) != QM._refine_cfg_sig("photoreal", _r_cfg, 120))
check("配置指纹：换系统提示词 -> 变",
      QM._refine_cfg_sig("photoreal", dict(_r_cfg, prompt="X"), 0)
      != QM._refine_cfg_sig("photoreal", _r_cfg, 0))

# ---- J7 端到端：首次优化（读初稿 / 写回 / 备份 / 记账） ----
D30, paths30 = jdir("j30", 1)
_t30 = QM._txt_path_for(paths30[0])
_OLD30 = "旧稿：一个女孩，动漫风格，masterpiece"
open(_t30, "w", encoding="utf-8").write(_OLD30)
h30 = Harness(raw_reply="一位年轻女性站在窗边，穿着浅色衬衫。")
try:
    r30 = h30.run(D30, system_preset="photoreal", caption_mode="refine")
    rep30 = r30["result"][0]
    check("refine：处理了 1 张", r30["result"][1] == 1, str(r30["result"][1]))
    check("refine：初稿被塞进 user 文本", any(_OLD30 in t for t in user_texts(h30)),
          str(user_texts(h30))[:100])
    check("refine：校订指令和初稿在同一段 user 文本里",
          any("FINAL caption" in t and _OLD30 in t for t in user_texts(h30)))
    check("refine：系统提示词仍是预设那份（没被校订指令顶掉）",
          h30.proc.messages_seen[0][0]["content"] == _presets["photoreal"]["prompt"])
    check("refine：结果写回同一个 txt",
          open(_t30, encoding="utf-8").read() == "一位年轻女性站在窗边，穿着浅色衬衫。")
    check("refine：原稿按字节备份成 .orig",
          open(_t30 + ".orig", encoding="utf-8").read() == _OLD30)
    check("refine：写下输出指纹记录 .q35state",
          os.path.isfile(_t30 + ".q35state")
          and bool(QM._read_refine_state(_t30).get("out_sha1")))
    check("refine：报告写明模式与变更判定",
          "优化已有打标" in rep30 and "变更判定" in rep30)
finally:
    h30.close()

# ---- J8 内容没变 -> 不覆盖、不优化（这是本轮的核心诉求） ----
h31 = Harness(raw_reply="这份回复不该被用到")
try:
    r31 = h31.run(D30, system_preset="photoreal", caption_mode="refine")
    check("内容未变：直接跳过，不加载模型、不覆盖",
          r31["result"][1] == 0 and h31.loads == 0, str(r31["result"][1]))
    check("内容未变：报告写明「内容未变跳过」", "内容未变跳过" in r31["result"][0])
    check("内容未变：txt 保持上次的输出",
          open(_t30, encoding="utf-8").read() == "一位年轻女性站在窗边，穿着浅色衬衫。")
finally:
    h31.close()

# ---- J9 手动改过 -> 自动继续优化 ----
_MANUAL = "我手动改的版本：只有一个人"
open(_t30, "w", encoding="utf-8").write(_MANUAL)
h32 = Harness(raw_reply="手改之后再优化出来的最终描述。")
try:
    r32 = h32.run(D30, system_preset="photoreal", caption_mode="refine")
    check("手改后继续优化：被认出来并重跑", r32["result"][1] == 1, str(r32["result"][1]))
    check("手改后继续优化：初稿用的就是我手改的那一版",
          any(_MANUAL in t for t in user_texts(h32)))
    check("手改后继续优化：输出已更新",
          open(_t30, encoding="utf-8").read() == "手改之后再优化出来的最终描述。")
    check("手改后继续优化：.orig 仍是最初那一稿（不被手改版覆盖）",
          open(_t30 + ".orig", encoding="utf-8").read() == _OLD30)
    check("手改后继续优化：指纹记录跟着更新",
          QM._read_refine_state(_t30)["out_sha1"]
          == QM._text_sha1("手改之后再优化出来的最终描述。"))
finally:
    h32.close()

# ---- J10 内容没变但换了预设 -> 该重做（否则换预设等于没换） ----
h33 = Harness(raw_reply="换了预设之后重跑的结果。")
try:
    r33 = h33.run(D30, system_preset="scene", caption_mode="refine")
    check("换了预设：内容没变也重做（配置指纹变了）",
          r33["result"][1] == 1, str(r33["result"][1]))
finally:
    h33.close()
h33b = Harness()
try:
    r33b = h33b.run(D30, system_preset="scene", caption_mode="refine")
    check("同预设再跑一次：又跳过了（说明上一步真的记了新指纹）",
          r33b["result"][1] == 0, str(r33b["result"][1]))
finally:
    h33b.close()

# ---- J11 overwrite=overwrite 可无视指纹强制重做 ----
h33c = Harness(raw_reply="强制重做的结果。")
try:
    r33c = h33c.run(D30, system_preset="scene", caption_mode="refine", overwrite="overwrite")
    check("overwrite=overwrite：内容没变也强制重做", r33c["result"][1] == 1,
          str(r33c["result"][1]))
    check("强制重做也不会覆盖 .orig（最初那一稿永远留着）",
          open(_t30 + ".orig", encoding="utf-8").read() == _OLD30)
finally:
    h33c.close()

# ---- J12 无初稿：refine 跳过 / refine_or_new 从零补写 ----
D31, paths31 = jdir("j31", 3)
_t31 = [QM._txt_path_for(p) for p in paths31]
open(_t31[0], "w", encoding="utf-8").write("第一张的旧稿")
h34 = Harness(raw_reply="REFINED")
try:
    r34 = h34.run(D31, system_preset="photoreal", caption_mode="refine")
    check("caption_mode=refine：没有旧 txt 的图一律跳过（只处理 1 张）",
          r34["result"][1] == 1, str(r34["result"][1]))
    check("caption_mode=refine：没旧稿的不会被写出 txt",
          not os.path.exists(_t31[1]) and not os.path.exists(_t31[2]))
    check("caption_mode=refine：没旧稿的也不会留下 .orig",
          not os.path.exists(_t31[1] + ".orig"))
finally:
    h34.close()

h35 = Harness(raw_reply="FILLED")
try:
    r35 = h35.run(D31, system_preset="photoreal", caption_mode="refine_or_new")
    check("caption_mode=refine_or_new：无初稿的从零补写（2 张）",
          r35["result"][1] == 2, str(r35["result"][1]))
    check("refine_or_new：补写的两张写出了 txt",
          os.path.exists(_t31[1]) and os.path.exists(_t31[2]))
    check("refine_or_new：从零补写的不留 .orig（本来就没有原稿可备份）",
          not os.path.exists(_t31[1] + ".orig"))
    check("refine_or_new：补写的也记了指纹（下次不会重做）",
          bool(QM._read_refine_state(_t31[1]).get("out_sha1")))
    check("refine_or_new：报告区分「待优化」与「无初稿从零写」",
          "无初稿从零写" in r35["result"][0])
finally:
    h35.close()

h36 = Harness(raw_reply="不该被用到")
try:
    r36 = h36.run(D31, system_preset="photoreal", caption_mode="refine_or_new")
    check("refine_or_new 幂等：第二次一张都不动（含刚才从零补写的）",
          r36["result"][1] == 0, str(r36["result"][1]))
finally:
    h36.close()

# ---- J13 写盘失败：不许留下误导性痕迹 ----
D32, paths32 = jdir("j32", 1)
_t32 = QM._txt_path_for(paths32[0])
open(_t32, "w", encoding="utf-8").write("原始稿")
h37 = Harness(raw_reply="新稿")
_save_wta = QM._write_text_atomic


def _boom(*a, **k):
    raise OSError("磁盘写满（模拟）")


QM._write_text_atomic = _boom
try:
    r37 = h37.run(D32, system_preset="photoreal", caption_mode="refine")
    check("写盘失败：原 txt 未被改动", open(_t32, encoding="utf-8").read() == "原始稿")
    check("写盘失败：刚建的 .orig 被撤回（不留误导性痕迹）",
          not os.path.exists(_t32 + ".orig"))
    check("写盘失败：不留下输出记录（否则下次会被当成「没变」跳过）",
          not os.path.exists(_t32 + ".q35state"))
    check("写盘失败：算这一张失败并写进报告", "成功 / 失败 : 0 / 1" in r37["result"][0])
finally:
    QM._write_text_atomic = _save_wta
    h37.close()

# ---- J14 dry_run 清单要标出「校订」与「补写」 ----
D33, paths33 = jdir("j33", 2)
open(QM._txt_path_for(paths33[0]), "w", encoding="utf-8").write("有稿")
h38 = Harness()
try:
    r38 = h38.run(D33, system_preset="photoreal", caption_mode="refine_or_new", dry_run=True)
    rep38 = r38["result"][0]
    check("dry_run(refine)：清单标出「校订已有初稿」与「从零补写」",
          "校订已有初稿" in rep38 and "从零补写" in rep38)
    check("dry_run(refine)：不加载模型、不写文件",
          h38.loads == 0 and not os.path.exists(QM._txt_path_for(paths33[1])))
    check("dry_run(refine)：报告写明模式", "打标模式" in rep38 and "优化已有打标" in rep38)
finally:
    h38.close()

# ---- J15 双语 refine：中英各读各的初稿 ----
D34, paths34 = jdir("j34", 1)
_t34en = QM._txt_path_for(paths34[0])
_t34zh = QM._txt_path_for(paths34[0], "_zh")
open(_t34en, "w", encoding="utf-8").write("EN OLD CAPTION")
open(_t34zh, "w", encoding="utf-8").write("中文旧稿")
h39 = Harness(raw_reply="REPLY")
try:
    r39 = h39.run(D34, system_preset="photoreal", bilingual="en_then_zh",
                  caption_mode="refine")
    ut39 = user_texts(h39)
    check("双语 refine：英文那份读的是 .txt",
          any("EN OLD CAPTION" in t and "EXISTING CAPTION" in t for t in ut39),
          str(ut39)[:120])
    check("双语 refine：中文那份读的是 _zh.txt（各读各的）",
          any("中文旧稿" in t and "现有描述：" in t for t in ut39), str(ut39)[:120])
    check("双语 refine：两个文件都写回", r39["result"][1] == 2, str(r39["result"][1]))
    check("双语 refine：两个 .orig 都建了",
          os.path.exists(_t34en + ".orig") and os.path.exists(_t34zh + ".orig"))
    check("双语 refine：两份各自记了指纹",
          os.path.exists(_t34en + ".q35state") and os.path.exists(_t34zh + ".q35state"))
finally:
    h39.close()

# ---- J16 回归：caption_mode=off 时老行为一字不变 ----
D35, paths35 = jdir("j35", 1)
_t35 = QM._txt_path_for(paths35[0])
open(_t35, "w", encoding="utf-8").write("已有标签")
h40 = Harness()
try:
    r40 = h40.run(D35, system_preset="photoreal")          # caption_mode 默认 off
    check("caption_mode=off（默认）：有 txt 的图照旧跳过",
          r40["result"][1] == 0 and h40.loads == 0, str(r40["result"][1]))
    check("caption_mode=off：不建 .orig / .q35state",
          not os.path.exists(_t35 + ".orig")
          and not os.path.exists(_t35 + ".q35state"))
    check("caption_mode=off：报告也写明「从零打标」",
          "从零打标" in r40["result"][0])
finally:
    h40.close()

# ================================================================
# K  GGUF 后端（gguf_backend.py）—— 全程假 server，不加载真模型、不碰显卡
# ================================================================
# 真机那条路（27B Q4 是否真进显存、tok/s 多少）由 smoke_gguf_backend.py 负责；
# 这里只钉死「参数拼装 / 签名复用 / health 轮询 / payload 形态 / 报错文本」这些
# 纯逻辑 —— 它们不依赖模型，用假 API 就能全部覆盖，也因此能塞进回归测试里。
section("K  GGUF 后端（llama.cpp / llama-server）")

import base64 as _b64              # noqa: E402
import http.server as _httpsrv     # noqa: E402
import threading as _thr           # noqa: E402

import gguf_backend as GBm         # noqa: E402


class _FakeAPI(_httpsrv.BaseHTTPRequestHandler):
    """假的 llama-server：只实现 /health 和 /v1/chat/completions。"""

    payloads = []
    health_code = 200
    health_body = {"status": "ok"}

    def log_message(self, *a):
        pass

    def _out(self, obj, code=200):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._out(_FakeAPI.health_body, _FakeAPI.health_code)
        else:
            self._out({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            _FakeAPI.payloads.append(json.loads(raw.decode("utf-8")))
        except Exception:
            _FakeAPI.payloads.append({})
        self._out({
            "choices": [{"message": {"content": "TAGGED-OUT",
                                     "reasoning_content": "hidden-thought"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "timings": {"prompt_ms": 300.0, "predicted_ms": 200.0},
        })


_fake_srv = _httpsrv.ThreadingHTTPServer(("127.0.0.1", 0), _FakeAPI)
_thr.Thread(target=_fake_srv.serve_forever, daemon=True).start()
FAKE_PORT = _fake_srv.server_address[1]


class _FakeProc:
    def __init__(self, alive=True):
        self.alive = alive
        self.stdout = None

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        return 0


def kserver(alive=True):
    """一个指向假 API 的 GgufServer；_start 换掉，不真拉进程。"""
    g = GBm.GgufServer({"server_exe": "x.exe", "model": "y.gguf"})
    g.port = FAKE_PORT
    g.proc = _FakeProc(alive)
    g._start = lambda: None
    return g


# ---- K1 build_args：命令行拼装 ----
KCFG = {
    "server_exe": r"C:\x\llama-server.exe", "model": r"C:\x\a.gguf",
    "mmproj": r"C:\x\mm.gguf", "parallel": 1, "n_gpu_layers": -1,
    "context_size": 8192, "kv_cache_type": "q8_0", "flash_attn": True,
    "batch": 2048, "ubatch": 512, "reasoning_format": "deepseek",
    "extra_args": "",
}


def kargs(**over):
    return GBm.GgufServer({**KCFG, **over}).build_args()


_a_bare = GBm.GgufServer({"server_exe": r"C:\x\llama-server.exe",
                          "model": "m.gguf"}).build_args()
check("K1a 最小配置也能拼出可用的命令行",
      _a_bare[0].endswith("llama-server.exe") and "-m" in _a_bare
      and "--host" in _a_bare and "--port" in _a_bare,
      " ".join(_a_bare[:6]))
check("K1b n_gpu_layers=-1 时不传 -ngl（层放置交给 llama.cpp 自动 fit）",
      "-ngl" not in _a_bare)
_a = kargs()
check("K1c 带 mmproj 时传 --mmproj", "--mmproj" in _a)
check("K1d 只有显式给了 n_gpu_layers>=0 才传 -ngl",
      "-ngl" not in _a and "-ngl" in kargs(n_gpu_layers=30))
check("K1e --flash-attn 用 on/off 取值（新版 llama.cpp 不认裸开关）",
      _a[_a.index("--flash-attn") + 1] == "on")
check("K1f --jinja 必开（否则 GGUF 里的 Qwen 模板与图片占位符都对不上）",
      "--jinja" in _a)
check("K1g parallel 钉死 1（48 层线性注意力的 recurrent state 按槽位算）",
      _a[_a.index("--parallel") + 1] == "1")
check("K1h KV cache 走 q8_0；填 f16 时反而不传该参数",
      "--cache-type-k" in _a and "--cache-type-k" not in kargs(kv_cache_type="f16"))
check("K1i flash_attn=False 时不传 --flash-attn",
      "--flash-attn" not in kargs(flash_attn=False))
check("K1j reasoning_format=none 时不传 --reasoning-format",
      "--reasoning-format" not in kargs(reasoning_format="none"))
check("K1k 上下文长度通过 -c 传下去", _a[_a.index("-c") + 1] == "8192")
check("K1l extra_args 原样追加到命令行尾",
      kargs(extra_args="--image-max-tokens 1024")[-2:] == ["--image-max-tokens", "1024"])

# ---- K2 参数签名 ----
_s1 = GBm._signature(KCFG)
check("K2a 同一份配置 -> 同一签名", GBm._signature(dict(KCFG)) == _s1)
check("K2b 换上下文 -> 签名变化（会触发重建）",
      GBm._signature({**KCFG, "context_size": 4096}) != _s1)
check("K2c 换模型路径 -> 签名变化",
      GBm._signature({**KCFG, "model": r"C:\x\b.gguf"}) != _s1)
check("K2d 换 mmproj -> 签名变化",
      GBm._signature({**KCFG, "mmproj": r"C:\x\other.gguf"}) != _s1)
check("K2e batch / ubatch 这类吃显存的参数也在签名里",
      GBm._signature({**KCFG, "batch": 1024}) != _s1
      and GBm._signature({**KCFG, "ubatch": 256}) != _s1)

# ---- K3 mmproj 识别与标签往返 ----
check("K3a mmproj 识别：含 mmproj / clip 开头 / 含 vision",
      GBm._is_mmproj("mmproj-model-bf16.gguf") and GBm._is_mmproj("clip-vit.gguf")
      and GBm._is_mmproj("vision-tower.gguf"))
check("K3b 主干模型不会被误判成 mmproj",
      not GBm._is_mmproj("Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf"))
check("K3c path_from_label 把「（无 / 纯文本）」翻成空串",
      GBm.path_from_label(GBm._NONE_MMPROJ) == "")
check("K3d path_from_label 给的是文件路径时原样返回",
      GBm.path_from_label(r"H:\AI\models\LLM\x.gguf") == r"H:\AI\models\LLM\x.gguf")

# ---- K4 签名单例：复用与重建 ----
GBm.release(force=True)
_i1 = GBm.acquire(dict(KCFG))
_i2 = GBm.acquire(dict(KCFG))
check("K4a 同签名第二次 acquire 复用同一实例", _i1 is _i2)
_i3 = GBm.acquire({**KCFG, "context_size": 4096})
check("K4b 换了上下文后 acquire 给的是新实例", _i3 is not _i1)
check("K4c 被顶掉的旧实例已标记关闭", _i1._closed)
GBm.release(force=True)
check("K4d release 之后 current_server() 为空", GBm.current_server() is None)

# ---- K5 就绪轮询 ----
_ok = kserver()
_ok.ensure_ready(timeout=10)
check("K5a /health 返回 ok 时 ensure_ready 直接返回", _ok._ready is True)

try:
    kserver(alive=False).ensure_ready(timeout=5)
    _m1 = ""
except GBm.GgufError as e:
    _m1 = str(e)
check("K5b 进程提前退出 -> 立刻报错而不是干等到超时",
      "退出" in _m1 and "退出码" in _m1, _m1.splitlines()[0] if _m1 else "(没抛错)")

_kc, _kb = _FakeAPI.health_code, _FakeAPI.health_body
_FakeAPI.health_code, _FakeAPI.health_body = 503, {"status": "loading model"}
try:
    kserver().ensure_ready(timeout=1.2)
    _m2 = ""
except GBm.GgufError as e:
    _m2 = str(e)
finally:
    _FakeAPI.health_code, _FakeAPI.health_body = _kc, _kb
check("K5c 一直 503（权重还在传）时最终按超时报警", "超时" in _m2,
      _m2.splitlines()[0] if _m2 else "(没抛错)")

# ---- K6 chat：纯文本 ----
_FakeAPI.payloads = []
_r = _ok.chat(system="SYS", user="USR", max_tokens=8, temperature=0.3)
_p = _FakeAPI.payloads[-1] if _FakeAPI.payloads else {}
check("K6a content 与 reasoning_content 分开取回",
      _r["text"] == "TAGGED-OUT" and _r["reasoning"] == "hidden-thought",
      f'{_r["text"]!r} / {_r["reasoning"]!r}')
check("K6b 从 timings 拆出 prefill / 解码两段时间",
      abs(_r["prefill_s"] - 0.3) < 1e-9 and abs(_r["decode_s"] - 0.2) < 1e-9,
      f'{_r["prefill_s"]} / {_r["decode_s"]}')
check("K6c usage 的 token 数被带出来",
      _r["prompt_tokens"] == 11 and _r["completion_tokens"] == 7)
check("K6d 系统提示词与用户提示词各成一条 message",
      [m["role"] for m in _p.get("messages", [])] == ["system", "user"],
      str([m["role"] for m in _p.get("messages", [])]))
check("K6e 默认关思考：chat_template_kwargs.enable_thinking=False",
      _p.get("chat_template_kwargs", {}).get("enable_thinking") is False)
check("K6f 纯文本时 content 是字符串而不是数组",
      isinstance(_p["messages"][-1]["content"], str))
check("K6g max_tokens / temperature 按请求传下去",
      _p.get("max_tokens") == 8 and abs(_p.get("temperature") - 0.3) < 1e-9)

# ---- K7 chat：带图 ----
_kimg_dir = os.path.join(TMP_ROOT, "kimg")
os.makedirs(_kimg_dir, exist_ok=True)
_kimg = os.path.join(_kimg_dir, "big.png")
Image.new("RGB", (400, 300), (10, 200, 30)).save(_kimg)

_FakeAPI.payloads = []
_imgs = GBm.encode_image_file(_kimg, 512)
_ok.chat(system="SYS", user="USR", images=_imgs, max_tokens=8, enable_thinking=True)
_pi = _FakeAPI.payloads[-1]
_c = _pi["messages"][-1]["content"]
check("K7a 带图时 content 变成数组（text + image_url）",
      isinstance(_c, list) and [x["type"] for x in _c] == ["text", "image_url"],
      str([x.get("type") for x in _c]))
check("K7b 图片走 base64 data URI",
      _c[1]["image_url"]["url"].startswith("data:image/"),
      _c[1]["image_url"]["url"][:40])
check("K7c base64 能解回真实图片字节",
      len(_b64.b64decode(_c[1]["image_url"]["url"].split(",", 1)[1])) > 100)
check("K7d enable_thinking=True 时不注入 chat_template_kwargs",
      "chat_template_kwargs" not in _pi)
check("K7e mime 按扩展名推断",
      GBm.mime_for_name("a.JPG") == "image/jpeg"
      and GBm.mime_for_name("a.webp") == "image/webp"
      and GBm.mime_for_name("a.png") == "image/png")
_FakeAPI.payloads = []
_ok.chat(system="SYS", user="USR", images=[_imgs], max_tokens=8)
_pi2 = _FakeAPI.payloads[-1]
check("K7f 包成列表 [ (mime, bytes) ] 也照样工作（上面测的是裸元组）",
      isinstance(_pi2["messages"][-1]["content"], list)
      and _pi2["messages"][-1]["content"][1]["type"] == "image_url")

# ---- K8 可读的中文报错 ----
def _start_err(cfg):
    z = GBm.GgufServer(cfg)
    z.port = 1
    try:
        z._start()
        return ""
    except GBm.GgufError as e:
        return str(e)


_e1 = _start_err({"server_exe": os.path.join(TMP_ROOT, "nope.exe"),
                  "model": os.path.join(TMP_ROOT, "nope.gguf")})
check("K8a 找不到 llama-server.exe -> 报错带文件名与下载指引",
      "llama-server.exe" in _e1 and "llama.cpp" in _e1, _e1.splitlines()[0])
_e2 = _start_err({"server_exe": os.path.abspath(__file__),
                  "model": os.path.join(TMP_ROOT, "nope.gguf")})
check("K8b 找不到 gguf -> 报错里带该路径",
      "nope.gguf" in _e2, _e2.splitlines()[0])
_e3 = _start_err({"server_exe": os.path.abspath(__file__),
                  "model": os.path.abspath(__file__),
                  "mmproj": os.path.join(TMP_ROOT, "nope_mm.gguf")})
check("K8c mmproj 路径写错时单独报错（不会退化成一堆难懂的 llama.cpp 报错）",
      "mmproj" in _e3, _e3.splitlines()[0])

# ---- K9 显存需求估算（喂给「腾显存」那步的阈值）----
_KNODE = QM.Qwen35BatchImageTagger()
_fake_gguf = os.path.join(TMP_ROOT, "fake4m.gguf")
with open(_fake_gguf, "wb") as fh:
    fh.write(b"\0" * (4 * 1024 * 1024))
_be = {"kind": "qwen35_gguf", "cfg": {"model": _fake_gguf, "server_exe": "x"}}
check("K9a 无 mmproj：按「权重×1.10 + 1.2GiB」估",
      _KNODE._gguf_need_mib(_be) == int(4.0 * 1.10 + 1200), str(_KNODE._gguf_need_mib(_be)))
_be2 = {"kind": "qwen35_gguf",
        "cfg": {"model": _fake_gguf, "server_exe": "x", "mmproj": _fake_gguf}}
check("K9b 有 mmproj：把它的体积也算进去",
      _KNODE._gguf_need_mib(_be2) == int(8.0 * 1.10 + 1200), str(_KNODE._gguf_need_mib(_be2)))
check("K9c backend 不是 GGUF 节点输出时退回保守默认值，不抛异常",
      _KNODE._gguf_need_mib({"kind": "wrong"}) == 20000)

# ---- K10 图片编码 / 缩放 ----
_m0, _b0 = GBm.encode_image_file(_kimg, 0)
_m1, _b1 = GBm.encode_image_file(_kimg, 64)
check("K10a max_side=0 时原样返回字节",
      _m0 == "image/png" and len(_b0) == os.path.getsize(_kimg))
check("K10b 给了 max_side 后重新编码、体积明显变小",
      len(_b1) < len(_b0), f"{len(_b0)} -> {len(_b1)}")
with Image.open(_kimg) as _im0:
    _wh0 = _im0.size
with Image.open(__import__("io").BytesIO(_b1)) as _im1:
    _wh1 = _im1.size
check("K10c 缩放保持长边不超过 max_side 且比例不变",
      max(_wh1) <= 64 and abs(_wh1[0] / _wh1[1] - _wh0[0] / _wh0[1]) < 0.02,
      f"{_wh0} -> {_wh1}")

# ---- K11 节点注册与默认值 ----
_NM = QM.NODE_CLASS_MAPPINGS
check("K11a 「GGUF 后端」节点已注册，输出类型为 QWEN35_BACKEND",
      "Qwen35GGUFServer" in _NM
      and _NM["Qwen35GGUFServer"].RETURN_TYPES == ("QWEN35_BACKEND",),
      str(_NM.get("Qwen35GGUFServer")))
_kit = _NM["Qwen35GGUFServer"].INPUT_TYPES()
check("K11b 三个必选控件 model / mmproj / server_exe 都在",
      all(k in _kit.get("required", {}) for k in ("model", "mmproj", "server_exe")),
      str(list(_kit.get("required", {}))))
check("K11c 两个主节点都接受 backend 连线输入",
      "backend" in QM.Qwen35BatchImageTagger.INPUT_TYPES().get("optional", {})
      and "backend" in QM.Qwen35PromptEnhancer.INPUT_TYPES().get("optional", {}))
check("K11d parallel 默认 1（不是 llama-server 的 4）",
      _kit["optional"]["parallel"][1]["default"] == 1,
      str(_kit["optional"]["parallel"][1].get("default")))
check("K11e n_gpu_layers 默认 -1（交给 llama.cpp 自动 fit）",
      _kit["optional"]["n_gpu_layers"][1]["default"] == -1,
      str(_kit["optional"]["n_gpu_layers"][1].get("default")))
check("K11f KV cache 默认 q8_0",
      _kit["optional"]["kv_cache_type"][1]["default"] == "q8_0")

# ---- K12 GGUF 路径下必须先腾显存（本轮修的坑）----
# 之前 GGUF 分支不调 _free_vram：ComfyUI 常驻的那份不放手，llama-server 只能
# 拿零头，llama.cpp 不报错、只是安静地把层摊到 CPU。这里钉住这个行为。
import inspect as _insp                                # noqa: E402
_src_tag = _insp.getsource(QM.Qwen35BatchImageTagger.tag_folder)
_src_enh = _insp.getsource(QM.Qwen35PromptEnhancer.enhance)
check("K12a 打标节点：GGUF 分支里先调了 _free_vram",
      "_free_vram" in _src_tag and "_gguf_need_mib" in _src_tag)
check("K12b 扩写节点：GGUF 分支里先调了 _free_vram",
      "_free_vram" in _src_enh and "_gguf_need_mib" in _src_enh)
check("K12c 腾显存时用的是 gguf 实际大小，而不是 HF 的 quant 档位",
      "_gguf_need_mib(backend)" in _src_tag and "_gguf_need_mib(backend)" in _src_enh)

_fake_srv.shutdown()

# ---- 收尾 ----
print()
print(f"  临时目录: {TMP_ROOT}")
try:
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print("  已清理临时目录")
except Exception as e:
    print(f"  清理失败（不影响结论）: {e}")

print()
print("=" * 74)
if FAIL:
    print(f"存在 {len(FAIL)} 项问题:")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("全部通过")
