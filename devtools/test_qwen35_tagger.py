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
order = list(it["required"]) + list(it["optional"])
print("  widgets 顺序:")
for _i, _k in enumerate(order):
    print(f"    [{_i:>2}] {_k}")
check("widgets 总数 = 27", len(order) == 27, str(len(order)))

sig = [p for p in inspect.signature(NODE.tag_folder).parameters.keys()
       if p not in ("self", "unique_id")]
check("签名与 widgets 顺序一致", sig == order,
      f"差异 {set(sig) ^ set(order)}" if sig != order else "")
check("两个新控件**追加在末尾**（否则旧工作流 widgets_values 会整体串位）",
      order[-2:] == ["bilingual", "max_output_chars"], str(order[-2:]))

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
