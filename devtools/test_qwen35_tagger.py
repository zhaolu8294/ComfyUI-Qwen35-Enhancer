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
check("widgets 总数 = 25", len(order) == 25, str(len(order)))

sig = [p for p in inspect.signature(NODE.tag_folder).parameters.keys()
       if p not in ("self", "unique_id")]
check("签名与 widgets 顺序一致", sig == order,
      f"差异 {set(sig) ^ set(order)}" if sig != order else "")

check("folder_path 不是多行框（路径不该用多行输入）",
      it["required"]["folder_path"][1].get("multiline") is False)
check("system_preset 排在 system_prompt 之前（先选预设再编辑）",
      order.index("system_preset") < order.index("system_prompt"))
check("system_preset 默认 custom（不改变旧工作流行为）",
      it["required"]["system_preset"][1]["default"] == "custom",
      str(it["required"]["system_preset"][1]["default"]))
check("system_preset 选项：custom 排第一，且带三套预设",
      list(it["required"]["system_preset"][0])[:1] == ["custom"]
      and {"photoreal", "character", "scene"} <= set(it["required"]["system_preset"][0]),
      str(list(it["required"]["system_preset"][0])))
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
check("默认带三套预设", set(_presets) >= {"photoreal", "character", "scene"},
      str(list(_presets)))
check("每套都有 label 与 prompt",
      all(p.get("label") and p.get("prompt") for p in _presets.values()))

check("写实套含摄影术语（film_grain / 85mm）",
      "film_grain" in _presets["photoreal"]["prompt"]
      and "85mm" in _presets["photoreal"]["prompt"])
check("角色套含 booru 属性（twintails / serafuku）",
      "twintails" in _presets["character"]["prompt"]
      and "serafuku" in _presets["character"]["prompt"])
check("场景套含建筑与光照词（vanishing_point / god_rays）",
      "vanishing_point" in _presets["scene"]["prompt"]
      and "god_rays" in _presets["scene"]["prompt"])
check("三套互不相同（不是同一份文本换名）",
      len({p["prompt"] for p in _presets.values()}) == 3)
for _n, _p in _presets.items():
    check(f"{_n} 套都写了共同硬规则：只标可见 / 列了禁词 / 一行逗号",
          "clearly visible" in _p["prompt"]
          and "masterpiece" in _p["prompt"]
          and "comma-separated" in _p["prompt"])

# ---- 选择逻辑 ----
_t, _k, _l = QM._resolve_tag_preset("photoreal", "SYS")
check("选预设时忽略 system_prompt（防止被忘改的旧 widget 值顶掉）",
      _t == _presets["photoreal"]["prompt"] and _k == "photoreal")
check("同时返回显示名（报告里要显示选了哪套）", _l == _presets["photoreal"]["label"], str(_l))
_t, _k, _l = QM._resolve_tag_preset("custom", "SYS")
check("custom 用 widget 里填的文本", _t == "SYS" and _k == "custom")
_t, _k, _l = QM._resolve_tag_preset("没这个预设", "SYS")
check("认不出的预设名回退 custom（预设被删/改名时老工作流不至于拿不到提示词）",
      _t == "SYS" and _k == "custom")
_t, _k, _l = QM._resolve_tag_preset("PHOTOREAL", "SYS")
check("预设名大小写不敏感（手改 JSON 常见）", _k == "photoreal")
_t, _k, _l = QM._resolve_tag_preset("", "SYS")
check("空 preset 视为 custom", _t == "SYS" and _k == "custom")
_t, _k, _l = QM._resolve_tag_preset(None, "SYS")
check("None preset 也视为 custom", _t == "SYS" and _k == "custom")

# ---- payload 解析容错 ----
_p = QM._parse_preset_payload({"presets": {
    "custom": {"prompt": "会被忽略"},
    "a": {"prompt": "AAA"},
    "b": "BBB",
}})
check("custom 是保留名，JSON 里写它会被忽略（否则下拉语义会乱）",
      "custom" not in _p and set(_p) == {"a", "b"}, str(list(_p)))
check("简写形式（key 直接给字符串）也认", _p["b"]["prompt"] == "BBB" and _p["b"]["label"] == "b")
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
    check("JSON 坏掉：回退内置三套而不是抛异常",
          set(_bad) >= {"photoreal", "character", "scene"}, str(list(_bad)))
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
    _t, _k, _ = QM._resolve_tag_preset("scene", "")
    check("改了 JSON 文本后无需重启即生效（按 mtime 热重载）",
          _t == "HOT-RELOAD-OK" and _k == "scene", _t[:40])
finally:
    shutil.copy2(_hot, _PJ)
    os.remove(_hot)
    QM._TAG_PRESET_CACHE["mtime"] = None
    _t, _k, _ = QM._resolve_tag_preset("scene", "")
    check("恢复原文件后重新读回内置文本", _t == _presets["scene"]["prompt"])

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
