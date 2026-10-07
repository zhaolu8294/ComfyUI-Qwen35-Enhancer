# -*- coding: utf-8 -*-
"""llama.cpp / GGUF 后端：用 GGUF 量化模型给两个主节点提供推理能力。

为什么走**外部进程 + HTTP**而不是别的路子：
  · `llama-cpp-python` 官方只发 CPU-only 的 win wheel，CUDA 版要自己编译；
    Python 3.13 + 新版 llama.cpp 编起来坑多，且升级二进制要重编整个包。
  · `ComfyUI-GGUF` 插件是给扩散模型写的 —— 它把 GGUF **反量化回浮点**再喂 torch。
    对 9B 无所谓，对 27B 就等于绕一圈回到 55GB 显存，GGUF 的意义全没了。
  · `transformers` 的 GGUF 支持同理，也是反量化路线。
  · llama.cpp 官方有 **Windows CUDA 预编译包**（含 cudart），解压即用、不用编译，
    llama-server 崩了也拖不垮 ComfyUI，还能单独开它的网页界面调参。

这个模块只做四件事：
  1. 找到 `llama-server.exe`（多候选路径，含环境变量）
  2. 扫描可用的 `.gguf`（主干与 mmproj 分开列）
  3. 按参数签名**单例**管理 server 子进程（签名变了就重建）
  4. 提供一个 `chat()`：纯文本与 base64 图片两种输入形态

关于显存：Qwen3.8-27B 是 27.78B dense，BF16 权重 55.56GB，24GB 卡**纯 GPU 装不下**，
必须用量化 GGUF。Q4_K_M 约 16.8GB + mmproj 0.9GB + KV + 缓冲 ≈ 19.7GB / 24GB。

关于那个 `--parallel 1`：Qwen3.8 有 48 层线性注意力（Gated DeltaNet），每层都要按
**序列槽**存 recurrent state。llama-server 的默认值不是 1 而是 4（`common/arg.cpp`
里 server 示例把 n_parallel 覆盖成 -1，再由 server.cpp 解析成 4），槽位一多这块就
凭空多吃 0.44 GiB。24GB 卡上没有富余，所以默认钉死 1。
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time

logger = logging.getLogger("Qwen35Enhancer")

try:
    import requests
except Exception:                                    # pragma: no cover
    requests = None


# ---------------------------------------------------------------------------
# llama-server 可执行文件的发现
# ---------------------------------------------------------------------------
def _node_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _models_dir():
    """ComfyUI 的 models 根目录；拿不到就按节点位置往上推三层。"""
    try:
        import folder_paths
        d = getattr(folder_paths, "models_dir", None)
        if d:
            return d
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.dirname(_node_dir())), "models")


# 解压 llama.cpp 的候选根目录。本机把二进制放在 H 盘（E 盘只剩几个 GB）。
_BINARY_ROOTS = (
    r"H:\AI\llama.cpp",
    r"F:\AI\llama.cpp",
    r"J:\AI\llama.cpp",
    r"G:\AI\llama.cpp",
    r"D:\AI\llama.cpp",
)


def _walk_for_server(root, max_depth=3):
    """在 root 下有限深度地找 llama-server.exe。

    官方 zip 解压出来一般是 `llama-b11461-bin-win-cuda-13.4-x64\\llama-server.exe`，
    用户也可能自己多套一层目录，所以这里不写死层级。
    """
    if not os.path.isdir(root):
        return []
    hits = []
    root = os.path.abspath(root)
    base_depth = root.rstrip("\\/").count(os.sep)
    for cur, dirs, files in os.walk(root):
        if cur.rstrip("\\/").count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
        for f in files:
            if f.lower() == "llama-server.exe":
                hits.append(os.path.join(cur, f))
    return hits


def find_llama_server(explicit=""):
    """按优先级找 llama-server.exe，找不到返回 ""（空串而不是 None，便于直接拼日志）。"""
    # 1) 用户显式指定（控件里填的）
    p = str(explicit or "").strip().strip('"')
    if p and os.path.isfile(p):
        return p
    # 2) 环境变量
    for key in ("QWEN35_LLAMA_SERVER", "LLAMA_SERVER_PATH"):
        v = os.environ.get(key)
        if v:
            v = v.strip().strip('"')
            if os.path.isfile(v):
                return v
            for hit in _walk_for_server(v):
                return hit
    # 3) 节点自带的 runtime 目录（若用户选择装在插件目录下）
    here = _node_dir()
    for sub in ("runtime", os.path.join("runtime", "bin")):
        cand = os.path.join(here, sub, "llama-server.exe")
        if os.path.isfile(cand):
            return cand
        for hit in _walk_for_server(os.path.join(here, sub), max_depth=2):
            return hit
    # 4) 约定根目录（H/F/J/G/D 盘）
    for root in _BINARY_ROOTS:
        for hit in sorted(_walk_for_server(root), reverse=True):
            return hit
    # 5) PATH 里有没有
    for name in ("llama-server", "llama-server.exe"):
        w = shutil.which(name)
        if w:
            return w
    return ""


# llama-server 候选列表的展示形式：``<短标签> -> <绝对路径>``。
# ComfyUI 的 COMBO 只存字符串，所以把路径编码进标签里，取用时再切出来。
_AUTO_SERVER = "auto (自动查找)"


def llama_server_choices():
    """给控件用的候选列表。第一项是自动查找。"""
    seen, out = set(), [_AUTO_SERVER]
    for root in _BINARY_ROOTS:
        for hit in sorted(_walk_for_server(root), reverse=True):
            if hit not in seen:
                seen.add(hit)
                out.append(hit)
    here = _node_dir()
    for hit in _walk_for_server(os.path.join(here, "runtime"), max_depth=3):
        if hit not in seen:
            seen.add(hit)
            out.append(hit)
    w = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if w and w not in seen:
        out.append(w)
    return out


def server_from_choice(choice):
    """把控件值翻成真实路径。自动项就交给 find_llama_server。"""
    c = str(choice or "").strip().strip('"')
    if c == _AUTO_SERVER or not c:
        return find_llama_server()
    if os.path.isfile(c):
        return c
    return find_llama_server(c)


# ---------------------------------------------------------------------------
# GGUF 文件扫描
# ---------------------------------------------------------------------------
def gguf_search_dirs():
    """找 .gguf 的目录列表，按优先级排。

    约定根目录是 `H:\\AI\\llama.cpp`，那么它同级/上一级的常见落点都扫一遍：
      · `H:\\AI\\LLM`          —— 直接平铺（本机实际用的就是这个）
      · `H:\\AI\\models\\LLM`  —— 早期规划的落点，保留兼容
    ComfyUI 自己的 `models/LLM` 也一并扫，方便以后把 gguf 放进去用 ComfyUI 统一管理。
    """
    dirs = []
    env = os.environ.get("QWEN35_GGUF_DIR")
    if env:
        for part in env.split(os.pathsep):
            part = part.strip().strip('"')
            if part:
                dirs.append(part)
    dirs.append(os.path.join(_models_dir(), "LLM"))
    for root in _BINARY_ROOTS:
        ai_root = os.path.dirname(root)               # H:\AI\llama.cpp -> H:\AI
        dirs.append(os.path.join(ai_root, "LLM"))     # H:\AI\LLM
        dirs.append(os.path.join(ai_root, "models", "LLM"))
    out, seen = [], set()
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def _is_mmproj(name):
    n = name.lower()
    return "mmproj" in n or n.startswith("clip") or "vision" in n


def list_gguf(include_subdirs=True, max_depth=2):
    """扫 GGUF。返回 (主干模型字典, mmproj 字典)，键是给人看的标签、值是绝对路径。

    只往下走 max_depth 层：模型的目录常常和别的东西混在一起（本机 `H:\\AI\\LLM`
    里就还躺着一整份 LM Studio 安装，它自带的 bundled-models 会被无深度限制的
    os.walk 一路挖出来，污染候选列表）。两层足够覆盖「直接平铺」和
    「按仓库/量化档位再分一层」这两种常见摆法。
    """
    mains, projs = {}, {}
    for base in gguf_search_dirs():
        if not os.path.isdir(base):
            continue
        if include_subdirs:
            root = os.path.abspath(base)
            base_depth = root.rstrip("\\/").count(os.sep)
            walker = os.walk(root)
            collected = []
            for cur, dirs, files in walker:
                if cur.rstrip("\\/").count(os.sep) - base_depth >= max_depth:
                    dirs[:] = []
                collected.append((cur, files))
        else:
            collected = [(base, os.listdir(base))]
        for cur, files in collected:
            for f in sorted(files):
                if not f.lower().endswith(".gguf"):
                    continue
                full = os.path.join(cur, f)
                try:
                    gib = os.path.getsize(full) / (1024 ** 3)
                except OSError:
                    continue
                rel = os.path.relpath(full, base).replace("\\", "/")
                label = f"{rel}  ({gib:.2f} GB)"
                if _is_mmproj(f):
                    projs[label] = full
                else:
                    mains[label] = full
    return mains, projs


_NONE_MMPROJ = "（无 / 纯文本）"


def gguf_model_choices():
    mains, _ = list_gguf()
    if not mains:
        return ["<no .gguf found — put one in models/LLM/>"]
    return list(mains.keys())


def gguf_mmproj_choices():
    _, projs = list_gguf()
    return [_NONE_MMPROJ] + list(projs.keys())


def path_from_label(label):
    """控件标签 -> 绝对路径。找不到就原样返回（用户可能直接填了路径）。"""
    s = str(label or "").strip().strip('"')
    if not s:
        return ""
    if os.path.isfile(s):
        return s
    if s == _NONE_MMPROJ:
        return ""
    mains, projs = list_gguf()
    for table in (mains, projs):
        if s in table:
            return table[s]
    # 标签里带了 "  (12.34 GB)" 尾巴，去掉再比一次
    stem = s.rsplit("  (", 1)[0].strip()
    for table in (mains, projs):
        for k, v in table.items():
            if k.rsplit("  (", 1)[0].strip() == stem:
                return v
    return s


# ---------------------------------------------------------------------------
# 下载：让节点自己把缺的文件取回来
# ---------------------------------------------------------------------------
# 为什么要有这段：Qwen3.8 要两个文件（主干 + mmproj），少下 mmproj 是个**静默失效**——
# llama-server 会高高兴兴加载成纯文本模型，打标时图片被直接丢掉，日志里看不出异常。
# 让节点在发现缺件时自己去拿，比让人记住去浏览器里点两下可靠。
#
# 用 hf-mirror.com（HuggingFace 国内镜像）而不是 huggingface.co：本机直连 HF 很慢，
# 镜像走 https 直连即可，不需要代理。
DEFAULT_MODEL_URL = (
    "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
    "/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf?download=true"
)
DEFAULT_MMPROJ_URL = (
    "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
    "/resolve/main/mmproj-model-bf16.gguf?download=true"
)


def filename_from_url(url):
    """从下载地址里取出文件名。带 query（`?download=true`）和 URL 编码都能处理。"""
    from urllib.parse import unquote, urlparse
    p = urlparse(str(url or "").strip()).path
    return unquote(os.path.basename(p)).strip()


def dir_of(path):
    """决定「新下下来的文件放哪」：优先跟已有文件同目录，否则返回 ""（交给默认目录）。"""
    p = str(path or "").strip()
    if p and os.path.isfile(p):
        return os.path.dirname(p)
    return ""


def default_download_dir():
    """默认落地目录：第一个存在且可写的搜索目录；都不行就建 ComfyUI 的 models/LLM。"""
    for d in gguf_search_dirs():
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    d = os.path.join(_models_dir(), "LLM")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def find_existing(name, max_depth=2):
    """在搜索路径里找同名文件。用户可能早就手动下过了，别重复下。"""
    target = str(name or "").strip().lower()
    if not target:
        return ""
    for base in gguf_search_dirs():
        if not os.path.isdir(base):
            continue
        root = os.path.abspath(base)
        base_depth = root.rstrip("\\/").count(os.sep)
        for cur, dirs, files in os.walk(root):
            if cur.rstrip("\\/").count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
            for f in files:
                if f.lower() == target:
                    return os.path.join(cur, f)
    return ""


def remote_size(url, timeout=30):
    """问服务器这个文件多大。拿不到就返回 -1（表示「无法校验」，不是错误）。"""
    if requests is None:
        return -1
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        n = r.headers.get("Content-Length")
        if n:
            return int(n)
    except Exception:
        pass
    # 有些镜像不吃 HEAD，用「只取第 0 个字节」换 Content-Range 里的总长度
    try:
        r = requests.get(url, stream=True, allow_redirects=True,
                         headers={"Range": "bytes=0-0"}, timeout=timeout)
        try:
            cr = r.headers.get("Content-Range") or ""      # bytes 0-0/931145888
            if "/" in cr:
                return int(cr.rsplit("/", 1)[1])
            n = r.headers.get("Content-Length")
            if n and r.status_code == 200:
                return int(n)
        finally:
            r.close()
    except Exception:
        pass
    return -1


def download_file(url, dest, on_status=None, resume=True, timeout=(30, 600)):
    """把 url 下到 dest。返回 (最终路径, "already" / "downloaded")。

    几个刻意的设计：
      · **先写 dest.part 再原子改名** —— 中途断了不会留下一个「看起来下好了但其实是半截」
        的文件，那种半截 gguf 会被当成正常模型喂给 llama.cpp，报一堆难懂的错。
      · **断点续传**：重跑一次会拿 .part 已有的大小发 Range 请求接着下。
        16GB 的模型在慢线上一断就得重来，这个钱不能省。
      · **下完校验总大小**，对不上就报错并保留 .part，方便再续。
    """
    if requests is None:
        raise GgufError("缺少 requests 库，无法下载")
    url = str(url or "").strip()
    if not url:
        raise GgufError("下载地址是空的")
    dest = os.path.abspath(dest)
    part = dest + ".part"
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    except OSError as e:
        raise GgufError(f"建不了目录 {os.path.dirname(dest)}：{e}") from e

    total = remote_size(url)

    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        if total <= 0 or os.path.getsize(dest) == total:
            return dest, "already"

    done = os.path.getsize(part) if (resume and os.path.isfile(part)) else 0
    if not resume and os.path.isfile(part):
        try:
            os.remove(part)
        except OSError:
            pass
        done = 0
    if total > 0 and done >= total:
        os.replace(part, dest)
        return dest, "already"

    headers = {"Range": f"bytes={done}-"} if done > 0 else {}
    t0 = time.time()
    try:
        r = requests.get(url, stream=True, allow_redirects=True, headers=headers,
                         timeout=timeout)
    except Exception as e:
        raise GgufError(f"下载失败（连不上）：{e}\n{url}") from e

    try:
        if r.status_code == 416 and done > 0:
            # 服务器说范围越界 = 本地 .part 已经够了
            r.close()
            if total > 0 and os.path.getsize(part) == total:
                os.replace(part, dest)
                return dest, "already"
            raise GgufError(
                f"服务器拒绝了续传请求，但本地 .part（{os.path.getsize(part):,} 字节）"
                f"和远端的 {total:,} 字节对不上。删掉\n  {part}\n再试一次。"
            )
        if r.status_code not in (200, 206):
            raise GgufError(f"下载失败：HTTP {r.status_code}\n{url}")
        # 200 = 服务器不认 Range，只能从头来
        if r.status_code == 200 and done > 0:
            done = 0
        mode = "ab" if (done > 0 and r.status_code == 206) else "wb"
        with open(part, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if on_status is not None:
                    try:
                        on_status(done, total, time.time() - t0)
                    except Exception:
                        pass
    finally:
        try:
            r.close()
        except Exception:
            pass

    size = os.path.getsize(part)
    if total > 0 and size != total:
        raise GgufError(
            f"下载不完整：拿到 {size:,} 字节，应该是 {total:,} 字节"
            f"（差 {total - size:,}）。\n再跑一次会从断点接着下，已下的部分不会白费。"
        )
    os.replace(part, dest)
    return dest, "downloaded"


def ensure_file(url, dest_dir="", on_status=None, resume=True):
    """确保 url 指向的文件在本地有一份，返回它的绝对路径。

    顺序：① 搜索路径里已有同名文件 -> 直接用；② 否则下到 dest_dir。
    """
    url = str(url or "").strip()
    if not url:
        return ""
    name = filename_from_url(url)
    if not name:
        raise GgufError(f"这个地址里看不出文件名，不知道存成什么好：\n{url}")
    hit = find_existing(name)
    if hit:
        logger.info("[Qwen35] 已有同名文件，跳过下载：%s", hit)
        return hit
    dest = os.path.join(dest_dir or default_download_dir(), name)
    path, how = download_file(url, dest, on_status=on_status, resume=resume)
    logger.info("[Qwen35] %s -> %s", "已存在" if how == "already" else "下载完成", path)
    return path


# ---------------------------------------------------------------------------
# 空闲端口
# ---------------------------------------------------------------------------
def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# 单例进程管理
# ---------------------------------------------------------------------------
class GgufError(RuntimeError):
    """GGUF 后端的问题，消息是给最终用户看的中文。"""


def _fmt_args_for_log(args):
    out = []
    for a in args:
        s = str(a)
        if len(s) > 90 and os.sep in s:
            s = os.path.basename(s)
        out.append(s)
    return " ".join(out)


class GgufServer:
    """一个 llama-server 子进程 + 它的 HTTP 客户端。

    生命周期由模块级的 `acquire()` 管：参数签名相同就复用同一个实例，
    签名变了（换模型 / 换上下文 / 换量化等）就把旧的收掉再起新的。
    24GB 显存塞不下两个 27B，所以这里**全局只保留一个** server。
    """

    #: 进程就绪最长等待。首次要 mmap 16.8GB 权重并上传到显存，慢盘上两分钟很正常。
    READY_TIMEOUT = 600.0

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.port = 0
        self.proc = None
        self._log_tail = []
        self._log_lock = threading.Lock()
        self._ready = False
        self._closed = False
        self._started_at = 0.0

    # -- 参数 ---------------------------------------------------------------
    def build_args(self):
        c = self.cfg
        args = [c["server_exe"], "-m", c["model"], "--host", "127.0.0.1",
                "--port", str(self.port),
                "--parallel", str(int(c.get("parallel", 1) or 1))]
        if c.get("mmproj"):
            args += ["--mmproj", c["mmproj"]]
        # n_gpu_layers < 0 表示"不传 -ngl"，把层放置整个交给 llama.cpp 自己 fit。
        # 显式传 -ngl 会关掉它的自动 fit（新版会打印 n_gpu_layers already set by user），
        # 一旦估错就可能静默退回 CPU —— 那速度是掉一到两个数量级的，很难发现。
        ngl = int(c.get("n_gpu_layers", -1))
        if ngl >= 0:
            args += ["-ngl", str(ngl)]
        ctx = int(c.get("context_size", 8192) or 0)
        if ctx > 0:
            args += ["-c", str(ctx)]
        kv = str(c.get("kv_cache_type", "q8_0") or "").strip()
        if kv and kv != "f16":
            args += ["--cache-type-k", kv, "--cache-type-v", kv]
        if c.get("flash_attn", True):
            args += ["--flash-attn", "on"]
        # -b/-ub 影响 prefill 吞吐；打标是单图短上下文，2048/512 是稳的取值。
        args += ["-b", str(int(c.get("batch", 2048) or 2048)),
                 "-ub", str(int(c.get("ubatch", 512) or 512))]
        # --jinja 必须开：否则 server 不会用 GGUF 里存的 Qwen chat template，
        # 多模态的图片占位符也对不上，图片会被当成纯文本丢掉。
        args += ["--jinja"]
        # Qwen3.8 是混合思考模型。把 think 块引到独立的 reasoning_content 字段，
        # content 里就只剩最终答案 —— 节点侧那段 strip_thinking 就成了兜底而已。
        # 注意读一次存进变量：早先写成「外层 .get 默认值、内层直接下标」，
        # 少传这个键时外层判定通过、内层就 KeyError 了。
        rf = str(c.get("reasoning_format", "deepseek") or "").strip()
        if rf and rf != "none":
            args += ["--reasoning-format", rf]
        # 官方推荐关掉这两个；请求里若要覆盖会以请求为准。
        args += ["--min-p", "0.0", "--repeat-penalty", "1.0"]
        extra = str(c.get("extra_args", "") or "").strip()
        if extra:
            args += _split_extra_args(extra)
        return args

    # -- 启动 ---------------------------------------------------------------
    def _start(self):
        if not os.path.isfile(self.cfg["server_exe"]):
            raise GgufError(
                f"找不到 llama-server.exe：{self.cfg['server_exe']}\n"
                "请下载 llama.cpp 的 Windows CUDA 包解压到 H:\\AI\\llama.cpp\\ ，"
                "或在「GGUF 后端」节点里手动指定 exe 路径。"
            )
        if not os.path.isfile(self.cfg["model"]):
            raise GgufError(f"找不到 GGUF 模型：{self.cfg['model']}")
        if self.cfg.get("mmproj") and not os.path.isfile(self.cfg["mmproj"]):
            raise GgufError(f"找不到 mmproj：{self.cfg['mmproj']}")

        self.port = _free_port()
        args = self.build_args()
        logger.info("[Qwen35] 启动 llama-server：%s", _fmt_args_for_log(args))

        creationflags = 0
        if os.name == "nt":
            # 别弹出黑框；也不要让子进程跟着 ComfyUI 的控制台一起被 Ctrl+C 波及太快。
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            self.proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=os.path.dirname(self.cfg["server_exe"]) or None,
                creationflags=creationflags,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as e:
            raise GgufError(
                f"无法启动 llama-server：{e}\n"
                "常见原因：缺 cudart 的 dll（要额外解压 cudart-llama-bin-win-cuda-*.zip "
                "到同一目录）、或路径里有特殊字符。"
            ) from e
        self._started_at = time.time()
        threading.Thread(target=self._drain_log, daemon=True).start()

    def _drain_log(self):
        """把 server 的 stdout 收进环形缓冲；出错时这段就是最有用的线索。"""
        try:
            for line in self.proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                with self._log_lock:
                    self._log_tail.append(line)
                    if len(self._log_tail) > 120:
                        del self._log_tail[:-120]
        except Exception:
            pass

    def log_tail(self, n=25):
        with self._log_lock:
            return "\n".join(self._log_tail[-n:])

    # -- 就绪 ---------------------------------------------------------------
    def ensure_ready(self, on_status=None, timeout=None):
        """启动并等到 /health 返回 ok。on_status 是给进度条报话用的回调。"""
        if self._ready and self.proc is not None and self.proc.poll() is None:
            return
        if requests is None:
            raise GgufError("缺少 requests 库，无法和 llama-server 通信")
        self._start()
        deadline = time.time() + float(timeout or self.READY_TIMEOUT)
        url = f"http://127.0.0.1:{self.port}/health"
        t0 = time.time()
        warned = 0.0
        while time.time() < deadline:
            code = self.proc.poll()
            if code is not None:
                raise GgufError(
                    f"llama-server 启动后退出（退出码 {code}）。最近输出：\n{self.log_tail()}"
                )
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    body = {}
                    try:
                        body = r.json()
                    except Exception:
                        body = {}
                    if str(body.get("status", "ok")).lower() in ("ok", "ready"):
                        self._ready = True
                        logger.info("[Qwen35] llama-server 就绪，端口 %s，耗时 %.1fs",
                                    self.port, time.time() - t0)
                        return
                # 503 = 还在加载权重，继续等
            except Exception:
                pass
            el = time.time() - t0
            if on_status and el - warned >= 5.0:
                warned = el
                on_status(f"正在加载 GGUF 模型到显存…已等 {el:.0f}s"
                          f"（首次 mmap 十几 GB 权重，慢盘上两分钟正常）")
            time.sleep(0.5)
        raise GgufError(
            f"等待 llama-server 就绪超时（{timeout or self.READY_TIMEOUT:.0f}s）。"
            f"最近输出：\n{self.log_tail()}"
        )

    # -- 推理 ---------------------------------------------------------------
    def chat(self, system="", user="", images=None, max_tokens=512,
             temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
             presence_penalty=0.0, repeat_penalty=1.0, seed=None,
             enable_thinking=False, timeout=900):
        """一次对话补全。

        images 是 [(mime, bytes), ...]（打标是一张，扩写最多四张）。

        返回 {"text", "reasoning", "prompt_tokens", "completion_tokens",
              "seconds", "prefill_s", "decode_s"}。

        prefill_s / decode_s 取自 llama.cpp 在响应里附带的 `timings` —— 它不是
        OpenAI 规范的一部分，但 llama-server 一直会给。有了它，日志里那句
        「prefill x.xs、解码 y.ys = z tok/s」在 GGUF 后端下才有意义；拿不到就退回
        「全部算作解码时间」，至少 tok/s 仍然是对的。
        """
        if requests is None:
            raise GgufError("缺少 requests 库，无法和 llama-server 通信")
        msgs = []
        if str(system or "").strip():
            msgs.append({"role": "system", "content": str(system)})
        if images:
            # 容错：`encode_image_file()` 返回的就是**单个** (mime, bytes)，
            # 而这里要的是列表。调用方漏掉那层 [] 时，报出来的会是
            # "too many values to unpack" 这种完全指错方向的错，所以这里直接认了。
            if (isinstance(images, tuple) and len(images) == 2
                    and isinstance(images[1], (bytes, bytearray))):
                images = [images]
            content = [{"type": "text", "text": str(user or "")}]
            for mime, blob in images:
                b64 = base64.b64encode(blob).decode("ascii")
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"}})
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": "user", "content": str(user or "")})

        payload = {
            "model": "local",
            "messages": msgs,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "presence_penalty": float(presence_penalty),
            "repeat_penalty": float(repeat_penalty),
            "stream": False,
            # 打标时同一张图很少重复，但扩写节点会带多图，
            # cache_prompt 对「系统提示词不变」的场景是白赚的。
            "cache_prompt": True,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if not enable_thinking:
            # Qwen 的 chat template 认这个开关；不认的模板会忽略它，
            # 那就靠节点侧 strip_thinking 兜底，不会出脏数据。
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=(20, float(timeout)))
        except Exception as e:
            raise GgufError(
                f"请求 llama-server 失败：{e}\n（若刚才手动关过它的窗口，"
                f"把本节点的 keep_alive 打开或重跑一次即可）"
            ) from e
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            raise GgufError(
                f"llama-server 返回 HTTP {r.status_code}：{r.text[:500]}"
            )
        try:
            data = r.json()
        except Exception as e:
            raise GgufError(f"llama-server 返回的不是 JSON：{r.text[:300]}") from e

        choices = data.get("choices") or []
        if not choices:
            raise GgufError(f"llama-server 没返回任何候选结果：{str(data)[:300]}")
        msg = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        tim = data.get("timings") or {}
        try:
            pre = float(tim.get("prompt_ms") or 0.0) / 1000.0
            dec = float(tim.get("predicted_ms") or 0.0) / 1000.0
        except (TypeError, ValueError):
            pre = dec = 0.0
        if pre <= 0.0 and dec <= 0.0:
            dec = dt                                  # 拿不到分解，就全算解码
        return {
            "text": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning_content") or "").strip(),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "seconds": dt,
            "prefill_s": pre,
            "decode_s": dec,
        }

    # -- 收尾 ---------------------------------------------------------------
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def close(self, timeout=8.0):
        p = self.proc
        self._ready = False
        self.proc = None
        self._closed = True
        if p is None:
            return
        try:
            if p.poll() is None:
                p.terminate()                         # Windows 上是 TerminateProcess
                try:
                    p.wait(timeout=timeout)
                except Exception:
                    p.kill()
                    try:
                        p.wait(timeout=3)
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            try:
                if p.stdout:
                    p.stdout.close()
            except Exception:
                pass


def _split_extra_args(text):
    """把附加参数串切成 argv。支持引号，方便路径带空格。"""
    import shlex
    try:
        return shlex.split(text, posix=False)
    except Exception:
        return [t for t in text.split() if t]


# ---------------------------------------------------------------------------
# 签名单例
# ---------------------------------------------------------------------------
_SIG_KEYS = ("server_exe", "model", "mmproj", "n_gpu_layers", "context_size",
             "parallel", "kv_cache_type", "flash_attn", "batch", "ubatch",
             "reasoning_format", "extra_args")

_CURRENT = {"sig": None, "server": None}
_LOCK = threading.RLock()


def _signature(cfg):
    parts = []
    for k in _SIG_KEYS:
        v = cfg.get(k)
        if k in ("server_exe", "model", "mmproj") and v:
            # 路径只取绝对路径本身，不去 stat（文件可能还没下完）
            v = os.path.abspath(str(v))
        parts.append(f"{k}={v}")
    return "|".join(parts)


def acquire(cfg):
    """拿到一个 GgufServer。签名一样就复用，不一样就把旧的关掉重起。

    复用条件看的是 `_closed` 而**不是** `alive()`：同一张图里节点可能被
    acquire 两次，而第一次拿到的实例还没启动进程（`proc is None`），
    这时 `alive()` 是 False —— 按它判断就会被白白丢掉、再新建一个。
    真正「进程已经死了」的情况由 ensure_ready 兜住：它看到 poll() 非 None
    会自己重启，所以这里不需要提前判死。
    """
    sig = _signature(cfg)
    with _LOCK:
        cur = _CURRENT.get("server")
        if cur is not None and _CURRENT.get("sig") == sig and not cur._closed:
            return cur
        if cur is not None:
            logger.info("[Qwen35] GGUF 参数变了，关闭旧的 llama-server（端口 %s）", cur.port)
            cur.close()
        srv = GgufServer(cfg)
        _CURRENT["sig"] = sig
        _CURRENT["server"] = srv
        return srv


def release(force=False):
    """关掉当前 server。force=False 时留给调用方判断要不要真的关。"""
    with _LOCK:
        cur = _CURRENT.get("server")
        _CURRENT["sig"] = None
        _CURRENT["server"] = None
    if cur is not None:
        try:
            cur.close()
        except Exception:
            pass
    return cur is not None


def current_server():
    with _LOCK:
        return _CURRENT.get("server")


@atexit.register
def _cleanup_on_exit():
    """ComfyUI 退出（含被关窗口）时别留孤儿进程占着 19GB 显存。"""
    try:
        release(force=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 图片编码
# ---------------------------------------------------------------------------
def encode_image_bytes(blob, mime="image/png"):
    """给已经拿到的字节流包一层，方便统一走 chat(images=...)。"""
    return (mime, blob)


def mime_for_name(name):
    n = str(name or "").lower()
    if n.endswith(".jpg") or n.endswith(".jpeg"):
        return "image/jpeg"
    if n.endswith(".webp"):
        return "image/webp"
    if n.endswith(".bmp"):
        return "image/bmp"
    return "image/png"


def encode_image_file(path, max_side=0, quality=92):
    """读图片文件 -> (mime, bytes)。max_side>0 时先等比缩到长边不超过它。

    缩放用 PIL，因为 llama.cpp 的视觉塔会按固定 patch 切块：原图太大时视觉 token
    数暴涨，prefill 时间跟着涨，而打标根本不需要那么高的分辨率。
    """
    mime = mime_for_name(path)
    with open(path, "rb") as fh:
        blob = fh.read()
    if not max_side or int(max_side) <= 0:
        return (mime, blob)
    try:
        from PIL import Image
    except Exception:
        return (mime, blob)
    try:
        with Image.open(io.BytesIO(blob)) as im:
            w, h = im.size
            longest = max(w, h)
            if longest <= int(max_side):
                return (mime, blob)
            scale = float(max_side) / float(longest)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            im = im.resize((nw, nh), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=int(quality))
            return ("image/jpeg", buf.getvalue())
    except Exception:
        return (mime, blob)


def encode_pil_images(images):
    """PIL Image 列表 -> [(mime, bytes), ...]。

    扩写节点的参考图已经在上游被 _collect_images 缩过、转成 PIL 了，
    这里不再二次缩放（那边按 max_image_side 控过尺寸），只负责编码。
    """
    out = []
    for im in images or []:
        try:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            out.append(("image/png", buf.getvalue()))
        except Exception:
            continue
    return out


def encode_image_tensor(image):
    """ComfyUI 的 IMAGE tensor（B,H,W,C / float 0~1）-> [(mime, bytes), ...]。

    逐张转，因为扩写节点最多收 4 张图，而 llama-server 的 content 数组里
    每张图就是一项，顺序和视觉 token 的排布有关，不能合并。
    """
    out = []
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return out
    try:
        arr = image
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3:                             # 单张 HWC
            arr = arr[None, ...]
        for i in range(arr.shape[0]):
            a = np.clip(arr[i] * 255.0, 0, 255).astype("uint8")
            if a.shape[-1] == 4:
                a = a[..., :3]
            im = Image.fromarray(a)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            out.append(("image/png", buf.getvalue()))
    except Exception:
        return out
    return out
