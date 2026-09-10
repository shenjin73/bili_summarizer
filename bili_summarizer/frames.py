"""Video frame analysis: screenshot extraction, deduplication, and OCR.

Pipeline for a downloaded video file:

1. ``extract_frames``  — ffmpeg grabs one frame every *interval* seconds (default 10).
2. ``dedupe_frames``   — perceptual-hash (dHash) comparison drops frames that are
                         visually near-identical to the last kept frame (static
                         slides, repeated title cards, ...).
3. ``ocr_frames``      — each unique frame is sent to a vision-capable LLM
                         (OpenAI-compatible API, e.g. local qwen3-vl via LM Studio)
                         which extracts all on-screen text and describes charts /
                         icons.

The OCR step needs a vision model; configure it with env vars:

    VISION_API_KEY   (falls back to the local chain keys)
    VISION_MODEL     (falls back to the local chain, e.g. Qwen3.8-27B-4bit)
    VISION_BASE_URL  (falls back to LOCAL_BASE_URLS, e.g. oMLX + LM Studio)
    OCR_CONCURRENCY  (worker threads; default 2 for local services, 6 for cloud)

Fallback chain (resolve_vision_provider): the local vision chain
(oMLX → LM Studio, from LOCAL_BASE_URLS / LOCAL_* in order) → DeepSeek
vision model (deepseek-flash, when DEEPSEEK_API_KEY is set) →
skip OCR (summary proceeds on subtitles alone).
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration defaults (overridable via env vars)
# ---------------------------------------------------------------------------
DEFAULT_INTERVAL = int(os.environ.get("FRAME_INTERVAL", "10"))   # seconds between screenshots
DEFAULT_DEDUP_THRESHOLD = int(os.environ.get("FRAME_DEDUP_THRESHOLD", "10"))  # dHash Hamming distance
DEFAULT_QN = int(os.environ.get("FRAME_QN", "64"))              # 16=360p 32=480p 64=720p 80=1080p

OCR_PROMPT = (
    "你是视频画面分析助手。这是一张从视频中截取的截图。\n"
    "请完成两件事：\n"
    "1. 提取画面中所有可见的文字（标题、字幕条、数据标签、代码、按钮文字等），"
    "按出现顺序列出，保留原始格式和数字。\n"
    "2. 简要描述画面中的图表、图标或图形元素（如折线图走势、柱状图对比、流程图结构等），"
    "说明它们表达了什么信息。\n"
    "如果画面是纯人物/风景且没有文字和图表，只回复：（无文字信息）\n"
    "输出要简洁、忠实于画面，不要推测画面中没有的内容。"
)


# ---------------------------------------------------------------------------
# 1. Frame extraction (ffmpeg)
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def extract_frames(
    video_path: Path,
    out_dir: Path,
    interval: int = DEFAULT_INTERVAL,
) -> list[tuple[Path, float]]:
    """Extract one frame every *interval* seconds from the video.

    Returns a list of ``(frame_path, timestamp_seconds)`` in time order.
    Raises ``RuntimeError`` when ffmpeg is missing or fails.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，请先安装（brew install ffmpeg）")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale frames from a previous run so the glob below only sees this run's output.
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink(missing_ok=True)
    pattern = str(out_dir / "frame_%03d.jpg")

    # showinfo logs the pts_time of every output frame to stderr, in order.
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-i", str(video_path),
        "-vf", f"fps=1/{max(1, interval)},showinfo",
        "-q:v", "2",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败: {proc.stderr[-500:]}")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        return []

    # Parse pts_time values from showinfo output, in order.
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]

    result = []
    for i, path in enumerate(frames):
        t = times[i] if i < len(times) else i * interval + interval / 2
        result.append((path, t))
    return result


# ---------------------------------------------------------------------------
# 2. Deduplication (perceptual dHash, pure PIL — no numpy needed)
# ---------------------------------------------------------------------------
def _dhash(path: Path, size: int = 8) -> int | None:
    """Compute a difference hash (size*size bits) of an image.

    Returns ``None`` if the image cannot be read.
    """
    from PIL import Image

    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None

    px = img.load()
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (1 if px[x, y] < px[x + 1, y] else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedupe_frames(
    frames: list[tuple[Path, float]],
    threshold: int = DEFAULT_DEDUP_THRESHOLD,
) -> list[tuple[Path, float]]:
    """Drop frames that are near-identical to the last *kept* frame.

    A frame is kept when its dHash differs from the previous kept frame by
    more than *threshold* bits (0-64).  Static slides therefore collapse to a
    single frame; genuinely new content is kept.
    """
    kept: list[tuple[Path, float]] = []
    last_hash: int | None = None

    for path, t in frames:
        h = _dhash(path)
        if h is None:
            continue  # unreadable frame — skip silently
        if last_hash is not None and _hamming(last_hash, h) <= threshold:
            path.unlink(missing_ok=True)  # duplicate — remove from disk
            continue
        kept.append((path, t))
        last_hash = h

    return kept


# ---------------------------------------------------------------------------
# 3. OCR via vision LLM (OpenAI-compatible API)
# ---------------------------------------------------------------------------
def _vision_config() -> dict:
    """Resolve the vision model config from env vars.

    Falls back to the local model settings (LOCAL_*) so a single local
    vision-capable model (e.g. qwen3.8-27b in LM Studio) can serve both the
    summarization and OCR roles.
    """
    api_key = os.environ.get("VISION_API_KEY") or os.environ.get("LOCAL_API_KEY", "")
    model = os.environ.get("VISION_MODEL") or os.environ.get("LOCAL_MODEL", "")
    base_url = (
        os.environ.get("VISION_BASE_URL")
        or os.environ.get("LOCAL_BASE_URL", "")
    )
    if not (model and base_url):
        raise RuntimeError(
            "未配置视觉模型。请设置 VISION_MODEL + VISION_BASE_URL"
            "（或不设 VISION_*，复用 LOCAL_BASE_URLS / LOCAL_MODELS 本地链）"
        )
    is_loopback = any(h in base_url for h in ("127.0.0.1", "localhost", "::1"))
    return {
        # Label by host: a VISION_* override may point at a cloud endpoint.
        "provider": "local" if is_loopback else "custom",
        "api_key": api_key or "lm-studio",  # placeholder; local servers ignore it
        "model": model,
        "base_url": base_url,
    }


def resolve_vision_provider() -> dict:
    """Resolve the vision model with automatic fallback.

    Chain: ``VISION_*`` single endpoint (when set) → the local chain
    (LOCAL_BASE_URLS / LOCAL_* — e.g. oMLX → LM Studio, in priority order) →
    DeepSeek vision model (when ``DEEPSEEK_API_KEY`` is set) → raise
    ``RuntimeError`` (OCR skipped, summary proceeds on subtitles alone).

    Each candidate is health-checked before use so an offline service fails
    fast instead of one slow failure per frame.
    """
    from .summarizer import (
        PROVIDER_DEFAULTS,
        check_endpoint,
        local_candidates,
        reason_label,
    )

    if os.environ.get("VISION_MODEL", "") or os.environ.get("VISION_BASE_URL", ""):
        try:
            primary = [_vision_config()]
        except RuntimeError as e:
            print(f"   ⚠️  {e}")
            primary = []
    else:
        primary = local_candidates()

    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    deepseek = None
    if ds_key:
        ds_defaults = PROVIDER_DEFAULTS["deepseek"]
        deepseek = {
            "provider": "deepseek",
            "api_key": ds_key,
            # Deliberately the provider-default vision model, ignoring the
            # LLM_MODEL override (which in .env may name a text-only model).
            "model": ds_defaults["model"],
            "base_url": ds_defaults.get("base_url", ""),
        }

    reasons = []
    for c in primary:
        status = check_endpoint(c["base_url"], c["model"], api_key=c.get("api_key"))
        if status == "ok":
            print(f"   视觉模型: {c['provider']} / {c['model']}（{c['base_url']}）")
            return c
        reasons.append(f"{c['base_url']} / {c['model']} {reason_label(status)}")

    if deepseek is not None:
        ds_status = check_endpoint(
            deepseek["base_url"], deepseek["model"], api_key=deepseek["api_key"]
        )
        if ds_status == "ok":
            if primary:
                print(
                    f"⚠️ 本地视觉模型不可用（{'；'.join(reasons)}），"
                    f"OCR 自动切换到 DeepSeek（{deepseek['model']}）"
                )
            else:
                print(f"   视觉模型: deepseek / {deepseek['model']}（本地未配置）")
            return deepseek
        # Also report the cloud-side reason. Without it the error below blames
        # only the local endpoint, hiding a renamed/retired DeepSeek model —
        # which silently disabled the OCR fallback and cost real debugging time.
        reasons.append(
            f"{deepseek['base_url']} / {deepseek['model']} {reason_label(ds_status)}"
        )

    if primary:
        raise RuntimeError(
            f"视觉模型不可用（{'；'.join(reasons)}），跳过 OCR（总结仅基于字幕）"
        )
    raise RuntimeError(
        "未配置可用的视觉模型（LOCAL_* / VISION_* 与 DEEPSEEK_API_KEY），跳过 OCR"
    )


def _ocr_one_frame(path: Path, cfg: dict) -> str:
    """Run OCR + chart description on a single frame image."""
    import base64

    from openai import OpenAI

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        suffix, "image/jpeg"
    )
    b64 = base64.b64encode(path.read_bytes()).decode()

    response = client.chat.completions.create(
        model=cfg["model"],
        max_tokens=1024,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _ocr_concurrency(cfg: dict) -> int:
    """Pick the OCR worker count.

    ``OCR_CONCURRENCY`` overrides everything. Otherwise: local services
    (e.g. LM Studio) serialise requests internally, so keep it low (2);
    cloud APIs tolerate more parallelism (6).
    """
    env = os.environ.get("OCR_CONCURRENCY", "")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    base_url = cfg.get("base_url", "")
    # Judge by host, not provider label: a VISION_* override pointing at a
    # cloud endpoint must get the cloud worker count.
    is_local = any(h in base_url for h in ("127.0.0.1", "localhost", "::1"))
    return 2 if is_local else 6


def _next_vision_provider(current: dict) -> dict | None:
    """Next health-checked vision candidate after *current*, or ``None``.

    Used for mid-batch failover: the remaining local chain endpoints (in
    priority order, skipping *current*), then the DeepSeek vision model.
    """
    from .summarizer import PROVIDER_DEFAULTS, check_endpoint, local_candidates

    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    ds = None
    if ds_key:
        ds_defaults = PROVIDER_DEFAULTS["deepseek"]
        ds = {
            "provider": "deepseek",
            "api_key": ds_key,
            # Provider-default vision model, ignoring any LLM_MODEL override.
            "model": ds_defaults["model"],
            "base_url": ds_defaults.get("base_url", ""),
        }

    for c in local_candidates():
        if c["base_url"] == current.get("base_url"):
            continue
        if check_endpoint(c["base_url"], c["model"], api_key=c.get("api_key")) == "ok":
            return c
    if ds is not None and check_endpoint(
        ds["base_url"], ds["model"], api_key=ds["api_key"]
    ) == "ok":
        return ds
    return None


def ocr_frames(
    frames: list[tuple[Path, float]],
    cfg: dict | None = None,
) -> list[dict]:
    """OCR every frame; returns ``[{timestamp, file, text}, ...]`` in time order.

    Frames are processed concurrently (ThreadPoolExecutor; worker count from
    ``OCR_CONCURRENCY``, default 2 for local services / 6 for cloud). Results
    stay in time order regardless of completion order. Individual failures are
    recorded as an error note instead of aborting the whole batch.

    Mid-batch failover: when a connection-level failure (the provider died)
    hits MORE than half of the frames, the failed frames are retried once on
    the next available vision provider (``_next_vision_provider``).
    """
    if cfg is None:
        # Health-check + automatic fallback (local chain → DeepSeek vision).
        # Raises RuntimeError when no vision service is available, so the
        # caller can skip OCR instead of one slow failure per frame.
        cfg = resolve_vision_provider()

    from concurrent.futures import ThreadPoolExecutor

    def _work(item: tuple[int, tuple[Path, float]], cfg_: dict | None = None) -> dict:
        use_cfg = cfg_ or cfg
        _, (path, t) = item
        mm, ss = divmod(int(t), 60)
        stamp = f"{mm:02d}:{ss:02d}"
        try:
            text = _ocr_one_frame(path, use_cfg)
            return {"timestamp": stamp, "file": path.name, "text": text}
        except Exception as e:
            from .summarizer import _is_connection_error

            return {
                "timestamp": stamp,
                "file": path.name,
                "text": f"（OCR 失败：{type(e).__name__}: {str(e)[:120]}）",
                "_conn_err": _is_connection_error(e),
            }

    workers = min(_ocr_concurrency(cfg), len(frames)) or 1
    print(f"   OCR 并发: {workers} 线程（OCR_CONCURRENCY 可调）")
    results: list[dict | None] = [None] * len(frames)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(_work, enumerate(frames))):
            results[i] = res

    # Mid-batch failover: provider died → retry the failed frames on the next
    # available vision provider. Run at most once; if the alternate also
    # fails, the frames keep their error notes.
    conn_failed = [i for i, r in enumerate(results) if r and r.get("_conn_err")]
    if conn_failed and len(conn_failed) * 2 > len(frames):
        alt = _next_vision_provider(cfg)
        if alt is not None:
            print(
                f"⚠️ OCR 中途 {cfg['base_url']} 不可用（{len(conn_failed)}/{len(frames)} "
                f"帧连接失败），切换到 {alt['base_url']} / {alt['model']} 重跑失败帧"
            )
            workers = min(_ocr_concurrency(alt), len(conn_failed)) or 1
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i, res in zip(
                    conn_failed,
                    pool.map(lambda j: _work((j, frames[j]), alt), conn_failed),
                ):
                    results[i] = res

    for r in results:
        if r:
            r.pop("_conn_err", None)
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 3b. Cross-frame dedup — compress OCR text before feeding it to the LLM
# ---------------------------------------------------------------------------
def compress_ocr_results(
    results: list[dict],
    similarity: float = 0.85,
) -> list[dict]:
    """Drop redundant frames/lines from OCR results to shrink the summary prompt.

    Rules, applied in time order:

    1. Frames with no information (``（无文字信息）`` / OCR failure notes) are
       dropped.
    2. Within each frame, only *new* lines survive: a line already seen
       verbatim in an earlier kept frame is dropped, as is a line that is a
       near-duplicate (SequenceMatcher ratio >= *similarity*) of any line in
       the previously kept frame (absorbs OCR noise between adjacent frames).
    3. A frame with no surviving lines is dropped entirely.

    The full, uncompressed results are still saved to disk by the caller —
    this compression only affects what is fed to the LLM.
    """
    import difflib

    def _norm(line: str) -> str:
        return re.sub(r"\s+", "", line)

    kept: list[dict] = []
    seen_exact: set[str] = set()
    prev_lines: list[str] = []

    for r in results:
        text = r["text"].strip()
        if not text or text.startswith("（无文字信息") or text.startswith("（OCR 失败"):
            continue

        # Keep only lines not seen before (information increment).
        new_lines = []
        for line in text.splitlines():
            key = _norm(line)
            if not key or key in seen_exact:
                continue
            if any(
                difflib.SequenceMatcher(None, key, p).ratio() >= similarity
                for p in prev_lines
            ):
                continue
            new_lines.append(line)
            seen_exact.add(key)
        if not new_lines:
            continue

        kept.append({**r, "text": "\n".join(new_lines)})
        prev_lines = [k for k in (_norm(l) for l in text.splitlines()) if k]

    return kept


# ---------------------------------------------------------------------------
# 4. High-level orchestration + formatting
# ---------------------------------------------------------------------------
def format_ocr_text(results: list[dict]) -> str:
    """Render OCR results as a timestamped plain-text block for the summarizer."""
    lines = []
    for r in results:
        text = r["text"].strip() or "（无文字信息）"
        lines.append(f"[{r['timestamp']}] {text}")
    return "\n".join(lines)


def analyze_video(
    video_path: Path,
    out_dir: Path,
    interval: int = DEFAULT_INTERVAL,
    dedup_threshold: int = DEFAULT_DEDUP_THRESHOLD,
) -> dict:
    """Full frame-analysis pipeline for one downloaded video file.

    Returns a dict with ``frames`` (kept unique frames), ``ocr_results``,
    and ``ocr_text``.  Raises ``RuntimeError`` on hard failures (no ffmpeg,
    no vision model configured).
    """
    frames = extract_frames(video_path, out_dir / "frames", interval=interval)
    if not frames:
        return {"frames": [], "ocr_results": [], "ocr_text": ""}

    unique = dedupe_frames(frames, threshold=dedup_threshold)
    ocr_results = ocr_frames(unique)

    return {
        "frames": unique,
        "ocr_results": ocr_results,
        "ocr_text": format_ocr_text(ocr_results),
    }
