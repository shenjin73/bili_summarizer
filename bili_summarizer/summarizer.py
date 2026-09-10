"""AI summarization of Bilibili video content.

Supports multiple LLM backends:

=========== ====================== ===================================
Provider    Env Vars               Default Model
=========== ====================== ===================================
Anthropic   ANTHROPIC_API_KEY      claude-sonnet-4-6
DeepSeek    DEEPSEEK_API_KEY       deepseek-chat
MiniMax     MINIMAX_API_KEY        abab6.5s-chat
OpenAI      OPENAI_API_KEY         gpt-4o
Custom      LLM_API_KEY            (must also set LLM_MODEL)
            LLM_BASE_URL
=========== ====================== ===================================

Auto-detection: the local chain first (``LOCAL_BASE_URLS`` / ``LOCAL_*``,
tried endpoint by endpoint in priority order), then cloud API keys in the
order above. Set ``LLM_PROVIDER`` to force a specific provider
(e.g. ``LLM_PROVIDER=deepseek``); ``LLM_PROVIDER=local`` uses the local
chain without any cloud fallback.
"""

import json
import os
import textwrap
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
你是一个专业的中文视频内容总结助手。你的任务是对B站视频进行全面、深入、详尽的总结。**你的总结将被读者当作视频的完整替代品来阅读，因此必须足够详细、信息密度足够高。**

## 核心要求

1. 用中文输出，Markdown 格式
2. **详细程度**：不少于 800 字。宁可多写，不可遗漏。每一个观点都要展开阐释，不要只列标题。
3. 使用二级标题（##）组织以下板块，每个板块至少写 3-5 句话：

### ## 视频概述
- 用 3-5 句话清晰描述视频讲了一件什么事
- 说明视频的背景（UP主是做什么的、这个视频属于什么系列、解决什么问题）
- 给出视频的核心结论或"一句话 takeaways"

### ## 核心主题
- 列出视频讨论的所有主要话题（通常 2-4 个）
- 每个主题用一段话展开：它是什么、为什么重要、视频中怎么讨论的

### ## 关键观点与论证
- 列出 UP 主的核心论点和结论（至少 4-6 条）
- **每条观点必须附带视频中的论证过程**：用了什么数据、举了什么例子、逻辑链条是什么
- 如果视频中有反驳某个常见误解的，请特别标注

### ## 重要细节
- 尽可能还原视频中提到的**具体数据**（数字、百分比、时间、对比结果等）
- 记录视频中提到的**案例、故事、类比**——这些是理解论点的关键
- 如果有代码、工具、方法论，详细记录下来
- **充分利用"画面文字与图表（OCR）"部分**：这是从视频截图中识别出的标题、数据标签、代码和图表描述，语音字幕里往往没有这些信息。画面中出现的术语定义、公式、参数、流程图结构都要纳入总结
- 注意视频中的图表、可视化内容描述了什么东西（OCR 部分会给出图表描述，请结合上下文解读）

### ## 评论视角（如有评论数据）
- 总结评论区的主要观点和讨论方向
- 是否有观众提出不同意见或补充信息
- 是否有课代表总结（常见于B站）

### ## 总体评价
- 视频的信息密度、逻辑严谨性、表达清晰度
- 适合什么水平的观众（入门/进阶/专业）
- 视频的优缺点
- 如果你看过 UP 主的其他视频，可以比较

### ## 参考价值评估（重要）
请从以下维度评估本视频的参考价值，每项给出1-5星评分并附一句话理由：
- **数据支撑度**：是否有具体数据、回测结果、统计检验支撑论点
- **方法可复现性**：观众能否根据视频内容复现分析方法或交易策略
- **观点独特性**：是否提出了反直觉、有深度、不同于市场共识的见解
- **实操指导性**：对实际投资的指导意义如何，是否给出可操作的建议
- **信息密度**：单位时间内传达的有价值信息量

最后给出**综合参考价值评级**（高/中/低），并用2-3句话总结适合什么类型的观众、是否需要结合其他资料学习。

## 风格要求
- **详细但不啰嗦**：每句话都有信息量，但不要用套话填充
- **忠实原视频**：不要编造视频中没有的内容；如果基于视频简介推断，请明确标注"基于简介推断"
- **保留数据**：任何提到的数字、统计结果都要保留
- **标注不确定性**：如果某个结论的视频依据不充分，请指出
- 如果视频没有字幕，请在开头明确标注"⚠️ 本总结基于视频简介和评论生成，非基于完整字幕，信息覆盖度有限"
- 如果既没有字幕也没有画面 OCR 数据，请在开头明确标注信息覆盖度有限

## 末尾
用列表格式附上视频基本信息（UP主、时长、播放量、点赞、投币、收藏、评论数、弹幕数、发布时间）
""")

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------
PROVIDER_DEFAULTS = {
    # Local OpenAI-compatible endpoints (e.g. oMLX, LM Studio). Checked FIRST.
    # The local "provider" is a CHAIN of endpoints, configured via LOCAL_*:
    #   LOCAL_BASE_URLS  comma-separated, in priority order (plural wins),
    #                    e.g. http://127.0.0.1:8000/v1,http://127.0.0.1:1234/v1
    #   LOCAL_MODELS     comma-separated model ids matched by position
    #                    (a single id applies to every endpoint)
    #   LOCAL_API_KEYS   comma-separated keys (a missing slot -> placeholder)
    # Legacy single-endpoint form: LOCAL_BASE_URL + LOCAL_MODEL (+ LOCAL_API_KEY).
    "local": {
        "env_key": "LOCAL_API_KEY",
        "model_env": "LOCAL_MODEL",
        "base_url_env": "LOCAL_BASE_URL",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        # Vision-capable: serves both text summarization and frame OCR fallback.
        # Was "deepseek-v4-flash-vision-exp" — that name is retired (requests to
        # it are now served by the current Flash model) and, crucially, it is no
        # longer returned by GET /models, so check_endpoint() reported
        # "model_missing" and the documented OCR fallback never fired.
        # "deepseek-flash" is the model the provider actually lists and
        # documents as image-capable:
        # https://api-docs.deepseek.com/zh-cn/guides/vision
        # resolve_vision_provider() pins this default for OCR even when
        # LLM_MODEL names a text-only model.
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
    },
    "minimax": {
        "env_key": "MINIMAX_API_KEY",
        "model": "abab6.5s-chat",
        "base_url": "https://api.minimax.chat/v1",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-4o",
    },
}

# Detection order for auto-detection (local first, then cloud providers)
DETECTION_ORDER = ["local", "anthropic", "deepseek", "minimax", "openai"]


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated env var value, dropping empty slots."""
    return [p.strip() for p in value.split(",") if p.strip()]


def local_candidates() -> list[dict]:
    """Build the local provider chain from LOCAL_* env vars, in priority order.

    Two config styles; the plural (chain) form wins when set:

    - Chain: ``LOCAL_BASE_URLS`` = comma-separated OpenAI-compatible
      endpoints, tried in order (e.g. oMLX then LM Studio).
      ``LOCAL_MODELS`` / ``LOCAL_API_KEYS`` are optional comma-separated
      lists matched by position; a single entry applies to every endpoint,
      and a missing key slot becomes the placeholder ``"lm-studio"``.
    - Single (legacy): ``LOCAL_BASE_URL`` + ``LOCAL_MODEL`` (+ optional
      ``LOCAL_API_KEY``) — one endpoint.

    Returns a list of config dicts (``provider``/``api_key``/``model``/
    ``base_url``), empty when no usable local endpoint is configured.
    """
    urls = _split_csv(os.environ.get("LOCAL_BASE_URLS", ""))
    models = _split_csv(os.environ.get("LOCAL_MODELS", ""))
    keys = _split_csv(os.environ.get("LOCAL_API_KEYS", ""))

    if not urls:
        single_url = os.environ.get("LOCAL_BASE_URL", "")
        if not single_url:
            return []
        urls = [single_url]
        if not models:
            models = [os.environ.get("LOCAL_MODEL", "")]
        if not keys:
            keys = [os.environ.get("LOCAL_API_KEY", "")]

    cands = []
    for i, url in enumerate(urls):
        model = models[i] if i < len(models) else (models[0] if models else "")
        if not model:
            continue  # endpoint without a model id is unusable
        key = keys[i] if i < len(keys) else (keys[0] if keys else "")
        cands.append(
            {
                "provider": "local",
                "api_key": key or "lm-studio",  # placeholder; most local servers ignore it
                "model": model,
                "base_url": url,
            }
        )
    return cands


def reason_label(status: str) -> str:
    """Human-readable label for a ``check_endpoint`` status."""
    return {
        "unreachable": "服务不在线",
        "model_missing": "模型未加载",
    }.get(status, status)


def _cloud_config(name: str) -> dict | None:
    """Build a cloud provider config, or None when its API key is unset."""
    cfg = PROVIDER_DEFAULTS[name]
    api_key = os.environ.get(cfg["env_key"], "")
    if not api_key:
        return None
    return {
        "provider": name,
        "api_key": api_key,
        # LLM_MODEL / LLM_BASE_URL act as overrides for the chosen provider.
        "model": os.environ.get("LLM_MODEL", cfg["model"]),
        "base_url": os.environ.get("LLM_BASE_URL", cfg.get("base_url", "")),
    }


# ---------------------------------------------------------------------------
# Availability check + automatic fallback
# ---------------------------------------------------------------------------
def check_endpoint(
    base_url: str, model: str, timeout: float = 5.0, api_key: str | None = None
) -> str:
    """Check an OpenAI-compatible endpoint's availability.

    Returns ``"ok"``, ``"unreachable"`` (service offline / bad status) or
    ``"model_missing"`` (service up but the model is not loaded).

    Some providers (e.g. DeepSeek) require an Authorization header even for
    the models list, so pass ``api_key`` when available.
    """
    import requests

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.get(
            base_url.rstrip("/") + "/models", timeout=timeout, headers=headers
        )
    except Exception:
        return "unreachable"
    if resp.status_code != 200:
        return "unreachable"
    try:
        ids = {m.get("id") for m in resp.json().get("data", [])}
    except Exception:
        return "ok"  # reachable but unparseable — let the real call decide
    if not ids:
        return "ok"  # some servers don't list models; assume ok
    return "ok" if model in ids else "model_missing"


def _deepseek_fallback() -> dict | None:
    """DeepSeek config for automatic fallback (always uses its own defaults)."""
    return _cloud_config("deepseek")


# Process-level cache: chain fingerprint -> resolved config. Avoids re-checking
# the endpoints and printing duplicate warnings when resolve_provider() is
# called more than once per run (e.g. main's announcement + summarize()).
_resolve_cache: dict[tuple, dict] = {}


def _chain_key(cands: list[dict]) -> tuple:
    return ("local-chain", *((c["base_url"], c["model"]) for c in cands))


def _walk_local_chain(cands: list[dict]) -> tuple[dict | None, list[str]]:
    """Health-check local endpoints in priority order.

    Returns ``(first_healthy_config_or_None, [reason per failed endpoint])``.
    """
    reasons: list[str] = []
    for c in cands:
        status = check_endpoint(c["base_url"], c["model"], api_key=c["api_key"])
        if status == "ok":
            return c, reasons
        reasons.append(f"{c['base_url']} / {c['model']} {reason_label(status)}")
    return None, reasons


def resolve_provider() -> dict:
    """Detect the LLM provider, with a local-first chain and cloud fallback.

    Resolution order:

    1. Forced ``LLM_PROVIDER`` (any name in ``PROVIDER_DEFAULTS``).
       ``LLM_PROVIDER=local`` uses the local chain and NEVER falls back to
       cloud — it raises when every local endpoint is down.
    2. Local chain (``LOCAL_BASE_URLS`` / ``LOCAL_*``), health-checked
       endpoint by endpoint in priority order; the first healthy one wins.
    3. DeepSeek (when ``DEEPSEEK_API_KEY`` is set and its endpoint is healthy).
    4. Remaining cloud providers in ``DETECTION_ORDER`` (only when no local
       endpoint is configured at all), then the generic ``LLM_*`` endpoint.

    Raises ``RuntimeError`` with a detailed message when nothing is usable.
    The resolution result (including a fallback) is cached per process, so one
    run stays on one provider.
    """
    # Allow explicit override
    forced = os.environ.get("LLM_PROVIDER", "").lower()
    if forced and forced in PROVIDER_DEFAULTS:
        if forced == "local":
            cands = local_candidates()
            if not cands:
                raise RuntimeError(
                    "LLM_PROVIDER=local 但 LOCAL_BASE_URLS / LOCAL_BASE_URL 未配置。"
                )
            chosen, reasons = _walk_local_chain(cands)
            if chosen is None:
                raise RuntimeError(
                    "LLM_PROVIDER=local，但所有本地端点不可用：" + "；".join(reasons)
                )
            return chosen
        cfg = _cloud_config(forced)
        if not cfg:
            raise RuntimeError(
                f"LLM_PROVIDER={forced} but {PROVIDER_DEFAULTS[forced]['env_key']} is not set."
            )
        return cfg

    cands = local_candidates()
    if cands:
        key = _chain_key(cands)
        cached = _resolve_cache.get(key)
        if cached is not None:
            return dict(cached)

        chosen, reasons = _walk_local_chain(cands)
        if chosen is None:
            ds = _deepseek_fallback()
            ds_ok = (
                ds is not None
                and check_endpoint(
                    ds["base_url"], ds["model"], api_key=ds["api_key"]
                )
                == "ok"
            )
            if ds_ok:
                print(
                    f"⚠️ 所有本地模型不可用（{'；'.join(reasons)}），"
                    f"自动切换到 DeepSeek（{ds['model']}）"
                )
                chosen = ds
            else:
                why = (
                    "未设置 DEEPSEEK_API_KEY 备用"
                    if ds is None
                    else "DeepSeek 备用也不可用"
                )
                raise RuntimeError(
                    f"所有本地模型不可用（{'；'.join(reasons)}），且{why}。"
                    "请启动本地服务或配置云端 Key。"
                )

        _resolve_cache[key] = chosen
        return dict(chosen)

    # No local endpoint configured — cloud providers in order, then generic.
    for name in DETECTION_ORDER:
        if name == "local":
            continue
        cfg = _cloud_config(name)
        if cfg:
            return cfg

    # Generic LLM_API_KEY + LLM_BASE_URL (remote OpenAI-compatible endpoint)
    generic_key = os.environ.get("LLM_API_KEY", "")
    if generic_key:
        model = os.environ.get("LLM_MODEL", "")
        if not model:
            raise RuntimeError(
                "LLM_API_KEY is set but LLM_MODEL is not. "
                "Please set LLM_MODEL to the model name."
            )
        return {
            "provider": "openai_compatible",
            "api_key": generic_key,
            "model": model,
            "base_url": os.environ.get("LLM_BASE_URL", ""),
        }

    raise RuntimeError(
        "未检测到任何 LLM 配置。请设置以下之一：\n"
        "  LOCAL_BASE_URLS + LOCAL_MODELS — 本地模型链（如 oMLX / LM Studio，默认优先）\n"
        "  ANTHROPIC_API_KEY            — Claude\n"
        "  DEEPSEEK_API_KEY             — DeepSeek（本地全部离线时的自动备用）\n"
        "  MINIMAX_API_KEY              — MiniMax\n"
        "  OPENAI_API_KEY               — OpenAI\n"
        "  LLM_API_KEY + LLM_MODEL + LLM_BASE_URL — 自定义兼容 API\n"
        "\n例如: export LOCAL_BASE_URLS=http://127.0.0.1:8000/v1,http://127.0.0.1:1234/v1 "
        "LOCAL_MODELS=Qwen3.8-27B-4bit,qwen3.8-27b-mlx LOCAL_API_KEYS=1234,lm-studio-local"
    )


def _is_connection_error(e: Exception) -> bool:
    """True for network-level failures (service went away mid-call)."""
    try:
        import openai

        if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
    except ImportError:
        pass
    import requests

    return isinstance(e, (requests.ConnectionError, requests.Timeout))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Per-section caps for the summary prompt (env-overridable). The sections are
# assembled as description → subtitles → OCR → comments, so a single hard cut
# on the whole text could chop off the OCR / comments sections when subtitles
# are long. Capping each section independently (head + tail kept, middle
# dropped) guarantees the later sections always survive.
MAX_DESC_CHARS = int(os.environ.get("SUMMARY_MAX_DESC_CHARS", "4000"))
MAX_SUBTITLE_CHARS = int(os.environ.get("SUMMARY_MAX_SUBTITLE_CHARS", "50000"))
MAX_OCR_CHARS = int(os.environ.get("SUMMARY_MAX_OCR_CHARS", "15000"))
MAX_COMMENT_CHARS = int(os.environ.get("SUMMARY_MAX_COMMENT_CHARS", "3000"))


def _cap_middle(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars*, keeping head + tail (middle dropped)."""
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 50)  # room for the marker
    return f"{text[: keep * 2 // 3]}\n\n[... 中间内容已截断 ...]\n\n{text[-keep // 3:]}"


def _build_content_text(
    meta: dict,
    subtitles_text: Optional[str],
    description: str,
    comments: list[dict],
    on_screen_text: Optional[str] = None,
) -> str:
    parts = []
    parts.append(f"# 视频标题\n{meta.get('title', '未知')}")

    if description:
        parts.append(f"\n# 视频简介\n{_cap_middle(description, MAX_DESC_CHARS)}")

    if subtitles_text:
        parts.append(f"\n# 字幕内容\n{_cap_middle(subtitles_text, MAX_SUBTITLE_CHARS)}")
    else:
        parts.append("\n# 注意\n此视频没有字幕，请基于其他可用信息进行总结。")

    if on_screen_text:
        parts.append(
            "\n# 画面文字与图表（OCR）\n"
            "以下是从视频截图中识别出的画面信息，每行以 [时间戳] 开头。"
            "语音字幕未覆盖的画面文字、数据标签和图表描述都在这里。\n"
            + _cap_middle(on_screen_text, MAX_OCR_CHARS)
        )

    if comments:
        comment_lines = []
        for c in comments[:10]:
            msg = c.get("content", {}).get("message", "")
            if msg:
                comment_lines.append(f"- {msg}")
        if comment_lines:
            parts.append(
                f"\n# 热门评论\n"
                + _cap_middle("\n".join(comment_lines), MAX_COMMENT_CHARS)
            )

    return "\n\n".join(parts)


def _format_subtitles(raw_subs: list[dict]) -> str:
    lines = []
    for entry in raw_subs:
        text = entry.get("content", "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _format_meta_info(meta: dict) -> str:
    owner = meta.get("owner", {})
    stat = meta.get("stat", {})
    duration_sec = meta.get("duration", 0)
    minutes = duration_sec // 60
    seconds = duration_sec % 60

    lines = [
        f"- **UP主**：{owner.get('name', '未知')}",
        f"- **时长**：{minutes}分{seconds}秒",
        f"- **播放量**：{stat.get('view', 0):,}",
        f"- **点赞**：{stat.get('like', 0):,}",
        f"- **投币**：{stat.get('coin', 0):,}",
        f"- **收藏**：{stat.get('favorite', 0):,}",
        f"- **评论数**：{stat.get('reply', 0):,}",
        f"- **弹幕数**：{stat.get('danmaku', 0):,}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backend: Anthropic (native SDK)
# ---------------------------------------------------------------------------
def _summarize_anthropic(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = []
    for block in message.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible (DeepSeek, MiniMax, OpenAI, custom)
# ---------------------------------------------------------------------------
def _summarize_openai_compatible(
    api_key: str,
    model: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    response = client.chat.completions.create(
        model=model,
        max_tokens=8192,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def summarize(
    meta: dict,
    description: str = "",
    subtitles: Optional[list[dict]] = None,
    comments: Optional[list[dict]] = None,
    on_screen_text: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate a Markdown summary of a Bilibili video.

    Automatically detects the LLM backend from environment variables.

    ``on_screen_text`` is optional timestamped OCR output (text + chart
    descriptions) extracted from video screenshots; it is merged into the
    prompt so on-screen information not covered by speech is included.

    Provider selection: the local chain (LOCAL_BASE_URLS / LOCAL_*, in
    priority order) is checked first; when every local endpoint is offline or
    its model is not loaded, automatically falls back to DeepSeek when
    ``DEEPSEEK_API_KEY`` is set.
    """
    provider_cfg = resolve_provider()

    # Allow caller to override model
    if model:
        provider_cfg["model"] = model

    # Build content
    subs_text = _format_subtitles(subtitles) if subtitles else None
    content_text = _build_content_text(
        meta, subs_text, description, comments or [], on_screen_text=on_screen_text
    )

    # Last-resort safety cut. The per-section caps in _build_content_text keep
    # the total below this in all realistic cases, so the OCR / comments
    # sections are never chopped off by an oversized subtitle block.
    max_chars = 80_000
    if len(content_text) > max_chars:
        content_text = content_text[:max_chars] + "\n\n[... 内容过长，已截断 ...]"

    user_prompt = (
        content_text + f"\n\n---\n\n# 视频信息\n{_format_meta_info(meta)}"
    )

    try:
        return _call_llm(provider_cfg, SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        # Safety net: if a local service dies mid-call, retry on the next
        # local endpoint in the chain, then on DeepSeek.
        if provider_cfg["provider"] == "local" and _is_connection_error(e):
            alts = [
                c
                for c in local_candidates()
                if c["base_url"] != provider_cfg["base_url"]
            ]
            for c in alts:
                if check_endpoint(c["base_url"], c["model"], api_key=c["api_key"]) == "ok":
                    print(
                        f"⚠️ 本地模型调用失败（{provider_cfg['base_url']}，"
                        f"{type(e).__name__}），切换到 {c['base_url']} / {c['model']} 重试"
                    )
                    return _call_llm(c, SYSTEM_PROMPT, user_prompt)
            fallback = _deepseek_fallback()
            if (
                fallback is not None
                and check_endpoint(
                    fallback["base_url"],
                    fallback["model"],
                    api_key=fallback["api_key"],
                )
                == "ok"
            ):
                print(
                    f"⚠️ 本地模型调用失败（{provider_cfg['base_url']}，{type(e).__name__}），"
                    f"自动切换到 DeepSeek 重试"
                )
                return _call_llm(fallback, SYSTEM_PROMPT, user_prompt)
        raise


def _call_llm(provider_cfg: dict, system_prompt: str, user_prompt: str) -> str:
    """Dispatch a summarization call to the right backend."""
    if provider_cfg["provider"] == "anthropic":
        return _summarize_anthropic(
            api_key=provider_cfg["api_key"],
            model=provider_cfg["model"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    # local, deepseek, minimax, openai, openai_compatible — all OpenAI-compatible
    return _summarize_openai_compatible(
        api_key=provider_cfg["api_key"],
        model=provider_cfg["model"],
        base_url=provider_cfg["base_url"],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
