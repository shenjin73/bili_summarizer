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

Auto-detection: checks API keys in the order above.
Set ``LLM_PROVIDER`` to force a specific provider
(e.g. ``LLM_PROVIDER=deepseek``).
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
- 注意视频中的图表、可视化内容描述了什么东西

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

## 末尾
用列表格式附上视频基本信息（UP主、时长、播放量、点赞、投币、收藏、评论数、弹幕数、发布时间）
""")

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------
PROVIDER_DEFAULTS = {
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
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


def _detect_provider() -> dict:
    """Detect which LLM provider to use from environment variables.

    Returns a dict with ``provider``, ``api_key``, ``model``, ``base_url``.
    """
    # Allow explicit override
    forced = os.environ.get("LLM_PROVIDER", "").lower()
    if forced and forced in PROVIDER_DEFAULTS:
        cfg = PROVIDER_DEFAULTS[forced]
        api_key = os.environ.get(cfg["env_key"], "")
        if not api_key:
            raise RuntimeError(
                f"LLM_PROVIDER={forced} but {cfg['env_key']} is not set."
            )
        return {
            "provider": forced,
            "api_key": api_key,
            "model": os.environ.get("LLM_MODEL", cfg["model"]),
            "base_url": os.environ.get("LLM_BASE_URL", cfg.get("base_url", "")),
        }

    # Auto-detect by checking API keys in order
    for name, cfg in PROVIDER_DEFAULTS.items():
        api_key = os.environ.get(cfg["env_key"], "")
        if api_key:
            return {
                "provider": name,
                "api_key": api_key,
                "model": os.environ.get("LLM_MODEL", cfg["model"]),
                "base_url": os.environ.get("LLM_BASE_URL", cfg.get("base_url", "")),
            }

    # Generic LLM_API_KEY + LLM_BASE_URL
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
        "未检测到任何 LLM API Key。请设置以下环境变量之一：\n"
        "  ANTHROPIC_API_KEY  — 使用 Claude\n"
        "  DEEPSEEK_API_KEY   — 使用 DeepSeek\n"
        "  MINIMAX_API_KEY    — 使用 MiniMax\n"
        "  OPENAI_API_KEY     — 使用 OpenAI\n"
        "  LLM_API_KEY + LLM_MODEL + LLM_BASE_URL — 自定义兼容 API\n"
        "\n例如: export DEEPSEEK_API_KEY=sk-xxxx"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_content_text(
    meta: dict,
    subtitles_text: Optional[str],
    description: str,
    comments: list[dict],
) -> str:
    parts = []
    parts.append(f"# 视频标题\n{meta.get('title', '未知')}")

    if description:
        parts.append(f"\n# 视频简介\n{description}")

    if subtitles_text:
        parts.append(f"\n# 字幕内容\n{subtitles_text}")
    else:
        parts.append("\n# 注意\n此视频没有字幕，请基于视频简介和评论进行总结。")

    if comments:
        comment_lines = []
        for c in comments[:10]:
            msg = c.get("content", {}).get("message", "")
            if msg:
                comment_lines.append(f"- {msg}")
        if comment_lines:
            parts.append(f"\n# 热门评论\n" + "\n".join(comment_lines))

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
    model: Optional[str] = None,
) -> str:
    """Generate a Markdown summary of a Bilibili video.

    Automatically detects the LLM backend from environment variables.
    """
    provider_cfg = _detect_provider()

    # Allow caller to override model
    if model:
        provider_cfg["model"] = model

    # Build content
    subs_text = _format_subtitles(subtitles) if subtitles else None
    content_text = _build_content_text(meta, subs_text, description, comments or [])

    # Safety truncation
    max_chars = 80_000
    if len(content_text) > max_chars:
        content_text = content_text[:max_chars] + "\n\n[... 内容过长，已截断 ...]"

    user_prompt = (
        content_text + f"\n\n---\n\n# 视频信息\n{_format_meta_info(meta)}"
    )

    provider = provider_cfg["provider"]

    if provider == "anthropic":
        return _summarize_anthropic(
            api_key=provider_cfg["api_key"],
            model=provider_cfg["model"],
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    else:
        # deepseek, minimax, openai, openai_compatible
        return _summarize_openai_compatible(
            api_key=provider_cfg["api_key"],
            model=provider_cfg["model"],
            base_url=provider_cfg["base_url"],
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
