"""Whisper audio transcription via mlx-whisper (Apple Silicon optimised)."""

import os
import re
from pathlib import Path

# Use HuggingFace mirror if HF_ENDPOINT is not explicitly set.
# hf-mirror.com is the standard mirror accessible in China.
# Set HF_ENDPOINT="" or HF_ENDPOINT="https://huggingface.co" to use the official endpoint.
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- Proxy-bypass sanitising -------------------------------------------------
# Some environments list loopback IPv6 in no_proxy, e.g.
#     NO_PROXY=localhost,127.0.0.1,::1,[::1]
# urllib/requests derive the bypass host by splitting an entry on ":", so a
# bracketed literal such as "[::1]" raises
#     ValueError: Invalid port: ':1]'
# which aborts the HuggingFace model fetch — and therefore transcription —
# even when the weights are already cached locally. HuggingFace is never
# reached over loopback, so these entries can be dropped safely. Plain
# hostnames and IPv4 (with optional port) are kept untouched.
_PROXY_BYPASS_RE = re.compile(r"^[A-Za-z0-9._*-]+(:\d+)?$")


def _sanitize_no_proxy() -> None:
    """Drop no_proxy entries that break URL parsing (bare/bracketed IPv6)."""
    for var in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(var)
        if not raw:
            continue
        kept, dropped = [], []
        for item in raw.split(","):
            entry = item.strip()
            if not entry:
                continue
            (kept if _PROXY_BYPASS_RE.match(entry) else dropped).append(entry)
        if dropped:
            os.environ[var] = ",".join(kept)
            print(
                f"   ⚠️  {var} 中的 {', '.join(dropped)} 无法被 URL 解析器处理"
                "（Invalid port），已临时剔除"
            )


def transcribe_audio(
    audio_path: Path,
    model_size: str = "",
    language: str = "zh",
) -> str:
    """Transcribe an audio file to plain text.

    Parameters
    ----------
    audio_path:
        Path to the audio file (any format supported by ffmpeg / Whisper).
    model_size:
        Whisper model variant, e.g. ``"tiny"``, ``"small"``, ``"medium"``,
        ``"large-v3"``, ``"large-v3-turbo"``. Empty (default) reads
        ``WHISPER_MODEL`` env var, falling back to ``"large-v3-turbo"`` —
        near-large accuracy at roughly 4-8x the speed on Apple Silicon.
    language:
        Language code hint (``"zh"`` for Chinese).

    Returns
    -------
    Plain-text transcription with one line per segment.
    """
    _sanitize_no_proxy()
    import mlx_whisper

    model_size = model_size or os.environ.get("WHISPER_MODEL", "large-v3-turbo")
    model_repo = f"mlx-community/whisper-{model_size}"

    print(f"   Whisper 模型: {model_repo}")

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        language=language,
        verbose=False,
    )

    # Build text from segments (one line per segment), falling back to full text
    segments = result.get("segments", [])
    if segments:
        lines = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                lines.append(text)
        return "\n".join(lines)

    return result.get("text", "").strip()
