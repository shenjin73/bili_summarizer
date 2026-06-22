"""Whisper audio transcription via mlx-whisper (Apple Silicon optimised)."""

import os
from pathlib import Path

# Use HuggingFace mirror if HF_ENDPOINT is not explicitly set.
# hf-mirror.com is the standard mirror accessible in China.
# Set HF_ENDPOINT="" or HF_ENDPOINT="https://huggingface.co" to use the official endpoint.
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def transcribe_audio(
    audio_path: Path,
    model_size: str = "medium",
    language: str = "zh",
) -> str:
    """Transcribe an audio file to plain text.

    Parameters
    ----------
    audio_path:
        Path to the audio file (any format supported by ffmpeg / Whisper).
    model_size:
        Whisper model variant: ``"tiny"``, ``"small"``, ``"medium"``, ``"large-v3"``.
        Default ``"medium"`` is a good balance for Chinese on Apple Silicon.
    language:
        Language code hint (``"zh"`` for Chinese).

    Returns
    -------
    Plain-text transcription with one line per segment.
    """
    import mlx_whisper

    model_repo = f"mlx-community/whisper-{model_size}"

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
