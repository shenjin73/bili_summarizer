"""File storage helpers for per-video subdirectories."""

import json
import os
from pathlib import Path

# Root directory where all video data is stored. Override with the
# VIDEOS_ROOT env var (e.g. for A/B comparison runs into separate trees).
ROOT = Path(os.environ.get("VIDEOS_ROOT") or (Path(__file__).resolve().parent.parent / "videos"))


def video_dir(bvid: str, creator: str | None = None) -> Path:
    """Return the subdirectory path for a given video BV id.

    When *creator* is given, videos are stored under ``<creator>/<bvid>/``
    so that different uploaders are kept in separate sub-folders.
    """
    base = ROOT / creator if creator else ROOT
    return base / bvid


def ensure_video_dir(bvid: str, creator: str | None = None) -> Path:
    """Create (if needed) and return the subdirectory for *bvid*."""
    path = video_dir(bvid, creator)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metadata(bvid: str, meta: dict, creator: str | None = None) -> Path:
    """Save raw video metadata as pretty-printed JSON."""
    path = ensure_video_dir(bvid, creator) / "metadata.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_description(bvid: str, text: str, creator: str | None = None) -> Path:
    """Save video description as plain text."""
    path = ensure_video_dir(bvid, creator) / "description.txt"
    path.write_text(text, encoding="utf-8")
    return path


def save_subtitles(bvid: str, data: dict | list, creator: str | None = None) -> Path:
    """Save subtitle data as JSON."""
    path = ensure_video_dir(bvid, creator) / "subtitles.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_subtitle_text(bvid: str, text: str, creator: str | None = None) -> Path:
    """Save subtitle plain-text transcript."""
    path = ensure_video_dir(bvid, creator) / "subtitles.txt"
    path.write_text(text, encoding="utf-8")
    return path


def save_comments(bvid: str, comments: list[dict], creator: str | None = None) -> Path:
    """Save comment list as JSON."""
    path = ensure_video_dir(bvid, creator) / "comments.json"
    path.write_text(json.dumps(comments, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_summary(bvid: str, text: str, creator: str | None = None) -> Path:
    """Save the markdown summary."""
    path = ensure_video_dir(bvid, creator) / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def save_ocr_text(bvid: str, text: str, creator: str | None = None) -> Path:
    """Save the timestamped on-screen text (OCR results)."""
    path = ensure_video_dir(bvid, creator) / "ocr_text.txt"
    path.write_text(text, encoding="utf-8")
    return path


def save_frames_manifest(
    bvid: str, results: list[dict], creator: str | None = None
) -> Path:
    """Save the per-frame OCR manifest (timestamp, file, text) as JSON."""
    path = ensure_video_dir(bvid, creator) / "frames.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_content(bvid: str, filename: str, data: str | bytes, creator: str | None = None) -> Path:
    """Generic save — write arbitrary content to a file in the video dir."""
    path = ensure_video_dir(bvid, creator) / filename
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path
