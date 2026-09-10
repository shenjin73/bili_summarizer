#!/usr/bin/env python3
"""Bili Summarizer — CLI tool to summarize Bilibili videos.

Usage:
    python -m bili_summarizer.main <bilibili_url> [options]
    python -m bili_summarizer.main BV1CNLQ6REu5

Options:
    --no-frames      跳过画面分析（截图 + OCR）
    --interval N     截图间隔秒数（默认 10，即每 10 秒截一帧）

Examples:
    python -m bili_summarizer.main https://www.bilibili.com/video/BV1CNLQ6REu5/
    python -m bili_summarizer.main https://b23.tv/xxxxx --no-frames
"""

import argparse
import os
import sys
import time
from pathlib import Path

from .fetcher import (
    download_audio,
    download_stream,
    extract_bvid,
    fetch_audio_url,
    fetch_comments,
    fetch_subtitle_content,
    fetch_subtitles,
    fetch_video_meta,
    fetch_video_url,
)
from .frames import (
    DEFAULT_INTERVAL,
    compress_ocr_results,
    dedupe_frames,
    extract_frames,
    format_ocr_text,
    ocr_frames,
)
from .storage import (
    save_comments,
    save_description,
    save_frames_manifest,
    save_metadata,
    save_ocr_text,
    save_subtitle_text,
    save_summary,
    video_dir,
)
from .summarizer import summarize
from .transcriber import transcribe_audio

_RUN_T0 = time.time()


def _stage(name: str) -> float:
    """Print a timestamped stage-start line; returns the start time."""
    print(f"[{time.strftime('%H:%M:%S')}] ▶ {name}")
    return time.time()


def _stage_done(name: str, t0: float) -> None:
    """Print a timestamped stage-end line with elapsed seconds."""
    print(f"[{time.strftime('%H:%M:%S')}] ✓ {name}（{time.time() - t0:.1f}s）")


def _extract_description(meta: dict) -> str:
    """Extract plain-text description from video metadata."""
    desc = meta.get("desc", "")
    desc_v2 = meta.get("desc_v2", [])
    if desc_v2:
        parts = [item.get("raw_text", "") for item in desc_v2 if item.get("raw_text")]
        return "\n".join(parts)
    return desc


def _analyze_frames(bvid: str, creator: str, cid: int, out_dir: Path, interval: int) -> str | None:
    """Download the video stream, screenshot every *interval* seconds, dedupe,
    and OCR each unique frame with the vision model.

    Returns timestamped on-screen text (str) or ``None`` when skipped/failed.
    The returned text is cross-frame deduplicated (``compress_ocr_results``)
    to shrink the summary prompt; the full OCR text is saved to disk as-is.
    Unique frames are kept under ``<out_dir>/frames/``; the video file itself
    is deleted afterwards.
    """
    print("🎬 画面分析（截图 + OCR）...")

    video_url = fetch_video_url(bvid, cid)
    if not video_url:
        print("   ⚠️ 无法获取视频流地址，跳过画面分析")
        return None

    print("   ⬇️ 下载视频流...")
    video_path = out_dir / "video.m4s"
    download_stream(video_url, video_path)
    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"   视频已下载: {size_mb:.1f} MB")

    try:
        t0 = _stage("抽帧")
        frames = extract_frames(video_path, out_dir / "frames", interval=interval)
        print(f"   每 {interval} 秒截图: {len(frames)} 张")
        _stage_done("抽帧", t0)
        if not frames:
            return None

        unique = dedupe_frames(frames)
        print(f"   去重后: {len(unique)} 张（重复画面已删除）")
        if not unique:
            return None

        # Safety cap for very long videos (evenly sample down to the cap)
        max_ocr = int(os.environ.get("FRAME_MAX_OCR", "80"))
        if len(unique) > max_ocr:
            step = len(unique) / max_ocr
            unique = [unique[int(i * step)] for i in range(max_ocr)]
            print(f"   画面过多，均匀采样至 {len(unique)} 张进行 OCR")

        print(f"   🔍 OCR 识别中（{len(unique)} 张，视觉模型）...")
        t0 = _stage("OCR")
        ocr_results = ocr_frames(unique)
        _stage_done(f"OCR（{len(ocr_results)} 张）", t0)

        save_ocr_text(bvid, format_ocr_text(ocr_results), creator)
        save_frames_manifest(bvid, ocr_results, creator)
        print("   ✓ ocr_text.txt + frames.json + frames/")

        # Compress for the summary prompt only; disk keeps the full text.
        compressed = compress_ocr_results(ocr_results)
        if len(compressed) < len(ocr_results):
            print(
                f"   摘要输入压缩: {len(ocr_results)} → {len(compressed)} 帧"
                "（跨帧去重，仅影响喂给 LLM 的文本）"
            )
        return format_ocr_text(compressed)
    finally:
        video_path.unlink(missing_ok=True)
        print("   ✓ 视频文件已清理")


def main():
    parser = argparse.ArgumentParser(
        description="Bili Summarizer — 总结 B站视频（字幕 + 画面 OCR）"
    )
    parser.add_argument("url", help="B站视频链接或BV号")
    parser.add_argument(
        "--no-frames", action="store_true", help="跳过画面分析（截图 + OCR）"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"截图间隔秒数（默认 {DEFAULT_INTERVAL}）",
    )
    args = parser.parse_args()

    url = args.url

    # --- Step 1: Parse BVID ---
    print(f"📎 解析链接: {url}")
    bvid = extract_bvid(url)
    print(f"   BVID: {bvid}")

    # --- Step 2: Fetch metadata ---
    t_meta = _stage("获取视频信息")
    print("📡 获取视频信息...")
    try:
        meta = fetch_video_meta(bvid)
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")
        sys.exit(1)
    _stage_done("获取视频信息", t_meta)

    title = meta.get("title", "未知标题")
    cid = meta.get("cid", 0)
    owner = meta.get("owner", {})
    creator = owner.get("name", "")  # UP主名称，用作子文件夹名
    stat = meta.get("stat", {})
    duration = meta.get("duration", 0)

    print(f"   标题: {title}")
    print(f"   UP主: {owner.get('name', '未知')}")
    print(f"   时长: {duration // 60}分{duration % 60}秒")
    print(f"   播放: {stat.get('view', 0):,} | 点赞: {stat.get('like', 0):,}")

    out_dir = video_dir(bvid, creator)

    # --- Step 3: Fetch description ---
    print("📝 提取视频简介...")
    description = _extract_description(meta)
    print(f"   简介长度: {len(description)} 字")

    # --- Step 4: Fetch subtitles (CC first, then Whisper fallback) ---
    t_subs = _stage("获取字幕/转写")
    print("🎤 获取字幕...")
    subtitles_raw = None
    subtitles_text = None
    try:
        subtitle_list = fetch_subtitles(bvid, cid)
        if subtitle_list:
            # Download the first available subtitle (prefer Chinese)
            sub_url = None
            for sub in subtitle_list:
                if "zh" in sub.get("lan", "").lower() or "cn" in sub.get("lan", "").lower():
                    sub_url = sub.get("subtitle_url")
                    break
            if not sub_url and subtitle_list:
                sub_url = subtitle_list[0].get("subtitle_url")

            if sub_url:
                subtitles_raw = fetch_subtitle_content(sub_url)
                if subtitles_raw:
                    subtitles_text = "\n".join(
                        entry.get("content", "") for entry in subtitles_raw
                    )
                    print(f"   字幕行数: {len(subtitles_raw)}")
                else:
                    print("   字幕内容为空")
            else:
                print("   字幕链接无效")

        if not subtitles_text:
            # --- Whisper fallback: download audio and transcribe ---
            print("   ⚠️  无CC字幕，尝试音频转写...")
            cached_subs = out_dir / "subtitles.txt"
            if cached_subs.exists() and cached_subs.read_text(
                encoding="utf-8"
            ).strip():
                # Resume: reuse a previous run's transcript instead of
                # re-transcribing (Whisper is the slowest stage).
                subtitles_text = cached_subs.read_text(encoding="utf-8")
                print(
                    f"   复用已有 subtitles.txt（{len(subtitles_text.splitlines())} 行），"
                    "跳过 Whisper 转写"
                )
            else:
                print("🎵 下载音频...")
                try:
                    audio_url = fetch_audio_url(bvid, cid)
                    if audio_url:
                        audio_path = out_dir / "audio.m4s"
                        download_audio(audio_url, audio_path)
                        size_mb = audio_path.stat().st_size / (1024 * 1024)
                        print(f"   音频已下载: {size_mb:.1f} MB")

                        print("🤫 Whisper 转写中...")
                        t0 = _stage("Whisper 转写")
                        subtitles_text = transcribe_audio(audio_path)
                        _stage_done("Whisper 转写", t0)
                        line_count = len(subtitles_text.splitlines())
                        print(f"   转写完成: {line_count} 行")

                        # Flush immediately: if the process is killed later
                        # (e.g. timeout), a re-run resumes from this file.
                        save_subtitle_text(bvid, subtitles_text, creator)
                        print("   ✓ subtitles.txt（已提前落盘，可断点续跑）")

                        # Clean up audio file
                        audio_path.unlink()
                        print("   ✓ 音频文件已清理")
                    else:
                        print("   ⚠️  无法获取音频地址")
                except Exception as e:
                    print(f"   ⚠️  音频转写失败: {e}")

    except Exception as e:
        print(f"   ⚠️  获取字幕失败: {e}")
    _stage_done("获取字幕/转写", t_subs)

    # --- Step 5: Frame analysis (screenshots + OCR) ---
    on_screen_text = None
    if not args.no_frames:
        t_frames = _stage("画面分析（截图+OCR）")
        try:
            on_screen_text = _analyze_frames(
                bvid, creator, cid, out_dir, interval=args.interval
            )
        except Exception as e:
            print(f"   ⚠️  画面分析失败: {e}")
        _stage_done("画面分析（截图+OCR）", t_frames)

    # --- Step 6: Fetch comments ---
    print("💬 获取评论...")
    comments = []
    try:
        comments = fetch_comments(bvid, count=20)
        print(f"   获取到 {len(comments)} 条评论")
    except Exception as e:
        print(f"   ⚠️  获取评论失败: {e}")

    # --- Step 7: Save everything ---
    print("💾 保存数据...")
    print(f"   目录: {out_dir}")

    save_metadata(bvid, meta, creator)
    print("   ✓ metadata.json")

    save_description(bvid, description, creator)
    print("   ✓ description.txt")

    if subtitles_text:
        save_subtitle_text(bvid, subtitles_text, creator)
        print("   ✓ subtitles.txt")
    if subtitles_raw:
        from .storage import save_subtitles
        save_subtitles(bvid, subtitles_raw, creator)
        print("   ✓ subtitles.json")

    if comments:
        save_comments(bvid, comments, creator)
        print("   ✓ comments.json")

    # --- Step 8: Summarize ---
    t_sum = _stage("AI 总结")
    print("🤖 AI 总结中...")
    try:
        # Show which provider is being used (includes local→DeepSeek fallback)
        from .summarizer import resolve_provider
        try:
            cfg = resolve_provider()
            print(f"   使用模型: {cfg['provider']} / {cfg['model']}")
        except Exception as e:
            print(f"   ⚠️  {e}")

        # Normalise subtitles: summarizer expects list[dict], but Whisper gives str
        subs_for_summary = subtitles_raw
        if not subs_for_summary and subtitles_text:
            subs_for_summary = [
                {"content": line}
                for line in subtitles_text.splitlines()
                if line.strip()
            ]

        summary = summarize(
            meta=meta,
            description=description,
            subtitles=subs_for_summary,
            comments=comments,
            on_screen_text=on_screen_text,
        )
        save_summary(bvid, summary, creator)
        print("   ✓ summary.md")
        print("\n" + "=" * 60)
        print(summary)
        print("=" * 60)
    except RuntimeError as e:
        print(f"   ⚠️  {e}")
        print(f"   数据已保存到 {out_dir}，可以稍后手动总结。")
    _stage_done("AI 总结", t_sum)

    print(f"\n✅ 完成！所有文件已保存到: {out_dir}")
    print(f"⏱️  总耗时: {time.time() - _RUN_T0:.1f}s")


if __name__ == "__main__":
    main()
