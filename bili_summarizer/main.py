#!/usr/bin/env python3
"""Bili Summarizer — CLI tool to summarize Bilibili videos.

Usage:
    python -m bili_summarizer.main <bilibili_url>
    python -m bili_summarizer.main BV1CNLQ6REu5

Examples:
    python -m bili_summarizer.main https://www.bilibili.com/video/BV1CNLQ6REu5/
    python -m bili_summarizer.main https://b23.tv/xxxxx
"""

import sys
from pathlib import Path

from .fetcher import (
    download_audio,
    extract_bvid,
    fetch_audio_url,
    fetch_comments,
    fetch_subtitle_content,
    fetch_subtitles,
    fetch_video_meta,
)
from .storage import (
    save_comments,
    save_description,
    save_metadata,
    save_subtitle_text,
    save_summary,
    video_dir,
)
from .summarizer import summarize
from .transcriber import transcribe_audio


def _extract_description(meta: dict) -> str:
    """Extract plain-text description from video metadata."""
    desc = meta.get("desc", "")
    desc_v2 = meta.get("desc_v2", [])
    if desc_v2:
        parts = [item.get("raw_text", "") for item in desc_v2 if item.get("raw_text")]
        return "\n".join(parts)
    return desc


def main():
    if len(sys.argv) < 2:
        print("用法: python -m bili_summarizer.main <B站视频链接或BV号>")
        print("示例: python -m bili_summarizer.main https://www.bilibili.com/video/BV1CNLQ6REu5/")
        sys.exit(1)

    url = sys.argv[1]

    # --- Step 1: Parse BVID ---
    print(f"📎 解析链接: {url}")
    bvid = extract_bvid(url)
    print(f"   BVID: {bvid}")

    # --- Step 2: Fetch metadata ---
    print("📡 获取视频信息...")
    try:
        meta = fetch_video_meta(bvid)
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")
        sys.exit(1)

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

    # --- Step 3: Fetch description ---
    print("📝 提取视频简介...")
    description = _extract_description(meta)
    print(f"   简介长度: {len(description)} 字")

    # --- Step 4: Fetch subtitles (CC first, then Whisper fallback) ---
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
            print("🎵 下载音频...")
            try:
                audio_url = fetch_audio_url(bvid, cid)
                if audio_url:
                    audio_path = video_dir(bvid, creator) / "audio.m4s"
                    download_audio(audio_url, audio_path)
                    size_mb = audio_path.stat().st_size / (1024 * 1024)
                    print(f"   音频已下载: {size_mb:.1f} MB")

                    print("🤫 Whisper 转写中...")
                    subtitles_text = transcribe_audio(audio_path)
                    line_count = len(subtitles_text.splitlines())
                    print(f"   转写完成: {line_count} 行")

                    # Clean up audio file
                    audio_path.unlink()
                    print("   ✓ 音频文件已清理")
                else:
                    print("   ⚠️  无法获取音频地址")
            except Exception as e:
                print(f"   ⚠️  音频转写失败: {e}")

    except Exception as e:
        print(f"   ⚠️  获取字幕失败: {e}")

    # --- Step 5: Fetch comments ---
    print("💬 获取评论...")
    comments = []
    try:
        comments = fetch_comments(bvid, count=20)
        print(f"   获取到 {len(comments)} 条评论")
    except Exception as e:
        print(f"   ⚠️  获取评论失败: {e}")

    # --- Step 6: Save everything ---
    print("💾 保存数据...")
    out_dir = video_dir(bvid, creator)
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

    # --- Step 7: Summarize ---
    print("🤖 AI 总结中...")
    try:
        # Show which provider is being used
        from .summarizer import _detect_provider
        try:
            cfg = _detect_provider()
            print(f"   使用模型: {cfg['provider']} / {cfg['model']}")
        except Exception:
            pass

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
        )
        save_summary(bvid, summary, creator)
        print("   ✓ summary.md")
        print("\n" + "=" * 60)
        print(summary)
        print("=" * 60)
    except RuntimeError as e:
        print(f"   ⚠️  {e}")
        print(f"   数据已保存到 {out_dir}，可以稍后手动总结。")

    print(f"\n✅ 完成！所有文件已保存到: {out_dir}")


if __name__ == "__main__":
    main()
