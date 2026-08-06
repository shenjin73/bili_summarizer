---
name: bili-summarize
description: >-
  Summarize a Bilibili video. Fetches subtitles (CC or Whisper), metadata,
  and comments, then generates an AI summary or provides a manual summary
  if the API is unavailable.
triggers:
  - Bilibili URL (bilibili.com/video/ or b23.tv)
  - BV number (BVxxxxxx)
  - "总结这个视频" / "summarize this video" with a Bilibili link
---

## Overview

Use the `bili_summarizer` CLI in `/Users/jin/Projects/bili_summarizer` to
fetch video content and generate a structured Chinese summary.

## Workflow

### Step 1 — Run the fetcher

```bash
cd /Users/jin/Projects/bili_summarizer
python3 -m bili_summarizer.main "<url_or_bvid>"
```

This fetches metadata, subtitles (CC first, Whisper fallback via `mlx-whisper`),
comments, and saves everything under `videos/<up主>/<bvid>/`.

The tool will:
- Print video metadata (title, uploader, duration, stats)
- Transcribe audio via Whisper if no CC subtitles are available (~30s–2min)
- Call the configured LLM to generate `summary.md`

### Step 2 — Present the result

If the LLM summary succeeds, read `summary.md` and present its content to
the user verbatim (it is already in polished Chinese Markdown).

If the LLM call fails (missing API key, rate limit, etc.), manually
summarize by reading the saved files:

1. `metadata.json` — title, stats, uploader info
2. `subtitles.txt` — full transcript (CC or Whisper)
3. `description.txt` — video description
4. `comments.json` — top comments

Then produce a summary covering: video overview, core topics, key arguments
with evidence, important data points, and overall assessment.

### Step 3 — Note limitations

- If subtitles came from Whisper, mention potential transcription errors
- If subtitles were unavailable entirely, clearly state that the summary is
  based on metadata/description/comments only

## Dependencies

- `requests` — Bilibili API calls with WBI signing
- `mlx-whisper` — Apple Silicon Whisper for audio transcription
- LLM backend (auto-detected from env vars): Anthropic, DeepSeek, MiniMax,
  OpenAI, or any OpenAI-compatible API
- API key configured in `/Users/jin/Projects/bili_summarizer/.env`

## Output location

```
videos/<up主名称>/<bvid>/
├── metadata.json
├── description.txt
├── subtitles.json     (CC raw data, if available)
├── subtitles.txt      (plain text transcript)
├── comments.json
└── summary.md
```
