# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`bili_summarizer` fetches Bilibili video content (metadata, subtitles, comments) and generates AI-powered summaries. It supports CC subtitles natively and falls back to Whisper transcription when needed.

## Commands

```bash
# Summarize a video (CLI)
cd /Users/jin/Projects/bili_summarizer
python3 -m bili_summarizer.main <url_or_bvid>

# Install dependencies
pip install -r requirements.txt
# Whisper transcription also needs: pip install mlx-whisper
```

## Architecture

Pipeline: `main.py` → `fetcher.py` (API calls) → `transcriber.py` (Whisper fallback) → `summarizer.py` (LLM summary) → `storage.py` (save output)

- **`fetcher.py`** — All Bilibili API interactions. Signs requests with WBI keys. Returns raw dicts/lists; does not save files.
- **`transcriber.py`** — Runs `mlx-whisper` (Apple Silicon optimized) on downloaded audio. Auto-sets `HF_ENDPOINT=hf-mirror.com` for China access.
- **`summarizer.py`** — Multi-provider LLM backend (Anthropic/DeepSeek/MiniMax/OpenAI/custom). Auto-detects from env vars; can force via `LLM_PROVIDER`. Long Chinese system prompt defines the output format.
- **`storage.py`** — Saves everything under `videos/<creator>/<bvid>/`. Up主 name becomes the parent directory.
- **`main.py`** — Orchestrator. Fetches CC subtitles first; if unavailable, downloads audio and invokes Whisper. Then runs LLM summarization.

## API Key Configuration

Copy `.env.example` → `.env`. Provider auto-detection checks keys in order: Anthropic → DeepSeek → MiniMax → OpenAI → custom (`LLM_API_KEY` + `LLM_MODEL` + `LLM_BASE_URL`).

## Whisper

Uses `mlx-whisper` (requires Apple Silicon). Model: `mlx-community/whisper-medium`. Downloads from HuggingFace mirror by default. Set `HF_ENDPOINT=https://huggingface.co` to use official endpoint.
