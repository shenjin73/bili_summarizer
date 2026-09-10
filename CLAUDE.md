# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`bili_summarizer` fetches Bilibili video content (metadata, subtitles, comments) and generates AI-powered summaries. It supports CC subtitles natively and falls back to Whisper transcription when needed. It also analyzes video frames: screenshots every 10s, perceptual-hash dedup of repeated frames, and vision-LLM OCR to capture on-screen text/charts that speech alone misses.

## Commands

```bash
# Summarize a video (CLI), from the repo root
cd bili_summarizer
python3 -m bili_summarizer.main <url_or_bvid>

# Install dependencies
pip install -r requirements.txt
# Whisper transcription also needs: pip install mlx-whisper
```

## Architecture

Pipeline: `main.py` → `fetcher.py` (API calls) → `transcriber.py` (Whisper fallback) + `frames.py` (screenshots/OCR) → `summarizer.py` (LLM summary) → `storage.py` (save output)

- **`fetcher.py`** — All Bilibili API interactions. Signs requests with WBI keys. Returns raw dicts/lists; does not save files. `fetch_audio_url` / `fetch_video_url` pick the best DASH stream; `download_stream` saves it.
- **`transcriber.py`** — Runs `mlx-whisper` (Apple Silicon optimized) on downloaded audio. Default model `large-v3-turbo` (`WHISPER_MODEL` env override). Auto-sets `HF_ENDPOINT=hf-mirror.com` for China access. `_sanitize_no_proxy()` drops no_proxy entries that URL parsers cannot handle (bare/bracketed IPv6 such as `::1` / `[::1]`); otherwise the HuggingFace fetch aborts with `ValueError: Invalid port: ':1]'` and transcription dies even when the weights are already cached.
- **`frames.py`** — Frame analysis: `extract_frames` (ffmpeg, one frame per N seconds via `fps=1/N,showinfo`, timestamps parsed from stderr), `dedupe_frames` (pure-PIL dHash, keeps a frame only if its Hamming distance from the last kept frame exceeds `FRAME_DEDUP_THRESHOLD`), `ocr_frames` (sends each unique JPEG as base64 to a vision LLM via the OpenAI SDK, **concurrently** via ThreadPoolExecutor — `OCR_CONCURRENCY` env, default 2 for local services / 6 for cloud; pre-checks the endpoint and fails fast when offline), `compress_ocr_results` (cross-frame dedup of OCR text — drops near-duplicate frames and already-seen lines; applied only to the summary prompt, the full OCR text is saved to disk). OCR model config: `VISION_API_KEY/MODEL/BASE_URL`, falling back to `LOCAL_*`.
- **`summarizer.py`** — Multi-provider LLM backend (local chain/Anthropic/DeepSeek/MiniMax/OpenAI/custom). `local_candidates()` builds the local chain from `LOCAL_BASE_URLS` + `LOCAL_MODELS` + `LOCAL_API_KEYS` (comma-separated, positional; the legacy singular `LOCAL_BASE_URL`/`LOCAL_MODEL`/`LOCAL_API_KEY` still works for a single endpoint; plural wins when both are set). `resolve_provider()` health-checks local endpoints in order (`GET /models` with the endpoint's own key, 5s timeout; DeepSeek requires the Bearer key even for `/models`), then falls back to DeepSeek. Can force via `LLM_PROVIDER` (`local` = chain only, never cloud). Long Chinese system prompt defines the output format. Accepts optional `on_screen_text` (timestamped OCR) merged into the prompt. Prompt sections are capped independently (`SUMMARY_MAX_DESC/SUBTITLE/OCR/COMMENT_CHARS`, defaults 4k/50k/15k/3k chars, head + tail kept) so oversized subtitles never chop off the OCR / comments sections; a final 80k-char safety cut remains. `ocr_frames` in `frames.py` additionally retries the failed frames once on the next available vision provider when a connection-level failure hits more than half of a batch.
- **`storage.py`** — Saves everything under `videos/<creator>/<bvid>/`. Up主 name becomes the parent directory. Includes `save_ocr_text` / `save_frames_manifest`.
- **`main.py`** — Orchestrator. Fetches CC subtitles first; if unavailable, downloads audio and invokes Whisper. Then frame analysis (skippable with `--no-frames`, interval via `--interval`). Finally runs LLM summarization.

## API Key Configuration

Copy `.env.example` → `.env`. Provider auto-detection order: **local** (`LOCAL_MODEL` + `LOCAL_BASE_URL`) → DeepSeek → other cloud → custom (`LLM_API_KEY` + `LLM_MODEL` + `LLM_BASE_URL`).

The current `.env` uses a **local** model via LM Studio (`LOCAL_MODEL=qwen3.8-27b-mlx`, `LOCAL_BASE_URL=http://127.0.0.1:1234/v1`, placeholder key) — a vision-capable model serving both summarization and frame OCR. **Automatic fallback**: `resolve_provider()` in `summarizer.py` health-checks the local endpoint (`GET /models`, 5s timeout); if it is offline or the model is not loaded, it switches to DeepSeek when `DEEPSEEK_API_KEY` is set (and prints a warning). A mid-call connection failure on local retries on DeepSeek. The DeepSeek default model is `deepseek-flash` (vision-capable), so frame OCR falls back to it too via `resolve_vision_provider()` in `frames.py` (local → DeepSeek vision → skip OCR only when both are down; summary then proceeds on subtitles). When every candidate fails, the raised error reports **both** the local and the cloud-side reason (`check_endpoint` distinguishes `unreachable` from `model_missing`), so a renamed/retired cloud model is no longer masked by a local-offline message. The resolution is cached per process so warnings print once. Frame-analysis knobs: `FRAME_INTERVAL` (default 10s), `FRAME_DEDUP_THRESHOLD` (dHash Hamming distance, default 10), `FRAME_QN` (stream quality, default 64=720p), `FRAME_MAX_OCR` (default 80), `OCR_CONCURRENCY` (default 2 local / 6 cloud).

Resume behavior: Whisper output is flushed to `subtitles.txt` immediately after transcription; a re-run of the same video reuses a non-empty `subtitles.txt` and skips transcription. `main.py` prints timestamped per-stage timing lines (`[HH:MM:SS] ▶/✓ stage（Ns）`) plus a total at the end.

Note: the `.env` loader in `__init__.py` strips trailing inline comments from unquoted values; keep `.env` values free of non-ASCII (API keys go into HTTP headers).

## Whisper

Uses `mlx-whisper` (requires Apple Silicon). Model: `mlx-community/whisper-large-v3-turbo` by default (`WHISPER_MODEL` env override). Downloads from HuggingFace mirror by default. Set `HF_ENDPOINT=https://huggingface.co` to use official endpoint.
