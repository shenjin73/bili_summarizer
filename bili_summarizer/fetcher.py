"""Bilibili API client for fetching video metadata, subtitles, and comments."""

import hashlib
import re
import time
import urllib.parse
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_VIDEO_INFO = "https://api.bilibili.com/x/web-interface/view"
API_PLAYER_V2 = "https://api.bilibili.com/x/player/v2"
API_COMMENTS = "https://api.bilibili.com/x/v2/reply/main"
API_SUBTITLE_LIST = "https://api.bilibili.com/x/player/v2"
API_SUBTITLE_SOS = "https://api.bilibili.com/x/player/sos/v3"
API_PLAYURL = "https://api.bilibili.com/x/player/playurl"

MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 52, 44, 34,
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_bvid(url: str) -> str:
    """Extract BV id from a Bilibili URL like /video/BV1CNLQ6REu5/"""
    m = re.search(r"(BV[0-9A-Za-z]{10})", url)
    if not m:
        raise ValueError(f"Cannot extract BVID from URL: {url}")
    return m.group(1)


def _get_mixin_key() -> tuple[str, str]:
    """Fetch the WBI mixin key needed for signed API requests."""
    resp = requests.get(
        "https://api.bilibili.com/x/web-interface/nav",
        headers=HEADERS,
        timeout=10,
    )
    data = resp.json().get("data", {})
    img_url = data.get("wbi_img", {}).get("img_url", "")
    sub_url = data.get("wbi_img", {}).get("sub_url", "")

    if not img_url or not sub_url:
        return "", ""

    # Extract the key segments from the CDN URLs
    img_key = img_url.rsplit("/", 1)[-1].split(".")[0]
    sub_key = sub_url.rsplit("/", 1)[-1].split(".")[0]
    raw = img_key + sub_key
    mixin = "".join(raw[i] for i in MIXIN_KEY_ENC_TAB if i < len(raw))[:32]
    return mixin, raw


def _wbi_sign(params: dict, mixin_key: str) -> dict:
    """Add w_rid and wts to params using WBI signing."""
    params["wts"] = int(time.time())
    # Sort by key
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    query = urllib.parse.urlencode(sorted_params)
    sign_str = query + mixin_key
    params["w_rid"] = hashlib.md5(sign_str.encode()).hexdigest()
    return params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_video_meta(bvid: str) -> dict:
    """Fetch basic video metadata (title, description, author, stats, cid).

    Returns the full ``data`` dict from the API response.
    """
    resp = requests.get(
        API_VIDEO_INFO,
        params={"bvid": bvid},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        raise RuntimeError(f"API error: {body.get('message', 'unknown')}")
    return body["data"]


def fetch_subtitles(bvid: str, cid: int) -> Optional[list[dict]]:
    """Fetch subtitle list for a video.

    Returns a list of subtitle dicts (each with ``lan``, ``subtitle_url``, etc.)
    or ``None`` when no subtitles are available.
    """
    mixin_key, _ = _get_mixin_key()
    params = _wbi_sign({"bvid": bvid, "cid": cid}, mixin_key)

    resp = requests.get(
        API_SUBTITLE_LIST,
        params=params,
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        return None

    subtitle_info = body.get("data", {}).get("subtitle", {})
    subtitles = subtitle_info.get("subtitles", [])
    if not subtitles:
        return None
    return subtitles


def fetch_subtitle_content(subtitle_url: str) -> list[dict]:
    """Download and parse a subtitle JSON file.

    Each entry: ``{"from": float, "to": float, "content": str}``
    """
    if subtitle_url.startswith("//"):
        subtitle_url = "https:" + subtitle_url
    resp = requests.get(subtitle_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json().get("body", [])


def fetch_comments(bvid: str, page: int = 1, count: int = 20) -> list[dict]:
    """Fetch top-level comments for a video.

    Each comment dict has ``content``, ``like``, ``member``, ``ctime``, etc.
    """
    resp = requests.get(
        API_COMMENTS,
        params={"oid": bvid, "type": 1, "mode": 3, "pagestr": f"pn={page}&ps={count}"},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        return []
    return body.get("data", {}).get("replies", [])


def fetch_audio_url(bvid: str, cid: int) -> str | None:
    """Fetch the best-quality DASH audio stream URL for a video.

    Returns the download URL (str) or ``None`` if unavailable.
    """
    mixin_key, _ = _get_mixin_key()
    params = _wbi_sign(
        {"bvid": bvid, "cid": cid, "qn": 64, "fnver": 0, "fnval": 4048, "fourk": 1},
        mixin_key,
    )
    resp = requests.get(API_PLAYURL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        return None

    dash = body.get("data", {}).get("dash", {})
    audio_items = dash.get("audio", [])
    if not audio_items:
        return None

    # Pick the highest quality audio (largest bandwidth / last item is usually best)
    best = max(audio_items, key=lambda a: a.get("bandwidth", 0))
    url = best.get("baseUrl") or best.get("base_url") or best.get("url", "")
    return url if url else None


def download_audio(audio_url: str, output_path: "Path") -> "Path":
    """Download an audio stream to *output_path*.

    Uses appropriate headers to avoid 403 from CDN.  Returns the output path.
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dl_headers = {
        **HEADERS,
        "Origin": "https://www.bilibili.com",
        "Accept": "*/*",
    }

    resp = requests.get(audio_url, headers=dl_headers, timeout=120, stream=True)
    resp.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return output_path
