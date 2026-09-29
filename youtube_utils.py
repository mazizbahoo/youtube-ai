"""YouTube helpers: parse links, fetch transcripts, search videos, pull comments."""
import os
import re

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi

load_dotenv()

# A video ID is always 11 characters from [A-Za-z0-9_-].
# Matches watch?v=ID, youtu.be/ID, /shorts/ID and /embed/ID.
VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)

_yt = None


def _client():
    """Build the YouTube Data API client once and reuse it."""
    global _yt
    if _yt is None:
        _yt = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
    return _yt


def get_video_id(url):
    """Return the 11-char video ID from a YouTube link, or None if it isn't one."""
    if not url:
        return None
    match = VIDEO_ID_RE.search(url.strip())
    return match.group(1) if match else None


def get_transcript(video_id, languages=("en",)):
    """Return the transcript as one string, or None if the video has no captions.

    Prefers the given languages; otherwise falls back to whatever language exists.
    """
    try:
        transcripts = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = transcripts.find_transcript(list(languages))
        except NoTranscriptFound:
            transcript = next(iter(transcripts))  # any language is better than nothing
        fetched = transcript.fetch()
    except Exception as e:
        print(f"[get_transcript] {type(e).__name__}")
        return None
    return " ".join(snippet.text for snippet in fetched)


def search_videos(query, n=25):
    """Search YouTube (costs 100 quota units) and return a DataFrame of results."""
    res = _client().search().list(
        q=query, part="snippet", type="video", maxResults=n
    ).execute()
    rows = []
    for item in res.get("items", []):
        snip = item["snippet"]
        rows.append({
            "video_id": item["id"]["videoId"],
            "title": snip["title"],
            "description": snip["description"],
            "channel": snip["channelTitle"],
            "thumbnail": snip["thumbnails"]["medium"]["url"],
        })
    return pd.DataFrame(rows)


def get_video_info(video_id):
    """Return {'title', 'description', 'channel'} for one video, or None if not found."""
    res = _client().videos().list(part="snippet", id=video_id).execute()
    items = res.get("items", [])
    if not items:
        return None
    snip = items[0]["snippet"]
    return {
        "title": snip["title"],
        "description": snip["description"],
        "channel": snip["channelTitle"],
    }


def get_comments(video_id, max_comments=200):
    """Return up to max_comments top-level comments, or None if comments are disabled."""
    comments = []
    page_token = None
    try:
        while len(comments) < max_comments:
            res = _client().commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                order="relevance", textFormat="plainText", pageToken=page_token,
            ).execute()
            for item in res.get("items", []):
                comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            page_token = res.get("nextPageToken")
            if not page_token:
                break
    except HttpError as e:
        # Don't print `e` itself: its message includes the request URL with the API key.
        print(f"[get_comments] HttpError {e.status_code}: {e.reason}")
        return None
    return comments[:max_comments]
