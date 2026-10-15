"""Is a SEARCH Candidate a platform video? (10-13)

A reverse image search often returns the page that *embeds* a video (a news article, a blog post) rather than the
video page itself. SEARCH's opener follows it with yt-dlp and records the platform video as the Candidate's final_url
(evidM27a video_194: an inspiremore.com article → youtube.com/watch?v=AcMmLmvOYTo, the original). Such a Candidate
keeps candidate_type="page", and INSPECT used to refuse to download it ("not a video candidate"), so neither the
SEARCH content_match nor compare_media could ever reach the original.

Shared by SEARCH (source_match), INSPECT (service) and the HI workspace hints; no imports from either package.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_YOUTUBE_VIDEO = re.compile(r"^https://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/)|youtu\.be/)"
                            r"[A-Za-z0-9_-]{11}")


def is_platform_video_url(url: Any) -> bool:
    return isinstance(url, str) and bool(_YOUTUBE_VIDEO.match(url))


def is_video_candidate(candidate: Mapping[str, Any]) -> bool:
    """candidate_type is video, or SEARCH resolved the page to a YouTube video page (final_url)."""

    if candidate.get("candidate_type") == "video":
        return True
    return is_platform_video_url(candidate.get("final_url"))


__all__ = ["is_platform_video_url", "is_video_candidate"]
