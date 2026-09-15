"""候选来源的保守去重规则。

URL 表面不同并不总是不同来源：协议和主机大小写、默认端口、fragment 都不应造成
重复 Candidate。视频平台 ID 比 URL 更稳定，因此能取得平台 ID 时优先使用它。
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from .contracts import CandidateSource


def normalize_url(url: str) -> str:
    """规范化 URL 中不影响资源身份的表示差异。"""

    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    # IPv6 主机在 netloc 中必须保留方括号。
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    netloc = host
    # HTTP 80 和 HTTPS 443 是默认端口，去掉它们可合并等价 URL。
    if port is not None and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))


def candidate_identity(candidate: CandidateSource) -> str:
    """为 Candidate 生成带类型前缀的稳定身份键。"""

    if candidate.platform and candidate.platform_video_id:
        return f"video:{candidate.platform.casefold()}:{candidate.platform_video_id}"
    if candidate.candidate_type == "image" and candidate.image_url:
        return f"image:{normalize_url(candidate.image_url)}|{normalize_url(candidate.source_url)}"
    return f"url:{normalize_url(candidate.source_url)}"
