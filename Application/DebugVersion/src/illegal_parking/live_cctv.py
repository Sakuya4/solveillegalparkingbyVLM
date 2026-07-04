from __future__ import annotations

import urllib.parse
import urllib.request
from html.parser import HTMLParser


class _MediaUrlParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        attr_map = {name.lower(): value for name, value in attrs if value}
        if tag.lower() in {"img", "source", "video", "iframe", "embed"} and "src" in attr_map:
            self._add(attr_map["src"])
        if tag.lower() == "param" and attr_map.get("name", "").lower() == "src" and "value" in attr_map:
            self._add(attr_map["value"])

    def _add(self, url: str) -> None:
        absolute = urllib.parse.urljoin(self.base_url, url)
        if absolute not in self.urls:
            self.urls.append(absolute)


def extract_media_urls(html: str, base_url: str) -> list[str]:
    parser = _MediaUrlParser(base_url)
    parser.feed(html)
    return parser.urls


def resolve_first_media_url(page_url: str, timeout_sec: float = 10.0) -> str | None:
    request = urllib.request.Request(
        page_url,
        headers={"User-Agent": "solveillegalparkingbyVLM live-cctv/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        html = response.read().decode("utf-8", errors="replace")
    urls = extract_media_urls(html, page_url)
    return urls[0] if urls else None
