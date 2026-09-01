from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class WebSearchError(RuntimeError):
    """A recoverable web-search failure."""


def _authority_score(url: str) -> int:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    authoritative = (
        "worldbank.org", "un.org", "who.int", "oecd.org", "imf.org",
        "europa.eu", "data.gov", "gov.br", "ibge.gov.br",
    )
    if any(host == domain or host.endswith(f".{domain}") for domain in authoritative):
        return 2
    if host.endswith(".gov") or ".gov." in host or host.endswith(".edu"):
        return 1
    return 0


def search_web(query: str, *, max_results: int = 5, timeout_seconds: int = 12) -> dict[str, Any]:
    clean_query = str(query or "").strip()
    if not clean_query:
        raise WebSearchError("A non-empty search query is required.")

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise WebSearchError("Web search is unavailable because TAVILY_API_KEY is not configured.")

    payload = json.dumps({
        "api_key": api_key,
        "query": clean_query,
        "include_answer": False,
        "search_depth": "advanced",
        "max_results": max(1, min(int(max_results), 5)),
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "DataPilot/1.0"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        raise WebSearchError(f"Web search provider failed: {exc}") from exc

    results: list[dict[str, str]] = []
    for item in body.get("results") or []:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = " ".join(str(item.get("content") or "").split())[:700]
        if title and url.startswith(("https://", "http://")):
            results.append({"title": title[:200], "url": url, "content": content})

    results.sort(key=lambda item: _authority_score(item["url"]), reverse=True)
    return {"query": clean_query, "results": results}
