from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class WebSearchError(RuntimeError):
    """A recoverable web-search failure."""


def _page_excerpt(raw_content: str, query: str) -> str:
    stop_words = {"the", "and", "for", "what", "which", "who", "was", "are", "with", "this", "from"}
    terms = set(re.findall(r"\b\w{3,}\b", query.casefold())) - stop_words
    if terms & {"won", "win", "winner", "champion", "result"}:
        terms.update(("won", "winner", "winners", "defeated", "beat", "champion", "champions", "final"))
    raw_content = re.sub(r"!\[[^\]]*\]\([^\n)]*\)", "", raw_content)
    raw_content = re.sub(r"\[([^\]\n]+)\]\([^\n)]*\)", r"\1", raw_content)
    passages = [
        " ".join(part.split()) for part in re.split(r"\n+|(?<=[.!?])\s+", raw_content)
    ]
    passages = list(dict.fromkeys(part for part in passages if len(part) >= 45))
    passages.sort(key=lambda part: len(terms & set(re.findall(r"\b\w{3,}\b", part.casefold()))), reverse=True)
    return "\n".join(part[:650] for part in passages[:3])


def search_web(query: str, *, max_results: int = 5, timeout_seconds: int = 18) -> dict[str, Any]:
    clean_query = " ".join(str(query or "").split())[:400]
    if not clean_query:
        raise WebSearchError("A non-empty search query is required.")

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise WebSearchError("Web search is unavailable because TAVILY_API_KEY is not configured.")

    payload = json.dumps({
        "query": clean_query,
        "include_answer": False,
        "include_raw_content": "markdown",
        "include_published_date": True,
        "chunks_per_source": 3,
        "search_depth": "advanced",
        "max_results": max(1, min(int(max_results), 5)),
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "DataPilot/1.0"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise WebSearchError(f"Tavily returned HTTP {exc.code}.") from exc
    except (OSError, ValueError) as exc:
        raise WebSearchError(f"Tavily could not return search results ({type(exc).__name__}).") from exc

    results: list[dict[str, str]] = []
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise WebSearchError("Tavily returned an unreadable search result.")
    seen_urls: set[str] = set()
    for item in body["results"]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip().split("#", 1)[0]
        try:
            parsed_url = urllib.parse.urlsplit(url)
        except ValueError:
            continue
        if not title or parsed_url.scheme not in ("https", "http") or not parsed_url.hostname or parsed_url.username is not None or url in seen_urls:
            continue
        snippet = " ".join(str(item.get("content") or "").split())[:2000]
        excerpt = _page_excerpt(str(item.get("raw_content") or ""), clean_query)
        if not snippet and not excerpt:
            continue
        seen_urls.add(url)
        results.append({
            "title": title[:200], "url": url, "content": snippet,
            "excerpt": excerpt, "published_date": str(item.get("published_date") or "")[:32],
        })

    return {"query": clean_query, "results": results}
