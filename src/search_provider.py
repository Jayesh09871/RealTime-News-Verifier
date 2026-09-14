"""Modular search provider abstraction supporting Tavily, SerpAPI, Google CSE, DuckDuckGo/Live News, and Mock."""

from __future__ import annotations

import abc
import html
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, quote_plus
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source_domain: str
    published_date: Optional[str] = None
    provider: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_search_queries(claim: str, category: str = "general") -> List[str]:
    """Generates concise, factual search queries prioritizing key named entities, numbers, and verbs."""
    if not claim:
        return []

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might",
        "must", "can", "could", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "up", "about", "into", "over", "after", "it", "its", "that", "this",
        "these", "those", "their", "they", "we", "he", "she", "also", "and", "or", "but"
    }

    words = re.findall(r"\b[A-Za-z0-9\-\$\%\.\,]+\b", claim)
    meaningful = [w for w in words if w.lower() not in stop_words]

    if not meaningful:
        return [claim]

    # Prioritize entities: uppercase words, numbers/codes, capitalized words
    priority_words = []
    other_words = []
    for w in meaningful:
        if w.isupper() or any(c.isdigit() for c in w) or "-" in w or w[0].isupper():
            if w not in priority_words:
                priority_words.append(w)
        else:
            if w not in other_words:
                other_words.append(w)

    selected = (priority_words + other_words)[:7]
    direct_query = " ".join(selected)

    # Query 2: Focused news query
    focused_selected = (priority_words[:4] + other_words[:2])
    focused_query = f"{' '.join(focused_selected)} news" if focused_selected else direct_query

    queries = [direct_query]
    if focused_query != direct_query:
        queries.append(focused_query)

    return queries


class SearchProvider(abc.ABC):
    """Abstract base class for all live search providers."""

    @abc.abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Executes a search query and returns structured SearchResults."""
        pass

    @abc.abstractmethod
    def get_provider_name(self) -> str:
        """Returns the identifier name of this provider."""
        pass

    @abc.abstractmethod
    def check_health(self) -> tuple[bool, str]:
        """Performs a quick connectivity check to verify provider health."""
        pass


class TavilySearchProvider(SearchProvider):
    """Tavily Search API implementation (specialized for AI agents & factual search)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEARCH_API_KEY", "")

    def get_provider_name(self) -> str:
        return "tavily"

    def check_health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Tavily API key not configured (SEARCH_API_KEY is empty)."
        try:
            # Low cost test search
            res = self.search("test", max_results=1)
            return True, "Tavily API operational."
        except Exception as e:
            return False, f"Tavily connection error: {e}"

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            raise ValueError("Tavily API key not provided.")

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_domains": [],
            "exclude_domains": [],
            "max_results": max_results,
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"Tavily API returned status {resp.status_code}: {resp.text}")

        data = resp.json()
        results = []
        for item in data.get("results", []):
            item_url = item.get("url", "")
            domain = urlparse(item_url).netloc.lower()
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item_url,
                    snippet=item.get("content", ""),
                    source_domain=domain,
                    published_date=item.get("published_date"),
                    provider=self.get_provider_name(),
                )
            )
        return results


class SerpApiSearchProvider(SearchProvider):
    """SerpAPI provider for Google News and Search."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEARCH_API_KEY", "")

    def get_provider_name(self) -> str:
        return "serpapi"

    def check_health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "SerpAPI key not configured."
        return True, "SerpAPI configured."

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            raise ValueError("SerpAPI key is missing.")

        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google_news",
            "num": max_results,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"SerpAPI returned status {resp.status_code}: {resp.text}")

        data = resp.json()
        results = []
        for item in data.get("news_results", [])[:max_results]:
            item_url = item.get("link", "")
            domain = urlparse(item_url).netloc.lower()
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item_url,
                    snippet=item.get("snippet", ""),
                    source_domain=domain,
                    published_date=item.get("date"),
                    provider=self.get_provider_name(),
                )
            )
        return results


class GoogleCustomSearchProvider(SearchProvider):
    """Google Custom Search JSON API provider."""

    def __init__(self, api_key: Optional[str] = None, cse_id: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEARCH_API_KEY", "")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID", "")

    def get_provider_name(self) -> str:
        return "google"

    def check_health(self) -> tuple[bool, str]:
        if not self.api_key or not self.cse_id:
            return False, "Google API Key or CSE ID missing."
        return True, "Google CSE configured."

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key or not self.cse_id:
            raise ValueError("Google API Key or CSE ID is missing.")

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(max_results, 10),
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"Google CSE returned status {resp.status_code}: {resp.text}")

        data = resp.json()
        results = []
        for item in data.get("items", []):
            item_url = item.get("link", "")
            domain = urlparse(item_url).netloc.lower()
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item_url,
                    snippet=item.get("snippet", ""),
                    source_domain=domain,
                    published_date=None,
                    provider=self.get_provider_name(),
                )
            )
        return results


class DuckDuckGoSearchProvider(SearchProvider):
    """Free live web and news search provider using Google News RSS & DuckDuckGo.

    Requires NO API keys, providing reliable real-time evidence out-of-the-box.
    """

    def get_provider_name(self) -> str:
        return "duckduckgo"

    def check_health(self) -> tuple[bool, str]:
        try:
            # Test connectivity to news feed
            test_url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(test_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if resp.status_code == 200:
                return True, "Live News RSS & Search endpoint operational."
            return False, f"HTTP {resp.status_code} from search endpoint."
        except Exception as e:
            return False, f"Live Search connection failed: {e}"

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        results: List[SearchResult] = []
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

        # 1. Query Live Google News RSS (zero-key live news evidence)
        try:
            rss_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(rss_url, headers=headers, timeout=8)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:max_results]:
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    pub_elem = item.find("pubDate")
                    desc_elem = item.find("description")
                    source_elem = item.find("source")

                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    raw_link = link_elem.text if link_elem is not None and link_elem.text else ""
                    pub_date = pub_elem.text if pub_elem is not None else None
                    desc = desc_elem.text if desc_elem is not None and desc_elem.text else ""

                    # Clean description html tags
                    clean_desc = re.sub(r"<[^>]+>", " ", desc)
                    clean_desc = html.unescape(clean_desc).strip()

                    # Extract source domain
                    source_name = source_elem.text if source_elem is not None and source_elem.text else ""
                    domain = ""
                    if source_elem is not None and source_elem.get("url"):
                        domain = urlparse(source_elem.get("url")).netloc.lower()
                    if not domain and raw_link:
                        domain = urlparse(raw_link).netloc.lower()
                    if not domain:
                        domain = source_name.lower().replace(" ", "") + ".com" if source_name else "news.google.com"

                    results.append(
                        SearchResult(
                            title=title,
                            url=raw_link or f"https://{domain}",
                            snippet=clean_desc or title,
                            source_domain=domain,
                            published_date=pub_date,
                            provider=self.get_provider_name(),
                        )
                    )
        except Exception as e:
            logger.warning(f"News RSS search failed for '{query}': {e}")

        # If we got results, return them up to max_results
        if results:
            return results[:max_results]

        # 2. Fallback: DuckDuckGo HTML search
        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            ddg_resp = requests.post(ddg_url, data={"q": query}, headers=headers, timeout=8)
            if ddg_resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(ddg_resp.text, "html.parser")
                web_results = soup.find_all("div", class_="result")
                for r in web_results[:max_results]:
                    title_tag = r.find("a", class_="result__a")
                    snippet_tag = r.find("a", class_="result__snippet")
                    if title_tag:
                        title = title_tag.get_text().strip()
                        link = title_tag.get("href", "")
                        snippet = snippet_tag.get_text().strip() if snippet_tag else ""
                        domain = urlparse(link).netloc.lower()
                        results.append(
                            SearchResult(
                                title=title,
                                url=link,
                                snippet=snippet,
                                source_domain=domain,
                                published_date=None,
                                provider=self.get_provider_name(),
                            )
                        )
        except Exception as e:
            logger.warning(f"DuckDuckGo HTML search fallback failed: {e}")

        return results[:max_results]


class MockSearchProvider(SearchProvider):
    """In-memory deterministic mock search provider for unit testing & offline demos."""

    def __init__(self, predefined_results: Optional[Dict[str, List[SearchResult]]] = None):
        self.predefined_results = predefined_results or {}
        self.is_healthy = True

    def get_provider_name(self) -> str:
        return "mock"

    def check_health(self) -> tuple[bool, str]:
        if self.is_healthy:
            return True, "Mock search provider operational."
        return False, "Mock provider forced unavailable."

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        # Return matched query if present
        query_lower = query.lower()
        for key, res in self.predefined_results.items():
            if key.lower() in query_lower or query_lower in key.lower():
                return res[:max_results]

        # Default realistic mock responses for testing
        return [
            SearchResult(
                title=f"Authoritative Report: {query}",
                url="https://www.reuters.com/world/article-test",
                snippet=f"Official government and institutional sources confirm events regarding: {query}.",
                source_domain="reuters.com",
                published_date="2026-09-12",
                provider="mock",
            ),
            SearchResult(
                title=f"BBC News Coverage on {query}",
                url="https://www.bbc.com/news/world-test",
                snippet=f"Comprehensive investigation corroborates details concerning {query}.",
                source_domain="bbc.com",
                published_date="2026-09-11",
                provider="mock",
            ),
        ][:max_results]


def get_search_provider(name: Optional[str] = None) -> SearchProvider:
    """Factory function to instantiate the configured SearchProvider."""
    provider_name = (name or os.getenv("SEARCH_PROVIDER", "duckduckgo")).lower().strip()
    api_key = os.getenv("SEARCH_API_KEY", "").strip()

    if provider_name == "tavily":
        if api_key:
            return TavilySearchProvider(api_key=api_key)
        logger.warning("SEARCH_PROVIDER is tavily but SEARCH_API_KEY is empty; falling back to duckduckgo.")
        return DuckDuckGoSearchProvider()

    elif provider_name == "serpapi":
        if api_key:
            return SerpApiSearchProvider(api_key=api_key)
        logger.warning("SEARCH_PROVIDER is serpapi but SEARCH_API_KEY is empty; falling back to duckduckgo.")
        return DuckDuckGoSearchProvider()

    elif provider_name == "google":
        cse_id = os.getenv("GOOGLE_CSE_ID", "")
        if api_key and cse_id:
            return GoogleCustomSearchProvider(api_key=api_key, cse_id=cse_id)
        logger.warning("SEARCH_PROVIDER is google but keys missing; falling back to duckduckgo.")
        return DuckDuckGoSearchProvider()

    elif provider_name == "mock":
        return MockSearchProvider()

    # Default provider: DuckDuckGo / Live News RSS (zero-key live search)
    return DuckDuckGoSearchProvider()
