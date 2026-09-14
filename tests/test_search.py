"""Unit tests for modular search providers and query generation."""

import pytest
from unittest.mock import patch, MagicMock
from src.search_provider import (
    generate_search_queries,
    MockSearchProvider,
    SearchResult,
    get_search_provider,
    TavilySearchProvider,
    DuckDuckGoSearchProvider,
)


def test_generate_search_queries():
    """Ensures search queries are concise and omit stop words."""
    claim = "India launched a new satellite on Monday to provide high speed internet."
    queries = generate_search_queries(claim)

    assert len(queries) >= 1
    # Check that common stop words are not clogging the primary query
    primary = queries[0].lower()
    assert "satellite" in primary
    assert "india" in primary
    assert "the" not in primary.split()


def test_mock_search_provider():
    """Verifies that MockSearchProvider returns deterministic structured results."""
    mock = MockSearchProvider()
    results = mock.search("India space launch", max_results=2)

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].source_domain == "reuters.com"
    assert results[0].provider == "mock"


def test_mock_search_provider_empty_and_custom():
    """Verifies predefined results in mock search provider."""
    custom_results = {
        "fusion test": [
            SearchResult(
                title="Fusion milestone achieved",
                url="https://nature.com/fusion",
                snippet="Lab achieved ignition.",
                source_domain="nature.com",
            )
        ]
    }
    mock = MockSearchProvider(predefined_results=custom_results)
    res = mock.search("fusion test")
    assert len(res) == 1
    assert res[0].source_domain == "nature.com"


@patch.dict("os.environ", {"SEARCH_PROVIDER": "tavily", "SEARCH_API_KEY": "test_tavily_key"})
def test_get_search_provider_tavily():
    """Verifies factory returns TavilySearchProvider when configured."""
    provider = get_search_provider()
    assert isinstance(provider, TavilySearchProvider)
    assert provider.get_provider_name() == "tavily"


@patch.dict("os.environ", {"SEARCH_PROVIDER": "duckduckgo", "SEARCH_API_KEY": ""})
def test_get_search_provider_duckduckgo():
    """Verifies factory defaults to DuckDuckGo when no keys set."""
    provider = get_search_provider()
    assert isinstance(provider, DuckDuckGoSearchProvider)
    assert provider.get_provider_name() == "duckduckgo"
