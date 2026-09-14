"""Unit tests for SSRF security validation and article extraction."""

import pytest
from unittest.mock import patch, MagicMock
from src.article_extractor import (
    validate_url_security,
    extract_article_from_url,
    process_manual_input,
    ExtractedArticle,
)


def test_ssrf_blocks_localhost_and_loopback():
    """Ensures SSRF validator rejects localhost, 127.0.0.1, and loopback ranges."""
    is_safe, err = validate_url_security("http://localhost:8080/secret")
    assert not is_safe
    assert "blocked" in err.lower() or "forbidden" in err.lower()

    is_safe, err = validate_url_security("http://127.0.0.1/admin")
    assert not is_safe

    is_safe, err = validate_url_security("http://127.0.1.5/status")
    assert not is_safe


def test_ssrf_blocks_private_and_cloud_metadata():
    """Ensures private IP spaces (10.x, 192.168.x) and cloud metadata are blocked."""
    is_safe, err = validate_url_security("http://10.0.0.1/dashboard")
    assert not is_safe

    is_safe, err = validate_url_security("http://192.168.1.1/router")
    assert not is_safe

    is_safe, err = validate_url_security("http://169.254.169.254/latest/meta-data")
    assert not is_safe


def test_ssrf_rejects_non_http_schemes():
    """Ensures non-HTTP schemes like file://, ftp://, gopher:// are disallowed."""
    is_safe, err = validate_url_security("file:///etc/passwd")
    assert not is_safe
    assert "unsupported scheme" in err.lower()

    is_safe, err = validate_url_security("ftp://ftp.example.com/file")
    assert not is_safe


def test_ssrf_accepts_valid_public_domain():
    """Ensures public web addresses pass SSRF check."""
    is_safe, err = validate_url_security("https://www.reuters.com/world/news")
    assert is_safe
    assert err is None


@patch("src.article_extractor.safe_fetch_html")
def test_extract_article_success(mock_fetch):
    """Tests successful extraction using mocked HTML."""
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>India Launches Earth Observation Satellite - Reuters</title>
        <meta name="author" content="Jane Doe">
        <meta property="article:published_time" content="2026-09-10">
    </head>
    <body>
        <article>
            <h1>India Launches Earth Observation Satellite</h1>
            <p>The Indian space agency successfully launched a new earth observation satellite on Monday from Sriharikota.</p>
            <p>The mission aims to bolster disaster management and agricultural monitoring capabilities across the country.</p>
        </article>
    </body>
    </html>
    """
    mock_fetch.return_value = (sample_html, None)

    article = extract_article_from_url("https://www.reuters.com/article-123")
    assert article.success
    assert "India Launches" in article.title
    assert "observation satellite" in article.body.lower()
    assert article.source_domain == "www.reuters.com"
    assert article.word_count > 15


@patch("src.article_extractor.safe_fetch_html")
def test_extract_article_timeout_and_errors(mock_fetch):
    """Tests graceful handling when remote server times out or fails."""
    mock_fetch.return_value = (None, "Connection to https://example.com timed out.")

    article = extract_article_from_url("https://example.com/timeout")
    assert not article.success
    assert "timed out" in article.error_message


def test_process_manual_input():
    """Tests manual headline and body submission."""
    headline = "Scientists Announce Clean Fusion Milestone"
    body = "Researchers at the National Ignition Facility announced an energy net gain in a controlled fusion test on Friday."

    article = process_manual_input(headline, body)
    assert article.success
    assert article.title == headline
    assert "National Ignition Facility" in article.body
    assert article.source_domain == "manual_entry"
    assert article.word_count > 10

    # Empty input handling
    empty_article = process_manual_input("", "")
    assert not empty_article.success
    assert "empty" in empty_article.error_message
