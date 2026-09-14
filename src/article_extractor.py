"""SSRF-protected article extraction module using trafilatura and BeautifulSoup fallback."""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
import trafilatura
from trafilatura.settings import use_config

from src.preprocess import clean_article_text

logger = logging.getLogger(__name__)

# Security & Limits
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 RealTimeNewsVerifier/1.0"


@dataclass
class ExtractedArticle:
    title: str
    body: str
    author: Optional[str]
    publication_date: Optional[str]
    source_domain: str
    canonical_url: str
    word_count: int
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SSRFValidationError(ValueError):
    """Raised when a URL targets a private, loopback, or forbidden network address."""
    pass


def validate_url_security(url: str) -> tuple[bool, Optional[str]]:
    """Validates URL scheme, syntax, and guards against SSRF attacks.

    Blocks:
    - Non-HTTP/HTTPS schemes (e.g. file://, gopher://, ftp://)
    - Localhost and loopback IPs (127.0.0.0/8, ::1)
    - Private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, fc00::/7)
    - Link-local and metadata endpoints (169.254.0.0/16, fe80::/10)
    - Broadcast/reserved ranges
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty."

    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Malformed URL: {e}"

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Unsupported scheme '{parsed.scheme}'. Only http:// and https:// URLs are allowed."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL does not contain a valid hostname."

    hostname_lower = hostname.lower()

    # Disallow common local keywords immediately
    forbidden_names = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"}
    if hostname_lower in forbidden_names or hostname_lower.endswith(".local") or hostname_lower.endswith(".internal"):
        return False, f"Access to local or private host '{hostname}' is blocked for security reasons."

    # DNS Resolution & IP Range Check
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return False, f"Cannot resolve domain '{hostname}': {e}"
    except Exception as e:
        return False, f"DNS resolution error for '{hostname}': {e}"

    for entry in addr_info:
        ip_str = entry[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            if ip_obj.is_loopback:
                return False, f"Loopback address '{ip_str}' is blocked."
            if ip_obj.is_private:
                return False, f"Private IP address '{ip_str}' is blocked."
            if ip_obj.is_link_local:
                return False, f"Link-local address '{ip_str}' is blocked."
            if ip_obj.is_reserved:
                return False, f"Reserved IP address '{ip_str}' is blocked."
            if ip_obj.is_multicast:
                return False, f"Multicast address '{ip_str}' is blocked."
            # Check AWS/GCP/Azure link-local metadata (169.254.169.254)
            if str(ip_obj) == "169.254.169.254":
                return False, "Cloud metadata IP address is strictly blocked."
        except ValueError:
            return False, f"Invalid IP address returned for host: {ip_str}"

    return True, None


def safe_fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[Optional[str], Optional[str]]:
    """Safely fetches HTML content with size limits and redirect SSRF verification."""
    is_valid, error = validate_url_security(url)
    if not is_valid:
        return None, error

    session = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}

    current_url = url
    try:
        # Stream response to strictly enforce byte limit without loading huge payloads into memory
        with session.get(current_url, headers=headers, timeout=timeout, stream=True, allow_redirects=False) as resp:
            # Handle manual redirect validation to protect against redirect-based SSRF
            redirect_count = 0
            while resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                redirect_count += 1
                if redirect_count > 5:
                    return None, "Too many HTTP redirects."
                redirect_url = resp.headers.get("Location")
                if not redirect_url:
                    return None, "Redirect location header missing."
                redirect_url = urljoin(current_url, redirect_url)
                is_safe, err = validate_url_security(redirect_url)
                if not is_safe:
                    return None, f"Redirect to unsafe URL blocked: {err}"
                current_url = redirect_url
                resp = session.get(current_url, headers=headers, timeout=timeout, stream=True, allow_redirects=False)

            if resp.status_code == 403:
                return None, "Access forbidden (HTTP 403). The website may be protected by a paywall or bot protection."
            if resp.status_code == 404:
                return None, "Article not found (HTTP 404)."
            if resp.status_code >= 400:
                return None, f"HTTP error {resp.status_code} occurred while fetching article."

            # Check content length header if provided
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                return None, f"Response size exceeds maximum allowed limit of {max_bytes / (1024*1024):.1f} MB."

            # Read up to max_bytes + 1
            chunks = []
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=16384):
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    return None, f"Response content exceeded maximum limit of {max_bytes / (1024*1024):.1f} MB."
                chunks.append(chunk)

            raw_bytes = b"".join(chunks)
            encoding = resp.encoding or resp.apparent_encoding or "utf-8"
            try:
                html_text = raw_bytes.decode(encoding, errors="replace")
            except Exception:
                html_text = raw_bytes.decode("utf-8", errors="replace")

            return html_text, None

    except requests.exceptions.Timeout:
        return None, f"Connection to {url} timed out after {timeout} seconds."
    except requests.exceptions.SSLError as e:
        return None, f"SSL/TLS certificate error: {e}"
    except requests.exceptions.ConnectionError as e:
        return None, f"Network connection failed: {e}"
    except Exception as e:
        return None, f"Unexpected error while fetching URL: {e}"


def extract_article_from_url(url: str) -> ExtractedArticle:
    """Extracts article content and metadata from a URL safely."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower() if parsed.netloc else "unknown"

    html_content, fetch_error = safe_fetch_html(url)
    if fetch_error or not html_content:
        return ExtractedArticle(
            title="",
            body="",
            author=None,
            publication_date=None,
            source_domain=domain,
            canonical_url=url,
            word_count=0,
            success=False,
            error_message=fetch_error or "Empty response received from web server.",
        )

    # 1. Primary extraction via trafilatura
    try:
        traf_config = use_config()
        traf_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "5")
        traf_result = trafilatura.bare_extraction(
            html_content,
            url=url,
            include_comments=False,
            include_tables=False,
            config=traf_config,
        )
    except Exception as e:
        logger.warning(f"Trafilatura extraction threw exception: {e}")
        traf_result = None

    if traf_result and traf_result.text and len(traf_result.text.strip()) > 50:
        cleaned_body = clean_article_text(traf_result.text)
        title = traf_result.title or ""
        author = traf_result.author
        date = traf_result.date
        canonical = traf_result.url or url

        if not title:
            # Fallback title extraction
            soup = BeautifulSoup(html_content, "html.parser")
            title_tag = soup.find("title") or soup.find("h1")
            title = title_tag.get_text().strip() if title_tag else "Untitled Article"

        words = cleaned_body.split()
        return ExtractedArticle(
            title=clean_article_text(title),
            body=cleaned_body,
            author=author,
            publication_date=date,
            source_domain=domain,
            canonical_url=canonical,
            word_count=len(words),
            success=True,
            error_message=None,
        )

    # 2. Fallback extraction via BeautifulSoup
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()

        # Extract author
        author = None
        author_meta = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", property="article:author")
        if author_meta and author_meta.get("content"):
            author = author_meta["content"].strip()

        # Extract date
        pub_date = None
        date_meta = (
            soup.find("meta", property="article:published_time")
            or soup.find("meta", attrs={"name": "pubdate"})
            or soup.find("meta", attrs={"name": "publish-date"})
        )
        if date_meta and date_meta.get("content"):
            pub_date = date_meta["content"].strip()

        # Canonical URL
        canonical_link = soup.find("link", rel="canonical")
        canonical = canonical_link["href"] if canonical_link and canonical_link.get("href") else url

        # Remove scripts, styles, forms, navs
        for el in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            el.decompose()

        # Target article container or main
        article_el = soup.find("article") or soup.find("main") or soup.find("div", class_=lambda c: c and "story" in c.lower())
        if article_el:
            paragraphs = [p.get_text().strip() for p in article_el.find_all("p") if len(p.get_text().strip()) > 20]
        else:
            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 20]

        raw_body = " ".join(paragraphs)
        cleaned_body = clean_article_text(raw_body)
        word_count = len(cleaned_body.split())

        if word_count < 15:
            return ExtractedArticle(
                title=title or "Untitled Article",
                body="",
                author=author,
                publication_date=pub_date,
                source_domain=domain,
                canonical_url=canonical,
                word_count=0,
                success=False,
                error_message="Could not extract substantial article text. The page may be paywalled, JavaScript-heavy, or empty.",
            )

        return ExtractedArticle(
            title=clean_article_text(title) if title else "Untitled Article",
            body=cleaned_body,
            author=author,
            publication_date=pub_date,
            source_domain=domain,
            canonical_url=canonical,
            word_count=word_count,
            success=True,
            error_message=None,
        )
    except Exception as e:
        return ExtractedArticle(
            title="",
            body="",
            author=None,
            publication_date=None,
            source_domain=domain,
            canonical_url=url,
            word_count=0,
            success=False,
            error_message=f"Extraction failure: {e}",
        )


def process_manual_input(headline: str, body: str) -> ExtractedArticle:
    """Processes user-provided manual headline and article body."""
    clean_title = clean_article_text(headline or "")
    clean_body = clean_article_text(body or "")

    if not clean_title and not clean_body:
        return ExtractedArticle(
            title="",
            body="",
            author=None,
            publication_date=None,
            source_domain="manual_entry",
            canonical_url="",
            word_count=0,
            success=False,
            error_message="Both headline and article body are empty.",
        )

    effective_title = clean_title or "Manual Submission"
    words = clean_body.split()

    return ExtractedArticle(
        title=effective_title,
        body=clean_body,
        author=None,
        publication_date=None,
        source_domain="manual_entry",
        canonical_url="",
        word_count=len(words),
        success=True,
        error_message=None,
    )
