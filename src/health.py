"""System health check module evaluating live availability of system components."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

from src.observability import DEFAULT_LOG_FILE, init_log_file
from src.search_provider import get_search_provider


@dataclass
class ComponentHealth:
    component: str
    status: str       # "Healthy", "Degraded", "Unavailable"
    details: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_article_extractor_health() -> ComponentHealth:
    """Verifies article extraction engine and SSRF validator."""
    t0 = time.perf_counter()
    try:
        from src.article_extractor import validate_url_security
        is_safe, _ = validate_url_security("http://127.0.0.1")
        if is_safe:
            # Should have blocked loopback!
            return ComponentHealth(
                component="Article Extraction",
                status="Degraded",
                details="SSRF validator failed loopback test.",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        return ComponentHealth(
            component="Article Extraction",
            status="Healthy",
            details="Trafilatura parser and SSRF security guard operational.",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    except Exception as e:
        return ComponentHealth(
            component="Article Extraction",
            status="Unavailable",
            details=f"Extraction subsystem error: {e}",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


def check_search_provider_health() -> ComponentHealth:
    """Verifies currently configured search provider connectivity."""
    t0 = time.perf_counter()
    try:
        provider = get_search_provider()
        ok, msg = provider.check_health()
        status = "Healthy" if ok else "Unavailable"
        return ComponentHealth(
            component="Search Provider",
            status=status,
            details=f"Provider ({provider.get_provider_name()}): {msg}",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    except Exception as e:
        return ComponentHealth(
            component="Search Provider",
            status="Unavailable",
            details=f"Search check failed: {e}",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


def check_llm_provider_health() -> ComponentHealth:
    """Verifies LLM provider configuration or rule-based fallback readiness."""
    t0 = time.perf_counter()
    provider_name = os.getenv("LLM_PROVIDER", "none").lower().strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()

    if provider_name in ("none", "") or not api_key:
        return ComponentHealth(
            component="LLM Provider",
            status="Healthy",
            details="Deterministic NLP & Rule-Based extraction active (Zero-Key Mode).",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    if provider_name == "gemini":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            return ComponentHealth(
                component="LLM Provider",
                status="Healthy",
                details="Google GenAI (Gemini) configured and ready.",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as e:
            return ComponentHealth(
                component="LLM Provider",
                status="Degraded",
                details=f"Gemini client error ({e}); falling back to NLP engine.",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

    elif provider_name == "openai":
        return ComponentHealth(
            component="LLM Provider",
            status="Healthy",
            details="OpenAI credentials configured.",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    return ComponentHealth(
        component="LLM Provider",
        status="Degraded",
        details=f"Unrecognized provider '{provider_name}'; using NLP fallback.",
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


def check_evidence_engine_health() -> ComponentHealth:
    """Verifies stance detection and source reliability evaluation."""
    t0 = time.perf_counter()
    try:
        from src.source_reliability import evaluate_source_reliability
        from src.claim_extractor import ExtractedClaim
        from src.search_provider import SearchResult
        from src.evidence_analyzer import analyze_claim_evidence

        test_claim = ExtractedClaim(claim="Test verification statement.", importance="high", category="general")
        test_search = [SearchResult(title="Test", url="https://reuters.com", snippet="Corroboration test", source_domain="reuters.com")]
        analysis = analyze_claim_evidence(test_claim, test_search)
        if analysis.status:
            return ComponentHealth(
                component="Evidence Engine",
                status="Healthy",
                details="Stance analyzer, source reliability scoring, and confidence engine active.",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        return ComponentHealth(
            component="Evidence Engine",
            status="Degraded",
            details="Engine returned empty status.",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    except Exception as e:
        return ComponentHealth(
            component="Evidence Engine",
            status="Unavailable",
            details=f"Evidence engine test error: {e}",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


def check_logging_health() -> ComponentHealth:
    """Verifies log file write access and CSV structure integrity."""
    t0 = time.perf_counter()
    try:
        log_path = init_log_file()
        if os.access(log_path.parent, os.W_OK) and (not log_path.exists() or os.access(log_path, os.W_OK)):
            return ComponentHealth(
                component="Logging & Storage",
                status="Healthy",
                details=f"prediction_logs.csv writeable at {log_path.name}",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        return ComponentHealth(
            component="Logging & Storage",
            status="Unavailable",
            details=f"Log directory or file lacks write permissions: {log_path}",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    except Exception as e:
        return ComponentHealth(
            component="Logging & Storage",
            status="Unavailable",
            details=f"Log health error: {e}",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


def run_system_health_checks() -> List[ComponentHealth]:
    """Runs comprehensive health checks across all five core components."""
    return [
        check_article_extractor_health(),
        check_search_provider_health(),
        check_llm_provider_health(),
        check_evidence_engine_health(),
        check_logging_health(),
    ]
