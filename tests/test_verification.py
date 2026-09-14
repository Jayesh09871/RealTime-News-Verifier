"""Unit tests for claim verification, status-to-label mapping, confidence, and verdict calculation."""

import pytest
from src.claim_extractor import ExtractedClaim
from src.evidence_analyzer import ClaimAnalysis, SourceEvidence
from src.verifier import (
    determine_overall_classification,
    calculate_verification_confidence,
    verify_news_article,
)
from src.search_provider import MockSearchProvider, SearchResult


def test_status_mapping_likely_true_to_real():
    """Verifies that LIKELY TRUE status maps directly to REAL."""
    claims = [
        ClaimAnalysis(
            claim="India launched an earth observation satellite.",
            importance="high",
            category="event",
            status="LIKELY TRUE",
            confidence=0.92,
            reason="Corroborated by Reuters and BBC.",
            supporting_sources=[
                SourceEvidence(
                    title="Reuters Report",
                    url="https://reuters.com/1",
                    snippet="Confirmed launch",
                    domain="reuters.com",
                    published_date="2026-09-12",
                    quality_level="HIGH",
                    reliability_score=0.92,
                    stance="SUPPORT",
                    stance_reason="Corroborates",
                )
            ],
            contradicting_sources=[],
        )
    ]
    final_label, status, explanation = determine_overall_classification(claims)
    assert final_label == "REAL"
    assert status == "LIKELY TRUE"
    assert "corroborated" in explanation.lower()

    conf_score, conf_pct = calculate_verification_confidence(claims, status)
    assert conf_pct >= 70
    assert 0.70 <= conf_score <= 1.0


def test_status_mapping_unverified_to_fake():
    """Verifies that UNVERIFIED status maps to FAKE with an explicit nuance explanation."""
    claims = [
        ClaimAnalysis(
            claim="Aliens were spotted in the central desert.",
            importance="high",
            category="event",
            status="UNVERIFIED",
            confidence=0.65,
            reason="No reliable evidence exists.",
            supporting_sources=[],
            contradicting_sources=[],
        )
    ]
    final_label, status, explanation = determine_overall_classification(claims)
    assert final_label == "FAKE"
    assert status == "UNVERIFIED"
    # Must NOT say 'definitely false'; must say 'could not find sufficient reliable evidence'
    assert "could not find sufficient reliable evidence" in explanation.lower()


def test_status_mapping_contradicted_to_fake():
    """Verifies that CONTRADICTED status maps to FAKE."""
    claims = [
        ClaimAnalysis(
            claim="Eiffel tower collapsed during a thunderstorm.",
            importance="high",
            category="event",
            status="CONTRADICTED",
            confidence=0.95,
            reason="Official French authorities confirm the tower is standing and unaffected.",
            supporting_sources=[],
            contradicting_sources=[
                SourceEvidence(
                    title="Fact Check: Eiffel Tower Hoax",
                    url="https://apnews.com/fact-check",
                    snippet="AP confirmed tower is open and standing.",
                    domain="apnews.com",
                    published_date="2026-09-12",
                    quality_level="HIGH",
                    reliability_score=0.94,
                    stance="CONTRADICT",
                    stance_reason="Refutes claim",
                )
            ],
        )
    ]
    final_label, status, explanation = determine_overall_classification(claims)
    assert final_label == "FAKE"
    assert status == "CONTRADICTED"
    assert "contradicts" in explanation.lower()


def test_full_pipeline_verification_with_mock():
    """Verifies the complete end-to-end pipeline using MockSearchProvider."""
    mock_results = {
        "satellite": [
            SearchResult(
                title="ISRO Launches Earth Observation Satellite EOS-08",
                url="https://www.isro.gov.in/eos08",
                snippet="ISRO has successfully launched the EOS-08 satellite on Monday.",
                source_domain="isro.gov.in",
                published_date="2026-09-12",
                provider="mock",
            ),
            SearchResult(
                title="India launches new satellite",
                url="https://www.reuters.com/world/india-satellite",
                snippet="Reuters reports India launched its latest observation satellite.",
                source_domain="reuters.com",
                published_date="2026-09-12",
                provider="mock",
            ),
        ]
    }
    mock_provider = MockSearchProvider(predefined_results=mock_results)

    headline = "India launches new satellite on Monday"
    body = "India successfully launched a new earth observation satellite on Monday to monitor agriculture."

    result = verify_news_article(
        headline=headline,
        body=body,
        search_provider=mock_provider,
    )

    assert result.final_label == "REAL"
    assert result.overall_status == "LIKELY TRUE"
    assert result.confidence_percent >= 70
    assert result.total_claims >= 1
    assert result.likely_true_count >= 1
    assert result.request_id.startswith("req_")
    assert "article_extraction" in result.pipeline_latencies_ms
