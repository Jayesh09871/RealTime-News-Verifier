"""Unit tests for checkable factual claim extraction and importance rating."""

import pytest
from src.claim_extractor import (
    extract_claims_rule_based,
    extract_claims,
    is_opinion_or_commentary,
    rate_claim_importance,
)


def test_extract_factual_claims():
    """Verifies that concrete factual claims with dates and numbers are extracted."""
    title = "India launched a new satellite on Monday"
    text = (
        "India launched a new satellite on Monday from Sriharikota. "
        "The satellite will provide internet access to 500 rural villages. "
        "In my opinion, this might be the most exciting launch ever. "
        "The space agency spent $45 million on the mission."
    )

    claims = extract_claims_rule_based(title, text, max_claims=5)
    assert len(claims) >= 2

    claim_texts = [c.claim.lower() for c in claims]
    assert any("satellite" in ct for ct in claim_texts)
    # Opinion should be ignored
    assert not any("most exciting launch ever" in ct for ct in claim_texts)


def test_opinion_detection():
    """Verifies opinion detection identifies rhetoric and speculation."""
    assert is_opinion_or_commentary("In my opinion, this policy is terrible.")
    assert is_opinion_or_commentary("I think the government might fail.")
    assert is_opinion_or_commentary("Critics speculate this is a jaw-dropping move.")
    assert not is_opinion_or_commentary("The president signed the bilateral trade agreement on Tuesday.")


def test_claim_importance_rating():
    """Verifies that central factual claims with numbers and dates receive high importance."""
    title = "SpaceX Launches 23 Starlink Satellites"
    high_sent = "SpaceX launched 23 Starlink satellites into orbit on Wednesday."
    imp, cat = rate_claim_importance(high_sent, title)
    assert imp in ("high", "medium")
    assert cat in ("event", "statistic", "announcement")

    low_sent = "Weather conditions were clear."
    imp_low, _ = rate_claim_importance(low_sent, title)
    assert imp_low == "low"


def test_empty_and_short_article_handling():
    """Verifies graceful handling when article is empty or contains no factual sentences."""
    claims = extract_claims_rule_based("", "")
    assert isinstance(claims, list)
    assert len(claims) == 0

    # Only title
    title_only = extract_claims_rule_based("Government passes new healthcare reform bill", "")
    assert len(title_only) == 1
    assert "healthcare reform" in title_only[0].claim.lower()
