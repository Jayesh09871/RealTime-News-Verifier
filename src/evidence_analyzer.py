"""Evidence analysis module comparing factual claims against retrieved web evidence."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

from src.claim_extractor import ExtractedClaim
from src.search_provider import SearchResult
from src.source_reliability import evaluate_source_reliability, SourceAssessment

logger = logging.getLogger(__name__)

# Patterns indicating explicit contradiction, debunking, or fact-check refutation
REFUTATION_PATTERNS = [
    r"(?i)\b(debunked|hoax|fact check:\s*false|false claim|fake claim|misleading|untrue|disproven|no evidence)\b",
    r"(?i)\b(denies|denied|refuted|refutes|denies reports|never happened|contrary to claims)\b",
    r"(?i)\b(did not occur|was not held|called off|fabricated|doctored|manipulated|parody|satire)\b",
    r"(?i)\b(suffers no damage|no damage|undamaged|unaffected|still standing|unharmed|safe and sound|intact)\b",
    r"(?i)\b(no reports of|not true|didn't happen|nothing happened to)\b",
]

# Patterns indicating strong confirmation and support
CORROBORATION_PATTERNS = [
    r"(?i)\b(confirmed|officially announced|successfully|reported that|according to|stated that|unveiled)\b",
    r"(?i)\b(signed the agreement|passed the bill|launched successfully|approved the)\b",
]

CRITICAL_EVENT_VERBS = [
    "collapsed", "destroy", "destroyed", "crash", "crashed", "stolen", "killed",
    "dead", "die", "died", "assassinated", "exploded", "attacked", "bombed",
    "arrested", "banned", "resigned", "resigns", "launched", "passes", "passed",
    "approved", "signed", "discovered", "won"
]


@dataclass
class SourceEvidence:
    title: str
    url: str
    snippet: str
    domain: str
    published_date: Optional[str]
    quality_level: str  # "HIGH", "MEDIUM", "LOW"
    reliability_score: float
    stance: str         # "SUPPORT", "CONTRADICT", "NEUTRAL"
    stance_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimAnalysis:
    claim: str
    importance: str     # "high", "medium", "low"
    category: str
    status: str         # "LIKELY TRUE", "UNVERIFIED", "CONTRADICTED"
    confidence: float   # 0.0 to 1.0 (Verification Confidence)
    reason: str
    supporting_sources: List[SourceEvidence] = field(default_factory=list)
    contradicting_sources: List[SourceEvidence] = field(default_factory=list)
    neutral_sources: List[SourceEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "importance": self.importance,
            "category": self.category,
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "supporting_sources": [s.to_dict() for s in self.supporting_sources],
            "contradicting_sources": [s.to_dict() for s in self.contradicting_sources],
            "neutral_sources": [s.to_dict() for s in self.neutral_sources],
        }


def detect_evidence_stance_rule_based(
    claim: str,
    search_res: SearchResult,
    assessment: SourceAssessment
) -> tuple[str, str]:
    """Determines whether an evidence snippet supports, contradicts, or is neutral to a claim."""
    combined_text = f"{search_res.title} {search_res.snippet}".lower()
    claim_lower = claim.lower()

    # If relevance is very low, mark neutral
    if assessment.relevance_score < 0.22:
        return "NEUTRAL", "Evidence snippet lacks sufficient direct relevance to the claim."

    # Check for direct refutation or contradiction markers
    is_refutation = any(re.search(pat, combined_text) for pat in REFUTATION_PATTERNS)
    if is_refutation:
        return "CONTRADICT", f"Source contains explicit refutation/debunking markers regarding '{claim[:40]}...'."

    # Check if numbers or key dates in claim are contradicted in snippet
    claim_numbers = set(re.findall(r"\b\d+[%kmb]?\b", claim_lower))
    evidence_numbers = set(re.findall(r"\b\d+[%kmb]?\b", combined_text))
    if claim_numbers and evidence_numbers and not claim_numbers.intersection(evidence_numbers):
        if assessment.relevance_score > 0.40 and any(w in combined_text for w in ["actually", "instead of", "not", "different"]):
            return "CONTRADICT", f"Evidence cites conflicting figures: {', '.join(list(evidence_numbers)[:3])} vs claim's {', '.join(list(claim_numbers)[:3])}."

    # Check critical event verbs (e.g. collapsed, killed, launched, resigned)
    claim_verbs = [v for v in CRITICAL_EVENT_VERBS if v in claim_lower]
    if claim_verbs:
        # If the claim asserts an action (e.g. collapsed), the evidence MUST mention that action to support it
        has_verb_in_evidence = any(v in combined_text for v in claim_verbs)
        if not has_verb_in_evidence:
            # If the evidence discusses the subject without confirming the critical action (e.g. storm at Eiffel Tower with no collapse)
            if any(w in combined_text for w in ["damage", "safe", "struck", "lightning", "weather"]):
                return "CONTRADICT", f"Sources reporting on this event make no mention of '{claim_verbs[0]}'."
            return "NEUTRAL", f"Source mentions related entities but does not corroborate '{claim_verbs[0]}'."

    # Check for corroboration / support
    if assessment.relevance_score >= 0.45:
        return "SUPPORT", f"Reliable reporting from {assessment.domain} corroborates the factual claim."

    return "NEUTRAL", "Source discusses related context but does not provide definitive confirmation or denial."


def analyze_claim_with_llm(
    claim: ExtractedClaim,
    search_results: List[SearchResult],
    api_key: str,
    provider: str = "gemini"
) -> Optional[ClaimAnalysis]:
    """Uses LLM to perform deep evidence reasoning strictly grounded in retrieved search results."""
    if not search_results:
        return None

    sources_summary = []
    for idx, r in enumerate(search_results[:6]):
        sources_summary.append(
            f"[{idx+1}] Source: {r.source_domain} | Date: {r.published_date or 'N/A'}\n"
            f"Title: {r.title}\n"
            f"Snippet: {r.snippet}\n"
        )
    sources_text = "\n".join(sources_summary)

    prompt = f"""You are an objective news verification analyst. Analyze whether the retrieved web evidence supports, contradicts, or fails to verify the following factual claim.
IMPORTANT: You MUST evaluate ONLY based on the retrieved evidence provided below. Do NOT use outside training knowledge to determine truth.

Claim to Verify: "{claim.claim}"
Claim Importance: {claim.importance}

Retrieved Web Evidence:
{sources_text}

For each source [1], [2], etc., classify its stance as:
- SUPPORT (if source confirms the factual statement)
- CONTRADICT (if source reports the opposite, debunks, or contradicts)
- NEUTRAL (if source is irrelevant or inconclusive)

Then decide the claim status:
- LIKELY TRUE: Reliable evidence clearly supports the claim.
- CONTRADICTED: Reliable evidence directly contradicts the claim.
- UNVERIFIED: Insufficient reliable evidence to confirm or deny.

Return ONLY a JSON object with this format:
{{
  "status": "LIKELY TRUE" | "UNVERIFIED" | "CONTRADICTED",
  "confidence": 0.0 to 1.0,
  "reason": "Clear, concise 1-2 sentence evidence-grounded justification",
  "source_stances": [
    {{"index": 1, "stance": "SUPPORT" | "CONTRADICT" | "NEUTRAL", "reason": "brief reason"}}
  ]
}}
"""
    try:
        if provider == "gemini":
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=os.getenv("LLM_MODEL", "gemini-3.5-flash"),
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
            )
            if resp.text:
                data = json.loads(resp.text)
                return _build_claim_analysis_from_llm(claim, search_results, data)

        elif provider == "openai":
            import requests
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                data = json.loads(resp.json()["choices"][0]["message"]["content"])
                return _build_claim_analysis_from_llm(claim, search_results, data)
    except Exception as e:
        logger.warning(f"LLM evidence analysis failed: {e}. Falling back to rule-based engine.")

    return None


def _build_claim_analysis_from_llm(
    claim: ExtractedClaim,
    search_results: List[SearchResult],
    data: Dict[str, Any]
) -> ClaimAnalysis:
    """Helper to assemble ClaimAnalysis from LLM response."""
    status = data.get("status", "UNVERIFIED").upper()
    if status not in ("LIKELY TRUE", "UNVERIFIED", "CONTRADICTED"):
        status = "UNVERIFIED"

    confidence = float(data.get("confidence", 0.70))
    confidence = round(min(max(confidence, 0.20), 0.98), 2)
    reason = data.get("reason", "Evidence analyzed against retrieved sources.")

    supporting = []
    contradicting = []
    neutral = []

    stances_map = {item.get("index", -1): item for item in data.get("source_stances", [])}

    for idx, r in enumerate(search_results):
        assess = evaluate_source_reliability(r.source_domain, r.snippet, claim.claim, r.title, r.published_date)
        stance_item = stances_map.get(idx + 1, {})
        stance = stance_item.get("stance", "NEUTRAL").upper()
        stance_reason = stance_item.get("reason", "Evaluated from source text.")

        evidence_item = SourceEvidence(
            title=r.title,
            url=r.url,
            snippet=r.snippet,
            domain=assess.domain,
            published_date=r.published_date,
            quality_level=assess.quality_level,
            reliability_score=assess.reliability_score,
            stance=stance,
            stance_reason=stance_reason,
        )
        if stance == "SUPPORT":
            supporting.append(evidence_item)
        elif stance == "CONTRADICT":
            contradicting.append(evidence_item)
        else:
            neutral.append(evidence_item)

    return ClaimAnalysis(
        claim=claim.claim,
        importance=claim.importance,
        category=claim.category,
        status=status,
        confidence=confidence,
        reason=reason,
        supporting_sources=supporting,
        contradicting_sources=contradicting,
        neutral_sources=neutral,
    )


def analyze_claim_evidence(
    claim: ExtractedClaim,
    search_results: List[SearchResult]
) -> ClaimAnalysis:
    """Analyzes retrieved web search results against a claim and assigns status & evidence confidence."""
    # Try LLM if configured
    llm_provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    if llm_provider in ("gemini", "openai") and llm_api_key:
        llm_analysis = analyze_claim_with_llm(claim, search_results, llm_api_key, provider=llm_provider)
        if llm_analysis is not None:
            return llm_analysis

    # Rule-based / NLP evidence evaluation engine
    supporting: List[SourceEvidence] = []
    contradicting: List[SourceEvidence] = []
    neutral: List[SourceEvidence] = []

    for r in search_results:
        assess = evaluate_source_reliability(r.source_domain, r.snippet, claim.claim, r.title, r.published_date)
        stance, reason = detect_evidence_stance_rule_based(claim.claim, r, assess)

        evidence_item = SourceEvidence(
            title=r.title,
            url=r.url,
            snippet=r.snippet,
            domain=assess.domain,
            published_date=r.published_date,
            quality_level=assess.quality_level,
            reliability_score=assess.reliability_score,
            stance=stance,
            stance_reason=reason,
        )

        if stance == "SUPPORT":
            supporting.append(evidence_item)
        elif stance == "CONTRADICT":
            contradicting.append(evidence_item)
        else:
            neutral.append(evidence_item)

    # Status Determination based on accumulated evidence
    if contradicting:
        # Reliable contradicting evidence present
        high_contradictions = [s for s in contradicting if s.quality_level in ("HIGH", "MEDIUM")]
        if high_contradictions:
            status = "CONTRADICTED"
            avg_rel = sum(s.reliability_score for s in high_contradictions) / len(high_contradictions)
            confidence = round(min(0.80 + (0.15 * avg_rel), 0.98), 2)
            domains = ", ".join({s.domain for s in high_contradictions[:2]})
            reason = f"Reliable sources ({domains}) directly conflict with or refute this claim."
        else:
            status = "CONTRADICTED"
            confidence = 0.70
            reason = "Available web evidence contradicts the claim, though source authority is moderate."

    elif supporting:
        # Supporting evidence present
        high_support = [s for s in supporting if s.quality_level == "HIGH"]
        med_support = [s for s in supporting if s.quality_level == "MEDIUM"]

        if high_support or len(med_support) >= 2:
            status = "LIKELY TRUE"
            # Confidence based on quality and count
            base = 0.85 if high_support else 0.78
            boost = min(len(supporting) * 0.03, 0.12)
            confidence = round(min(base + boost, 0.98), 2)
            top_domains = ", ".join({s.domain for s in (high_support + med_support)[:3]})
            reason = f"Corroborated by reliable sources ({top_domains}) reporting consistent details."
        elif med_support:
            status = "LIKELY TRUE"
            confidence = 0.72
            reason = f"Supported by regional reporting from {med_support[0].domain}."
        else:
            # Only LOW quality support
            status = "UNVERIFIED"
            confidence = 0.65
            reason = "The system could not find sufficient reliable evidence to confirm this claim (only low-authority mentions found)."

    else:
        # No supporting or contradicting evidence
        status = "UNVERIFIED"
        confidence = 0.75
        reason = "The system could not find sufficient reliable evidence to support the claim in current web sources."

    return ClaimAnalysis(
        claim=claim.claim,
        importance=claim.importance,
        category=claim.category,
        status=status,
        confidence=confidence,
        reason=reason,
        supporting_sources=supporting,
        contradicting_sources=contradicting,
        neutral_sources=neutral,
    )
