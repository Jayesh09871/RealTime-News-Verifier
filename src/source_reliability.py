"""Source reliability assessment layer analyzing domain authority, quality tiers, relevance, and recency."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import datetime

# High-authority wire services and global news agencies
WIRE_SERVICES = {
    "reuters.com", "apnews.com", "afp.com", "bloomberg.com", "upi.com", "ansa.it",
    "dpa-international.com", "kyodonews.net", "tass.com", "xinhuanet.com", "ptinews.com"
}

# Established major news organizations with institutional fact-checking standards
ESTABLISHED_NEWS = {
    "bbc.com", "bbc.co.uk", "nytimes.com", "wsj.com", "washingtonpost.com", "theguardian.com",
    "ft.com", "economist.com", "npr.org", "pbs.org", "cbsnews.com", "nbcnews.com", "abcnews.go.com",
    "cnn.com", "time.com", "usatoday.com", "theatlantic.com", "aljazeera.com", "dw.com", "france24.com",
    "hindustantimes.com", "thehindu.com", "indianexpress.com", "timesofindia.indiatimes.com",
    "japantimes.co.jp", "scmp.com", "smh.com.au", "cbc.ca", "nature.com", "science.org",
    "space.com", "newscientist.com", "scientificamerican.com", "politico.com", "thehill.com"
}

# Major international organizations
INTERNATIONAL_ORGS = {
    "un.org", "who.int", "nasa.gov", "isro.gov.in", "esa.int", "wto.org", "imf.org",
    "worldbank.org", "cdc.gov", "nih.gov", "fda.gov", "noaa.gov", "interpol.int"
}

# Known satire, spoof, or chronically questionable content farms
QUESTIONABLE_OR_SATIRE = {
    "theonion.com", "babylonbee.com", "nationalreport.net", "worldnewsdailyreport.com",
    "newsexaminer.net", "infowars.com", "beforeitsnews.com", "naturalnews.com",
    "wnd.com", "breitbart.com", "thegatewaypundit.com", "dailywire.com"
}

# User-generated or social media content platforms
SOCIAL_OR_UGC = {
    "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com",
    "reddit.com", "quora.com", "medium.com", "substack.com", "tumblr.com"
}


@dataclass
class SourceAssessment:
    domain: str
    quality_level: str  # "HIGH", "MEDIUM", "LOW"
    reliability_score: float  # 0.0 to 1.0
    category: str
    relevance_score: float  # 0.0 to 1.0
    recency_note: str
    is_primary_or_institutional: bool
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_domain(domain: str) -> str:
    """Strips www. and ports from domain string."""
    if not domain:
        return "unknown"
    d = domain.lower().strip()
    if d.startswith("www."):
        d = d[4:]
    if ":" in d:
        d = d.split(":")[0]
    return d


def calculate_relevance(claim: str, snippet: str, title: str = "") -> float:
    """Calculates relevance of evidence snippet/title to the claim using token overlap."""
    if not claim or (not snippet and not title):
        return 0.0

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might",
        "must", "can", "could", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "up", "about", "into", "over", "after", "it", "its", "that", "this",
        "these", "those", "their", "they", "we", "he", "she", "also", "and", "or", "but"
    }

    claim_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9\-\.]{3,}\b", claim) if w.lower() not in stop_words]
    claim_words = set(claim_tokens)

    evidence_text = f"{title} {snippet}".lower()
    evidence_words = set(re.findall(r"\b[a-zA-Z0-9\-\.]{3,}\b", evidence_text))

    if not claim_words:
        return 0.5

    intersection = claim_words.intersection(evidence_words)
    overlap_ratio = len(intersection) / len(claim_words)

    return min(max(overlap_ratio, 0.0), 1.0)


def evaluate_source_reliability(
    domain: str,
    snippet: str = "",
    claim: str = "",
    title: str = "",
    published_date: Optional[str] = None
) -> SourceAssessment:
    """Evaluates the reliability level and quality tier of a source without making absolute claims."""
    norm_domain = normalize_domain(domain)
    category = "general_web"
    is_primary = False
    base_score = 0.50
    level = "MEDIUM"
    reasons = []

    # 1. Government domains
    if (
        norm_domain.endswith(".gov")
        or ".gov." in norm_domain
        or norm_domain.endswith(".mil")
        or norm_domain in INTERNATIONAL_ORGS
    ):
        category = "government_or_international"
        is_primary = True
        base_score = 0.95
        level = "HIGH"
        reasons.append("Official government or recognized international agency domain.")

    # 2. Educational & Research Institutions
    elif norm_domain.endswith(".edu") or ".edu." in norm_domain or norm_domain.endswith(".ac.uk"):
        category = "academic_institution"
        is_primary = True
        base_score = 0.90
        level = "HIGH"
        reasons.append("Accredited university or research institution domain.")

    # 3. Global Wire Services
    elif any(norm_domain == ws or norm_domain.endswith("." + ws) for ws in WIRE_SERVICES):
        category = "wire_service"
        is_primary = True
        base_score = 0.92
        level = "HIGH"
        reasons.append("Global news wire service adhering to stringent editorial standards.")

    # 4. Major Established News Organizations
    elif any(norm_domain == en or norm_domain.endswith("." + en) for en in ESTABLISHED_NEWS):
        category = "established_news"
        is_primary = False
        base_score = 0.85
        level = "HIGH"
        reasons.append("Major established news publisher with documented editorial standards.")

    # 5. Questionable or Satire
    elif any(norm_domain == q or norm_domain.endswith("." + q) for q in QUESTIONABLE_OR_SATIRE):
        category = "satire_or_questionable"
        is_primary = False
        base_score = 0.15
        level = "LOW"
        reasons.append("Domain known for satire, sensationalism, or low editorial accountability.")

    # 6. User-generated / Social media
    elif any(norm_domain == s or norm_domain.endswith("." + s) for s in SOCIAL_OR_UGC):
        category = "social_or_user_generated"
        is_primary = False
        base_score = 0.30
        level = "LOW"
        reasons.append("User-generated platform; content lacks editorial fact-checking.")

    # 7. General Web / Regional
    else:
        category = "general_web"
        is_primary = False
        base_score = 0.55
        level = "MEDIUM"
        reasons.append("Standard web publication or regional outlet.")

    # Relevance adjustment
    relevance = calculate_relevance(claim, snippet, title)
    if relevance < 0.20:
        base_score *= 0.65
        reasons.append("Low text alignment with the specific claim.")
    elif relevance >= 0.60:
        base_score = min(base_score + 0.05, 0.98)
        reasons.append("Strong textual correspondence to the claim.")

    # Recency check
    recency_note = "Date not specified"
    if published_date:
        recency_note = f"Reported: {published_date}"
        # If publication year is current or recent
        if any(yr in published_date for yr in ["2024", "2025", "2026"]):
            reasons.append("Recent publication timestamp.")
        elif any(yr in published_date for yr in ["2015", "2016", "2017", "2018", "2019"]):
            reasons.append("Older archival report; may not reflect current status.")
            base_score *= 0.85

    # Final tier assignment
    final_score = round(min(max(base_score, 0.05), 0.99), 2)
    if final_score >= 0.75:
        level = "HIGH"
    elif final_score >= 0.45:
        level = "MEDIUM"
    else:
        level = "LOW"

    return SourceAssessment(
        domain=norm_domain,
        quality_level=level,
        reliability_score=final_score,
        category=category,
        relevance_score=round(relevance, 2),
        recency_note=recency_note,
        is_primary_or_institutional=is_primary,
        justification=" ".join(reasons),
    )
