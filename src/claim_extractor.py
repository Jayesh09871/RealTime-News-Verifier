"""Checkable factual claim extraction using LLM or rule-based NLP fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedClaim:
    claim: str
    importance: str  # "high", "medium", "low"
    category: str    # "event", "announcement", "statistic", "quote", "statement", "general"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Subjective / Opinion phrase patterns to ignore or downweight
OPINION_PATTERNS = [
    r"(?i)\b(i think|i believe|we believe|in my opinion|in our view|some argue|critics suggest)\b",
    r"(?i)\b(it seems|it feels like|arguably|perhaps|maybe|undoubtedly the greatest)\b",
    r"(?i)\b(could possibly be|predicted to|experts speculate|rumored to)\b",
    r"(?i)\b(wonderful|terrible|shocking|unbelievable|jaw-dropping|must-see)\b",
]

# Factual action verbs indicating verifiable occurrences
FACTUAL_VERBS = [
    "announced", "signed", "launched", "passed", "approved", "discovered",
    "reported", "confirmed", "arrested", "voted", "resigned", "acquired",
    "tested", "died", "won", "appointed", "released", "stated", "unveiled",
    "declared", "introduced", "banned", "awarded", "published", "built"
]


def is_opinion_or_commentary(sentence: str) -> bool:
    """Detects if a sentence is primarily an opinion, speculation, or emotional commentary."""
    for pattern in OPINION_PATTERNS:
        if re.search(pattern, sentence):
            return True
    return False


def rate_claim_importance(sentence: str, title: str) -> tuple[str, str]:
    """Rates the importance ('high', 'medium', 'low') and category of a factual sentence."""
    s_lower = sentence.lower()
    t_words = set(re.findall(r"\b\w{4,}\b", title.lower()))
    s_words = set(re.findall(r"\b\w{4,}\b", s_lower))

    # Overlap with title core terms elevates importance
    title_overlap = len(t_words.intersection(s_words))

    has_numbers = bool(re.search(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+[%kmb]?|\$\d+)\b", sentence))
    has_dates = bool(re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|\b20\d\d\b)\b", s_lower))
    has_action_verb = any(verb in s_lower for verb in FACTUAL_VERBS)

    category = "general"
    if any(k in s_lower for k in ["announced", "declared", "statement", "spokesperson", "said", "confirmed"]):
        category = "announcement"
    elif any(k in s_lower for k in ["launched", "tested", "passed", "signed", "won", "appointed", "held"]):
        category = "event"
    elif has_numbers and any(k in s_lower for k in ["percent", "%", "million", "billion", "dollars", "rupees", "increase", "decrease", "total"]):
        category = "statistic"

    if title_overlap >= 2 or (has_action_verb and (has_dates or has_numbers)):
        return "high", category
    elif has_action_verb or has_numbers or has_dates:
        return "medium", category
    return "low", category


def extract_claims_rule_based(title: str, text: str, max_claims: int = 6) -> List[ExtractedClaim]:
    """Heuristic / NLP rule-based extractor that isolates checkable factual claims.

    Used when no external LLM API key is configured or as an instant fallback.
    """
    claims: List[ExtractedClaim] = []
    seen_claims = set()

    # If title is factual, it's often the central claim
    if title and not is_opinion_or_commentary(title) and len(title.split()) >= 4:
        importance, cat = rate_claim_importance(title, title)
        claims.append(ExtractedClaim(claim=title, importance="high", category=cat))
        seen_claims.add(title.lower())

    # Split body into distinct sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    scored_sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        words = s_clean.split()
        if len(words) < 5 or len(words) > 40:
            continue
        if s_clean.lower() in seen_claims:
            continue
        if is_opinion_or_commentary(s_clean):
            continue

        importance, cat = rate_claim_importance(s_clean, title)
        # Score for ranking: high=3, medium=2, low=1
        score = 3 if importance == "high" else (2 if importance == "medium" else 1)
        scored_sentences.append((score, importance, cat, s_clean))

    # Sort by score descending
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    for _, imp, cat, s_clean in scored_sentences:
        if len(claims) >= max_claims:
            break
        # Avoid duplicate claims with high word overlap
        words_set = set(s_clean.lower().split())
        if any(len(words_set.intersection(set(c.claim.lower().split()))) / max(len(words_set), 1) > 0.7 for c in claims):
            continue
        claims.append(ExtractedClaim(claim=s_clean, importance=imp, category=cat))

    # If still no claims extracted, take the title or first non-empty sentence
    if not claims and title:
        claims.append(ExtractedClaim(claim=title, importance="high", category="general"))
    elif not claims and text:
        first_sentence = text.split(".")[0].strip()
        if first_sentence:
            claims.append(ExtractedClaim(claim=first_sentence, importance="high", category="general"))

    return claims


def extract_claims_with_gemini(title: str, text: str, api_key: str, model_name: str = "gemini-3.5-flash") -> Optional[List[ExtractedClaim]]:
    """Extracts factual claims using Google GenAI SDK."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = f"""You are an objective news fact-checking assistant. Extract the 3 to 6 most important checkable factual claims from this news article.

Article Title: {title}
Article Text: {text[:4000]}

Focus STRICTLY on:
- Concrete events (what happened, when, where)
- Official announcements (who said/signed/passed what)
- Verifiable statistics, numbers, and measurements
- Named people and organizations involved in specific actions

IGNORE:
- Personal opinions, emotional language, rhetoric, speculation, vague commentary

Return ONLY a valid JSON object matching this schema:
{{
  "claims": [
    {{
      "claim": "Concise, checkable declarative statement",
      "importance": "high" | "medium" | "low",
      "category": "event" | "announcement" | "statistic" | "quote" | "statement" | "general"
    }}
  ]
}}
"""
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        if not response.text:
            return None

        data = json.loads(response.text)
        claims_data = data.get("claims", [])
        extracted = []
        for c in claims_data:
            claim_str = c.get("claim", "").strip()
            if claim_str:
                extracted.append(
                    ExtractedClaim(
                        claim=claim_str,
                        importance=c.get("importance", "medium").lower(),
                        category=c.get("category", "general").lower(),
                    )
                )
        return extracted if extracted else None
    except Exception as e:
        logger.warning(f"Gemini claim extraction failed: {e}. Falling back to rule-based extractor.")
        return None


def extract_claims_with_openai(title: str, text: str, api_key: str, model_name: str = "gpt-4o-mini") -> Optional[List[ExtractedClaim]]:
    """Extracts factual claims using OpenAI API via HTTP requests."""
    import requests
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        prompt = f"""Extract 3 to 6 key checkable factual claims from this article.
Title: {title}
Text: {text[:4000]}

Focus strictly on concrete verifiable events, dates, numbers, official announcements, and named entities.
Ignore opinions and commentary.

Return a JSON object with this exact format:
{{"claims": [{{"claim": "string", "importance": "high"|"medium"|"low", "category": "event"|"announcement"|"statistic"|"general"}}]}}"""

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a professional fact-checker. Respond in valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            claims_data = parsed.get("claims", [])
            extracted = [
                ExtractedClaim(
                    claim=c.get("claim", "").strip(),
                    importance=c.get("importance", "medium").lower(),
                    category=c.get("category", "general").lower(),
                )
                for c in claims_data if c.get("claim")
            ]
            return extracted if extracted else None
    except Exception as e:
        logger.warning(f"OpenAI claim extraction failed: {e}")
    return None


def extract_claims(title: str, text: str, max_claims: int = 5) -> List[ExtractedClaim]:
    """Primary claim extraction dispatcher: tries configured LLM, gracefully falls back to rule-based NLP."""
    llm_provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    llm_api_key = os.getenv("LLM_API_KEY", "").strip()

    # 1. Try Gemini
    if llm_provider == "gemini" and llm_api_key:
        model = os.getenv("LLM_MODEL", "gemini-3.5-flash")
        claims = extract_claims_with_gemini(title, text, llm_api_key, model)
        if claims:
            return claims[:max_claims]

    # 2. Try OpenAI
    if llm_provider == "openai" and llm_api_key:
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        claims = extract_claims_with_openai(title, text, llm_api_key, model)
        if claims:
            return claims[:max_claims]

    # 3. Deterministic rule-based NLP fallback (zero-key, high quality)
    return extract_claims_rule_based(title, text, max_claims=max_claims)
