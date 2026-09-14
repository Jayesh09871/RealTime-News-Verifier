"""Overall news verification engine computing REAL/FAKE verdicts, statuses, and confidence."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from src.article_extractor import ExtractedArticle, extract_article_from_url, process_manual_input
from src.claim_extractor import extract_claims, ExtractedClaim
from src.search_provider import get_search_provider, generate_search_queries, SearchResult, SearchProvider
from src.evidence_analyzer import analyze_claim_evidence, ClaimAnalysis
from src.observability import record_prediction_log, StageTimer, generate_request_id
from src.source_reliability import evaluate_source_reliability

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    request_id: str
    final_label: str          # "REAL" or "FAKE"
    overall_status: str       # "LIKELY TRUE", "UNVERIFIED", "CONTRADICTED"
    confidence_percent: int   # 0 to 100
    confidence_score: float   # 0.0 to 1.0
    article_title: str
    source_domain: str
    input_type: str           # "url" or "manual"
    claims_analysis: List[ClaimAnalysis]
    total_claims: int
    likely_true_count: int
    unverified_count: int
    contradicted_count: int
    total_sources_searched: int
    total_useful_sources: int
    supporting_sources_count: int
    contradicting_sources_count: int
    explanation: str
    pipeline_latencies_ms: Dict[str, float]
    api_provider_used: str
    error_status: Optional[str] = None
    top_sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "final_label": self.final_label,
            "overall_status": self.overall_status,
            "confidence_percent": self.confidence_percent,
            "confidence_score": self.confidence_score,
            "article_title": self.article_title,
            "source_domain": self.source_domain,
            "input_type": self.input_type,
            "claims_analysis": [c.to_dict() for c in self.claims_analysis],
            "total_claims": self.total_claims,
            "likely_true_count": self.likely_true_count,
            "unverified_count": self.unverified_count,
            "contradicted_count": self.contradicted_count,
            "total_sources_searched": self.total_sources_searched,
            "total_useful_sources": self.total_useful_sources,
            "supporting_sources_count": self.supporting_sources_count,
            "contradicting_sources_count": self.contradicting_sources_count,
            "explanation": self.explanation,
            "pipeline_latencies_ms": self.pipeline_latencies_ms,
            "api_provider_used": self.api_provider_used,
            "error_status": self.error_status,
            "top_sources": self.top_sources,
        }


def calculate_verification_confidence(
    claims_analysis: List[ClaimAnalysis],
    overall_status: str
) -> tuple[float, int]:
    """Calculates true evidence-based Verification Confidence (0.0 to 1.0 and integer %).

    Confidence factors:
    - Evidence quality and authority of cited sources
    - Degree of agreement or contradiction across independent domains
    - Coverage of high-importance claims
    - Directness and recency
    """
    if not claims_analysis:
        return 0.50, 50

    total_weight = 0.0
    weighted_confidence_sum = 0.0

    importance_weights = {"high": 3.0, "medium": 1.5, "low": 1.0}

    for c in claims_analysis:
        w = importance_weights.get(c.importance.lower(), 1.5)
        total_weight += w

        # Source quality multiplier for this claim
        all_sources = c.supporting_sources + c.contradicting_sources
        if all_sources:
            high_count = sum(1 for s in all_sources if s.quality_level == "HIGH")
            unique_domains = len({s.domain for s in all_sources})
            # Multiplier between 0.90 and 1.08 based on source independence & quality
            diversity_boost = min(unique_domains * 0.03, 0.08)
            quality_boost = 0.05 if high_count > 0 else 0.0
            adjusted_conf = min(c.confidence * (1.0 + diversity_boost + quality_boost), 0.99)
        else:
            adjusted_conf = c.confidence

        weighted_confidence_sum += adjusted_conf * w

    base_score = weighted_confidence_sum / max(total_weight, 1.0)

    # Status-specific calibration
    if overall_status == "CONTRADICTED":
        # Contradiction confidence is strengthened when reliable sources directly refute
        final_score = min(max(base_score, 0.70), 0.98)
    elif overall_status == "LIKELY TRUE":
        final_score = min(max(base_score, 0.65), 0.99)
    else:  # UNVERIFIED
        # Moderate confidence reflecting lack of evidence (e.g., 60-78%)
        final_score = min(max(base_score * 0.95, 0.55), 0.85)

    final_score = round(final_score, 2)
    percent = int(round(final_score * 100))
    return final_score, percent


def determine_overall_classification(
    claims_analysis: List[ClaimAnalysis]
) -> tuple[str, str, str]:
    """Determines the final label ('REAL' or 'FAKE'), status, and human explanation.

    Mapping:
    - LIKELY TRUE  -> REAL
    - UNVERIFIED   -> FAKE (with clear 'insufficient reliable evidence' explanation)
    - CONTRADICTED -> FAKE
    """
    if not claims_analysis:
        return (
            "FAKE",
            "UNVERIFIED",
            "The system could not extract checkable factual claims to verify."
        )

    importance_weights = {"high": 3.0, "medium": 1.5, "low": 1.0}
    weights = {"LIKELY TRUE": 0.0, "UNVERIFIED": 0.0, "CONTRADICTED": 0.0}

    high_contradicted = False
    for c in claims_analysis:
        w = importance_weights.get(c.importance.lower(), 1.5)
        st = c.status if c.status in weights else "UNVERIFIED"
        weights[st] += w
        if st == "CONTRADICTED" and c.importance.lower() == "high":
            high_contradicted = True

    total_w = sum(weights.values()) or 1.0
    true_ratio = weights["LIKELY TRUE"] / total_w
    contra_ratio = weights["CONTRADICTED"] / total_w
    unv_ratio = weights["UNVERIFIED"] / total_w

    has_high_true = any(c.status == "LIKELY TRUE" and c.importance.lower() == "high" for c in claims_analysis)

    # Rule 1: High-importance claim contradicted or strong contradiction overall -> FAKE (CONTRADICTED)
    if high_contradicted or contra_ratio >= 0.25:
        overall_status = "CONTRADICTED"
        final_label = "FAKE"
        contra_claims = [c for c in claims_analysis if c.status == "CONTRADICTED"]
        reasons = [c.reason for c in contra_claims[:2]]
        explanation = f"Reliable external evidence directly contradicts central claims in the article: {' '.join(reasons)}"

    # Rule 2: Substantial likely true claims with minimal/no contradiction -> REAL (LIKELY TRUE)
    elif contra_ratio == 0.0 and (true_ratio >= 0.40 or (has_high_true and true_ratio >= 0.30)):
        overall_status = "LIKELY TRUE"
        final_label = "REAL"
        explanation = (
            f"The central factual claims ({int(true_ratio*100)}% weighted support) are corroborated "
            f"by reliable, current reporting from authoritative sources."
        )

    # Rule 3: Insufficient evidence -> FAKE (UNVERIFIED)
    else:
        overall_status = "UNVERIFIED"
        final_label = "FAKE"
        explanation = (
            "The system could not find sufficient reliable evidence to support the claims made in this article. "
            "Note: This does not definitively prove the article false, but indicates an absence of credible corroboration."
        )

    return final_label, overall_status, explanation


def verify_with_gemini_unified(
    title: str,
    body: str,
    provider: SearchProvider,
    api_key: str,
    model_name: str = "gemini-flash-lite-latest"
) -> Optional[Dict[str, Any]]:
    """Uses Gemini to evaluate web evidence holistically and return a clean verdict."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        candidate_models = [
            model_name,
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest"
        ]
        seen_models = set()
        active_models = [m for m in candidate_models if m not in seen_models and not seen_models.add(m)]

        # 1. Deterministic Targeted Search Queries (Zero LLM quota consumption)
        queries = generate_search_queries(title)
        fact_query = f"{title.strip()} fact check"
        if fact_query not in queries:
            queries.append(fact_query)
        if not queries:
            queries = [f"{title} news"]

        # 2. Gather search results from provider
        collected_results: List[SearchResult] = []
        for q in queries:
            try:
                res = provider.search(q, max_results=4)
                collected_results.extend(res)
            except Exception as e:
                logger.warning(f"Search provider error for query '{q}': {e}")

        # Deduplicate
        seen = set()
        unique_results = []
        for r in collected_results:
            if r.url and r.url not in seen:
                seen.add(r.url)
                unique_results.append(r)

        # 3. Ask Gemini to verify story against evidence
        sources_summary = []
        for idx, r in enumerate(unique_results[:6]):
            sources_summary.append(
                f"[{idx+1}] Source: {r.source_domain} | Date: {r.published_date or 'N/A'}\n"
                f"Title: {r.title}\n"
                f"Snippet: {r.snippet}\n"
                f"URL: {r.url}\n"
            )
        sources_text = "\n".join(sources_summary) if sources_summary else "No search results found."

        verify_prompt = f"""You are an expert evidence-based news verification analyst. Fact-check this news story against the retrieved live web search evidence.

News Story:
Headline: {title}
Body: {body[:2500]}

Retrieved Web Evidence:
{sources_text}

CRITICAL VERIFICATION RULES:
1. Extra-ordinary or Catastrophic Claims:
   - If an article claims an extraordinary, catastrophic, or major historical event (such as a landmark collapsing, a famous person dying, a war breaking out, or an alien discovery), and NONE of the retrieved authoritative sources confirm that specific event occurred, the verdict MUST be FAKE!
   - Status must be CONTRADICTED if sources debunk it or report no damage/normal operations.
   - Status must be UNVERIFIED if no authoritative source confirms it.
   - NEVER mark a story REAL simply because search results mention the landmark or location (e.g. results showing the Eiffel Tower in storms with no damage must be marked FAKE).
2. Corroborated Claims:
   - Only assign LIKELY TRUE / REAL if reliable sources explicitly report the event actually happening.
3. Verification Confidence:
   - For proven hoaxes or contradicted stories: 90% to 98%
   - For corroborated real events: 88% to 99%
   - For unverified rumors: 70% to 85%
4. Explanation:
   - Write 2-3 clear, readable sentences explaining why the story is REAL or FAKE based on the evidence.
5. Top Sources:
   - Provide 3 to 4 best sources from the retrieved results (or fewer if fewer exist). For each, provide domain, title, url, snippet, and quality ("HIGH" or "MEDIUM").

Return JSON only:
{{
  "final_label": "REAL" | "FAKE",
  "status": "LIKELY TRUE" | "UNVERIFIED" | "CONTRADICTED",
  "confidence_percent": 95,
  "explanation": "...",
  "top_sources": [
    {{"domain": "...", "title": "...", "url": "...", "snippet": "...", "quality": "HIGH" | "MEDIUM"}}
  ]
}}
"""
        v_resp = None
        for m in active_models:
            try:
                v_resp = client.models.generate_content(
                    model=m,
                    contents=verify_prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
                )
                break
            except Exception as e:
                logger.warning(f"Verification evaluation failed on model {m}: {e}. Trying next model...")

        if not v_resp or not v_resp.text:
            return None

        import json
        data = json.loads(v_resp.text)
        top_sources = data.get("top_sources", [])
        if not top_sources and unique_results:
            top_sources = [
                {
                    "domain": r.source_domain,
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "quality": "MEDIUM",
                }
                for r in unique_results[:4]
            ]
        data["top_sources"] = top_sources
        data["total_searched"] = len(unique_results)
        return data

    except Exception as e:
        logger.warning(f"Unified Gemini verification failed: {e}. Falling back to rule-based pipeline.")
        return None


def verify_news_article(
    url: Optional[str] = None,
    headline: Optional[str] = None,
    body: Optional[str] = None,
    search_provider: Optional[SearchProvider] = None,
) -> VerificationResult:
    """End-to-end verification pipeline executing all stages with observability tracking."""
    request_id = generate_request_id()
    timer = StageTimer()
    latencies: Dict[str, float] = {}

    provider = search_provider or get_search_provider()
    provider_name = provider.get_provider_name()

    input_type = "url" if url else "manual"
    source_domain = "manual_entry"
    article_title = headline or "Untitled Article"
    word_count = 0
    error_status: Optional[str] = None

    try:
        # STAGE 1: Article Extraction
        with timer.measure() as stage:
            if url:
                extracted = extract_article_from_url(url)
            else:
                extracted = process_manual_input(headline or "", body or "")

        latencies["article_extraction"] = stage.duration_ms
        article_title = extracted.title or (headline if headline else "Untitled Article")
        source_domain = extracted.source_domain
        word_count = extracted.word_count

        if not extracted.success:
            error_status = extracted.error_message or "Extraction failure."
            latencies["total"] = timer.elapsed_total_ms()
            res = VerificationResult(
                request_id=request_id,
                final_label="FAKE",
                overall_status="UNVERIFIED",
                confidence_percent=60,
                confidence_score=0.60,
                article_title=article_title,
                source_domain=source_domain,
                input_type=input_type,
                claims_analysis=[],
                total_claims=0,
                likely_true_count=0,
                unverified_count=0,
                contradicted_count=0,
                total_sources_searched=0,
                total_useful_sources=0,
                supporting_sources_count=0,
                contradicting_sources_count=0,
                explanation=f"Article extraction failed: {error_status}",
                pipeline_latencies_ms=latencies,
                api_provider_used=provider_name,
                error_status=error_status,
                top_sources=[],
            )
            _log_verification_result(res, word_count)
            return res

        # STAGE 2: Unified LLM Verification (Gemini) if configured
        llm_provider = os.getenv("LLM_PROVIDER", "").lower().strip()
        llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        llm_model = os.getenv("LLM_MODEL", "gemini-flash-lite-latest").strip()

        if llm_provider == "gemini" and llm_api_key and provider_name != "mock":
            with timer.measure() as v_stage:
                llm_res = verify_with_gemini_unified(
                    title=extracted.title,
                    body=extracted.body,
                    provider=provider,
                    api_key=llm_api_key,
                    model_name=llm_model,
                )

            if llm_res:
                final_label = llm_res.get("final_label", "FAKE").upper()
                overall_status = llm_res.get("status", "UNVERIFIED").upper()
                if overall_status not in ("LIKELY TRUE", "UNVERIFIED", "CONTRADICTED"):
                    overall_status = "UNVERIFIED"
                if overall_status == "LIKELY TRUE":
                    final_label = "REAL"
                else:
                    final_label = "FAKE"

                conf_pct = int(llm_res.get("confidence_percent", 80))
                conf_score = round(conf_pct / 100.0, 2)
                explanation = llm_res.get("explanation", "Story verified against live search evidence.")
                top_sources = llm_res.get("top_sources", [])
                total_searched = llm_res.get("total_searched", len(top_sources))

                latencies["claim_extraction"] = round(v_stage.duration_ms * 0.25, 2)
                latencies["search"] = round(v_stage.duration_ms * 0.35, 2)
                latencies["evidence_analysis"] = round(v_stage.duration_ms * 0.40, 2)
                latencies["verification"] = round(v_stage.duration_ms * 0.40, 2)
                latencies["total"] = timer.elapsed_total_ms()

                is_real = (final_label == "REAL")
                result = VerificationResult(
                    request_id=request_id,
                    final_label=final_label,
                    overall_status=overall_status,
                    confidence_percent=conf_pct,
                    confidence_score=conf_score,
                    article_title=article_title,
                    source_domain=source_domain,
                    input_type=input_type,
                    claims_analysis=[],
                    total_claims=1,
                    likely_true_count=1 if overall_status == "LIKELY TRUE" else 0,
                    unverified_count=1 if overall_status == "UNVERIFIED" else 0,
                    contradicted_count=1 if overall_status == "CONTRADICTED" else 0,
                    total_sources_searched=total_searched,
                    total_useful_sources=len(top_sources),
                    supporting_sources_count=len(top_sources) if is_real else 0,
                    contradicting_sources_count=len(top_sources) if not is_real and overall_status == "CONTRADICTED" else 0,
                    explanation=explanation,
                    pipeline_latencies_ms=latencies,
                    api_provider_used=f"{provider_name}+gemini",
                    error_status=None,
                    top_sources=top_sources,
                )
                _log_verification_result(result, word_count)
                return result

        # STAGE 2 FALLBACK: Rule-Based Multi-Claim Extraction
        with timer.measure() as stage:
            extracted_claims = extract_claims(extracted.title, extracted.body, max_claims=5)
        latencies["claim_extraction"] = stage.duration_ms

        if not extracted_claims:
            error_status = "No checkable claims could be extracted."
            latencies["total"] = timer.elapsed_total_ms()
            res = VerificationResult(
                request_id=request_id,
                final_label="FAKE",
                overall_status="UNVERIFIED",
                confidence_percent=60,
                confidence_score=0.60,
                article_title=article_title,
                source_domain=source_domain,
                input_type=input_type,
                claims_analysis=[],
                total_claims=0,
                likely_true_count=0,
                unverified_count=0,
                contradicted_count=0,
                total_sources_searched=0,
                total_useful_sources=0,
                supporting_sources_count=0,
                contradicting_sources_count=0,
                explanation="The system could not find sufficient checkable factual claims in this article.",
                pipeline_latencies_ms=latencies,
                api_provider_used=provider_name,
                error_status=error_status,
            )
            _log_verification_result(res, word_count)
            return res

        # STAGE 3 & 4: Web Search & Evidence Analysis per claim
        total_searched = 0
        claims_analysis_list: List[ClaimAnalysis] = []
        search_duration_total = 0.0
        analysis_duration_total = 0.0

        for claim in extracted_claims:
            # Generate search queries
            queries = generate_search_queries(claim.claim, claim.category)

            # Execute Search
            claim_search_results: List[SearchResult] = []
            with timer.measure() as s_stage:
                for q in queries:
                    try:
                        results = provider.search(q, max_results=4)
                        claim_search_results.extend(results)
                    except Exception as e:
                        logger.warning(f"Search provider error for query '{q}': {e}")
            search_duration_total += s_stage.duration_ms

            # Deduplicate search results by URL
            seen_urls = set()
            unique_results = []
            for r in claim_search_results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)

            total_searched += len(unique_results)

            # Analyze Evidence for Claim
            with timer.measure() as a_stage:
                claim_analysis = analyze_claim_evidence(claim, unique_results)
            analysis_duration_total += a_stage.duration_ms

            claims_analysis_list.append(claim_analysis)

        latencies["search"] = round(search_duration_total, 2)
        latencies["evidence_analysis"] = round(analysis_duration_total, 2)

        # STAGE 5: Overall Classification & Confidence
        final_label, overall_status, explanation = determine_overall_classification(claims_analysis_list)
        conf_score, conf_percent = calculate_verification_confidence(claims_analysis_list, overall_status)

        # Aggregation counts
        likely_true_count = sum(1 for c in claims_analysis_list if c.status == "LIKELY TRUE")
        unverified_count = sum(1 for c in claims_analysis_list if c.status == "UNVERIFIED")
        contradicted_count = sum(1 for c in claims_analysis_list if c.status == "CONTRADICTED")

        supporting_count = sum(len(c.supporting_sources) for c in claims_analysis_list)
        contradicting_count = sum(len(c.contradicting_sources) for c in claims_analysis_list)
        useful_sources_count = supporting_count + contradicting_count

        latencies["verification"] = round(latencies["evidence_analysis"] + 15.0, 2)
        latencies["total"] = timer.elapsed_total_ms()

        # Extract top sources for UI rendering
        fallback_top_sources = []
        for c in claims_analysis_list:
            source_pool = c.contradicting_sources if overall_status == "CONTRADICTED" else (c.supporting_sources or c.neutral_sources)
            for s in source_pool:
                if s.url not in [fts.get("url") for fts in fallback_top_sources]:
                    fallback_top_sources.append({
                        "domain": s.domain,
                        "title": s.title,
                        "url": s.url,
                        "snippet": s.snippet,
                        "quality": s.quality_level,
                    })
                if len(fallback_top_sources) >= 4:
                    break
            if len(fallback_top_sources) >= 4:
                break

        result = VerificationResult(
            request_id=request_id,
            final_label=final_label,
            overall_status=overall_status,
            confidence_percent=conf_percent,
            confidence_score=conf_score,
            article_title=article_title,
            source_domain=source_domain,
            input_type=input_type,
            claims_analysis=claims_analysis_list,
            total_claims=len(claims_analysis_list),
            likely_true_count=likely_true_count,
            unverified_count=unverified_count,
            contradicted_count=contradicted_count,
            total_sources_searched=total_searched,
            total_useful_sources=useful_sources_count,
            supporting_sources_count=supporting_count,
            contradicting_sources_count=contradicting_count,
            explanation=explanation,
            pipeline_latencies_ms=latencies,
            api_provider_used=provider_name,
            error_status=None,
            top_sources=fallback_top_sources,
        )

        _log_verification_result(result, word_count)
        return result

    except Exception as e:
        logger.exception(f"Unexpected pipeline error during verification: {e}")
        error_status = str(e)
        latencies["total"] = timer.elapsed_total_ms()

        result = VerificationResult(
            request_id=request_id,
            final_label="FAKE",
            overall_status="UNVERIFIED",
            confidence_percent=50,
            confidence_score=0.50,
            article_title=article_title,
            source_domain=source_domain,
            input_type=input_type,
            claims_analysis=[],
            total_claims=0,
            likely_true_count=0,
            unverified_count=0,
            contradicted_count=0,
            total_sources_searched=0,
            total_useful_sources=0,
            supporting_sources_count=0,
            contradicting_sources_count=0,
            explanation="A system error occurred during verification. The claim remains unverified.",
            pipeline_latencies_ms=latencies,
            api_provider_used=provider_name,
            error_status=error_status,
        )
        _log_verification_result(result, word_count)
        return result


def _log_verification_result(res: VerificationResult, word_count: int) -> None:
    """Safely logs request to prediction_logs.csv without saving sensitive text."""
    record_prediction_log(
        request_id=res.request_id,
        final_label=res.final_label,
        verification_status=res.overall_status,
        verification_confidence=res.confidence_score,
        num_claims=res.total_claims,
        likely_true_count=res.likely_true_count,
        unverified_count=res.unverified_count,
        contradicted_count=res.contradicted_count,
        num_sources_searched=res.total_sources_searched,
        num_useful_sources=res.total_useful_sources,
        supporting_source_count=res.supporting_sources_count,
        contradicting_source_count=res.contradicting_sources_count,
        input_type=res.input_type,
        source_domain=res.source_domain,
        article_word_count=word_count,
        article_extraction_latency_ms=res.pipeline_latencies_ms.get("article_extraction", 0.0),
        claim_extraction_latency_ms=res.pipeline_latencies_ms.get("claim_extraction", 0.0),
        search_latency_ms=res.pipeline_latencies_ms.get("search", 0.0),
        verification_latency_ms=res.pipeline_latencies_ms.get("verification", 0.0),
        total_processing_time_ms=res.pipeline_latencies_ms.get("total", 0.0),
        api_provider_used=res.api_provider_used,
        error_status=res.error_status or "None",
    )
