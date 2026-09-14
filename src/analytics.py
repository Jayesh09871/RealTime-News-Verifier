"""Analytics module for parsing prediction logs, computing metrics, and preparing dashboard data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from src.observability import DEFAULT_LOG_FILE


def load_logs_dataframe(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Loads prediction logs into a sanitized Pandas DataFrame."""
    target_path = filepath or DEFAULT_LOG_FILE
    if not target_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(target_path, keep_default_na=False)
        if df.empty:
            return pd.DataFrame()

        # Clean types
        numeric_cols = [
            "verification_confidence", "num_claims", "likely_true_count", "unverified_count",
            "contradicted_count", "num_sources_searched", "num_useful_sources",
            "supporting_source_count", "contradicting_source_count", "article_word_count",
            "article_extraction_latency_ms", "claim_extraction_latency_ms",
            "search_latency_ms", "verification_latency_ms", "total_processing_time_ms"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Clean string columns
        string_cols = ["final_label", "verification_status", "input_type", "source_domain", "api_provider_used", "error_status"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Parse timestamp
        if "timestamp_utc" in df.columns:
            df["timestamp_clean"] = df["timestamp_utc"].str.replace(" UTC", "", regex=False)
            df["datetime"] = pd.to_datetime(df["timestamp_clean"], errors="coerce")

        return df
    except Exception:
        return pd.DataFrame()


def get_analytics_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes all required verification, claim, evidence, performance, and reliability metrics."""
    if df.empty:
        return {"has_data": False}

    total_articles = len(df)
    real_count = int((df["final_label"] == "REAL").sum())
    fake_count = int((df["final_label"] == "FAKE").sum())
    real_pct = round((real_count / total_articles) * 100, 1) if total_articles > 0 else 0.0
    fake_pct = round((fake_count / total_articles) * 100, 1) if total_articles > 0 else 0.0

    total_claims = int(df["num_claims"].sum())
    avg_confidence = round(df["verification_confidence"].mean() * 100, 1) if total_articles > 0 else 0.0

    # Claim metrics
    likely_true_count = int(df["likely_true_count"].sum())
    unverified_count = int(df["unverified_count"].sum())
    contradicted_count = int(df["contradicted_count"].sum())
    avg_claims_per_article = round(total_claims / total_articles, 1) if total_articles > 0 else 0.0

    # Most common claim status
    status_counts = {"LIKELY TRUE": likely_true_count, "UNVERIFIED": unverified_count, "CONTRADICTED": contradicted_count}
    most_common_status = max(status_counts, key=status_counts.get) if total_claims > 0 else "None"

    # Evidence metrics
    total_sources = int(df["num_sources_searched"].sum())
    avg_sources = round(total_sources / total_articles, 1) if total_articles > 0 else 0.0
    supporting_sources = int(df["supporting_source_count"].sum())
    contradicting_sources = int(df["contradicting_source_count"].sum())

    # Top source domains from logs
    domain_counts = df["source_domain"].value_counts().to_dict()

    # Performance metrics
    avg_total_time = round(df["total_processing_time_ms"].mean(), 1)
    avg_extraction_time = round(df["article_extraction_latency_ms"].mean(), 1)
    avg_claim_time = round(df["claim_extraction_latency_ms"].mean(), 1)
    avg_search_time = round(df["search_latency_ms"].mean(), 1)
    avg_verification_time = round(df["verification_latency_ms"].mean(), 1)
    fastest_time = round(df["total_processing_time_ms"].min(), 1)
    slowest_time = round(df["total_processing_time_ms"].max(), 1)

    # Reliability metrics
    failed_requests = int((df["error_status"].astype(str) != "None").sum())
    successful_requests = total_articles - failed_requests
    error_rate = round((failed_requests / total_articles) * 100, 1) if total_articles > 0 else 0.0

    extraction_failures = int(df["error_status"].str.contains("extraction", case=False, na=False).sum())
    search_failures = int(df["error_status"].str.contains("search", case=False, na=False).sum())
    llm_failures = int(df["error_status"].str.contains("llm", case=False, na=False).sum())
    timeout_errors = int(df["error_status"].str.contains("timed? ?out", case=False, na=False).sum())
    rate_limit_errors = int(df["error_status"].str.contains("rate.?limit", case=False, na=False).sum())

    return {
        "has_data": True,
        "overview": {
            "total_articles": total_articles,
            "real_count": real_count,
            "fake_count": fake_count,
            "real_percentage": real_pct,
            "fake_percentage": fake_pct,
            "total_claims": total_claims,
            "avg_confidence": avg_confidence,
        },
        "claims": {
            "likely_true_count": likely_true_count,
            "unverified_count": unverified_count,
            "contradicted_count": contradicted_count,
            "avg_claims_per_article": avg_claims_per_article,
            "most_common_status": most_common_status,
        },
        "evidence": {
            "total_sources": total_sources,
            "avg_sources_per_article": avg_sources,
            "supporting_sources": supporting_sources,
            "contradicting_sources": contradicting_sources,
            "top_domains": domain_counts,
        },
        "performance": {
            "avg_total_ms": avg_total_time,
            "avg_extraction_ms": avg_extraction_time,
            "avg_claim_ms": avg_claim_time,
            "avg_search_ms": avg_search_time,
            "avg_verification_ms": avg_verification_time,
            "fastest_ms": fastest_time,
            "slowest_ms": slowest_time,
        },
        "reliability": {
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "error_rate_pct": error_rate,
            "extraction_failures": extraction_failures,
            "search_failures": search_failures,
            "llm_failures": llm_failures,
            "timeout_errors": timeout_errors,
            "rate_limit_errors": rate_limit_errors,
        }
    }


def filter_recent_verifications(
    df: pd.DataFrame,
    result_filter: str = "ALL",
    status_filter: str = "ALL",
    input_filter: str = "ALL",
    max_rows: int = 50,
) -> pd.DataFrame:
    """Filters recent verifications table for UI display."""
    if df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    if result_filter != "ALL":
        filtered = filtered[filtered["final_label"] == result_filter]

    if status_filter != "ALL":
        filtered = filtered[filtered["verification_status"] == status_filter]

    if input_filter != "ALL":
        filtered = filtered[filtered["input_type"] == input_filter]

    # Select columns required for recent verification table (Section 27)
    display_cols = [
        "timestamp_utc", "request_id", "final_label", "verification_status",
        "verification_confidence", "num_claims", "num_sources_searched",
        "total_processing_time_ms", "input_type"
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]

    filtered = filtered[available_cols].iloc[::-1].head(max_rows)
    return filtered
