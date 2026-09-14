"""Observability module managing Request IDs, high-precision timers, CSV logging, and API telemetry."""

from __future__ import annotations

import csv
import datetime
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

# Default CSV Log Location
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "prediction_logs.csv"

CSV_HEADERS = [
    "timestamp_utc",
    "request_id",
    "final_label",
    "verification_status",
    "verification_confidence",
    "num_claims",
    "likely_true_count",
    "unverified_count",
    "contradicted_count",
    "num_sources_searched",
    "num_useful_sources",
    "supporting_source_count",
    "contradicting_source_count",
    "input_type",
    "source_domain",
    "article_word_count",
    "article_extraction_latency_ms",
    "claim_extraction_latency_ms",
    "search_latency_ms",
    "verification_latency_ms",
    "total_processing_time_ms",
    "api_provider_used",
    "error_status",
]

_csv_lock = threading.Lock()


def generate_request_id() -> str:
    """Generates a concise, unique trace ID for a verification request."""
    return f"req_{uuid.uuid4().hex[:10]}"


class StageTimerContext:
    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)


class StageTimer:
    """Multi-stage high-precision timer."""

    def __init__(self):
        self._global_start = time.perf_counter()

    def measure(self) -> StageTimerContext:
        """Context manager to measure a pipeline stage in milliseconds."""
        return StageTimerContext()

    def elapsed_total_ms(self) -> float:
        """Returns total elapsed time in milliseconds since timer creation."""
        return round((time.perf_counter() - self._global_start) * 1000.0, 2)


class ApiTelemetryTracker:
    """Thread-safe external API telemetry metrics collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._providers: Dict[str, Dict[str, Any]] = {}

    def record_call(
        self,
        provider: str,
        latency_ms: float,
        success: bool = True,
        is_timeout: bool = False,
        is_rate_limit: bool = False,
    ):
        with self._lock:
            if provider not in self._providers:
                self._providers[provider] = {
                    "request_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "total_latency_ms": 0.0,
                    "timeout_count": 0,
                    "rate_limit_count": 0,
                }
            stats = self._providers[provider]
            stats["request_count"] += 1
            if success:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
            stats["total_latency_ms"] += latency_ms
            if is_timeout:
                stats["timeout_count"] += 1
            if is_rate_limit:
                stats["rate_limit_count"] += 1

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            result = {}
            for name, data in self._providers.items():
                reqs = data["request_count"]
                avg_lat = round(data["total_latency_ms"] / max(reqs, 1), 2)
                result[name] = {
                    "provider": name,
                    "request_count": reqs,
                    "success_count": data["success_count"],
                    "failure_count": data["failure_count"],
                    "average_latency_ms": avg_lat,
                    "timeout_count": data["timeout_count"],
                    "rate_limit_count": data["rate_limit_count"],
                }
            return result


# Global singleton telemetry
global_telemetry = ApiTelemetryTracker()


def init_log_file(filepath: Optional[Path] = None) -> Path:
    """Ensures logs directory and CSV header row exist."""
    target_path = filepath or DEFAULT_LOG_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists() or target_path.stat().st_size == 0:
        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    return target_path


def record_prediction_log(
    request_id: str,
    final_label: str,
    verification_status: str,
    verification_confidence: float,
    num_claims: int,
    likely_true_count: int,
    unverified_count: int,
    contradicted_count: int,
    num_sources_searched: int,
    num_useful_sources: int,
    supporting_source_count: int,
    contradicting_source_count: int,
    input_type: str,
    source_domain: str,
    article_word_count: int,
    article_extraction_latency_ms: float,
    claim_extraction_latency_ms: float,
    search_latency_ms: float,
    verification_latency_ms: float,
    total_processing_time_ms: float,
    api_provider_used: str,
    error_status: str = "None",
    filepath: Optional[Path] = None,
) -> None:
    """Logs verification request metadata to CSV without storing article text or credentials."""
    target_path = init_log_file(filepath)
    utc_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    row = [
        utc_timestamp,
        request_id,
        final_label,
        verification_status,
        round(verification_confidence, 2),
        num_claims,
        likely_true_count,
        unverified_count,
        contradicted_count,
        num_sources_searched,
        num_useful_sources,
        supporting_source_count,
        contradicting_source_count,
        input_type,
        source_domain,
        article_word_count,
        round(article_extraction_latency_ms, 2),
        round(claim_extraction_latency_ms, 2),
        round(search_latency_ms, 2),
        round(verification_latency_ms, 2),
        round(total_processing_time_ms, 2),
        api_provider_used,
        error_status,
    ]

    with _csv_lock:
        with open(target_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
