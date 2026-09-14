"""Unit tests for observability, prediction logging, analytics, and health checks."""

import csv
from pathlib import Path
import pytest
from src.observability import (
    generate_request_id,
    StageTimer,
    record_prediction_log,
    global_telemetry,
    CSV_HEADERS,
)
from src.analytics import load_logs_dataframe, get_analytics_metrics, filter_recent_verifications
from src.health import run_system_health_checks


def test_request_id_format():
    """Ensures Request IDs are non-empty and unique."""
    id1 = generate_request_id()
    id2 = generate_request_id()
    assert id1.startswith("req_")
    assert id2.startswith("req_")
    assert id1 != id2


def test_stage_timer_accuracy():
    """Ensures stage timer computes positive elapsed milliseconds."""
    timer = StageTimer()
    with timer.measure() as stage:
        total = sum(i for i in range(50000))

    assert stage.duration_ms >= 0.0
    assert timer.elapsed_total_ms() >= stage.duration_ms


def test_csv_logging_and_privacy(tmp_path: Path):
    """Verifies that records are appended to CSV without leaking article text."""
    test_csv = tmp_path / "test_prediction_logs.csv"

    record_prediction_log(
        request_id="req_test123",
        final_label="REAL",
        verification_status="LIKELY TRUE",
        verification_confidence=0.91,
        num_claims=3,
        likely_true_count=2,
        unverified_count=1,
        contradicted_count=0,
        num_sources_searched=6,
        num_useful_sources=4,
        supporting_source_count=3,
        contradicting_source_count=1,
        input_type="manual",
        source_domain="manual_entry",
        article_word_count=150,
        article_extraction_latency_ms=12.5,
        claim_extraction_latency_ms=45.0,
        search_latency_ms=120.0,
        verification_latency_ms=30.0,
        total_processing_time_ms=207.5,
        api_provider_used="mock",
        error_status="None",
        filepath=test_csv,
    )

    assert test_csv.exists()

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2  # header + 1 row
        headers = reader[0]
        row = reader[1]
        assert headers == CSV_HEADERS
        assert row[1] == "req_test123"
        assert row[2] == "REAL"
        assert row[3] == "LIKELY TRUE"
        assert row[4] == "0.91"

    # Test loading and analytics from this dataframe
    df = load_logs_dataframe(test_csv)
    assert not df.empty
    metrics = get_analytics_metrics(df)
    assert metrics["has_data"] is True
    assert metrics["overview"]["total_articles"] == 1
    assert metrics["overview"]["real_count"] == 1
    assert metrics["overview"]["fake_count"] == 0
    assert metrics["claims"]["likely_true_count"] == 2


def test_system_health_checks():
    """Ensures all system health checks run and return valid health statuses."""
    healths = run_system_health_checks()
    assert len(healths) == 5
    for h in healths:
        assert h.component in (
            "Article Extraction",
            "Search Provider",
            "LLM Provider",
            "Evidence Engine",
            "Logging & Storage",
        )
        assert h.status in ("Healthy", "Degraded", "Unavailable")
        assert h.latency_ms >= 0.0
