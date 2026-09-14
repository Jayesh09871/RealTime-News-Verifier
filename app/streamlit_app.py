"""Real-Time News Verification System - Streamlit Web Application."""

from __future__ import annotations

import os
import sys
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import altair as alt
from dotenv import load_dotenv
import pandas as pd
import streamlit as st


load_dotenv()

from src.article_extractor import validate_url_security
from src.verifier import verify_news_article, VerificationResult
from src.analytics import load_logs_dataframe, get_analytics_metrics, filter_recent_verifications
from src.health import run_system_health_checks
from src.search_provider import get_search_provider
from src.observability import global_telemetry


st.set_page_config(
    page_title="Real-Time News Verifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    /* Global Styles & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Header */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }

    /* Verdict Banners */
    .verdict-box-real {
        background: linear-gradient(135deg, #065f46 0%, #059669 50%, #10b981 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.4);
        margin: 18px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .verdict-box-fake {
        background: linear-gradient(135deg, #881337 0%, #be123c 50%, #e11d48 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(225, 29, 72, 0.4);
        margin: 18px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .verdict-title {
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: 2px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .verdict-meta {
        font-size: 1.15rem;
        font-weight: 600;
        opacity: 0.95;
        margin-top: 6px;
    }

    /* Metrics & Badge Pills */
    .status-badge-true {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-unverified {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-contradicted {
        background-color: #ffe4e6;
        color: #9f1239;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .tier-badge-high {
        background-color: #e0e7ff;
        color: #3730a3;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .tier-badge-med {
        background-color: #f1f5f9;
        color: #334155;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .tier-badge-low {
        background-color: #fee2e2;
        color: #991b1b;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }

    /* Cards */
    .claim-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .source-item {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.9rem;
    }

    /* Latency Chips */
    .latency-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 0.85rem;
        color: #475569;
        margin: 14px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def render_sidebar():
    """Renders persistent sidebar navigation, settings, and quick system status."""
    st.sidebar.markdown("### 🛡️ News Verifier")
    st.sidebar.caption("Evidence-Based Real-Time Verification")

    page = st.sidebar.radio(
        "Navigation",
        ["🔍 News Verifier", "📊 Analytics & Observability"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### System Configuration")

    provider = get_search_provider()
    st.sidebar.markdown(f"**Search Engine:** `{provider.get_provider_name()}`")

    llm_choice = os.getenv("LLM_PROVIDER", "none").lower()
    llm_desc = "Gemini AI" if llm_choice == "gemini" else ("OpenAI" if llm_choice == "openai" else "Rule-based NLP (Zero-Key)")
    st.sidebar.markdown(f"**Claim Extractor:** `{llm_desc}`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### About the Architecture")
    st.sidebar.caption(
        "This application does **not** rely on training datasets or static ML classifiers (no WELFake, TF-IDF, or Random Forest). "
        "Every article is verified in real time: **Article → Checkable Claims → Live Web Search → Evidence Analysis → REAL / FAKE**."
    )

    return page


def render_verifier_page():
    """Page 1: News Verifier - Primary verification interface."""
    st.markdown('<div class="hero-title">Real-Time News Verification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Extract checkable claims from news articles and verify them against live web evidence and authoritative sources.</div>',
        unsafe_allow_html=True,
    )

    # Example Preset Buttons for Instant Testing
    st.markdown("##### ⚡ Quick Try Examples")
    col1, col2 = st.columns(2)
    preset_url = None
    preset_headline = None
    preset_body = None

    if col1.button("🟢 Real: ISRO Satellite Launch", use_container_width=True):
        st.session_state["manual_headline"] = "ISRO Successfully Launches Earth Observation Satellite EOS-05"
        st.session_state["manual_body"] = (
            "The Indian Space Research Organisation (ISRO) has successfully placed the EOS-05 Earth observation satellite "
            "in geosynchronous orbit on Monday from Sriharikota. The mission aims to bolster disaster management and agricultural monitoring."
        )
        st.session_state["headline_input_box"] = st.session_state["manual_headline"]
        st.session_state["body_input_box"] = st.session_state["manual_body"]
        st.session_state["url_input_box"] = ""

    if col2.button("🔴 Fake: Eiffel Tower Collapsed", use_container_width=True):
        st.session_state["manual_headline"] = "Eiffel Tower Collapsed in Severe Paris Storm"
        st.session_state["manual_body"] = (
            "Officials in Paris reported that the Eiffel Tower collapsed completely on Monday following an unprecedented lightning strike. "
            "The French government declared a national state of emergency as engineers inspect the ruins."
        )
        st.session_state["headline_input_box"] = st.session_state["manual_headline"]
        st.session_state["body_input_box"] = st.session_state["manual_body"]
        st.session_state["url_input_box"] = ""

    # Input Mode Tabs (Option A: Manual Article, Option B: Article URL)
    input_tab1, input_tab2 = st.tabs(["✍️ Option A — Manual Article", "🌐 Option B — Article URL"])

    headline_input = ""
    body_input = ""
    url_input = ""

    with input_tab1:
        st.markdown("**Enter Article Headline & Body:**")
        headline_input = st.text_input(
            "Headline",
            value=st.session_state.get("manual_headline", ""),
            placeholder="Enter article headline...",
            key="headline_input_box",
        )
        body_input = st.text_area(
            "Article Body",
            value=st.session_state.get("manual_body", ""),
            placeholder="Paste the full article body text here...",
            height=150,
            key="body_input_box",
        )

    with input_tab2:
        st.markdown("**Enter News Article URL:**")
        url_input = st.text_input(
            "Article URL",
            placeholder="https://example.com/news/article-headline",
            label_visibility="collapsed",
            key="url_input_box",
        )
        st.caption("Protected with SSRF guards. Supports HTTP/HTTPS news publications.")

    verify_btn = st.button("🚀 Verify News", type="primary", use_container_width=True)

    if verify_btn:
        has_manual = bool((headline_input and headline_input.strip()) or (body_input and body_input.strip()))
        has_url = bool(url_input and url_input.strip())

        if not has_manual and not has_url:
            st.error("Please enter an Article Headline/Body or provide a valid News Article URL.")
            return

        # Perform Verification with Progress Bar
        with st.status("🔍 Analyzing article through verification pipeline...", expanded=True) as status:
            st.write("1. 📥 Extracting & sanitizing article content...")
            st.write("2. 🧩 Extracting checkable factual claims...")
            st.write("3. 🌐 Executing real-time web search across authoritative sources...")
            st.write("4. ⚖️ Analyzing evidence stance & source reliability...")
            st.write("5. 🎯 Computing REAL / FAKE classification and verification confidence...")

            if has_manual and not has_url:
                result = verify_news_article(headline=headline_input.strip(), body=body_input.strip())
            elif has_url and not has_manual:
                result = verify_news_article(url=url_input.strip())
            else:
                # If both are present, prioritize manual if preset/custom entered, or URL if custom
                result = verify_news_article(headline=headline_input.strip(), body=body_input.strip())

            status.update(label="✅ Verification Complete!", state="complete", expanded=False)

       
        st.session_state["latest_result"] = result

        # DISPLAY RESULTS
        render_verification_verdict(result)


def render_verification_verdict(result: VerificationResult):
    """Renders the main REAL / FAKE visual banner, confidence, status, and claim breakdown."""
    st.markdown("---")


    if result.final_label == "REAL":
        st.markdown(
            f"""
            <div class="verdict-box-real">
                <div class="verdict-title">🟢 REAL</div>
                <div class="verdict-meta">
                    Status: <strong>{result.overall_status}</strong> &nbsp;|&nbsp; 
                    Verification Confidence: <strong>{result.confidence_percent}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-box-fake">
                <div class="verdict-title">🔴 FAKE</div>
                <div class="verdict-meta">
                    Status: <strong>{result.overall_status}</strong> &nbsp;|&nbsp; 
                    Verification Confidence: <strong>{result.confidence_percent}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Explanation callout
    st.markdown("### 📝 Verification Explanation")
    if result.overall_status == "UNVERIFIED":
        st.warning(
            f"{result.explanation}\n\n"
            "*Note: UNVERIFIED does not mean the information is proven false—it indicates "
            "that current reliable web search evidence is insufficient to corroborate the claim.*"
        )
    elif result.overall_status == "CONTRADICTED":
        st.error(f"{result.explanation}")
    else:
        st.success(f"{result.explanation}")

    # Gather 3-4 Sources to display
    display_sources = []
    if result.top_sources:
        display_sources = result.top_sources[:4]
    elif result.claims_analysis:
        for c in result.claims_analysis:
            pool = c.supporting_sources if result.final_label == "REAL" else (c.contradicting_sources or c.neutral_sources)
            for s in pool:
                if s.url not in [ds.get("url") for ds in display_sources]:
                    display_sources.append({
                        "domain": s.domain,
                        "title": s.title,
                        "url": s.url,
                        "snippet": s.snippet,
                        "quality": s.quality_level,
                    })
                if len(display_sources) >= 4:
                    break
            if len(display_sources) >= 4:
                break

    # Display Sources Section
    if result.final_label == "REAL" or result.overall_status == "LIKELY TRUE":
        st.markdown("### 🟢 Supporting Sources")
    elif result.overall_status == "CONTRADICTED":
        st.markdown("### 🔴 Contradicting Sources")
    else:
        st.markdown("### 🔍 Retrieved Web Sources")

    if display_sources:
        for idx, s in enumerate(display_sources):
            q_tier = s.get("quality", "HIGH")
            tier_class = (
                "tier-badge-high"
                if q_tier == "HIGH"
                else ("tier-badge-med" if q_tier == "MEDIUM" else "tier-badge-low")
            )
            st.markdown(
                f"""
                <div class="source-item">
                    <span class="{tier_class}">{q_tier} QUALITY</span> &nbsp;
                    <strong><a href="{s.get('url', '#')}" target="_blank" rel="noopener noreferrer">{s.get('domain', 'source')}</a></strong> — {s.get('title', '')}<br>
                    <span style="color: #475569; font-size: 0.88rem;">"{s.get('snippet', '')}"</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No authoritative sources could be retrieved for this submission.")

    # Technical Details & Latency (Clean collapsed expander)
    with st.expander("⚙️ Technical Performance & Observability", expanded=False):
        lats = result.pipeline_latencies_ms
        st.markdown(
            f"""
            <div class="latency-bar">
                <span>⏱️ <strong>Request ID:</strong> <code>{result.request_id}</code></span>
                <span>Extraction: <strong>{lats.get('article_extraction', 0)} ms</strong></span>
                <span>Claim/Query: <strong>{lats.get('claim_extraction', 0)} ms</strong></span>
                <span>Web Search: <strong>{lats.get('search', 0)} ms</strong></span>
                <span>Verification: <strong>{lats.get('verification', 0)} ms</strong></span>
                <span>Total: <strong>{lats.get('total', 0)} ms</strong></span>
                <span>Engine: <code>{result.api_provider_used}</code></span>
            </div>
            """,
            unsafe_allow_html=True,
        )




def render_analytics_page():
    """Page 3: Analytics & Observability Dashboard."""
    st.markdown('<div class="hero-title">Analytics & Observability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Live monitoring of verification throughput, accuracy distributions, latency benchmarks, and system health.</div>',
        unsafe_allow_html=True,
    )

    # 1. System Health Section
    st.markdown("### 🏥 System Health")
    health_results = run_system_health_checks()
    h_cols = st.columns(len(health_results))

    for idx, h in enumerate(health_results):
        dot_color = "🟢" if h.status == "Healthy" else ("🟡" if h.status == "Degraded" else "🔴")
        with h_cols[idx]:
            st.metric(
                label=f"{dot_color} {h.component}",
                value=h.status,
                delta=f"{h.latency_ms:.1f} ms",
                delta_color="off",
            )
            st.caption(h.details)

    st.markdown("---")

    # Load Logs Dataframe
    df = load_logs_dataframe()
    metrics = get_analytics_metrics(df)

    if not metrics.get("has_data", False):
        st.info("No analytics data available yet.\n\nVerify some articles to populate the dashboard.")
        return

    overview = metrics["overview"]
    claims = metrics["claims"]
    evidence = metrics["evidence"]
    perf = metrics["performance"]
    rel = metrics["reliability"]

    # 2. Verification Overview KPIs
    st.markdown("### 📈 Verification Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Articles", overview["total_articles"])
    kpi2.metric("REAL Count", f"{overview['real_count']} ({overview['real_percentage']}%)")
    kpi3.metric("FAKE Count", f"{overview['fake_count']} ({overview['fake_percentage']}%)")
    kpi4.metric("Claims Checked", overview["total_claims"])
    kpi5.metric("Avg Confidence", f"{overview['avg_confidence']}%")

    # 3. Analytics Charts Row 1
    st.markdown("### 📊 Distribution Analytics")
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        st.markdown("##### REAL vs FAKE Distribution")
        rf_df = pd.DataFrame({
            "Verdict": ["REAL", "FAKE"],
            "Count": [overview["real_count"], overview["fake_count"]],
        })
        chart_rf = (
            alt.Chart(rf_df)
            .mark_bar(cornerRadius=6)
            .encode(
                x=alt.X("Verdict:N", title="Verdict"),
                y=alt.Y("Count:Q", title="Articles"),
                color=alt.Color("Verdict:N", scale=alt.Scale(domain=["REAL", "FAKE"], range=["#10b981", "#ef4444"])),
            )
            .properties(height=260)
        )
        st.altair_chart(chart_rf, use_container_width=True)

    with ch_col2:
        st.markdown("##### Claim Status Breakdown")
        claim_df = pd.DataFrame({
            "Status": ["LIKELY TRUE", "UNVERIFIED", "CONTRADICTED"],
            "Count": [claims["likely_true_count"], claims["unverified_count"], claims["contradicted_count"]],
        })
        chart_claims = (
            alt.Chart(claim_df)
            .mark_bar(cornerRadius=6)
            .encode(
                x=alt.X("Status:N", title="Claim Status"),
                y=alt.Y("Count:Q", title="Total Claims"),
                color=alt.Color("Status:N", scale=alt.Scale(domain=["LIKELY TRUE", "UNVERIFIED", "CONTRADICTED"], range=["#10b981", "#f59e0b", "#ef4444"])),
            )
            .properties(height=260)
        )
        st.altair_chart(chart_claims, use_container_width=True)

    # 4. Analytics Charts Row 2: Performance Latency
    st.markdown("### ⏱️ Performance & Latencies")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Avg Total Latency", f"{perf['avg_total_ms']:.0f} ms")
    p2.metric("Article Extraction", f"{perf['avg_extraction_ms']:.0f} ms")
    p3.metric("Claim Extraction", f"{perf['avg_claim_ms']:.0f} ms")
    p4.metric("Web Search", f"{perf['avg_search_ms']:.0f} ms")
    p5.metric("Evidence Analysis", f"{perf['avg_verification_ms']:.0f} ms")

    if "datetime" in df.columns and df["datetime"].notna().any():
        st.markdown("##### Processing Latency Over Time (ms)")
        perf_chart = (
            alt.Chart(df)
            .mark_line(point=True, color="#3b82f6")
            .encode(
                x=alt.X("timestamp_clean:N", title="Time (UTC)"),
                y=alt.Y("total_processing_time_ms:Q", title="Total Time (ms)"),
                tooltip=["request_id", "total_processing_time_ms", "final_label"],
            )
            .properties(height=240)
        )
        st.altair_chart(perf_chart, use_container_width=True)

    # 5. Top Source Domains & Error Analytics
    st.markdown("### 🌐 Evidence & Reliability")
    rel_col1, rel_col2 = st.columns(2)

    with rel_col1:
        st.markdown("##### Top Source Domains")
        if evidence["top_domains"]:
            dom_df = pd.DataFrame(
                list(evidence["top_domains"].items())[:8],
                columns=["Domain", "Frequency"]
            )
            dom_chart = (
                alt.Chart(dom_df)
                .mark_bar(cornerRadius=4, color="#6366f1")
                .encode(
                    x=alt.X("Frequency:Q", title="Articles Checked"),
                    y=alt.Y("Domain:N", sort="-x", title="Domain"),
                )
                .properties(height=240)
            )
            st.altair_chart(dom_chart, use_container_width=True)
        else:
            st.caption("No domain records available.")

    with rel_col2:
        st.markdown("##### System Reliability Metrics")
        st.metric("Success Rate", f"{100.0 - rel['error_rate_pct']:.1f}%")
        st.write(f"- **Successful Requests:** {rel['successful_requests']}")
        st.write(f"- **Failed Requests:** {rel['failed_requests']}")
        st.write(f"- **Extraction Failures:** {rel['extraction_failures']}")
        st.write(f"- **Search Failures:** {rel['search_failures']}")
        st.write(f"- **Timeout Errors:** {rel['timeout_errors']}")

    # 6. Filterable Recent Verifications Table
    st.markdown("---")
    st.markdown("### 📋 Recent Verification History")

    f_c1, f_c2, f_c3 = st.columns(3)
    filter_res = f_c1.selectbox("Filter Verdict", ["ALL", "REAL", "FAKE"])
    filter_stat = f_c2.selectbox("Filter Status", ["ALL", "LIKELY TRUE", "UNVERIFIED", "CONTRADICTED"])
    filter_inp = f_c3.selectbox("Filter Input", ["ALL", "url", "manual"])

    recent_table = filter_recent_verifications(
        df,
        result_filter=filter_res,
        status_filter=filter_stat,
        input_filter=filter_inp,
    )

    if not recent_table.empty:
        st.dataframe(recent_table, use_container_width=True, hide_index=True)
    else:
        st.caption("No log entries match the selected filters.")


def main():
    selected_page = render_sidebar()

    if selected_page == "🔍 News Verifier":
        render_verifier_page()
    elif selected_page == "📊 Analytics & Observability":
        render_analytics_page()


if __name__ == "__main__":
    main()
