# Real-Time News Verification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Testing: Pytest](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)](https://docs.pytest.org/)

A production-grade, evidence-based web application that verifies news articles in real time. Rather than relying on traditional, stale machine-learning fake-news classifiers or static datasets, this system extracts checkable factual claims from articles and evaluates them against live web evidence and authoritative sources.

> **Crucial Project Architecture Rule**:
> This system does **not** use a fake-news training dataset (no WELFake, Kaggle fake news corpora, TF-IDF, Logistic Regression, Random Forests, or neural-network classifiers). It performs **evidence-based real-time verification** by extracting claims and comparing them with current external sources.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Key Features](#3-key-features)
4. [How It Works](#4-how-it-works)
5. [Core Architecture](#5-core-architecture)
6. [Claim Verification Process](#6-claim-verification-process)
7. [REAL vs. FAKE Classification Logic](#7-real-vs-fake-classification-logic)
8. [Claim Evidence Statuses](#8-claim-evidence-statuses)
9. [Verification Confidence](#9-verification-confidence)
10. [Analytics & Observability Dashboard](#10-analytics--observability-dashboard)
11. [Technology Stack](#11-technology-stack)
12. [Project Structure](#12-project-structure)
13. [Installation](#13-installation)
14. [Environment Variables](#14-environment-variables)
15. [Running Locally](#15-running-locally)
16. [API & Provider Configuration](#16-api--provider-configuration)
17. [Automated Testing](#17-automated-testing)
18. [Security & SSRF Mitigation](#18-security--ssrf-mitigation)
19. [Limitations](#19-limitations)
20. [Future Improvements](#20-future-improvements)

---

## 1. Overview

The **Real-Time News Verification System** ingests a news article via a URL or manual text submission, isolates verifiable factual claims (events, dates, statistics, official announcements), searches the live web for corroborate or contradictory reporting from authoritative sources, and renders a decisive **REAL** or **FAKE** verdict with genuine evidence confidence.

---

## 2. Problem Statement

Traditional fake news detection systems suffer from severe structural shortcomings:
* **Data Staleness**: Classifiers trained on historical datasets (e.g. 2016-2020 election corpora) cannot fact-check breaking news or contemporary events.
* **Superficial Stylometric Bias**: TF-IDF and bag-of-words models mistake sensationalist vocabulary for falsity, failing to verify actual real-world facts.
* **Black-Box Hallucination**: Static neural networks often hallucinate explanations or claim events occurred without citing external sources.

This project resolves these failures through **real-time evidence retrieval**, grounding every decision in cited, live, reputable external sources.

---

## 3. Key Features

* **Dual Input Flexibility**: Verify news directly via URL (with automated metadata extraction) or via manual Headline + Body submission.
* **Strictly Evidence-Grounded**: No training datasets, synthetic fake ML metrics, or static weights.
* **SSRF-Protected Article Extraction**: Hardened URL parsing using `trafilatura` and `BeautifulSoup4` that prevents server-side request forgery (SSRF) and blocks loopback, private IP ranges, and cloud metadata endpoints.
* **Checkable Claim Decomposition**: Isolates checkable factual statements (events, numbers, dates, institutions) while discarding subjective rhetoric and speculative opinions.
* **Modular Search Provider Abstraction**: Pluggable support for Tavily, SerpAPI, Google Custom Search, DuckDuckGo / Live News RSS (zero-key mode), and deterministic Mock search for automated testing.
* **Hierarchical Source Reliability Layer**: Categorizes domains into `HIGH`, `MEDIUM`, and `LOW` authority tiers (wire services, `.gov`, `.edu`, established global news vs. questionable blogs/social media).
* **Decisive REAL / FAKE Verdicts**: Clear primary verdict with internal statuses (`LIKELY TRUE`, `UNVERIFIED`, `CONTRADICTED`).
* **Genuine Verification Confidence**: Evidence-based confidence score computed from source counts, quality tiers, independence, and agreement.
* **Comprehensive Analytics & Observability**: Dedicated dashboard tracking request latencies, system throughput, error distributions, top cited domains, and live component health.

---

## 4. How It Works

```text
Article URL / Text
       │
       ▼
[Article Extractor]  ──(SSRF Guard, Trafilatura & BS4)
       │
       ▼
[Text Preprocessor]  ──(HTML decoding, entity preservation)
       │
       ▼
[Claim Extractor]    ──(Checkable facts, dates, numbers, verbs)
       │
       ▼
[Query Generator]    ──(Concise factual search phrases)
       │
       ▼
[Live Web Search]    ──(Tavily / SerpAPI / Google / DuckDuckGo)
       │
       ▼
[Reliability Layer]  ──(Tier classification, recency, relevance)
       │
       ▼
[Evidence Analyzer]  ──(SUPPORT vs. CONTRADICT vs. NEUTRAL)
       │
       ▼
[Verifier Engine]    ──(Weighted aggregation & status mapping)
       │
       ├──► 🟢 REAL  or  🔴 FAKE Verdict Card
       ├──► Claim Cards with Clickable Source Links
       └──► Logs to logs/prediction_logs.csv (Request ID tracked)
```

---

## 5. Core Architecture

The architecture maintains strict separation of concerns across dedicated modules:

1. **Extraction Layer** (`src/article_extractor.py`, `src/preprocess.py`): Validates URLs, enforces SSRF safety, extracts body and metadata, normalizes text while preserving factual numbers and names.
2. **Claim Decomposition Layer** (`src/claim_extractor.py`): Uses LLMs (Gemini / OpenAI) or intelligent rule-based NLP to parse checkable factual propositions.
3. **Retrieval & Evidence Layer** (`src/search_provider.py`, `src/source_reliability.py`, `src/evidence_analyzer.py`): Dispatches queries across modular search backends, evaluates domain quality, and determines evidence stance.
4. **Verification & Verdict Layer** (`src/verifier.py`): Maps claim-level findings into overall **REAL** or **FAKE** verdicts.
5. **Observability & Telemetry Layer** (`src/observability.py`, `src/analytics.py`, `src/health.py`): Logs request telemetry, computes metrics, and verifies subsystem availability.
6. **User Interface** (`app/streamlit_app.py`): High-end, reactive Streamlit dashboard providing immediate visual insights.

---

## 6. Claim Verification Process

For every extracted claim:
1. **Search Query Formulation**: 2 targeted queries are generated by removing stop words and isolating named entities, dates, and action verbs.
2. **Evidence Gathering**: Web search returns up to 5 top candidate articles with titles, URLs, publication dates, and snippets.
3. **Source Reliability Assessment**: Each source domain is scored:
   * **HIGH**: Wire services (Reuters, AP), government (`.gov`), academic institutions (`.edu`), international agencies (ISRO, NASA, WHO).
   * **MEDIUM**: Established regional news, general trade publications.
   * **LOW**: Unverified blogs, social media platforms, clickbait domains.
4. **Stance Detection**: Evidence snippets are categorized into `SUPPORT`, `CONTRADICT`, or `NEUTRAL` based on direct refutation markers, numerical contradictions, or factual corroboration.
5. **Claim Status Assignment**: The claim is assigned one of three mutually exclusive internal statuses: `LIKELY TRUE`, `UNVERIFIED`, or `CONTRADICTED`.

---

## 7. REAL vs. FAKE Classification Logic

The primary verdict shown to the user is strictly **REAL** or **FAKE**:

| Internal Status | Final Verdict | Display Description |
| :--- | :---: | :--- |
| **LIKELY TRUE** | 🟢 **REAL** | Reliable, authoritative external reporting corroborates the central claims. |
| **UNVERIFIED** | 🔴 **FAKE** | The system could not find sufficient reliable evidence in current sources to corroborate the claims. |
| **CONTRADICTED** | 🔴 **FAKE** | Reliable external evidence directly conflicts with or refutes central claims in the article. |

### Crucial Nuance for `UNVERIFIED`:
When an article is marked **FAKE** due to unverified claims, the system explicitly clarifies:
> *"The system could not find sufficient reliable evidence to support the claim."*
It never falsely asserts that the information has been proven untrue, respecting scientific fact-checking standards.

---

## 8. Claim Evidence Statuses

Each individual claim receives one of three internal statuses:

1. **LIKELY TRUE**: Corroborated by high-authority sources or multiple independent news outlets reporting matching events.
2. **UNVERIFIED**: Insufficient reliable evidence could be located on the live web (e.g. unknown local events, obscure rumors, or paywalled items).
3. **CONTRADICTED**: Reliable reporting directly refutes, debunks, or contradicts the statement.

---

## 9. Verification Confidence

Verification Confidence is **never fabricated, random, or generated from a static model**:
* Ranges between **50% and 99%**.
* Factors in:
  1. **Source Authority**: High-tier domains boost confidence.
  2. **Source Diversity**: Multiple independent domains reporting the same facts increase confidence.
  3. **Agreement vs. Contradiction**: Ratio of supporting to contradicting snippets.
  4. **Claim Importance**: Heavy weighting given to central headlines versus minor details.
  5. **Recency**: Recent corroborating dates raise confidence for contemporary events.

---

## 10. Analytics & Observability Dashboard

A dedicated, full-featured analytics suite monitors the verification pipeline in real time:

* **Verification Overview**: Total articles processed, REAL vs. FAKE counts and percentages, total claims checked, average verification confidence.
* **Claim Analytics**: Breakdown of Likely True, Unverified, and Contradicted counts, average claims per article, and modal claim status.
* **Evidence Analytics**: Total sources searched, average sources per article, supporting vs. contradicting counts, and top cited source domains.
* **Pipeline Performance Benchmarks**: High-precision latency tracking across every phase (Article Extraction, Claim Extraction, Search, Evidence Analysis, Total Processing Time).
* **Reliability & Error Tracking**: Success vs. failure rates, extraction errors, timeouts, rate limits, and provider failures.
* **Recent Verification History**: Filterable table showing past verifications by result, status, input type, confidence, and latency.
* **System Health Section**: Live operational indicators for Article Extraction, Search Provider, LLM Provider, Evidence Engine, and Logging.

---

## 11. Technology Stack

* **Language**: Python 3.10+
* **Frontend UI**: Streamlit with custom CSS and Altair interactive visual charts
* **Article Extraction**: `trafilatura`, `beautifulsoup4`, `lxml`
* **Networking & Security**: `requests`, `urllib3`, `ipaddress`, `socket` (SSRF protection)
* **Data Processing**: `pandas`, `altair`
* **Search Integrations**: Tavily API, SerpAPI, Google Custom Search, DuckDuckGo / Live News RSS
* **LLM Engine (Optional)**: Google GenAI (`google.genai`), OpenAI API, plus built-in zero-key rule-based NLP fallback
* **Testing**: `pytest`, `pytest-mock`

---

## 12. Project Structure

```text
Real-Time-News-Verification/
│
├── app/
│   └── streamlit_app.py         # Multi-view Streamlit web application
│
├── src/
│   ├── __init__.py              # Package export
│   ├── article_extractor.py     # SSRF-protected article extractor (trafilatura & bs4)
│   ├── preprocess.py            # Text cleaning and factual preservation
│   ├── claim_extractor.py       # Checkable factual claim extractor (LLM + NLP rule fallback)
│   ├── search_provider.py       # Modular search providers (Tavily, SerpApi, Google, DuckDuckGo, Mock)
│   ├── source_reliability.py    # Source quality scoring, domain tiering, and recency analysis
│   ├── evidence_analyzer.py     # Stance detection (Support / Contradict / Neutral) and claim status
│   ├── verifier.py              # Overall REAL/FAKE verdict, status mapping, and confidence engine
│   ├── observability.py         # Request IDs, latency timing, prediction_logs.csv, telemetry
│   ├── analytics.py             # Log aggregation, statistical metrics, and chart preparation
│   └── health.py                # Live health checks for all subsystems
│
├── tests/
│   ├── test_extraction.py       # SSRF security and article extraction tests
│   ├── test_claims.py           # Checkable claims and importance tests
│   ├── test_search.py           # Modular search provider and query generator tests
│   ├── test_verification.py     # Verdict mapping, confidence, and pipeline tests
│   └── test_observability.py    # Logging privacy, telemetry, analytics, and health tests
│
├── logs/
│   └── prediction_logs.csv      # Structured verification log (strict PII & text privacy)
│
├── requirements.txt             # Project dependencies
├── .env.example                 # Environment configuration template
├── .gitignore                   # Version control ignore definitions
├── README.md                    # Comprehensive documentation
└── run_app.py                   # Convenience application launcher
```

---

## 13. Installation

### 1. Clone the repository
```bash
git clone https://github.com/Jayesh09871/RealTime-News-Verifier.git
cd RealTime-News-Verifier
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 14. Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your parameters as desired:
```env
# Search Provider (duckduckgo, tavily, serpapi, google, mock)
SEARCH_PROVIDER=duckduckgo
SEARCH_API_KEY=

# Google Custom Search ID (required only if SEARCH_PROVIDER=google)
GOOGLE_CSE_ID=

# LLM Provider (none, gemini, openai)
LLM_PROVIDER=none
LLM_API_KEY=
LLM_MODEL=gemini-2.0-flash

# System Settings
APP_ENV=development
LOG_LEVEL=INFO
SSRF_PROTECTION_ENABLED=true
MAX_ARTICLE_BYTES=5242880
REQUEST_TIMEOUT_SECONDS=10
```

> **Zero-Key Mode**: If no API keys are configured, the system automatically uses **DuckDuckGo / Live News RSS** for live search and the **NLP Rule-Based Claim Extractor**, enabling full out-of-the-box verification without any paid credentials!

---

## 15. Running Locally

Start the Streamlit application using either command:

### Option A: Using the convenience launcher
```bash
python run_app.py
```

### Option B: Using Streamlit directly
```bash
streamlit run app/streamlit_app.py
```

Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 16. API Configuration

* **Tavily Search**: Set `SEARCH_PROVIDER=tavily` and `SEARCH_API_KEY=tvly-...`.
* **SerpAPI**: Set `SEARCH_PROVIDER=serpapi` and `SEARCH_API_KEY=...`.
* **Google Custom Search**: Set `SEARCH_PROVIDER=google`, `SEARCH_API_KEY=...`, and `GOOGLE_CSE_ID=...`.
* **Google Gemini**: Set `LLM_PROVIDER=gemini` and `LLM_API_KEY=AIza...`.
* **OpenAI**: Set `LLM_PROVIDER=openai` and `LLM_API_KEY=sk-...`.

---

## 17. Automated Testing

The project includes an extensive test suite with 100% mocked external calls, requiring no live network connection or API keys:

```bash
pytest tests/ -v
```

All 24 unit and integration tests validate:
* SSRF protection blocking localhost, 127.0.0.1, private CIDR blocks, and non-HTTP protocols.
* Article extraction, fallback parsing, and timeout handling.
* Factual claim extraction, opinion filtering, and importance grading.
* Search query generation and MockSearchProvider determinism.
* Status-to-label mapping (`LIKELY TRUE` -> REAL, `UNVERIFIED` -> FAKE, `CONTRADICTED` -> FAKE).
* Verification confidence calculations and end-to-end pipeline execution.
* CSV privacy enforcement (ensuring full article text and keys are never logged).
* System health checks and analytics metric computations.

---

## 18. Security & SSRF Mitigation

To protect against Server-Side Request Forgery (SSRF) and malicious user inputs:
1. **Scheme Validation**: Only `http://` and `https://` protocols are allowed.
2. **DNS Pre-Resolution**: Hostnames are resolved via `socket.getaddrinfo` before connection.
3. **Private IP Filtering**: Loopback (`127.0.0.0/8`, `::1`), private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local (`169.254.0.0/16`), and cloud metadata IPs (`169.254.169.254`) are strictly blocked.
4. **Redirect Inspection**: HTTP redirects (301, 302, 307, 308) are followed manually, re-validating the destination IP on every hop (up to a 5-hop limit).
5. **Payload Size Capping**: HTTP responses are streamed and limited to 5 MB to prevent memory exhaustion attacks.
6. **Strict Timeouts**: Enforces a 10-second socket timeout to prevent Slowloris attacks.

---

## 19. Limitations

While this system delivers state-of-the-art evidence-based verification, users should understand inherent operational boundaries:
* **Evidence Lag**: News breaking within the last few minutes may not yet have sufficient authoritative coverage indexed on the web.
* **Paywalled Media**: Certain subscription-only publications may block full-text extraction.
* **Subjective Assertions**: Complex philosophical, moral, or purely political interpretations cannot be resolved through factual corroboration.
* **Absence of Proof vs. Proof of Absence**: An `UNVERIFIED` result signifies an absence of credible corroborating reporting, not definitive proof of fabrication.

---

## 20. Future Improvements

* **Fact-Checking Database Connectors**: Direct integration with Google Fact Check Tools API and ClaimReview schema repositories.
* **Cross-Lingual Verification**: Multilingual claim translation allowing foreign-language reporting to be cross-checked against global wires.
* **Image & Deepfake Forensics**: Reverse-image search and EXIF metadata verification for accompanying media.
* **Browser Extension**: Manifest V3 extension allowing instant 1-click verification directly inside browser tabs.
* **Human-in-the-Loop Review Queue**: Workflow allowing community fact-checkers to audit and annotate ambiguous verification cases.

---

## License

This project is licensed under the MIT License.
