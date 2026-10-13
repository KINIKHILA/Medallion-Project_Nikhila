# 🏅 Agentic Medallion Pipeline

An AI-assisted retail analytics pipeline that profiles uploaded data, proposes medallion-layer transformation rules for human approval, and produces analytics-ready tables and business reports.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [REST API](#rest-api)
- [Configuration](#configuration)
- [FAQ](#faq)

## Overview

Traditional data engineering often requires engineers to inspect each source, write and maintain transformation code, validate mappings, orchestrate jobs, and build reports across multiple tools.

With this project, you upload CSV or Excel workbooks, describe the business question, review and approve each proposed STTM (Source-to-Target Mapping), and receive a report with tables and charts. Each Excel worksheet is processed as a separate dataset.

The agents automatically profile data, propose transformation rules, ingest and cleanse records, join and aggregate Gold data, query it with SQL, and save report artifacts. Human approval gates keep generated mappings reviewable before each layer executes.

## Architecture

```text
+------------------+
| Streamlit UI     |
| Upload / Approve |
+--------+---------+
         |
         v
+------------------+
| Orchestrator     |
| Supervisor       |
+--------+---------+
         |
         +------------------+------------------+
         |                  |                  |
         v                  v                  v
+----------------+  +----------------+  +----------------+
| Profiler / STTM|  | Bronze / Silver|  | Gold / Reporter|
| generation     |  | agents         |  | agents         |
+--------+-------+  +--------+-------+  +--------+-------+
         |                   |                   |
         +-------------------+-------------------+
                             v
+-----------------------------------------------------+
| Storage: landing CSVs, Parquet layers, STTM,        |
| profiles, reports, audit JSONL, traces, Chroma     |
+-----------------------------------------------------+
```

| Phase | Agents | Outputs | Human gate |
| --- | --- | --- | --- |
| 1. Profile and Bronze STTM | Profiler, STTM Generator | Profile JSON, Bronze STTM CSV | Approve Bronze rules |
| 2. Bronze and Silver STTM | Bronze, STTM Generator | Bronze Parquet, Silver STTM CSV | Approve Silver rules |
| 3. Silver and Gold STTM | Silver, STTM Generator | Silver Parquet, Gold STTM CSV | Approve Gold rules |
| 4. Gold and report | Gold, Reporter | Gold Parquet, HTML and JSON report | Pipeline complete |

## How It Works

### Medallion flow

```text
Raw CSV/XLSX data -> Bronze -> Silver -> Gold -> Executive Report
```

| Layer | Purpose | Typical output |
| --- | --- | --- |
| Raw / Landing | Original uploaded CSV/XLSX files, kept available for inspection | `data/landing/` |
| Bronze | Raw values with minimal STTM-directed changes and load metadata | `data/bronze_layer/*_bronze.parquet` |
| Silver | Cleansed, deduplicated, standardized data with surrogate keys | `data/silver_layer/*_silver.parquet` |
| Gold | Joined and aggregated tables shaped for business questions | `data/gold_layer/*.parquet` |
| Report | SQL-backed analysis and Plotly visualizations | `reports/report_*.html` and `.json` |

Each specialist follows the agentic pattern **THINK → INSPECT → PLAN → ACT → VERIFY**. Tool calls perform the actual file inspection, transformations, queries, and persistence; the orchestrator checks their outputs and records phase status.

### Data Quality Gates

`core/data_quality.py` checks generated Bronze, Silver, and Gold Parquet files between stages. It reports null ratios and numeric IQR outliers, validates minimum/expected row counts and required columns when configured, and assigns a 0–1 score. The default minimum score is `0.70`; scores below the threshold or hard errors fail the pipeline before the next stage. The Streamlit review panel displays the score and individual issues.

### Error Recovery

Transient provider/network errors are retried up to three attempts with exponential backoff. Other failures record the failed phase, exception type, context, and traceback in the audit log and pipeline state. The UI offers **Retry Failed Step** to resume with the same run ID or **Start Over** to clear the current session; retries can repeat writes for the failed phase.

### Error Recovery

The supervisor retries recognized transient provider/network failures up to three attempts with exponential backoff. Other failures retain the failed phase, error code/type, safe context, and traceback in pipeline state and the audit JSONL. Streamlit displays the failure details and offers **Retry Failed Step** to reuse the same run ID, or **Start Over** to reset; retries may overwrite deterministic outputs for that phase.

## Project Structure

```text
medallion-pipeline/
├── agents/
│   ├── bronze_agent.py       # Raw CSV ingestion to Bronze Parquet
│   ├── gold_agent.py         # Gold joins and aggregations
│   ├── orchestrator.py       # Supervisor, phase tools, and state transitions
│   ├── profiler.py           # CSV schema and quality profiling
│   ├── reporter.py           # DuckDB queries and HTML/JSON reports
│   ├── silver_agent.py       # Silver cleansing and standardization
│   └── sttm_generator.py     # Bronze, Silver, and Gold mapping rules
├── api/
│   └── main.py               # FastAPI upload, run, status, approval, and report endpoints
├── app/
│   ├── streamlit_app.py      # Upload, approvals, progress, audit, and report UI
│   └── pages/
│       └── 1_Admin_Dashboard.py # Authenticated operational dashboard
├── core/
│   ├── admin_dashboard.py    # Cross-workspace run and health aggregation
│   ├── audit.py              # JSONL audit events
│   ├── cache.py              # Per-user content-addressed profile and STTM cache
│   ├── config.py             # Environment settings and project paths
│   ├── data_lineage.py       # Per-run column lineage graph and Graphviz output
│   ├── data_quality.py       # Null, row-count, and numeric anomaly checks
│   ├── data_io.py            # CSV and multi-sheet XLSX loading
│   ├── email_notifications.py # Optional SMTP report delivery
│   ├── exceptions.py         # Structured pipeline and phase errors
│   ├── memory.py             # Persistent Chroma document utility
│   ├── natural_language_query.py # Gold-scoped NLQ, SQL validation, and results
│   ├── observability.py      # Agent execution traces
│   ├── performance.py        # Per-run duration, RSS, token, and cost metrics
│   ├── retry.py              # Transient-error retry policy
│   ├── state.py              # Pydantic pipeline state model
│   ├── sttm_versions.py      # Immutable STTM history, comparison, and restore
│   └── user_workspace.py     # Context-local per-user data paths
├── data/
│   ├── landing/              # Source and sample CSV/XLSX files
│   ├── bronze_layer/         # Generated Bronze Parquet
│   ├── silver_layer/         # Generated Silver Parquet
│   ├── gold_layer/           # Generated Gold Parquet
│   ├── profiles/             # Profile JSON files
│   ├── sttm/                 # STTM CSV files
│   ├── sttm_versions/        # Immutable STTM snapshots by layer
│   ├── traces/               # Agent trace JSON files
│   ├── metrics/              # Per-run performance metrics
│   └── cache/                # Reusable profile and STTM JSON entries
├── reports/                  # Generated HTML and JSON reports
├── audit_logs/               # Per-run audit JSONL files
├── tests/                    # Unit tests for agents and utilities
│   ├── test_data_quality.py  # Data quality score and gate tests
│   ├── test_cache.py         # Cache keys, invalidation, and workspace isolation
│   ├── test_performance.py   # Performance metrics and cost summary tests
│   ├── test_admin_dashboard.py # Admin aggregation and authentication tests
│   ├── test_data_lineage.py  # Column-level lineage and persistence tests
│   └── test_natural_language_query.py # SQL policy and prompt-context tests
├── tools/                    # Additional utilities
├── .env.example              # Environment-variable template
├── .gitignore                # Excludes secrets and generated artifacts
├── run_pipeline.py           # Headless command-line pipeline runner
├── SCHEDULING.md             # Cron and Windows Task Scheduler instructions
├── requirements.txt          # Python dependencies
├── API.md                    # REST API routes and curl examples
└── test_import.py            # Quick orchestrator import check
```

Generated files such as Parquet outputs, profiles, STTM files, reports, audit logs, traces, and `.chroma` data are ignored by Git. CSVs under `data/landing/` remain trackable as sample/source data.

## Agents

- **Profiler**: previews CSV files and Excel worksheets, calculates column statistics, and writes a combined profile JSON using `inspect_files_tool` and `profiler_tool`.
- **STTM Generator**: inspects layer context and creates the requested mapping CSV through context inspection and layer-specific generation tools.
- **Bronze Agent**: applies Bronze column mappings/casts, adds `_load_timestamp` and `_source_file`, and writes Parquet through `inspect_task_tool` and `bronze_ingestion_tool`.
- **Silver Agent**: applies approved renames, null handling, deduplication, type/date/text normalization, surrogate keys, and approved-column filtering through `inspect_task_tool` and `silver_ingestion_tool`.
- **Gold Agent**: groups rules by target table, joins on shared `_id` columns, applies aggregations, filters output columns, and writes Gold Parquet through `inspect_task_tool` and `gold_aggregation_tool`.
- **Reporter**: loads Gold Parquet into DuckDB, runs read-only SQL, renders Plotly charts, and submits report analysis through four ordered tools.

The `core.memory` module is an optional persistent Chroma document store. Its current query helper performs text matching over stored documents; it does not calculate semantic embeddings.

### Custom STTM Transformations

The STTM editor allows editing `transformation_type` and `transformation_logic`. Custom types are `expression`, `regex_replace`, and `lookup`. Expressions use a restricted Python-like grammar (column names, arithmetic, comparisons, conditionals, and approved functions such as `round`, `lower`, `concat`, and `coalesce`); arbitrary Python, imports, attribute access, and filesystem/network operations are not executed.

Examples for the `transformation_logic` cell:

```text
expression: quantity * unit_price
regex_replace: {"pattern":"-","replacement":""}
lookup: {"mapping":{"Prem":"Premium","Std":"Standard"},"default":"Other"}
```

Regex logic is JSON with `pattern` and optional `replacement`; regex execution has a timeout. Lookup logic is JSON with a `mapping` object and optional `default`. The app validates selected custom rules before saving them, and the agents validate again at execution time.

### STTM Rule History

Generated, approved, and restored rules are stored as immutable JSON snapshots under `data/users/<username>/sttm_versions/<layer>/` (or `data/sttm_versions/` without an active user). The review screen lists each version's timestamp, run, reason, and rule count, and shows a unified diff between the current rules and a selected version. **Restore selected version** writes those rules back to the active STTM file while creating a new version linked to the restored version; version creation, unchanged rules, and restores are recorded in the run's audit log.

## Setup and Installation

### Prerequisites

- Python 3.10 or newer
- Git
- An API key for one supported model provider: Groq, Google Gemini, OpenAI, or GitHub Models

### Clone and install (Windows PowerShell)

```powershell
git clone https://github.com/KINIKHILA/Accelerate-with-AI.git
cd Accelerate-with-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, you can invoke the environment's Python directly with `.\.venv\Scripts\python.exe`.

### Other platforms

```bash
git clone https://github.com/KINIKHILA/Accelerate-with-AI.git
cd Accelerate-with-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Running the Pipeline

From the project root, with the virtual environment activated and `.env` configured:

```bash
streamlit run app/streamlit_app.py
```

Upload one or more CSV or XLSX files and enter a business question. Every worksheet in an XLSX workbook is handled as an independent dataset named `<workbook>_<sheet>`; each produces its own Bronze and Silver Parquet file. Review the proposed STTM rows at each approval gate; selected rows are saved back to the STTM CSV before the next phase runs. After Gold generation, the app displays the executive report and provides an HTML download.

Below the report, users can ask follow-up questions against that run's Gold tables. Recent questions and result samples carry context across turns; results appear inline with an optional trend/comparison chart and expandable SQL.

The report page also displays a Graphviz lineage diagram from uploaded files and source columns through Bronze, Silver, and Gold transformations to the report. The graph is rebuilt and saved under `data/users/<username>/traces/lineage_<run_id>.json` (or `data/traces/` without an active user) whenever run state is saved, so the artifact grows with each completed phase. In Streamlit, filter by stage, select a node to highlight it and inspect its mapping/logic, or download the DOT graph.

SQL is generated from the available Gold schema, accepted only as one read-only `SELECT`, and executed against in-memory registered Gold DataFrames with DuckDB external file access disabled. Results are capped at 500 rows and questions, SQL, outcomes, and failures are written to the run audit log.

A quick import check is also available:

```bash
python test_import.py
```

For headless runs and scheduled automation, see [SCHEDULING.md](SCHEDULING.md). Full unattended execution requires the explicit `--auto-approve` flag because it accepts generated STTM rules without a human review.

## REST API

Start the API alongside Streamlit in a second terminal:

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive endpoint explorer. See [API.md](API.md) for upload, run, approval, status, and report-download examples. The API and UI share user-scoped JSON run state through the mounted data directory.

## Configuration

Copy `.env.example` to `.env`, choose a provider, and add the corresponding secret key. Only the selected provider's key is needed.

| Setting | Purpose |
| --- | --- |
| `LLM_PROVIDER` | Select `groq`, `gemini`, `openai`, or `github`; if omitted, configured keys are detected in GitHub, OpenAI, Groq, Gemini priority order |
| `GROQ_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN` | Credentials for the selected model provider |
| `GROQ_MODEL`, `GEMINI_MODEL`, `OPENAI_MODEL`, `GITHUB_MODEL` | Provider model names; defaults are provided in `.env.example` |
| `GITHUB_BASE_URL` | GitHub Models inference endpoint |
| `EMBEDDING_MODEL` | Configured embedding model name for optional memory integrations |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | Required credentials for the Streamlit Admin Dashboard; leave blank to disable it |

The Admin Dashboard is available as a separate Streamlit page. It aggregates persisted runs and metrics across the default workspace and all `data/users/*` workspaces, and displays run statuses, failure rates/categories, agent durations/API calls, system memory, and disk usage for the data volume. Configure both admin values in `.env`; the login form compares credentials in constant time. This is application-level login, not full identity management; use HTTPS and a trusted reverse proxy or identity provider before exposing the app outside a trusted network, and never commit `.env`.

### PostgreSQL Gold Export

Install the PostgreSQL packages from `requirements.txt` (`SQLAlchemy` and `psycopg[binary]`). Set `DATABASE_URL` to enable export after each Gold Parquet table is written, and optionally set `GOLD_DB_SCHEMA` (defaults to `public`). For example:

```dotenv
DATABASE_URL=postgresql+psycopg://db_user:URL_ENCODED_PASSWORD@db.example.com:5432/analytics?sslmode=require
GOLD_DB_SCHEMA=public
```

The URL format is `postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE`. URL-encode reserved characters in the username or password (for example, `@` becomes `%40`). Use TLS (`sslmode=require` when supported), a dedicated database user with only the required schema/table permissions, and never commit the URL or password. Keep `DATABASE_URL` blank to disable database export.

Gold tables are created automatically from the DataFrame columns. Each run uses a full refresh (`if_exists="replace"`), so matching tables are replaced rather than appended and duplicated.

### Email Notifications

Email delivery is optional and uses Python's standard-library SMTP support, so no additional package is required. Leave `SMTP_HOST` or `SMTP_TO` empty to disable notifications. To enable delivery, set `SMTP_HOST`, `SMTP_PORT` (usually `587`), `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_TO`, and `SMTP_USE_TLS=true` in `.env`.

For Gmail, use `smtp.gmail.com` on port `587` with TLS. Create an App Password under Google Account **Security → 2-Step Verification → App passwords**; use that as `SMTP_PASSWORD`, not your normal account password. For Microsoft 365, use `smtp.office365.com` on port `587` with TLS and obtain mailbox SMTP credentials from your administrator; SMTP AUTH may need to be enabled. For other providers, get the SMTP host, port, and credentials from the email hosting provider.

`SMTP_PASSWORD` is a secret. Keep it only in `.env`, never commit it, and revoke/rotate it if it is exposed. After the HTML report is generated, the pipeline sends it as an attachment when SMTP is configured; SMTP failures are recorded in the audit log but do not undo the completed pipeline run.

### Performance Monitoring

Each agent trace records elapsed time, sampled peak process RSS, and API-call/token usage. Metrics are stored as JSONL under `data/metrics/` or the active user's metrics directory, summarized on the report page, and downloadable as JSON. Process RSS is shared by concurrent work in the same Streamlit process, so it is an operational estimate rather than a per-user memory measurement.

Estimated LLM cost is shown only when both `LLM_INPUT_COST_PER_1M_TOKENS` and `LLM_OUTPUT_COST_PER_1M_TOKENS` are configured in `.env`. Set the current provider/model prices in USD per million tokens; leave them blank if unknown. Prices change, so verify them with your provider before relying on estimates.

### Caching

Profiles are cached by source-file SHA-256, task description, profiler prompt, and configured model. Bronze, Silver, and Gold STTM rows are cached by normalized input context, task/business intent, prompt, and model. Cache entries are isolated under `data/users/<username>/cache/` when a user workspace is active; run metrics show cache hits and misses in the performance panel.

Changed file contents or model/prompt inputs naturally produce a new key. To manually invalidate the active user's cache, use **Clear my cache** in the run performance panel or call `core.cache.clear_cache()`; pass a namespace such as `profile` or `sttm_gold` to clear only that cache type.

Never commit `.env` or real credentials. The `.gitignore` excludes the secret file.

## FAQ

**Do I need API keys for all providers?**  
No. Configure one provider and its key. An explicit `LLM_PROVIDER` takes precedence over automatic detection.

**What happens at an approval gate?**  
Review the proposed STTM rows and select the transformations to approve. Only selected rows are kept for the next pipeline phase.

**Where are generated files saved?**  
Profiles go to `data/profiles/`, STTM files to `data/sttm/`, layer Parquet files to their corresponding `data/*_layer/` folders, reports to `reports/`, audit events to `audit_logs/`, and traces to `data/traces/`.

**Can I upload Excel workbooks with multiple worksheets?**  
Yes. Each worksheet is profiled and ingested separately, using a table name based on the workbook and sheet names. The workbook is read-only; the original file remains in `data/landing/`.

**How are multiple users' files separated?**  
The Streamlit app asks for a username and stores that session's data under `data/users/<username>/`, including landing files, layer outputs, reports, traces, Chroma storage, and audit history. The username selector is a workspace label, not secure authentication; put the app behind a trusted network or real authentication provider before handling private data.

**Can I keep sample CSVs in Git?**  
Yes. CSV files in `data/landing/` are intentionally not ignored; generated CSVs elsewhere under `data/` are ignored.

**Does the memory module currently do semantic search?**  
No. It stores documents in Chroma without an embedding function and provides simple text matching; semantic embedding/search can be added separately.

**What if imports fail?**  
Activate the project virtual environment, install `requirements.txt`, then run `python test_import.py` to see the import traceback.

**What if email is not sent?**  
Confirm `SMTP_HOST`, `SMTP_TO`, `SMTP_FROM` (or `SMTP_USERNAME`), and TLS settings in `.env`. For authenticated servers, provide both `SMTP_USERNAME` and `SMTP_PASSWORD`; Gmail requires an App Password.
