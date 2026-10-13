# Agentic Medallion Pipeline: Setup Checklist

Use this checklist from the project root before running the pipeline. On this machine, Python 3.12.0 is available; `pytest` is not installed in the active interpreter until project requirements are installed.

## 1. Expected Project Files

Tracked source and sample-data files should include:

```text
README.md
requirements.txt
.env.example
.gitignore
SETUP_CHECKLIST.md
SCHEDULING.md
DOCKER.md
API.md
Dockerfile
docker-compose.yml
.dockerignore
run_pipeline.py
test_import.py
api/__init__.py
api/main.py
app/pages/1_Admin_Dashboard.py
agents/__init__.py
agents/bronze_agent.py
agents/gold_agent.py
agents/orchestrator.py
agents/profiler.py
agents/reporter.py
agents/silver_agent.py
agents/sttm_generator.py
app/streamlit_app.py
core/__init__.py
core/audit.py
core/admin_dashboard.py
core/cache.py
core/config.py
core/data_lineage.py
core/data_quality.py
core/data_io.py
core/email_notifications.py
core/exceptions.py
core/memory.py
core/natural_language_query.py
core/observability.py
core/performance.py
core/retry.py
core/run_store.py
core/state.py
core/sttm_versions.py
core/user_workspace.py
data/landing/products.csv
data/landing/sales_data.csv
data/landing/stores.csv
tests/__init__.py
tests/test_audit.py
tests/test_data_quality.py
tests/test_bronze.py
tests/test_gold.py
tests/test_profiler.py
tests/test_reporter.py
tests/test_silver.py
tests/test_sttm.py
tests/test_excel_ingestion.py
tests/test_email_notifications.py
tests/test_user_workspace.py
tests/test_retry.py
tests/test_performance.py
tests/test_admin_dashboard.py
tests/test_natural_language_query.py
tests/test_data_lineage.py
```

These generated-output directories should also exist, though they may be empty before the first run:

```text
data/bronze_layer/
data/silver_layer/
data/gold_layer/
data/profiles/
data/sttm/
data/traces/
reports/
audit_logs/
tools/
.chroma/
```

`.env` is created locally from `.env.example` and should not be committed.

## 2. Verify Environment and Install

Run these commands in Windows PowerShell from the project root:

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Python should be version 3.10 or newer. If PowerShell blocks virtual-environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

## 3. Configure a Model Provider

Edit `.env` and set one provider plus its matching key. For example:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-key
```

Supported providers are `groq`, `gemini`, `openai`, and `github`. Their credentials are `GROQ_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, and `GITHUB_TOKEN`, respectively. Model names and the GitHub Models endpoint are listed in `.env.example`. If `LLM_PROVIDER` is omitted, configured keys are detected in GitHub, OpenAI, Groq, then Gemini priority order.

Optional email notification settings are `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_TO`, and `SMTP_USE_TLS`. Leave the host or recipient blank to disable notifications. SMTP passwords are sensitive; Gmail users should create an App Password rather than using their normal account password.

## 4. Run Verification Commands

With `.venv` activated:

```powershell
python test_import.py
python -m pytest -q
```

For command-line runs and schedules, see `SCHEDULING.md`; add `--auto-approve` only when unattended acceptance of STTM rules is approved.

The import check should print `SUCCESS! All imports worked.` The test suite should finish with a passing summary; import failures usually mean the virtual environment is inactive or `pip install -r requirements.txt` did not complete.

Start the app from the same project root:

```powershell
streamlit run app/streamlit_app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`.

## 5. Test the Full Workflow

Upload the included samples from `data/landing/` or test an Excel workbook:

- `sales_data.csv` contains transactions, duplicate IDs, inconsistent labels, mixed date formats, missing stores, and returns.
- `products.csv` contains product names, categories, and list prices.
- `stores.csv` contains store names, cities, states, and regions.
- An `.xlsx` workbook can contain multiple sheets; each sheet is profiled and ingested as its own named dataset.

For Excel, table names are generated as `<workbook-stem>_<sanitized-sheet-name>`; for example, `retail.xlsx` with a `North Sales` sheet becomes `retail_North_Sales`. Each sheet receives its own STTM mapping and Bronze Parquet output.

Example business question:

> Which product categories generated the most sales by region, and how did returns affect revenue by customer segment?

Expected progression:

1. **Upload & Intent:** The app saves CSVs under `data/landing/`, profiles their schemas and quality, and creates Bronze STTM rules.
2. **Bronze STTM Review:** Review and select mapping rules; approving them runs Bronze ingestion and proposes Silver STTM rules.
3. **Silver STTM Review:** Approving cleansing rules handles configured nulls, types, text, and duplicates, then proposes Gold STTM rules.
4. **Gold STTM Review:** Approving join and aggregation rules creates Gold Parquet tables and queries them for the report.
5. **Executive Report:** The HTML report appears in the app and can be downloaded; a JSON companion is also saved under `reports/`.

Generated outputs are saved under `data/bronze_layer/`, `data/silver_layer/`, `data/gold_layer/`, `data/profiles/`, `data/sttm/`, `data/traces/`, `audit_logs/`, and `reports/`.

## 6. Common Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Missing API key or provider error | `LLM_PROVIDER` and key do not match, or the key is blank | Set one provider in `.env` and its matching key; restart Streamlit after editing. |
| `ModuleNotFoundError` during imports | Dependencies were installed into another Python environment | Activate `.venv`, run `python -m pip install -r requirements.txt`, then `python test_import.py`. |
| Streamlit command not found or app will not start | Virtual environment is inactive, wrong working directory, or Streamlit install failed | Activate `.venv`, change to the project root, install requirements, then run `python -m streamlit run app/streamlit_app.py`. |
| Agent fails during a phase | Invalid STTM, missing input columns/files, API failure, or malformed model output | Read the on-screen error, verify selected STTM rows and provider credentials, then inspect `audit_logs/<run_id>.jsonl` and `data/traces/trace_<agent>_<run-id-prefix>.json`. |
| No report is generated | Gold output or report step failed | Check Gold Parquet files and audit/trace records; confirm the business question can be answered from Gold columns. |
| Parquet read/write error | PyArrow is missing or input is not valid Parquet | Activate `.venv`, reinstall requirements, and verify the upstream output file exists. |

Never put API keys in source files or commit `.env`; `.gitignore` excludes it.