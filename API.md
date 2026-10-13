# REST API

FastAPI runs alongside Streamlit at `http://localhost:8000`; interactive API docs are available at `/docs`. The API and UI share run state through user-scoped JSON files under `data/users/<username>/runs/`.

## User Header

Every endpoint requires `X-Username`. It selects the user's isolated workspace; it is **not authentication** because callers can choose another username. Keep the API on a trusted network or add OAuth/API-key authentication at a gateway before exposing it externally.

## Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/uploads` | Upload one or more CSV/XLSX files to the user's landing folder (50 MB per file) |
| `POST` | `/api/runs` | Start a background pipeline run |
| `POST` | `/api/runs/{run_id}/approve` | Approve the current Bronze, Silver, or Gold STTM gate |
| `GET` | `/api/runs/{run_id}` | Get current state, phase, outputs, and error details |
| `GET` | `/api/runs/{run_id}/report` | Download the completed HTML report |

## Example Calls

Upload CSV files:

```bash
curl -X POST http://localhost:8000/api/uploads \
  -H "X-Username: alice" \
  -F "files=@data/landing/sales_data.csv" \
  -F "files=@data/landing/products.csv" \
  -F "files=@data/landing/stores.csv"
```

Use the returned server-side `files` paths to start a run. Keep `auto_approve` false to stop at human approval gates:

```bash
curl -X POST http://localhost:8000/api/runs \
  -H "X-Username: alice" \
  -H "Content-Type: application/json" \
  -d '{"input_files":["/app/data/users/alice/landing/sales_data.csv","/app/data/users/alice/landing/products.csv","/app/data/users/alice/landing/stores.csv"],"business_intent":"Which categories generated the most revenue by region?","auto_approve":false}'
```

To run unattended, set `auto_approve` to `true`; this accepts every generated STTM without a person reviewing it. Otherwise, poll the returned `run_id` and approve each gate when the state is `awaiting_bronze_sttm_approval`, `awaiting_silver_sttm_approval`, or `awaiting_gold_sttm_approval`:

```bash
curl -X POST http://localhost:8000/api/runs/RUN_ID/approve \
  -H "X-Username: alice" \
  -H "Content-Type: application/json" \
  -d '{"phase":"bronze"}'

curl http://localhost:8000/api/runs/RUN_ID -H "X-Username: alice"
```

Use `phase` values `bronze`, `silver`, and `gold` at their matching approval gates. Download the report when the run state is `completed`:

```bash
curl -L http://localhost:8000/api/runs/RUN_ID/report \
  -H "X-Username: alice" \
  -o report.html
```

The service uses FastAPI background tasks for convenience. For long-running production workloads that must survive API process restarts, move background execution to a durable worker/queue and protect the API with real authentication.