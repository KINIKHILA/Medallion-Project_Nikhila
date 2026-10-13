# Docker Deployment

## Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine plus the Docker Compose plugin (Linux).
- A configured `.env` file with one LLM provider and key. Copy `.env.example` to `.env`; never bake credentials into the image.

## Build and Run

From the project root, in PowerShell:

```powershell
Copy-Item .env.example .env  # Skip if .env already exists, then edit it.
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f app
docker compose logs -f api
```

Open `http://localhost:8501` for Streamlit and `http://localhost:8000/docs` for the FastAPI Swagger UI. The `app` container runs Streamlit on port `8501`; the `api` container runs Uvicorn on port `8000`.

Stop the service with:

```powershell
docker compose down
```

To rebuild after changing code or dependencies:

```powershell
docker compose up --build -d
```

## Persistence and Configuration

Compose injects `.env` at runtime and mounts `data/`, `reports/`, `audit_logs/`, and `.chroma/` from the host. User workspaces, Parquet layers, profiles, STTM files, traces, and metrics live below `data/`; landing samples are included in the image while generated output folders are excluded from the build context.

The image does not include `.env`. Keep the file out of source control and restrict access to it. For PostgreSQL, use a database host reachable from inside the container; `localhost` inside the container refers to the container itself, not the host machine. Use TLS and a least-privilege database account.

## Troubleshooting

- Check startup output with `docker compose logs -f app`.
- If port `8501` is already occupied, change the left side of the Compose mapping, for example `8502:8501`, and open `http://localhost:8502`.
- If the container cannot write mounted output directories, check that the host directories exist and are writable by Docker.
- Verify the provider key is set in `.env`; do not add secret values to the Dockerfile or Compose file.