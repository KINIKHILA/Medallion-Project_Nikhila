import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL = os.getenv("GITHUB_MODEL") or "openai/gpt-4.1-mini"
GITHUB_BASE_URL = os.getenv(
	"GITHUB_BASE_URL", "https://models.inference.ai.azure.com"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "models/text-embedding-004"

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "")
SMTP_TO = os.getenv("SMTP_TO", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {
	"1",
	"true",
	"yes",
	"on",
}

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
GOLD_DB_SCHEMA = os.getenv("GOLD_DB_SCHEMA", "public").strip() or "public"
LLM_INPUT_COST_PER_1M_TOKENS = os.getenv("LLM_INPUT_COST_PER_1M_TOKENS", "")
LLM_OUTPUT_COST_PER_1M_TOKENS = os.getenv("LLM_OUTPUT_COST_PER_1M_TOKENS", "")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

_VALID_PROVIDERS = {"groq", "gemini", "openai", "github"}
_configured_provider = os.getenv("LLM_PROVIDER", "").strip().lower()

if _configured_provider:
	if _configured_provider not in _VALID_PROVIDERS:
		raise ValueError(
			"LLM_PROVIDER must be one of: groq, gemini, openai, github"
		)
	LLM_PROVIDER = _configured_provider
else:
	_provider_keys = (
		("github", GITHUB_TOKEN),
		("openai", OPENAI_API_KEY),
		("groq", GROQ_API_KEY),
		("gemini", GOOGLE_API_KEY),
	)
	LLM_PROVIDER = next(
		(provider for provider, api_key in _provider_keys if api_key and api_key.strip()),
		None,
	)

DATA_DIR = BASE_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze_layer"
SILVER_DIR = DATA_DIR / "silver_layer"
GOLD_DIR = DATA_DIR / "gold_layer"
REPORTS_DIR = BASE_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"
CHROMA_DIR = BASE_DIR / ".chroma"
METRICS_DIR = DATA_DIR / "metrics"
CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = DATA_DIR / "run_states"

for directory in (
	DATA_DIR,
	LANDING_DIR,
	PROFILES_DIR,
	STTM_DIR,
	BRONZE_DIR,
	SILVER_DIR,
	GOLD_DIR,
	REPORTS_DIR,
	AUDIT_DIR,
	CHROMA_DIR,
	METRICS_DIR,
	CACHE_DIR,
	RUNS_DIR,
):
	directory.mkdir(parents=True, exist_ok=True)
