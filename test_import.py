"""Quick sanity check for the pipeline orchestrator imports."""

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

try:
	from agents.orchestrator import (
		run_bronze_to_silver_sttm,
		run_gold_and_report,
		run_silver_to_gold_sttm,
		run_until_bronze_sttm,
	)
except Exception as error:
	print(f"❌ IMPORT ERROR: {error}")
	traceback.print_exc()
else:
	print("✅ SUCCESS! All imports worked.")