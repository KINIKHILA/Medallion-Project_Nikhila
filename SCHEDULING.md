# Scheduled Pipeline Runs

`run_pipeline.py` runs the existing orchestrator without Streamlit. It accepts one or more CSV/XLSX paths and a business question, logs to the console and a file, and returns an exit code suitable for cron or Task Scheduler.

## Parameters and Manual Test

Run from the project root with the virtual environment active:

```powershell
python run_pipeline.py `
  --input data/landing/sales_data.csv data/landing/products.csv data/landing/stores.csv `
  --business-intent "Which categories generated the most revenue by region?" `
  --log-file audit_logs/manual_run.log
```

Use repeated workbook/file paths after `--input`; `.xlsx` workbooks are expanded into one dataset per worksheet. The business question is passed to profiling/report generation. Without `--auto-approve`, the CLI stops after producing Bronze STTM and exits with code `2`. To run all phases unattended, explicitly add `--auto-approve`:

```powershell
python run_pipeline.py `
  --input data/landing/sales_data.csv data/landing/products.csv data/landing/stores.csv `
  --business-intent "Which categories generated the most revenue by region?" `
  --auto-approve `
  --log-file audit_logs/scheduled_run.log
```

Exit codes: `0` means complete, `1` means validation/agent failure, `2` means stopped for human STTM approval, and `130` means interrupted. Logs include timestamps, phase progress, and full unexpected exception tracebacks. Phase-level details are also written to `audit_logs/<run_id>.jsonl`.

**Auto-approval warning:** Scheduled runs accept generated STTM rules without a person reviewing them. Test with non-production data first and enable this only when unattended approval is acceptable.

## Cron (Linux/macOS)

Use absolute paths in cron and ensure the scheduled account can read `.env`, the input files, and write the output folders. Edit the user's crontab with `crontab -e`, then add a daily 02:00 run (replace paths and question as needed):

```cron
0 2 * * * cd /opt/medallion-pipeline && flock -n /tmp/idamp-pipeline.lock /opt/medallion-pipeline/.venv/bin/python /opt/medallion-pipeline/run_pipeline.py --input /opt/medallion-pipeline/data/landing/sales_data.csv /opt/medallion-pipeline/data/landing/products.csv /opt/medallion-pipeline/data/landing/stores.csv --business-intent "Which categories generated the most revenue by region?" --auto-approve --log-file /opt/medallion-pipeline/audit_logs/scheduled_run.log >> /opt/medallion-pipeline/audit_logs/cron_output.log 2>&1
```

`flock` prevents overlapping runs on systems that provide it. If it is unavailable, use your scheduler's equivalent no-overlap option.

## Windows Task Scheduler

Create a task and configure:

1. **Trigger:** choose the desired schedule, such as daily at 2:00 AM.
2. **Action:** Start a program.
3. **Program/script:** `C:\path\to\medallion-pipeline\.venv\Scripts\python.exe`
4. **Add arguments:**

   ```text
   "C:\path\to\medallion-pipeline\run_pipeline.py" --input "C:\path\to\medallion-pipeline\data\landing\sales_data.csv" "C:\path\to\medallion-pipeline\data\landing\products.csv" "C:\path\to\medallion-pipeline\data\landing\stores.csv" --business-intent "Which categories generated the most revenue by region?" --auto-approve --log-file "C:\path\to\medallion-pipeline\audit_logs\scheduled_run.log"
   ```

5. **Start in:** `C:\path\to\medallion-pipeline`
6. In task settings, choose **Do not start a new instance** if the task is already running.

Run the task once manually from Task Scheduler before relying on the schedule. Confirm the task account has access to `.env`, the input files, and output directories.

## Credentials and Logs

The CLI loads provider keys and optional SMTP settings from the project `.env` through `core.config`; do not put API keys or SMTP passwords in scheduled command arguments. Protect `.env` and scheduled log files with appropriate filesystem permissions. If report-email SMTP delivery is configured, email failures are recorded in the audit log and do not invalidate a completed pipeline run.