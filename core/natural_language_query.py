import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.reporter import _make_llm, _table_name
from core.audit import AuditLogger
from core.performance import record_api_call

MAX_RESULT_ROWS = 500


def load_gold_tables(gold_paths: Sequence[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import pandas as pd

    tables: Dict[str, Any] = {}
    catalog = []
    for gold_path in gold_paths:
        path = Path(gold_path)
        if not path.is_file() or path.suffix.lower() != ".parquet":
            raise ValueError(f"Gold table is not an available Parquet file: {path.name}")
        table_name = _table_name(path)
        if table_name in tables:
            raise ValueError(f"Gold files resolve to duplicate table name '{table_name}'.")
        dataframe = pd.read_parquet(path)
        tables[table_name] = dataframe
        catalog.append(
            {
                "table_name": table_name,
                "columns": [
                    {"name": str(column), "dtype": str(dtype)}
                    for column, dtype in dataframe.dtypes.items()
                ],
                "row_count": len(dataframe),
            }
        )
    if not tables:
        raise ValueError("This run has no Gold tables available for questions.")
    return tables, catalog


def build_sql_prompt(
    question: str,
    catalog: Sequence[Dict[str, Any]],
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    schema_entries = []
    for table in catalog:
        columns = ", ".join(
            column["name"] + (f" {column['dtype']}" if column.get("dtype") else "")
            for column in table["columns"]
        )
        schema_entries.append(f"- {table['table_name']} ({columns})")
    schema = "\n".join(schema_entries)
    prior_turns = []
    for message in (history or [])[-8:]:
        role = str(message.get("role", "")).upper()
        content = str(message.get("content", "")).strip()
        if role in {"USER", "ASSISTANT"} and content:
            prior_turns.append(f"{role}: {content}")
        if role == "ASSISTANT" and message.get("sql"):
            prior_turns.append(f"Previous SQL: {message['sql']}")
        if role == "ASSISTANT" and message.get("sample_rows"):
            prior_turns.append(
                "Previous result sample: "
                + json.dumps(message["sample_rows"], ensure_ascii=False, default=str)
            )
    history_text = "\n".join(prior_turns) or "No earlier questions in this chat."
    return f"""You translate a user's analytics question into one DuckDB SQL query.
Return SQL only, with no markdown or explanation.
Use only the listed Gold tables and columns. Never query files, system catalogs,
other databases, or external sources. Generate exactly one read-only SELECT query.
Use a CTE if needed. For time trends, use available date columns and order by time.
For comparisons, group by the relevant category. Limit detail results to 500 rows.

Gold table schemas:
{schema}

Recent conversation:
{history_text}

Current question:
{question}
"""


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    text = str(content).strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if text.lower().startswith("sql\n"):
        text = text[4:].strip()
    return text


def validate_read_only_sql(connection: Any, sql: str) -> str:
    query = str(sql or "").strip()
    if not query:
        raise ValueError("The generated SQL query was empty.")
    statements = connection.extract_statements(query)
    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")
    statement_type = getattr(statements[0].type, "name", str(statements[0].type))
    if str(statement_type).rsplit(".", 1)[-1].upper() != "SELECT":
        raise ValueError("Only read-only SELECT queries are allowed.")
    if re.search(
        r"\b(read_csv|read_parquet|read_json|read_text|glob|httpfs|sqlite_scan|postgres_scan)\s*\(",
        query,
        flags=re.IGNORECASE,
    ):
        raise ValueError("External data access is not allowed in follow-up queries.")
    return query[:-1].rstrip() if query.endswith(";") else query


def answer_gold_question(
    question: str,
    tables: Dict[str, Any],
    catalog: Sequence[Dict[str, Any]],
    run_id: str,
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    import duckdb

    if not question.strip():
        raise ValueError("Enter a question about the Gold data.")

    audit_logger = AuditLogger(run_id)
    connection = duckdb.connect(
        database=":memory:",
        config={"enable_external_access": "false"},
    )
    try:
        for table_name, dataframe in tables.items():
            connection.register(table_name, dataframe)
        prompt = build_sql_prompt(question, catalog, history)
        response = _make_llm().invoke(prompt)
        record_api_call(run_id, "natural_language_query", response)
        sql = validate_read_only_sql(connection, _message_text(response))
        audit_logger.log(
            "natural_language_query",
            "sql_generated",
            run_id=run_id,
            question=question,
            sql=sql,
            gold_tables=[table["table_name"] for table in catalog],
        )
        bounded_query = f"SELECT * FROM ({sql}) AS _nlq_result LIMIT {MAX_RESULT_ROWS + 1}"
        dataframe = connection.execute(bounded_query).fetchdf()
        truncated = len(dataframe) > MAX_RESULT_ROWS
        if truncated:
            dataframe = dataframe.iloc[:MAX_RESULT_ROWS].copy()
        audit_logger.log(
            "natural_language_query",
            "query_completed",
            run_id=run_id,
            row_count=len(dataframe),
            truncated=truncated,
        )
        return {
            "sql": sql,
            "dataframe": dataframe,
            "truncated": truncated,
            "row_count": len(dataframe),
        }
    except Exception as error:
        audit_logger.log(
            "natural_language_query",
            "query_failed",
            run_id=run_id,
            question=question,
            error=str(error),
        )
        raise
    finally:
        connection.close()


def suggest_chart(question: str, dataframe: Any) -> Optional[Dict[str, str]]:
    if dataframe.empty or len(dataframe.columns) < 2:
        return None
    numeric_columns = list(dataframe.select_dtypes(include="number").columns)
    if not numeric_columns:
        return None
    value_column = str(numeric_columns[0])
    category_columns = [str(column) for column in dataframe.columns if str(column) != value_column]
    if not category_columns:
        return None
    question_text = question.casefold()
    chart_type = "line" if any(word in question_text for word in ("trend", "over time", "monthly")) else "bar"
    return {
        "type": chart_type,
        "x": category_columns[0],
        "y": value_column,
    }
