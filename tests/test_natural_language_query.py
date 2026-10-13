from types import SimpleNamespace

import pandas as pd
import pytest

from core import natural_language_query as query


class _FakeConnection:
    def __init__(self, statement_types):
        self.statement_types = statement_types

    def extract_statements(self, sql):
        return [SimpleNamespace(type=SimpleNamespace(name=value)) for value in self.statement_types]


def test_prompt_includes_gold_schema_and_previous_turn_context():
    prompt = query.build_sql_prompt(
        "What about East?",
        [{"table_name": "category_sales", "columns": [{"name": "region"}, {"name": "sales"}]}],
        [
            {"role": "user", "content": "Compare regions."},
            {
                "role": "assistant",
                "content": "Found 2 rows.",
                "sql": "SELECT region, SUM(sales) FROM category_sales GROUP BY region",
                "sample_rows": [{"region": "East", "sales": 20}],
            },
        ],
    )

    assert "category_sales (region, sales)" in prompt
    assert "What about East?" in prompt
    assert "Previous result sample" in prompt


def test_read_only_sql_validation_allows_single_select_and_strips_terminator():
    sql = "SELECT region, SUM(sales) FROM category_sales GROUP BY region;"

    assert query.validate_read_only_sql(_FakeConnection(["SELECT"]), sql) == sql[:-1]


@pytest.mark.parametrize(
    "sql,statement_types,message",
    [
        ("UPDATE category_sales SET sales = 0", ["UPDATE"], "read-only SELECT"),
        ("SELECT 1; DROP TABLE category_sales", ["SELECT", "DROP"], "one SQL statement"),
        ("SELECT * FROM read_parquet('secret.parquet')", ["SELECT"], "External data access"),
    ],
)
def test_read_only_sql_validation_rejects_unsafe_queries(sql, statement_types, message):
    with pytest.raises(ValueError, match=message):
        query.validate_read_only_sql(_FakeConnection(statement_types), sql)


def test_suggest_chart_selects_line_for_trends_and_bar_for_comparison():
    dataframe = pd.DataFrame({"month": ["2026-01", "2026-02"], "sales": [10, 12]})

    assert query.suggest_chart("Show the sales trend over time", dataframe) == {
        "type": "line",
        "x": "month",
        "y": "sales",
    }
    assert query.suggest_chart("Compare East and West", dataframe)["type"] == "bar"
