import csv
import json
from pathlib import Path

from core import data_lineage, run_store, user_workspace


STTM_COLUMNS = [
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
]


def _write_rules(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=STTM_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_builds_column_and_transformation_lineage_to_report(tmp_path):
    source_path = tmp_path / "orders.csv"
    source_path.write_text("order_id\n1\n", encoding="utf-8")
    bronze_path = tmp_path / "orders_bronze.parquet"
    silver_path = tmp_path / "orders_silver.parquet"
    gold_path = tmp_path / "sales.parquet"
    for output_path in (bronze_path, silver_path, gold_path):
        output_path.touch()
    report_path = tmp_path / "report.html"
    report_path.touch()

    bronze_sttm = _write_rules(
        tmp_path / "bronze.csv",
        [
            {
                "source_table": "orders",
                "source_column": "order_id",
                "target_table": "orders",
                "target_column": "order_id",
                "transformation_type": "copy",
                "transformation_logic": "Copy order identifier.",
            }
        ],
    )
    silver_sttm = _write_rules(
        tmp_path / "silver.csv",
        [
            {
                "source_table": "orders_bronze",
                "source_column": "order_id",
                "target_table": "orders",
                "target_column": "order_id",
                "transformation_type": "cast",
                "transformation_logic": "Cast to integer.",
            },
            {
                "target_table": "orders",
                "target_column": "pk_orders_silver_id",
                "transformation_type": "surrogate_key",
                "transformation_logic": "Generate a surrogate key.",
            },
        ],
    )
    gold_sttm = _write_rules(
        tmp_path / "gold.csv",
        [
            {
                "source_table": "orders_silver",
                "source_column": "order_id",
                "target_table": "sales",
                "target_column": "order_id",
                "transformation_type": "group_key",
                "transformation_logic": "Group by order identifier.",
            }
        ],
    )

    graph = data_lineage.build_lineage_graph(
        {
            "run_id": "lineage-test",
            "status": "completed",
            "uploaded_files": [str(source_path)],
            "sttm_bronze_path": str(bronze_sttm),
            "sttm_silver_path": str(silver_sttm),
            "sttm_gold_path": str(gold_sttm),
            "bronze_output_paths": [str(bronze_path)],
            "silver_output_paths": [str(silver_path)],
            "gold_output_paths": [str(gold_path)],
            "report_path": str(report_path),
        }
    )

    node_kinds = {node["kind"] for node in graph["nodes"]}
    node_labels = {node["label"] for node in graph["nodes"]}
    assert {"source_file", "source_column", "transformation", "layer_column", "output_file", "report", "generated"} <= node_kinds
    assert "Source: orders.order_id" in "\n".join(node["details"] for node in graph["nodes"])
    assert "Cast to integer." in "\n".join(node["details"] for node in graph["nodes"])
    assert "report.html" in node_labels
    assert any(edge["label"] == "summarized in" for edge in graph["edges"])
    bronze_output = next(
        node for node in graph["nodes"]
        if node["kind"] == "output_file" and "orders_bronze.parquet" in node["label"]
    )
    bronze_target_column = next(
        node for node in graph["nodes"]
        if node["kind"] == "layer_column"
        and node.get("layer") == "bronze"
        and "orders.order_id" in node["label"]
    )
    bronze_input_column = next(
        node for node in graph["nodes"]
        if node["kind"] == "layer_column"
        and node.get("layer") == "bronze"
        and "orders_bronze.order_id" in node["label"]
    )
    assert {
        (bronze_target_column["id"], bronze_output["id"]),
        (bronze_output["id"], bronze_input_column["id"]),
    } <= {(edge["source"], edge["target"]) for edge in graph["edges"]}
    assert "digraph lineage" in data_lineage.to_graphviz(graph)

    gold_graph = data_lineage.filter_lineage_graph(graph, "Gold")
    assert any(node.get("layer") == "gold" for node in gold_graph["nodes"])


def test_run_state_save_persists_lineage_per_user(tmp_path, monkeypatch):
    monkeypatch.setattr(user_workspace, "DATA_DIR", tmp_path / "data")
    source_path = tmp_path / "input.csv"
    source_path.write_text("id\n1\n", encoding="utf-8")
    state = {
        "run_id": "run-123",
        "status": "running",
        "uploaded_files": [str(source_path)],
    }

    with user_workspace.use_user_workspace("alice"):
        state_path = run_store.save_run_state(state)
        saved_state = run_store.get_run_state("run-123")
        lineage_path = Path(saved_state["lineage_path"])
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))

    assert state_path == tmp_path / "data" / "users" / "alice" / "runs" / "run-123.json"
    assert lineage_path == tmp_path / "data" / "users" / "alice" / "traces" / "lineage_run-123.json"
    assert lineage["run_id"] == "run-123"
    assert lineage["nodes"][0]["kind"] == "source_file"
