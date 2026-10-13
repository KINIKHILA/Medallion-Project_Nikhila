import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.config import DATA_DIR
from core.user_workspace import user_directory

_LAYER_KEYS = (
    ("bronze", "sttm_bronze_path", "bronze_output_paths"),
    ("silver", "sttm_silver_path", "silver_output_paths"),
    ("gold", "sttm_gold_path", "gold_output_paths"),
)
_NODE_COLORS = {
    "source_file": "#dceeff",
    "source_column": "#d6f2e3",
    "transformation": "#ffe4ad",
    "layer_column": "#e0e3ff",
    "output_file": "#e8e8e8",
    "report": "#ccebd7",
    "generated": "#f3dcf7",
}


def _node_id(kind: str, *parts: str) -> str:
    value = "|".join((kind, *parts))
    return f"{kind}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:14]}"


def _canonical_table_name(value: str, layer: Optional[str] = None) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    if layer and name.endswith(f"_{layer}"):
        name = name[: -(len(layer) + 1)]
    return name


def _read_rules(path_value: Any, warnings: List[str], layer: str) -> List[Dict[str, str]]:
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.is_file():
        warnings.append(f"{layer.title()} STTM file is missing: {path.name}")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))
    except (OSError, csv.Error) as error:
        warnings.append(f"Could not read {layer.title()} STTM: {error}")
        return []


def _source_files_for_table(source_files: Sequence[str], table_name: str) -> List[str]:
    normalized = _canonical_table_name(table_name)
    matches = []
    for source_file in source_files:
        stem = _canonical_table_name(Path(source_file.split("#", 1)[0]).stem)
        if normalized == stem or normalized.startswith(f"{stem}_"):
            matches.append(source_file)
    return matches


def build_lineage_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[tuple, Dict[str, str]] = {}
    warnings: List[str] = []

    def add_node(
        kind: str,
        label: str,
        details: str,
        layer: Optional[str] = None,
        key: Optional[str] = None,
    ) -> str:
        node_id = key or _node_id(kind, label, details)
        node = {
            "id": node_id,
            "kind": kind,
            "label": label,
            "details": details,
            "layer": layer,
        }
        nodes[node_id] = node
        return node_id

    def add_edge(source: str, target: str, label: str = "") -> None:
        if source == target:
            return
        edges[(source, target, label)] = {"source": source, "target": target, "label": label}

    run_id = str(state.get("run_id") or "unknown")
    uploaded_files = [str(path) for path in state.get("uploaded_files", [])]
    source_file_nodes = {}
    for source_file in uploaded_files:
        source_file_nodes[source_file] = add_node(
            "source_file",
            Path(source_file.split("#", 1)[0]).name,
            source_file,
            key=_node_id("source_file", source_file),
        )

    rules_by_layer = {
        layer: _read_rules(state.get(sttm_key), warnings, layer)
        for layer, sttm_key, _ in _LAYER_KEYS
    }
    target_columns_by_layer: Dict[str, List[tuple]] = {layer: [] for layer, _, _ in _LAYER_KEYS}
    input_columns_by_layer: Dict[str, List[tuple]] = {layer: [] for layer, _, _ in _LAYER_KEYS}

    for layer, _, _ in _LAYER_KEYS:
        previous_layer = {"bronze": None, "silver": "bronze", "gold": "silver"}[layer]
        for rule_index, rule in enumerate(rules_by_layer[layer]):
            source_table = str(rule.get("source_table") or "").strip()
            source_column = str(rule.get("source_column") or "").strip()
            target_table = str(rule.get("target_table") or f"{layer}_dataset").strip()
            target_column = str(rule.get("target_column") or "").strip()
            transformation_type = str(rule.get("transformation_type") or "copy").strip()
            transformation_logic = str(rule.get("transformation_logic") or "").strip()

            if previous_layer is None:
                source_node = None
                if source_column:
                    source_key = _node_id("source_column", source_table, source_column)
                    source_node = add_node(
                        "source_column",
                        f"{source_table}.{source_column}" if source_table else source_column,
                        f"Source table: {source_table or 'unknown'}\nSource column: {source_column}",
                        key=source_key,
                    )
                    matched_files = _source_files_for_table(uploaded_files, source_table)
                    if not matched_files:
                        warnings.append(
                            f"No uploaded file matched Bronze source table '{source_table}'."
                        )
                    for matched_file in matched_files:
                        if matched_file in source_file_nodes:
                            add_edge(source_file_nodes[matched_file], source_node, "contains")
            else:
                source_node = None
                if source_column:
                    normalized_source_table = _canonical_table_name(source_table, previous_layer)
                    source_key = _node_id(
                        "layer_column",
                        previous_layer,
                        normalized_source_table,
                        source_column,
                        "input",
                    )
                    source_node = add_node(
                        "layer_column",
                        f"{previous_layer.title()}: {source_table}.{source_column}",
                        f"Layer: {previous_layer.title()}\nTable: {source_table}\nColumn: {source_column}",
                        layer=previous_layer,
                        key=source_key,
                    )
                    input_columns_by_layer[layer].append(
                        (source_node, normalized_source_table)
                    )

            transformation_key = _node_id(layer, run_id, str(rule_index), target_table, target_column)
            transformation_node = add_node(
                "transformation",
                f"{layer.title()}: {transformation_type}",
                "\n".join(
                    part
                    for part in (
                        f"Layer: {layer.title()}",
                        f"Source: {source_table}.{source_column}" if source_column else "Source: generated",
                        f"Target: {target_table}.{target_column}" if target_column else f"Target table: {target_table}",
                        f"Logic: {transformation_logic}" if transformation_logic else "",
                    )
                    if part
                ),
                layer=layer,
                key=transformation_key,
            )
            if source_node is None:
                source_node = add_node(
                    "generated",
                    f"Generated {layer.title()} value",
                    f"Generated by {transformation_type} for {target_table}.{target_column}",
                    layer=layer,
                    key=_node_id("generated", layer, run_id, str(rule_index)),
                )
            add_edge(source_node, transformation_node, "input")

            target_key = _node_id(
                "layer_column",
                layer,
                _canonical_table_name(target_table),
                target_column,
                "output",
            )
            target_node = add_node(
                "layer_column",
                f"{layer.title()}: {target_table}.{target_column}",
                f"Layer: {layer.title()}\nTable: {target_table}\nColumn: {target_column}",
                layer=layer,
                key=target_key,
            )
            add_edge(transformation_node, target_node, "produces")
            target_columns_by_layer[layer].append(
                (target_key, _canonical_table_name(target_table))
            )

    output_nodes: Dict[str, List[str]] = {layer: [] for layer, _, _ in _LAYER_KEYS}
    for layer, _, output_key in _LAYER_KEYS:
        for output_path in state.get(output_key, []) or []:
            output_path = str(output_path)
            path = Path(output_path)
            output_node = add_node(
                "output_file",
                f"{layer.title()} output: {path.name}",
                f"Layer: {layer.title()}\nPath: {output_path}",
                layer=layer,
                key=_node_id("output_file", layer, output_path),
            )
            output_nodes[layer].append(output_node)
            output_table = _canonical_table_name(path.stem, layer)
            for column_node, target_table in target_columns_by_layer[layer]:
                if (
                    not output_table
                    or output_table == target_table
                    or re.fullmatch(rf"{re.escape(target_table)}_\d+", output_table)
                ):
                    add_edge(column_node, output_node, "written to")

    for layer_index, (layer, _, _) in enumerate(_LAYER_KEYS):
        if layer_index == 0:
            continue
        previous_layer = _LAYER_KEYS[layer_index - 1][0]
        for output_node in output_nodes[previous_layer]:
            output_details = nodes[output_node]["details"].splitlines()
            output_path = next(
                (line.removeprefix("Path: ") for line in output_details if line.startswith("Path: ")),
                "",
            )
            output_table = _canonical_table_name(Path(output_path).stem, previous_layer)
            for column_node, source_table in input_columns_by_layer[layer]:
                if (
                    not output_table
                    or output_table == source_table
                    or re.fullmatch(rf"{re.escape(source_table)}_\d+", output_table)
                ):
                    add_edge(output_node, column_node, "feeds")

    report_path = str(state.get("report_path") or "")
    if report_path:
        report_node = add_node("report", Path(report_path).name, report_path)
        for gold_output in output_nodes["gold"]:
            add_edge(gold_output, report_node, "summarized in")

    return {
        "run_id": run_id,
        "status": str(state.get("status") or "unknown"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "warnings": warnings,
    }


def persist_lineage(state: Dict[str, Any]) -> Path:
    graph = build_lineage_graph(state)
    run_id = re.sub(r"[^A-Za-z0-9._-]", "_", graph["run_id"])
    lineage_dir = user_directory("traces", DATA_DIR / "traces")
    lineage_dir.mkdir(parents=True, exist_ok=True)
    path = lineage_dir / f"lineage_{run_id}.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{run_id}_lineage_", suffix=".tmp", dir=lineage_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(graph, target, indent=2, ensure_ascii=False)
            target.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _dot_quote(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def to_graphviz(graph: Dict[str, Any], focus_node: Optional[str] = None) -> str:
    lines = [
        "digraph lineage {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.25", nodesep="0.35", ranksep="0.75", splines="ortho"];',
        '  node [shape="box", style="rounded,filled", fontname="Arial", fontsize="10", margin="0.12,0.08"];',
        '  edge [fontname="Arial", fontsize="8", color="#687782", arrowsize="0.7"];',
    ]
    for node in graph.get("nodes", []):
        attributes = {
            "label": node["label"],
            "tooltip": node.get("details", ""),
            "fillcolor": _NODE_COLORS.get(node.get("kind"), "#ffffff"),
        }
        if node.get("id") == focus_node:
            attributes["penwidth"] = "3"
            attributes["color"] = "#d34b2f"
        encoded = ", ".join(
            f"{key}={_dot_quote(value)}" for key, value in attributes.items()
        )
        lines.append(f"  {_dot_quote(node['id'])} [{encoded}];")
    for edge in graph.get("edges", []):
        label = f" [label={_dot_quote(edge['label'])}]" if edge.get("label") else ""
        lines.append(
            f"  {_dot_quote(edge['source'])} -> {_dot_quote(edge['target'])}{label};"
        )
    lines.append("}")
    return "\n".join(lines)


def filter_lineage_graph(graph: Dict[str, Any], layer: str) -> Dict[str, Any]:
    if layer == "All":
        return graph
    normalized = layer.casefold()
    if normalized == "source":
        focused = {
            node["id"] for node in graph["nodes"]
            if node.get("kind") in {"source_file", "source_column"}
        }
    elif normalized == "report":
        focused = {node["id"] for node in graph["nodes"] if node.get("kind") == "report"}
    else:
        focused = {
            node["id"] for node in graph["nodes"]
            if str(node.get("layer") or "").casefold() == normalized
        }
    included = set(focused)
    for edge in graph["edges"]:
        if edge["source"] in focused or edge["target"] in focused:
            included.update((edge["source"], edge["target"]))
    return {
        **graph,
        "nodes": [node for node in graph["nodes"] if node["id"] in included],
        "edges": [
            edge for edge in graph["edges"]
            if edge["source"] in included and edge["target"] in included
        ],
    }
