"""Tests for CSV and multi-sheet XLSX ingestion support."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from agents import bronze_agent, profiler
from core.data_io import read_tabular_datasets


def _make_workbook(tmp_path: Path) -> Path:
	workbook_path = tmp_path / "retail.xlsx"
	with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
		pd.DataFrame({"order_id": [1, 2], "amount": [10.0, 20.0]}).to_excel(
			writer, sheet_name="North Sales", index=False
		)
		pd.DataFrame({"order_id": [3], "amount": [30.0]}).to_excel(
			writer, sheet_name="Returns", index=False
		)
	return workbook_path


def _make_sttm(tmp_path: Path) -> Path:
	sttm_path = tmp_path / "bronze_sttm.csv"
	columns = [
		"source_schema",
		"source_table",
		"source_column",
		"target_schema",
		"target_table",
		"target_column",
		"transformation_type",
		"transformation_logic",
	]
	with sttm_path.open("w", newline="", encoding="utf-8") as sttm_file:
		writer = csv.DictWriter(sttm_file, fieldnames=columns)
		writer.writeheader()
		for table_name in ("retail_North_Sales", "retail_Returns"):
			for column in ("order_id", "amount"):
				writer.writerow(
					{
						"source_table": table_name,
						"source_column": column,
						"target_table": table_name,
						"target_column": column,
						"transformation_type": "copy",
						"transformation_logic": "Copy without changes.",
					}
				)
	return sttm_path


def test_excel_loader_returns_one_dataset_per_sheet(tmp_path):
	workbook_path = _make_workbook(tmp_path)

	datasets = read_tabular_datasets(workbook_path)

	assert [dataset["table_name"] for dataset in datasets] == [
		"retail_North_Sales",
		"retail_Returns",
	]
	assert [dataset["sheet_name"] for dataset in datasets] == ["North Sales", "Returns"]
	assert [len(dataset["dataframe"]) for dataset in datasets] == [2, 1]


def test_profiler_inspects_each_excel_sheet(tmp_path):
	workbook_path = _make_workbook(tmp_path)

	inspection = profiler._inspect_files([workbook_path])
	statistics = profiler._compute_stats([workbook_path])

	assert [item["table_name"] for item in inspection["files"]] == [
		"retail_North_Sales",
		"retail_Returns",
	]
	assert [item["shape"][0] for item in statistics["files"]] == [2, 1]


def test_bronze_writes_each_excel_sheet_as_parquet(tmp_path, monkeypatch):
	workbook_path = _make_workbook(tmp_path)
	sttm_path = _make_sttm(tmp_path)
	bronze_dir = tmp_path / "bronze"
	monkeypatch.setattr(bronze_agent, "BRONZE_DIR", bronze_dir)

	output_paths = bronze_agent._apply_bronze_rules(
		[workbook_path], sttm_path, "excel-run"
	)

	assert [Path(path).name for path in output_paths] == [
		"retail_North_Sales_bronze.parquet",
		"retail_Returns_bronze.parquet",
	]
	first_output = pd.read_parquet(output_paths[0])
	second_output = pd.read_parquet(output_paths[1])
	assert len(first_output) == 2
	assert len(second_output) == 1
	assert first_output["_source_file"].iloc[0].endswith("#North Sales")
	assert second_output["_source_file"].iloc[0].endswith("#Returns")


def test_csv_loader_remains_unchanged(tmp_path):
	csv_path = tmp_path / "sales.csv"
	csv_path.write_text("order_id\n1\n", encoding="utf-8")

	datasets = read_tabular_datasets(csv_path)

	assert len(datasets) == 1
	assert datasets[0]["table_name"] == "sales"
	assert datasets[0]["sheet_name"] is None