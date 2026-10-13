"""Tests for pipeline row-count, null, and numeric-anomaly quality checks."""

import pandas as pd

from core.data_quality import assess_dataframe_quality


class TestAssessDataFrameQuality:
	def test_reports_nulls_and_numeric_anomalies(self):
		dataframe = pd.DataFrame(
			{
				"customer_id": [1, 2, 3, 4, 5],
				"revenue": [10.0, 11.0, 12.0, 13.0, 1000.0],
				"segment": ["loyal", None, "standard", "budget", "loyal"],
			}
		)

		result = assess_dataframe_quality(dataframe, "silver", "customers")

		assert result["passed"] is True
		assert result["null_count"] == 1
		assert result["numeric_outlier_count"] >= 1
		assert {issue["code"] for issue in result["issues"]} >= {
			"null_values_found",
			"numeric_outliers_found",
		}

	def test_fails_when_quality_score_is_too_low(self):
		dataframe = pd.DataFrame({"empty_column": [None, None, None]})

		result = assess_dataframe_quality(
			dataframe,
			"gold",
			"empty_table",
			minimum_score=0.8,
		)

		assert result["passed"] is False
		assert result["score"] < result["minimum_score"]

	def test_fails_when_row_count_is_outside_expected_range(self):
		dataframe = pd.DataFrame({"id": [1]})

		result = assess_dataframe_quality(
			dataframe,
			"bronze",
			"small_table",
			expected_row_range=(2, 10),
		)

		assert result["passed"] is False
		assert any(issue["code"] == "row_count_out_of_range" for issue in result["issues"])

	def test_fails_when_required_column_is_missing(self):
		dataframe = pd.DataFrame({"id": [1]})

		result = assess_dataframe_quality(
			dataframe,
			"silver",
			"orders",
			required_columns=["order_id"],
		)

		assert result["passed"] is False
		assert any(issue["code"] == "required_column_missing" for issue in result["issues"])