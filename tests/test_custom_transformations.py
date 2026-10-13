import json

import pandas as pd
import pytest

from core.custom_transformations import (
	CustomTransformationError,
	apply_custom_transformation,
	validate_custom_rule,
)


def test_safe_expression_uses_dataframe_columns():
	dataframe = pd.DataFrame({"quantity": [2, 3], "unit_price": [4.5, 10.0]})
	rule = {
		"source_column": "",
		"target_column": "total",
		"transformation_type": "expression",
		"transformation_logic": "quantity * unit_price",
	}

	assert apply_custom_transformation(dataframe, rule).tolist() == [9.0, 30.0]


def test_regex_replace_uses_json_spec():
	dataframe = pd.DataFrame({"sku": ["AB-12", "CD-34"]})
	rule = {
		"source_column": "sku",
		"target_column": "sku_clean",
		"transformation_type": "regex_replace",
		"transformation_logic": json.dumps({"pattern": "-", "replacement": ""}),
	}

	assert apply_custom_transformation(dataframe, rule).tolist() == ["AB12", "CD34"]


def test_lookup_uses_declared_mapping_and_default():
	dataframe = pd.DataFrame({"segment": ["Prem", "unknown"]})
	rule = {
		"source_column": "segment",
		"target_column": "normalized_segment",
		"transformation_type": "lookup",
		"transformation_logic": json.dumps(
			{"mapping": {"Prem": "Premium", "Std": "Standard"}, "default": "Other"}
		),
	}

	assert apply_custom_transformation(dataframe, rule).tolist() == ["Premium", "Other"]


def test_unsafe_python_expression_is_rejected():
	rule = {
		"target_column": "result",
		"transformation_type": "expression",
		"transformation_logic": "__import__('os').system('whoami')",
	}

	assert validate_custom_rule(rule) == [
		"Expression contains an unsupported or overly complex operation."
	]
	with pytest.raises(CustomTransformationError):
		apply_custom_transformation(pd.DataFrame({"value": [1]}), rule)


def test_expression_rejects_unknown_columns():
	rule = {
		"target_column": "result",
		"transformation_type": "expression",
		"transformation_logic": "missing_column + 1",
	}

	assert validate_custom_rule(rule, ["value"]) == [
		"Expression references unknown column(s): missing_column"
	]