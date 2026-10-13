from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

FilePath = Union[str, Path]
DEFAULT_MINIMUM_QUALITY_SCORE = 0.70


def _issue(
    severity: str,
    code: str,
    message: str,
    dataset: str,
    column: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "dataset": dataset,
        "column": column,
    }


def assess_dataframe_quality(
    dataframe: Any,
    layer: str,
    dataset: str,
    min_rows: int = 1,
    expected_row_range: Optional[Tuple[int, Optional[int]]] = None,
    required_columns: Sequence[str] = (),
    minimum_score: float = DEFAULT_MINIMUM_QUALITY_SCORE,
) -> Dict[str, Any]:
    """Score a DataFrame and report row-count, null, and numeric-outlier issues."""
    import pandas as pd

    row_count, column_count = dataframe.shape
    issues = []
    if row_count < min_rows:
        issues.append(
            _issue(
                "error",
                "row_count_below_minimum",
                f"Expected at least {min_rows} rows; found {row_count}.",
                dataset,
            )
        )
    if expected_row_range is not None:
        lower_bound, upper_bound = expected_row_range
        if row_count < lower_bound or (upper_bound is not None and row_count > upper_bound):
            issues.append(
                _issue(
                    "error",
                    "row_count_out_of_range",
                    f"Expected row count in [{lower_bound}, {upper_bound}]; found {row_count}.",
                    dataset,
                )
            )

    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    for column in missing_columns:
        issues.append(
            _issue(
                "error",
                "required_column_missing",
                f"Required column '{column}' is missing.",
                dataset,
                column,
            )
        )

    cell_count = row_count * max(column_count, 1)
    null_count = int(dataframe.isna().sum().sum()) if row_count and column_count else 0
    null_ratio = null_count / cell_count if cell_count else 1.0
    if null_count:
        issues.append(
            _issue(
                "warning",
                "null_values_found",
                f"Found {null_count} null value(s) ({null_ratio:.1%} of cells).",
                dataset,
            )
        )
    for column in required_columns:
        if column in dataframe.columns:
            column_nulls = int(dataframe[column].isna().sum())
            if column_nulls:
                issues.append(
                    _issue(
                        "error",
                        "required_column_has_nulls",
                        f"Required column '{column}' contains {column_nulls} null value(s).",
                        dataset,
                        column,
                    )
                )

    numeric_value_count = 0
    outlier_count = 0
    for column in dataframe.select_dtypes(include="number").columns:
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        numeric_value_count += len(values)
        if len(values) < 4:
            continue
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        if not interquartile_range:
            continue
        lower_fence = first_quartile - 1.5 * interquartile_range
        upper_fence = third_quartile + 1.5 * interquartile_range
        column_outliers = int(((values < lower_fence) | (values > upper_fence)).sum())
        if column_outliers:
            outlier_count += column_outliers
            issues.append(
                _issue(
                    "warning",
                    "numeric_outliers_found",
                    f"Found {column_outliers} IQR outlier(s) in '{column}'.",
                    dataset,
                    str(column),
                )
            )

    numeric_outlier_ratio = outlier_count / numeric_value_count if numeric_value_count else 0.0
    score = max(0.0, 1.0 - (0.6 * null_ratio) - (0.4 * numeric_outlier_ratio))
    if row_count < min_rows or missing_columns or any(issue["severity"] == "error" for issue in issues):
        score = min(score, 0.0)
    passed = score >= minimum_score and not any(
        issue["severity"] == "error" for issue in issues
    )

    return {
        "layer": layer,
        "dataset": dataset,
        "row_count": int(row_count),
        "column_count": int(column_count),
        "null_count": null_count,
        "null_ratio": round(null_ratio, 4),
        "numeric_outlier_count": outlier_count,
        "numeric_outlier_ratio": round(numeric_outlier_ratio, 4),
        "score": round(score, 4),
        "minimum_score": minimum_score,
        "passed": passed,
        "issues": issues,
    }


def assess_parquet_files(
    file_paths: Sequence[FilePath],
    layer: str,
    min_rows: int = 1,
    expected_row_range: Optional[Tuple[int, Optional[int]]] = None,
    required_columns: Sequence[str] = (),
    minimum_score: float = DEFAULT_MINIMUM_QUALITY_SCORE,
) -> Dict[str, Any]:
    """Assess Parquet outputs and aggregate their quality results."""
    import pandas as pd

    datasets = []
    for file_path in file_paths:
        path = Path(file_path)
        try:
            dataframe = pd.read_parquet(path)
            datasets.append(
                assess_dataframe_quality(
                    dataframe,
                    layer=layer,
                    dataset=path.name,
                    min_rows=min_rows,
                    expected_row_range=expected_row_range,
                    required_columns=required_columns,
                    minimum_score=minimum_score,
                )
            )
        except Exception as error:
            datasets.append(
                {
                    "layer": layer,
                    "dataset": path.name,
                    "row_count": 0,
                    "column_count": 0,
                    "null_count": 0,
                    "null_ratio": 0.0,
                    "numeric_outlier_count": 0,
                    "numeric_outlier_ratio": 0.0,
                    "score": 0.0,
                    "minimum_score": minimum_score,
                    "passed": False,
                    "issues": [
                        _issue("error", "parquet_read_failed", str(error), path.name)
                    ],
                }
            )

    average_score = (
        sum(dataset["score"] for dataset in datasets) / len(datasets) if datasets else 0.0
    )
    issues = [issue for dataset in datasets for issue in dataset["issues"]]
    return {
        "layer": layer,
        "passed": bool(datasets) and all(dataset["passed"] for dataset in datasets),
        "score": round(average_score, 4),
        "minimum_score": minimum_score,
        "datasets": datasets,
        "issues": issues,
    }
