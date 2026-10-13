import ast
import json
import math
import operator
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import regex

CUSTOM_TRANSFORMATION_TYPES = {"expression", "regex_replace", "lookup"}
MAX_EXPRESSION_LENGTH = 512
MAX_REGEX_LENGTH = 256
MAX_LOOKUP_ITEMS = 1000
REGEX_TIMEOUT_SECONDS = 0.05

_ALLOWED_NODES = {
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.IfExp,
    ast.Call,
}
_SAFE_FUNCTIONS = {
    "abs",
    "round",
    "lower",
    "upper",
    "title",
    "strip",
    "concat",
    "coalesce",
    "to_number",
    "to_text",
}
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}


class CustomTransformationError(ValueError):
    """Raised when a custom STTM transformation is unsafe or malformed."""


def _transformation_type(rule: Dict[str, Any]) -> str:
    return str(rule.get("transformation_type", "")).strip().lower().replace("-", "_").replace(" ", "_")


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_expression(
    expression: str, available_columns: Optional[Sequence[str]] = None
) -> ast.Expression:
    if not expression or len(expression) > MAX_EXPRESSION_LENGTH:
        raise CustomTransformationError(
            f"Expression must contain 1 to {MAX_EXPRESSION_LENGTH} characters."
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise CustomTransformationError(f"Invalid expression syntax: {error.msg}") from error

    nodes = list(ast.walk(tree))
    if len(nodes) > 100 or any(type(node) not in _ALLOWED_NODES for node in nodes):
        raise CustomTransformationError(
            "Expression contains an unsupported or overly complex operation."
        )

    referenced_columns = set()
    for node in nodes:
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise CustomTransformationError("Private or dunder names are not allowed.")
            if node.id not in _SAFE_FUNCTIONS:
                referenced_columns.add(node.id)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
                raise CustomTransformationError("Only approved transformation functions may be called.")
            if node.keywords:
                raise CustomTransformationError("Keyword arguments are not supported in expressions.")

    if available_columns is not None:
        missing = referenced_columns.difference(str(column) for column in available_columns)
        if missing:
            raise CustomTransformationError(
                "Expression references unknown column(s): " + ", ".join(sorted(missing))
            )
    return tree


def _parse_json_logic(logic: str) -> Dict[str, Any]:
    if not logic or len(logic) > 16_384:
        raise CustomTransformationError("Transformation JSON is empty or too large.")
    try:
        value = json.loads(logic)
    except json.JSONDecodeError as error:
        raise CustomTransformationError(f"Transformation logic must be valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise CustomTransformationError("Transformation logic must be a JSON object.")
    return value


def validate_custom_rule(
    rule: Dict[str, Any], available_columns: Optional[Sequence[str]] = None
) -> List[str]:
    """Validate one custom STTM rule; return errors without executing it."""
    kind = _transformation_type(rule)
    if kind not in CUSTOM_TRANSFORMATION_TYPES:
        return []

    logic = str(rule.get("transformation_logic", ""))
    try:
        target_column = str(rule.get("target_column", "")).strip()
        source_column = str(rule.get("source_column", "")).strip()
        if not target_column:
            raise CustomTransformationError("Custom transformation requires target_column.")
        if kind != "expression" and not source_column:
            raise CustomTransformationError(
                f"{kind} transformation requires source_column."
            )
        if (
            kind != "expression"
            and available_columns is not None
            and source_column not in available_columns
            and target_column not in available_columns
        ):
            raise CustomTransformationError(
                f"Transformation source column '{source_column}' is unavailable."
            )
        if kind == "expression":
            _parse_expression(logic, available_columns)
        elif kind == "regex_replace":
            spec = _parse_json_logic(logic)
            pattern = spec.get("pattern")
            replacement = spec.get("replacement", "")
            if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_REGEX_LENGTH:
                raise CustomTransformationError(
                    f"Regex pattern must contain 1 to {MAX_REGEX_LENGTH} characters."
                )
            if not isinstance(replacement, str) or len(replacement) > MAX_REGEX_LENGTH:
                raise CustomTransformationError("Regex replacement must be a short string.")
            regex.compile(pattern)
        elif kind == "lookup":
            spec = _parse_json_logic(logic)
            mapping = spec.get("mapping")
            if not isinstance(mapping, dict) or len(mapping) > MAX_LOOKUP_ITEMS:
                raise CustomTransformationError(
                    f"Lookup logic requires a mapping with at most {MAX_LOOKUP_ITEMS} entries."
                )
            if any(
                not isinstance(value, (str, int, float, bool, type(None)))
                for value in mapping.values()
            ):
                raise CustomTransformationError("Lookup results must be scalar JSON values.")
            if "default" in spec and not isinstance(
                spec["default"], (str, int, float, bool, type(None))
            ):
                raise CustomTransformationError("Lookup default must be a scalar JSON value.")
    except CustomTransformationError as error:
        return [str(error)]
    except regex.error as error:
        return [f"Invalid regular expression: {error}"]
    return []


def validate_custom_rules(
    rules: Iterable[Dict[str, Any]], available_columns: Optional[Sequence[str]] = None
) -> List[str]:
    errors = []
    for index, rule in enumerate(rules, start=1):
        for error in validate_custom_rule(rule, available_columns):
            errors.append(f"Rule {index}: {error}")
    return errors


def _is_null_expression(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _call_safe_function(name: str, args: List[Any]) -> Any:
    if name == "coalesce":
        return next((value for value in args if not _is_null_expression(value)), None)
    if name == "concat":
        return "".join("" if _is_null_expression(value) else str(value) for value in args)
    if name == "to_number":
        if not args or _is_null_expression(args[0]):
            return None
        try:
            return float(args[0])
        except (TypeError, ValueError):
            return None
    if name == "to_text":
        return "" if not args or _is_null_expression(args[0]) else str(args[0])
    if name in {"lower", "upper", "title", "strip"}:
        if not args or _is_null_expression(args[0]):
            return None
        text = str(args[0])
        return {
            "lower": text.lower,
            "upper": text.upper,
            "title": text.title,
            "strip": text.strip,
        }[name]()
    if name == "abs":
        return abs(args[0])
    if name == "round":
        return round(*args)
    raise CustomTransformationError(f"Function '{name}' is not allowed.")


def _evaluate(node: ast.AST, row: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in row:
            return row[node.id]
        if node.id in _SAFE_FUNCTIONS:
            return node.id
        raise CustomTransformationError(f"Unknown expression column '{node.id}'.")
    if isinstance(node, ast.BinOp):
        return _BINARY_OPERATORS[type(node.op)](_evaluate(node.left, row), _evaluate(node.right, row))
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, row)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.Not):
            return not value
    if isinstance(node, ast.BoolOp):
        values = [_evaluate(value, row) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, row)
        for operation, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, row)
            if not _COMPARE_OPERATORS[type(operation)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _evaluate(node.body if _evaluate(node.test, row) else node.orelse, row)
    if isinstance(node, ast.Call):
        name = node.func.id
        args = [_evaluate(argument, row) for argument in node.args]
        return _call_safe_function(name, args)
    raise CustomTransformationError(f"Unsupported expression node: {type(node).__name__}.")


def apply_custom_transformation(dataframe: Any, rule: Dict[str, Any]) -> pd.Series:
    """Apply a validated expression, regex replacement, or lookup to a DataFrame."""
    kind = _transformation_type(rule)
    target_column = str(rule.get("target_column", "")).strip()
    source_column = str(rule.get("source_column", "")).strip()
    logic = str(rule.get("transformation_logic", ""))
    errors = validate_custom_rule(rule, list(dataframe.columns))
    if errors:
        raise CustomTransformationError("; ".join(errors))
    if not target_column:
        raise CustomTransformationError("Custom transformation requires target_column.")

    if kind == "expression":
        tree = _parse_expression(logic, list(dataframe.columns))
        records = dataframe.to_dict(orient="records")
        values = [_evaluate(tree.body, row) for row in records]
        return pd.Series(values, index=dataframe.index)

    source_column = source_column if source_column in dataframe.columns else target_column
    if source_column not in dataframe.columns:
        raise CustomTransformationError(
            f"Custom transformation source column '{source_column}' is missing."
        )
    source_values = dataframe[source_column]

    if kind == "regex_replace":
        spec = _parse_json_logic(logic)
        pattern = spec["pattern"]
        replacement = spec.get("replacement", "")

        def replace(value: Any) -> Any:
            if _is_null(value):
                return value
            return regex.sub(
                pattern,
                replacement,
                str(value),
                timeout=REGEX_TIMEOUT_SECONDS,
            )

        return source_values.map(replace)

    spec = _parse_json_logic(logic)
    mapping = spec["mapping"]
    default = spec.get("default")
    return source_values.map(
        lambda value: value if _is_null(value) else mapping.get(str(value), default)
    )
