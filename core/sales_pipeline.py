from calendar import month_abbr, month_name
import html
from io import BytesIO
import json
import re
from pathlib import Path
from shutil import copyfile
from typing import Callable, Mapping

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


SALES_COLUMNS = {
    "transaction_id", "transaction_date", "store_id", "product_id",
    "product_name", "category", "quantity", "unit_price", "total_amount",
    "customer_segment", "payment_method",
}
SOURCE_COLUMNS = {
    "sales_data.csv": SALES_COLUMNS,
    "products.csv": {"product_id", "product_name"},
    "stores.csv": {"store_id", "store_name", "region"},
}
QUESTION_DIMENSIONS = (
    (r"\b(products?|items?)\b", "product_name", "Product"),
    (r"\b(stores?|locations?)\b", "store_name", "Store"),
    (r"\bregions?\b", "region", "Region"),
    (r"\b(categor(?:y|ies))\b", "category", "Category"),
    (r"\bsegments?\b", "customer_segment", "Customer segment"),
    (r"\bpayments?|methods?\b", "payment_method", "Payment method"),
)


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def clean_sales(sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    require_columns(sales, SALES_COLUMNS, "sales_data.csv")
    require_columns(stores, {"store_id", "store_name", "region"}, "stores.csv")
    sales = sales.copy()
    for column in sales.columns:
        sales[column] = sales[column].astype("string").str.strip()
    sales = sales.drop_duplicates().reset_index(drop=True)

    if sales["transaction_id"].isna().any() or sales["transaction_id"].eq("").any():
        raise ValueError("Sales contain a missing transaction_id")
    if sales["transaction_id"].duplicated().any():
        raise ValueError("Sales contain conflicting rows for a transaction_id")
    if sales["product_id"].isna().any() or sales["product_id"].eq("").any():
        raise ValueError("Sales contain a missing product_id")

    dates = pd.to_datetime(sales["transaction_date"], format="mixed", errors="coerce")
    quantities = pd.to_numeric(sales["quantity"], errors="coerce")
    prices = pd.to_numeric(sales["unit_price"], errors="coerce")
    amounts = pd.to_numeric(sales["total_amount"], errors="coerce")
    if dates.isna().any() or quantities.isna().any() or prices.isna().any() or amounts.isna().any() or (quantities % 1 != 0).any():
        raise ValueError("Sales contain invalid dates, quantities, or amounts")

    sales["transaction_date"] = dates.dt.strftime("%Y-%m-%d")
    sales["quantity"] = quantities.astype("int64")
    sales["unit_price"] = prices.astype("float64")
    sales["total_amount"] = amounts.astype("float64")
    sales["category"] = sales["category"].str.casefold().map({
        "elec.": "Electronics", "electronics": "Electronics",
        "stationery": "Stationery", "kitchen": "Kitchen", "accessories": "Accessories",
    }).fillna(sales["category"].str.title())
    sales["customer_segment"] = sales["customer_segment"].str.casefold().map({
        "prem": "Premium", "premium": "Premium", "std": "Standard",
        "standard": "Standard", "bgt": "Budget", "budget": "Budget", "loyal": "Loyal",
    }).fillna(sales["customer_segment"].str.title())
    sales["store_id"] = sales["store_id"].fillna("").replace("", "Unknown")

    stores = stores[["store_id", "store_name", "region"]].copy()
    stores["store_id"] = stores["store_id"].astype("string").str.strip()
    if stores["store_id"].duplicated().any():
        raise ValueError("Store catalog has duplicate store_id values")
    enriched = sales.merge(stores, on="store_id", how="left", validate="many_to_one")
    enriched[["store_name", "region"]] = enriched[["store_name", "region"]].fillna("Unknown")
    return enriched


def build_gold(silver: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(products, {"product_id", "product_name"}, "products.csv")
    products = products[["product_id", "product_name"]].copy()
    if products["product_id"].duplicated().any():
        raise ValueError("Product catalog has duplicate product_id values")

    summary = silver.assign(is_return=silver["quantity"] < 0).groupby("category", as_index=False).agg(
        transactions=("transaction_id", "count"),
        units=("quantity", "sum"),
        returns=("is_return", "sum"),
        net_revenue=("total_amount", "sum"),
    ).sort_values("net_revenue", ascending=False).reset_index(drop=True)
    summary["net_revenue"] = summary["net_revenue"].round(2)

    sold = silver[["product_id", "product_name"]].drop_duplicates(subset="product_id")
    catalog = sold.merge(products, on="product_id", how="left", suffixes=("_sales", "_catalog"), validate="many_to_one")
    missing = catalog["product_name_catalog"].isna()
    mismatch = catalog["product_name_sales"].str.casefold() != catalog["product_name_catalog"].fillna("").str.casefold()
    issues = catalog.loc[missing | mismatch].copy()
    issues["issue"] = "Name mismatch"
    issues.loc[missing, "issue"] = "Missing from catalog"
    return summary, issues.rename(columns={"product_name_sales": "sales_product_name", "product_name_catalog": "catalog_product_name"}).reset_index(drop=True)


def run_pipeline(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    landing = root / "data" / "landing"
    sources = ("sales_data.csv", "products.csv", "stores.csv")
    sales, products, stores = (pd.read_csv(landing / name, dtype=str, keep_default_na=False) for name in sources)
    silver = clean_sales(sales, stores)
    summary, issues = build_gold(silver, products)

    bronze_dir = root / "data" / "bronze_layer"
    silver_dir = root / "data" / "silver_layer"
    gold_dir = root / "data" / "gold_layer"
    for directory in (bronze_dir, silver_dir, gold_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name in sources:
        copyfile(landing / name, bronze_dir / name)
    silver.to_csv(silver_dir / "sales_clean.csv", index=False)
    summary.to_csv(gold_dir / "category_sales.csv", index=False)
    issues.to_csv(gold_dir / "product_catalog_issues.csv", index=False)
    return silver, summary, issues


def save_template(root: Path, layer: str, frames: Mapping[str, pd.DataFrame], rules: list[str]) -> None:
    directory = root / "data" / "sttm"
    directory.mkdir(parents=True, exist_ok=True)
    template = {
        "layer": layer,
        "outputs": {name: {"columns": list(frame.columns), "rows": len(frame)} for name, frame in frames.items()},
        "rules": rules,
    }
    (directory / f"{layer}_template.json").write_text(json.dumps(template, indent=2), encoding="utf-8")


def create_bronze(root: Path, uploads: Mapping[str, bytes]) -> dict[str, pd.DataFrame]:
    missing = SOURCE_COLUMNS.keys() - uploads.keys()
    if missing:
        raise ValueError(f"Missing upload: {', '.join(sorted(missing))}")
    frames = {}
    for name, columns in SOURCE_COLUMNS.items():
        data = uploads[name]
        if not data or len(data) > 10 * 1024 * 1024:
            raise ValueError(f"{name} must be a nonempty CSV under 10 MB")
        try:
            frame = pd.read_csv(BytesIO(data), dtype=str, keep_default_na=False)
        except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            raise ValueError(f"{name} is not a valid CSV") from error
        require_columns(frame, columns, name)
        frames[name] = frame

    directory = root / "data" / "bronze_layer"
    directory.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_COLUMNS:
        (directory / name).write_bytes(uploads[name])
    save_template(root, "bronze", frames, ["Preserve uploaded CSV bytes without changing rows or columns"])
    return frames


def create_silver(root: Path, bronze: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    silver = clean_sales(bronze["sales_data.csv"], bronze["stores.csv"])
    directory = root / "data" / "silver_layer"
    directory.mkdir(parents=True, exist_ok=True)
    silver.to_csv(directory / "sales_clean.csv", index=False)
    save_template(root, "silver", {"sales_clean.csv": silver}, [
        "Remove exact duplicate transactions; reject conflicting transaction IDs",
        "Normalize dates, categories, segments and numeric fields; retain signed returns",
        "Enrich stores; label missing or unmatched stores Unknown",
    ])
    return silver


def create_gold(root: Path, silver: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, issues = build_gold(silver, products)
    directory = root / "data" / "gold_layer"
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / "category_sales.csv", index=False)
    issues.to_csv(directory / "product_catalog_issues.csv", index=False)
    save_template(root, "gold", {"category_sales.csv": summary, "product_catalog_issues.csv": issues}, [
        "Sum signed amounts and units by category; count return rows",
        "Report catalog names that disagree with sales and missing product IDs",
    ])
    return summary, issues


def filter_question_period(question: str, silver: pd.DataFrame) -> pd.DataFrame:
    months = {name.casefold(): number for number in range(1, 13) for name in (month_name[number], month_abbr[number])}
    named_period = re.search(rf"\b({'|'.join(months)})\s+(20\d{{2}})\b", question)
    iso_period = re.search(r"\b(20\d{2})-(0[1-9]|1[0-2])\b", question)
    year = re.search(r"\b(20\d{2})\b", question)
    if year is None or "transaction_date" not in silver:
        return silver
    dates = pd.to_datetime(silver["transaction_date"], errors="coerce")
    requested_year = int(named_period.group(2) if named_period else iso_period.group(1) if iso_period else year.group(1))
    selected = dates.dt.year == requested_year
    if named_period or iso_period:
        requested_month = months[named_period.group(1)] if named_period else int(iso_period.group(2))
        selected &= dates.dt.month == requested_month
    return silver.loc[selected]


def requested_dimension(question: str) -> tuple[str, str] | None:
    return next(((column, label) for pattern, column, label in QUESTION_DIMENSIONS if re.search(pattern, question)), None)


def question_chart(question: str, silver: pd.DataFrame, issues: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    topics = question.casefold()
    if re.search(r"\b(catalog|mismatch|issues?|quality)\b", topics) and not requested_dimension(topics):
        chart = issues["issue"].value_counts().rename_axis("issue").reset_index(name="value")
        return "Catalog issues", chart
    filtered = filter_question_period(topics, silver)
    dimension = requested_dimension(topics) or ("category", "Category")
    column, label = dimension
    if column not in filtered or filtered.empty:
        return f"Net revenue by {label.lower()}", pd.DataFrame(columns=[column, "value"])
    if column in ("store_name", "region"):
        filtered = filtered.loc[filtered[column] != "Unknown"]
    if re.search(r"\b(returns?|refunds?)\b", topics):
        metric, measure = "Returns", (filtered["quantity"] < 0).groupby(filtered[column]).sum()
    elif re.search(r"\b(units?|quantit(?:y|ies)|sold)\b", topics):
        metric, measure = "Net units", filtered.groupby(column)["quantity"].sum()
    elif re.search(r"\b(transactions?|orders?)\b", topics) and not re.search(r"\b(revenue|sales|amount)\b", topics):
        metric, measure = "Transactions", filtered.groupby(column).size()
    else:
        metric, measure = "Net revenue", filtered.groupby(column)["total_amount"].sum()
    count = re.search(r"\b(?:top|bottom)\s+(\d+)\b", topics)
    limit = min(int(count.group(1)), 10) if count else 10
    ascending = bool(re.search(r"\b(lowest|least|bottom|fewest|smallest)\b", topics))
    chart = measure.sort_values(ascending=ascending).head(limit).rename("value").reset_index()
    return f"{metric} by {label.lower()}", chart


def answer_business_question(question: str, silver: pd.DataFrame, summary: pd.DataFrame, issues: pd.DataFrame) -> tuple[str, list[str]]:
    topics = question.casefold()
    unsupported = "I cannot answer this question from the uploaded sales, product, and store fields. Ask about net revenue, transactions, net units, returns, category, product, store, region, segment, payment method, or catalog issues."
    if any(re.search(rf"\b{term}\b", topics) for term in ("why", "forecast", "predict", "profit", "margin", "prefer", "preference", "cause")):
        return unsupported, []

    if re.search(r"\b(last|previous|this|recent|today|yesterday|quarter)\b", topics):
        return "I cannot answer relative-period questions. Specify a year or a month and year instead.", []
    year = re.search(r"\b(20\d{2})\b", topics)
    if year:
        if "transaction_date" not in silver:
            return unsupported, []
        silver = filter_question_period(topics, silver)
        if silver.empty:
            return "No transactions matched the requested period.", []

    ranking_count = re.search(r"\b(?:top|bottom)\s+(\d+)\b", topics)
    limit = int(ranking_count.group(1)) if ranking_count else 10
    if not 1 <= limit <= 10:
        return "I cannot answer rankings outside the top or bottom 1 to 10. Choose a number in that range.", []

    quality = bool(re.search(r"\b(catalog|mismatch|issues?|quality)\b", topics))
    if year and quality:
        return "I cannot answer period-specific catalog quality: catalog issues are calculated across all uploaded sales.", []
    if quality and not re.search(r"\b(categor(?:y|ies)|stores?|regions?|segments?)\b", topics):
        details = [f"- {issue}: {count}" for issue, count in issues["issue"].value_counts().items()]
        return f"{len(issues)} sold product IDs have catalog issues.", ["## Catalog quality", *details]

    dimension = requested_dimension(topics)
    if dimension:
        column, label = dimension
        if column not in silver:
            return unsupported, []
        unassigned_revenue = None
        if column in ("store_name", "region"):
            unassigned = silver[column].eq("Unknown")
            unassigned_revenue = silver.loc[unassigned, "total_amount"].sum()
            silver = silver.loc[~unassigned]
            if silver.empty:
                return f"No named {label.lower()} values were available; ${unassigned_revenue:,.2f} of revenue is unassigned.", []
        if re.search(r"\b(returns?|refunds?)\b", topics):
            metric, measure, unit = "returns", (silver["quantity"] < 0).groupby(silver[column]).sum(), "returns"
        elif re.search(r"\b(units?|quantit(?:y|ies)|sold)\b", topics):
            metric, measure, unit = "net units", silver.groupby(column)["quantity"].sum(), "net units"
        elif re.search(r"\b(transactions?|orders?)\b", topics) and not re.search(r"\b(revenue|sales|amount)\b", topics):
            metric, measure, unit = "transactions", silver.groupby(column).size(), "transactions"
        else:
            metric, measure, unit = "net revenue", silver.groupby(column)["total_amount"].sum(), "net revenue"
        ranking = measure.sort_values(ascending=bool(re.search(r"\b(lowest|least|bottom|fewest|smallest)\b", topics)))
        if ranking.empty:
            return "No rows match this business question.", []
        def display(value: float) -> str:
            return f"${value:,.2f}" if metric == "net revenue" else f"{value:,.0f}"
        title = f"{label} {unit}"
        details = [f"- {name}: {display(value)}{' net' if metric == 'net revenue' else ''}" for name, value in ranking.head(limit).items()]
        if unassigned_revenue:
            details.append(f"- ${unassigned_revenue:,.2f} of unassigned revenue is excluded from {label.lower()} rankings")
        if re.search(r"\b(highest|most|top|largest|best|lowest|least|bottom|fewest|smallest)\b", topics):
            direction = "fewest" if "fewest" in topics else "lowest" if re.search(r"\b(lowest|least|bottom|smallest)\b", topics) else "most" if "most" in topics else "highest"
            verb = "sold" if metric == "net units" else "generated" if metric == "net revenue" else "had"
            answer = f"{ranking.index[0]} {verb} the {direction} {metric}: {display(ranking.iloc[0])}."
        else:
            answer = f"{metric.capitalize()} by {label.lower()} is shown below; the total is {display(measure.sum())}."
        if quality:
            details.extend(["", "## Catalog quality", *[f"- {issue}: {count}" for issue, count in issues["issue"].value_counts().items()]])
        return answer, [f"## {title}", *details]

    if re.search(r"\b(returns?|refunds?)\b", topics):
        count = int((silver["quantity"] < 0).sum())
        amount = silver.loc[silver["quantity"] < 0, "total_amount"].sum()
        return f"There were {count} return transactions, totaling ${amount:,.2f} in signed revenue.", []
    if re.search(r"\b(transactions?|orders?)\b", topics):
        return f"There were {len(silver):,} unique transactions after exact duplicates were removed.", []
    if re.search(r"\b(revenue|sales|amount|total)\b", topics):
        revenue = silver["total_amount"].sum() if year else summary["net_revenue"].sum()
        return f"Total net revenue was ${revenue:,.2f}, including returns.", []
    return unsupported, []


def generate_report(prompt: str, silver: pd.DataFrame, summary: pd.DataFrame, issues: pd.DataFrame) -> str:
    request = prompt.strip()
    if not request or len(request) > 2000:
        raise ValueError("Enter a report prompt of 1 to 2000 characters")
    answer, evidence = answer_business_question(request, silver, summary, issues)
    lines = ["# Sales pipeline report", "", f"**Business question:** {request.replace(chr(10), ' ')}", "", "## Answer", "", answer]
    if evidence:
        lines.extend(["", *evidence])
    lines.extend(["", "Generated from the uploaded CSVs with deterministic calculations; no AI model was used."])
    return "\n".join(lines) + "\n"


def export_visual_report(question: str, silver: pd.DataFrame, summary: pd.DataFrame, issues: pd.DataFrame) -> tuple[str, list[Figure]]:
    answer, _ = answer_business_question(question, silver, summary, issues)
    title, chart = question_chart(question, silver, issues)
    figures = []
    if not answer.startswith(("I cannot", "No transactions")) and not chart.empty:
        dimension = chart.columns[0]
        chart = chart.copy()
        chart[dimension] = chart[dimension].map(lambda label: html.escape(str(label)))
        breakdown = px.bar(chart, x=dimension, y="value", title=title, labels={dimension: "", "value": title})
        breakdown.update_traces(marker_color="#28766c")
        figures.append(breakdown)
        filtered = filter_question_period(question.casefold(), silver)
        trend = filtered.groupby("transaction_date", as_index=False)["total_amount"].sum().sort_values("transaction_date")
        if not trend.empty:
            daily = px.line(trend, x="transaction_date", y="total_amount", markers=True, title="Daily net revenue", labels={"transaction_date": "Date", "total_amount": "Net revenue"})
            daily.update_traces(line_color="#d16c4f")
            figures.append(daily)
    for figure in figures:
        figure.update_layout(template="plotly_white", font_family="Segoe UI", margin=dict(l=30, r=30, t=60, b=30))
    charts = "".join(figure.to_html(full_html=False, include_plotlyjs="inline" if index == 0 else False) for index, figure in enumerate(figures))
    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Sales pipeline report</title><style>body{font:16px 'Segoe UI',sans-serif;color:#26353b;background:#f3f6f5;margin:0}"
        "main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:32px}h2{color:#28766c}"
        ".chart{background:white;border:1px solid #dde5e3;border-radius:6px;margin:20px 0;padding:12px}"
        "small{color:#56676e}</style></head><body><main><h1>Sales pipeline report</h1>"
        f"<p><strong>Business question:</strong> {html.escape(question)}</p><h2>Answer</h2><p>{html.escape(answer)}</p>"
        f"<p><strong>Transactions:</strong> {len(silver):,} &nbsp; <strong>Net revenue:</strong> ${summary['net_revenue'].sum():,.2f}"
        f" &nbsp; <strong>Catalog issues:</strong> {len(issues)}</p>"
        f"<section class='chart'>{charts}</section>" if figures else ""
    )
    if not figures:
        document = (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Sales pipeline report</title></head><body><main>"
            f"<h1>Sales pipeline report</h1><p>{html.escape(question)}</p><h2>Answer</h2><p>{html.escape(answer)}</p>"
        )
    return document + "<small>Calculated locally from approved CSV layers. No AI model was used.</small></main></body></html>", figures


def advance_review(root: Path, state: dict | None, uploads: Mapping[str, bytes], prompt: str) -> dict:
    if not prompt.strip() or len(prompt) > 2000:
        raise ValueError("Enter a report prompt of 1 to 2000 characters")
    if state is None:
        return {"stage": "bronze", "bronze": create_bronze(root, uploads)}
    if state["stage"] == "bronze":
        return {**state, "stage": "silver", "silver": create_silver(root, state["bronze"])}
    if state["stage"] == "silver":
        summary, issues = create_gold(root, state["silver"], state["bronze"]["products.csv"])
        return {**state, "stage": "gold", "summary": summary, "issues": issues}
    if state["stage"] == "gold":
        report = generate_report(prompt, state["silver"], state["summary"], state["issues"])
        directory = root / "reports"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sales_report.md").write_text(report, encoding="utf-8")
        return {**state, "stage": "complete", "report": report}
    raise ValueError("Pipeline is already complete")


def run_uploaded_flow(root: Path, uploads: Mapping[str, bytes], prompt: str, on_stage: Callable[[str], None] | None = None) -> dict:
    state = None
    for stage in ("Bronze", "Silver", "Gold", "complete"):
        state = advance_review(root, state, uploads, prompt)
        if on_stage and stage != "complete":
            on_stage(stage)
    return state