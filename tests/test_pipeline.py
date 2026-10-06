import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from core.sales_pipeline import advance_review, build_gold, clean_sales, export_visual_report, generate_report, question_chart, run_pipeline, run_uploaded_flow


class SalesPipelineTest(unittest.TestCase):
    def test_clean_sales_deduplicates_normalizes_and_keeps_returns(self):
        sales = pd.DataFrame(
            [
                {"transaction_id": "T1", "transaction_date": "01/15/2024", "store_id": "S01", "product_id": "P1", "product_name": "Mouse", "category": "Elec.", "quantity": "2", "unit_price": "10", "total_amount": "20", "customer_segment": "Prem", "payment_method": "Cash"},
                {"transaction_id": "T1", "transaction_date": "01/15/2024", "store_id": "S01", "product_id": "P1", "product_name": "Mouse", "category": "Elec.", "quantity": "2", "unit_price": "10", "total_amount": "20", "customer_segment": "Prem", "payment_method": "Cash"},
                {"transaction_id": "T2", "transaction_date": "2024-01-16", "store_id": "", "product_id": "P2", "product_name": "Pen", "category": "stationery", "quantity": "-1", "unit_price": "5", "total_amount": "-5", "customer_segment": "Std", "payment_method": "Card"},
            ]
        )
        stores = pd.DataFrame([{"store_id": "S01", "store_name": "Main", "region": "East", "city": "Boston", "state": "MA"}])

        silver = clean_sales(sales, stores)

        self.assertEqual(len(silver), 2)
        self.assertEqual(silver["category"].tolist(), ["Electronics", "Stationery"])
        self.assertEqual(silver["customer_segment"].tolist(), ["Premium", "Standard"])
        self.assertEqual(silver["transaction_date"].tolist(), ["2024-01-15", "2024-01-16"])
        self.assertEqual(silver["store_name"].tolist(), ["Main", "Unknown"])
        self.assertEqual(silver["quantity"].tolist(), [2, -1])

    def test_gold_summarizes_net_sales_and_flags_catalog_conflicts(self):
        silver = pd.DataFrame([
            {"transaction_id": "T1", "product_id": "P1", "product_name": "Mouse", "category": "Electronics", "quantity": 2, "total_amount": 20.0},
            {"transaction_id": "T2", "product_id": "P2", "product_name": "Pen", "category": "Stationery", "quantity": -1, "total_amount": -5.0},
        ])
        products = pd.DataFrame([
            {"product_id": "P1", "product_name": "Phone", "category": "Electronics", "standard_price": "799"},
        ])

        summary, issues = build_gold(silver, products)

        self.assertEqual(summary.set_index("category").loc["Stationery", "net_revenue"], -5.0)
        self.assertEqual(summary.set_index("category").loc["Stationery", "returns"], 1)
        self.assertEqual(set(issues["issue"]), {"Name mismatch", "Missing from catalog"})

    def test_rejects_conflicting_transaction_ids(self):
        sales = pd.DataFrame([
            {"transaction_id": "T1", "transaction_date": "2024-01-15", "store_id": "S01", "product_id": "P1", "product_name": "Mouse", "category": "Electronics", "quantity": "1", "unit_price": "10", "total_amount": "10", "customer_segment": "Loyal", "payment_method": "Cash"},
            {"transaction_id": "T1", "transaction_date": "2024-01-15", "store_id": "S01", "product_id": "P1", "product_name": "Mouse", "category": "Electronics", "quantity": "2", "unit_price": "10", "total_amount": "20", "customer_segment": "Loyal", "payment_method": "Cash"},
        ])
        stores = pd.DataFrame(columns=["store_id", "store_name", "region"])

        with self.assertRaisesRegex(ValueError, "conflicting rows"):
            clean_sales(sales, stores)

    def test_run_pipeline_writes_layers_from_isolated_landing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            landing = root / "data" / "landing"
            landing.mkdir(parents=True)
            source = (
                "transaction_id,transaction_date,store_id,product_id,product_name,category,quantity,unit_price,total_amount,customer_segment,payment_method\n"
                "T1,2024-01-15,S01,P1,Mouse,Elec.,2,10,20,Prem,Cash\n"
                "T1,2024-01-15,S01,P1,Mouse,Elec.,2,10,20,Prem,Cash\n"
            )
            (landing / "sales_data.csv").write_text(source, encoding="utf-8")
            (landing / "stores.csv").write_text("store_id,store_name,region\nS01,Main,East\n", encoding="utf-8")
            (landing / "products.csv").write_text("product_id,product_name\nP1,Phone\n", encoding="utf-8")

            silver, gold, issues = run_pipeline(root)

            self.assertEqual((root / "data" / "bronze_layer" / "sales_data.csv").read_text(encoding="utf-8"), source)
            self.assertEqual(len(pd.read_csv(root / "data" / "silver_layer" / "sales_clean.csv")), 1)
            self.assertEqual(gold.loc[0, "net_revenue"], 20)
            self.assertEqual(issues.loc[0, "issue"], "Name mismatch")
            self.assertEqual(len(silver), 1)

    def test_uploaded_flow_creates_templates_in_order_and_exports_prompt_report(self):
        files = {
            "sales_data.csv": (
                "transaction_id,transaction_date,store_id,product_id,product_name,category,quantity,unit_price,total_amount,customer_segment,payment_method\n"
                "T1,2024-01-15,S01,P1,Mouse,Elec.,2,10,20,Prem,Cash\n"
                "T1,2024-01-15,S01,P1,Mouse,Elec.,2,10,20,Prem,Cash\n"
            ).encode(),
            "stores.csv": b"store_id,store_name,region\nS01,Main,East\n",
            "products.csv": b"product_id,product_name\nP1,Phone\n",
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stages = []
            result = run_uploaded_flow(root, files, "Summarize category sales and catalog issues", stages.append)

            self.assertEqual(stages, ["Bronze", "Silver", "Gold"])
            self.assertEqual(result["summary"].loc[0, "net_revenue"], 20)
            self.assertIn("Summarize category sales and catalog issues", result["report"])
            self.assertIn("Electronics", result["report"])
            self.assertIn("Name mismatch", result["report"])
            self.assertNotIn("T1,2024-01-15", result["report"])
            self.assertTrue((root / "reports" / "sales_report.md").is_file())
            for stage in ("bronze", "silver", "gold"):
                template = json.loads((root / "data" / "sttm" / f"{stage}_template.json").read_text(encoding="utf-8"))
                self.assertEqual(template["layer"], stage)
            self.assertEqual((root / "data" / "bronze_layer" / "sales_data.csv").read_bytes(), files["sales_data.csv"])

    def test_uploaded_flow_rejects_missing_files_before_writing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Missing upload"):
                run_uploaded_flow(root, {"sales_data.csv": b"invalid"}, "Sales report")
            self.assertFalse((root / "data" / "bronze_layer").exists())

    def test_each_review_approval_gates_the_next_layer_and_report(self):
        files = {
            "sales_data.csv": (
                "transaction_id,transaction_date,store_id,product_id,product_name,category,quantity,unit_price,total_amount,customer_segment,payment_method\n"
                "T1,2024-01-15,S01,P1,Mouse,Electronics,2,10,20,Premium,Cash\n"
            ).encode(),
            "stores.csv": b"store_id,store_name,region\nS01,Main,East\n",
            "products.csv": b"product_id,product_name\nP1,Mouse\n",
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            state = advance_review(root, None, files, "Which store generated the most revenue?")
            self.assertEqual(state["stage"], "bronze")
            self.assertTrue((root / "data" / "sttm" / "bronze_template.json").exists())
            self.assertFalse((root / "data" / "silver_layer" / "sales_clean.csv").exists())

            state = advance_review(root, state, files, "Which store generated the most revenue?")
            self.assertEqual(state["stage"], "silver")
            self.assertTrue((root / "data" / "sttm" / "silver_template.json").exists())
            self.assertFalse((root / "data" / "gold_layer" / "category_sales.csv").exists())

            state = advance_review(root, state, files, "Which store generated the most revenue?")
            self.assertEqual(state["stage"], "gold")
            self.assertTrue((root / "data" / "sttm" / "gold_template.json").exists())
            self.assertFalse((root / "reports" / "sales_report.md").exists())

            state = advance_review(root, state, files, "Which store generated the most revenue?")
            self.assertEqual(state["stage"], "complete")
            self.assertIn("Main generated the most net revenue: $20.00", state["report"])
            self.assertTrue((root / "reports" / "sales_report.md").exists())
            with self.assertRaisesRegex(ValueError, "already complete"):
                advance_review(root, state, files, "Which store generated the most revenue?")

    def test_report_selects_requested_store_section(self):
        silver = pd.DataFrame([{"quantity": 1, "store_name": "Main", "total_amount": 12.5}])
        summary = pd.DataFrame([{"category": "Electronics", "net_revenue": 12.5, "transactions": 1, "returns": 0}])
        issues = pd.DataFrame(columns=["issue"])

        report = generate_report("Show store sales", silver, summary, issues)

        self.assertIn("Main: $12.50 net", report)
        self.assertNotIn("## Catalog quality", report)
        with self.assertRaisesRegex(ValueError, "report prompt"):
            generate_report(" ", silver, summary, issues)

    def test_report_answers_ranked_business_question_from_sales_values(self):
        silver = pd.DataFrame([
            {"quantity": 2, "store_name": "Main", "product_name": "Mouse", "category": "Electronics", "total_amount": 20.0},
            {"quantity": 1, "store_name": "Outlet", "product_name": "Pen", "category": "Stationery", "total_amount": 7.0},
            {"quantity": -1, "store_name": "Outlet", "product_name": "Pen", "category": "Stationery", "total_amount": -2.0},
        ])
        summary = pd.DataFrame([
            {"category": "Electronics", "net_revenue": 20.0, "transactions": 1, "returns": 0},
            {"category": "Stationery", "net_revenue": 5.0, "transactions": 2, "returns": 1},
        ])
        issues = pd.DataFrame(columns=["issue"])

        store_report = generate_report("Which store generated the highest revenue?", silver, summary, issues)
        product_report = generate_report("Which product sold the most units?", silver, summary, issues)

        self.assertIn("## Answer", store_report)
        self.assertIn("Main generated the highest net revenue: $20.00", store_report)
        self.assertIn("Mouse sold the most net units: 2", product_report)

    def test_report_explains_unsupported_questions_without_fabricating_an_answer(self):
        silver = pd.DataFrame([{"quantity": 1, "total_amount": 12.5}])
        summary = pd.DataFrame([{"category": "Electronics", "net_revenue": 12.5, "transactions": 1, "returns": 0}])
        issues = pd.DataFrame(columns=["issue"])

        report = generate_report("Why did customers prefer this brand?", silver, summary, issues)

        self.assertIn("cannot answer", report.lower())
        self.assertNotIn("## Category sales", report)

    def test_report_honors_requested_month_and_top_count(self):
        silver = pd.DataFrame([
            {"transaction_date": "2024-01-12", "product_name": "January Item", "quantity": 1, "total_amount": 500.0},
            {"transaction_date": "2024-02-04", "product_name": "Mouse", "quantity": 1, "total_amount": 30.0},
            {"transaction_date": "2024-02-08", "product_name": "Cable", "quantity": 1, "total_amount": 20.0},
            {"transaction_date": "2024-02-09", "product_name": "Pen", "quantity": 1, "total_amount": 10.0},
        ])
        summary = pd.DataFrame([{"category": "Mixed", "net_revenue": 560.0, "transactions": 4, "returns": 0}])
        issues = pd.DataFrame(columns=["issue"])

        report = generate_report("What are the top 2 products by revenue in February 2024?", silver, summary, issues)

        self.assertIn("Mouse generated the highest net revenue: $30.00", report)
        self.assertIn("- Cable: $20.00 net", report)
        self.assertNotIn("January Item", report)
        self.assertNotIn("- Pen:", report)
        self.assertIn("No transactions matched", generate_report("Top product in 2025?", silver, summary, issues))

    def test_store_rankings_exclude_unknown_but_disclose_unassigned_revenue(self):
        silver = pd.DataFrame([
            {"store_name": "Unknown", "quantity": 1, "total_amount": 100.0},
            {"store_name": "Main", "quantity": 1, "total_amount": 50.0},
        ])
        summary = pd.DataFrame([{"category": "Electronics", "net_revenue": 150.0, "transactions": 2, "returns": 0}])

        report = generate_report("Which store had the highest revenue?", silver, summary, pd.DataFrame(columns=["issue"]))

        self.assertIn("Main generated the highest net revenue: $50.00", report)
        self.assertIn("$100.00 of unassigned revenue", report)

    def test_question_chart_matches_ranked_answer_and_period(self):
        sales = pd.DataFrame([
            {"transaction_date": "2024-01-01", "store_name": "Unknown", "category": "Electronics", "quantity": 1, "total_amount": 100.0},
            {"transaction_date": "2024-02-01", "store_name": "Main", "category": "Electronics", "quantity": 1, "total_amount": 25.0},
            {"transaction_date": "2024-02-02", "store_name": "Outlet", "category": "Stationery", "quantity": 1, "total_amount": 10.0},
        ])
        issues = pd.DataFrame([{"issue": "Name mismatch"}])

        label, chart = question_chart("Top 1 store by revenue in February 2024", sales, issues)
        self.assertEqual(label, "Net revenue by store")
        self.assertEqual(chart["store_name"].tolist(), ["Main"])
        self.assertEqual(chart["value"].tolist(), [25.0])

        label, chart = question_chart("How many catalog issues?", sales, issues)
        self.assertEqual(label, "Catalog issues")
        self.assertEqual(chart["value"].tolist(), [1])

    def test_visual_report_exports_answer_and_chart_as_safe_standalone_html(self):
        sales = pd.DataFrame([{"transaction_date": "2024-01-15", "store_name": "Main", "category": "Electronics", "quantity": 2, "total_amount": 20.0}])
        summary = pd.DataFrame([{"category": "Electronics", "net_revenue": 20.0, "transactions": 1, "returns": 0}])
        issues = pd.DataFrame(columns=["issue"])
        question = "Which store has the highest revenue? <script>alert(1)</script>"

        document, figures = export_visual_report(question, sales, summary, issues)

        self.assertIn("Main generated the highest net revenue: $20.00", document)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)
        self.assertNotIn("<script>alert(1)</script>", document)
        self.assertIn("plotly.js", document)
        self.assertEqual(len(figures), 2)


if __name__ == "__main__":
    unittest.main()