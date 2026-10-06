# Medallion Pipeline

A local sales pipeline with sample CSVs in `data/landing/`. Upload your own sales, products, and stores CSVs to run the interactive flow. Uploaded data and generated outputs are excluded from Git; do not commit real customer data or credentials.

From the project root on Windows, create a virtual environment with `py -3.12 -m venv .venv` if `.venv` does not already exist. Then install dependencies and start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

In Command Prompt, `streamlit run app\streamlit_app.py` works from the project root through `streamlit.cmd`. If a browser does not open, visit the `Local URL` printed in the terminal (usually `http://localhost:8501`). Streamlit's first-run email prompt is disabled for this project; restart any older session still waiting at `Email:`.

Upload `sales_data.csv`, `products.csv`, and `stores.csv` in their respective fields (up to 10 MB each), then enter a business question and click **Start flow**. Review Bronze's source preview and schema/rules template and click **Approve Bronze layer** to generate Silver. Review cleaned sales and approve Silver to generate Gold. Review category sales and catalog issues and approve Gold to generate the report. Each stage can be downloaded before approval; the report is unavailable until all three approvals are complete.

The final view has a question-specific ranking chart and a daily net revenue chart. **Export visual report** downloads a self-contained `business_report.html` with its charts; **Download text report** downloads `sales_report.md`. Each browser flow writes to an isolated `runs/<id>/` directory so earlier exports cannot count as approved stages. Changing the uploads or the question on submission starts a new review.

The report directly answers supported questions about net revenue, transaction counts, net units, returns, categories, products, stores, regions, customer segments, payment methods, and catalog issues. It supports top/bottom 1 to 10 rankings and explicit years or months such as `February 2024` or `2024-02`. Unknown store/region revenue is disclosed but excluded from named-location rankings. If the question cannot be answered from the uploaded fields, the report says so rather than inventing an answer. Calculations are local and deterministic; no CSV data is sent to an AI provider.

## Layers

| Layer | Files | Behavior |
| --- | --- | --- |
| Landing | `data/landing/{sales_data,products,stores}.csv` | Included samples for local tests and the optional `run_pipeline` function. The web app uses uploads instead. |
| Bronze | `data/bronze_layer/*.csv` | Byte-for-byte copies of the three uploaded inputs. |
| Silver | `data/silver_layer/sales_clean.csv` | Exact duplicates removed; conflicting transaction IDs rejected; dates, categories, segments, and numeric fields normalized; stores enriched, unknown stores marked `Unknown`. Negative quantities and amounts remain signed as returns. |
| Gold | `data/gold_layer/category_sales.csv`, `product_catalog_issues.csv` | Category transaction counts, net units, return counts, net revenue; missing or conflicting product catalog names reported separately. Sales product names are not overwritten by the catalog because the provided files disagree on their IDs. |

For interactive runs, templates are written to `runs/<id>/data/sttm/{bronze,silver,gold}_template.json`; the Markdown report is written to `runs/<id>/reports/sales_report.md`. These are generated files and are ignored by Git. The optional `run_pipeline` helper still writes layers beneath the top-level `data/` folders.

Invalid dates, quantities, amounts, missing required columns/IDs, and conflicting IDs stop the run with an error. Gold revenue uses the supplied `total_amount`, including negative returns. Store IDs absent from the store catalog remain visible as unknown.

Run all local tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```