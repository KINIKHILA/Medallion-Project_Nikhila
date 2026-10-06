import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
	sys.path.insert(0, str(root))

from core.sales_pipeline import advance_review, export_visual_report

st.set_page_config(page_title="Medallion Pipeline", layout="wide")
st.title("Medallion Pipeline")

st.subheader("1. Upload files")
sales_file = st.file_uploader("Sales transactions", type="csv", key="sales")
products_file = st.file_uploader("Product catalog", type="csv", key="products")
stores_file = st.file_uploader("Store catalog", type="csv", key="stores")

files = {"sales_data.csv": sales_file, "products.csv": products_file, "stores.csv": stores_file}
uploaded_all = all(file is not None for file in files.values())
prompt = ""
start_flow = False
if uploaded_all:
	with st.form("business_question_form"):
		st.subheader("2. Business question")
		prompt = st.text_input("Business question", placeholder="Which store generated the highest net revenue?", max_chars=2000)
		start_flow = st.form_submit_button("Start flow", type="primary")
else:
	st.button("Start flow", disabled=True)
uploads = {name: file.getvalue() for name, file in files.items() if file is not None}
signature = (tuple((name, hashlib.sha256(data).hexdigest()) for name, data in uploads.items()), prompt)
if st.session_state.get("flow_signature") != signature:
	st.session_state.pop("review_state", None)
	st.session_state.pop("review_root", None)
	st.session_state["flow_signature"] = signature

if start_flow and not prompt.strip():
	st.error("Enter a business question before starting the flow.")
if start_flow and prompt.strip():
	run_root = root / "runs" / uuid4().hex
	try:
		state = advance_review(run_root, None, uploads, prompt)
	except (OSError, ValueError, pd.errors.ParserError) as error:
		st.error(str(error))
	else:
		st.session_state["review_state"] = state
		st.session_state["review_root"] = run_root

state = st.session_state.get("review_state")
if state is not None:
	run_root = st.session_state["review_root"]
	stages = ("bronze", "silver", "gold")
	approved_count = len(stages) if state["stage"] == "complete" else stages.index(state["stage"])
	for index, (column, stage_name) in enumerate(zip(st.columns(3), stages)):
		if index < approved_count:
			column.success(f"{stage_name.title()} approved")
		elif state["stage"] == stage_name:
			column.info(f"{stage_name.title()} awaiting approval")
		else:
			column.write(f"{stage_name.title()} pending")

	if state["stage"] in stages:
		stage_name = state["stage"]
		st.subheader(f"{stages.index(stage_name) + 3}. {stage_name.title()} layer review and approve")
		template_file = run_root / "data" / "sttm" / f"{stage_name}_template.json"
		st.json(json.loads(template_file.read_text(encoding="utf-8")), expanded=True)
		st.download_button(f"Download {stage_name} template", template_file.read_bytes(), template_file.name, "application/json")
		if stage_name == "bronze":
			source = st.selectbox("Review source file", list(files))
			st.dataframe(state["bronze"][source].head(100), hide_index=True)
		elif stage_name == "silver":
			st.dataframe(state["silver"].head(100), hide_index=True)
			st.download_button("Download cleaned sales", state["silver"].to_csv(index=False), "sales_clean.csv", "text/csv")
		else:
			st.dataframe(state["summary"], hide_index=True)
			st.dataframe(state["issues"].head(100), hide_index=True)
			st.download_button("Download category sales", state["summary"].to_csv(index=False), "category_sales.csv", "text/csv")
		if st.button(f"Approve {stage_name.title()} layer", type="primary"):
			try:
				st.session_state["review_state"] = advance_review(run_root, state, uploads, prompt)
			except (OSError, ValueError, pd.errors.ParserError) as error:
				st.error(str(error))
			else:
				st.rerun()
	else:
		st.subheader("6. Business report")
		visual_report, figures = export_visual_report(prompt, state["silver"], state["summary"], state["issues"])
		st.markdown(state["report"])
		for figure in figures:
			st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
		st.download_button("Export visual report", visual_report, "business_report.html", "text/html", type="primary")
		st.download_button("Download text report", state["report"], "sales_report.md", "text/markdown")
