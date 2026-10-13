"""Authenticated operational overview for administrators."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from core.admin_dashboard import (
    admin_auth_configured,
    collect_admin_statistics,
    get_system_health,
    verify_admin_credentials,
)
from core.config import DATA_DIR

st.set_page_config(page_title="Admin Dashboard", page_icon="📊", layout="wide")


def _render_login() -> None:
    st.title("Admin Dashboard")
    if not admin_auth_configured():
        st.error("Admin access is disabled until ADMIN_USERNAME and ADMIN_PASSWORD are configured.")
        st.caption("Set both values in the local .env file, then restart the Streamlit app.")
        st.stop()

    with st.form("admin_login"):
        username = st.text_input("Admin username")
        password = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if verify_admin_credentials(username, password):
            st.session_state.admin_authenticated = True
            st.rerun()
        st.error("Invalid username or password.")
    st.stop()


if not st.session_state.get("admin_authenticated", False):
    _render_login()

header_col, action_col = st.columns([5, 1])
with header_col:
    st.title("Admin Dashboard")
    st.caption("Operational view across the default workspace and all user workspaces.")
with action_col:
    if st.button("Sign out", use_container_width=True):
        st.session_state.pop("admin_authenticated", None)
        st.rerun()

statistics = collect_admin_statistics()
health = get_system_health(DATA_DIR)

metric_cols = st.columns(4)
metric_cols[0].metric("Pipeline runs", statistics["total_runs"])
metric_cols[1].metric("Running", statistics["active_runs"])
metric_cols[2].metric("Failed", statistics["failed_runs"])
metric_cols[3].metric("Failure rate", f"{statistics['failure_rate']:.1%}")

st.subheader("System Health")
health_cols = st.columns(2)
with health_cols[0]:
    if "memory_used_percent" in health:
        st.metric(
            "Memory used",
            f"{health['memory_used_percent']:.1f}%",
            f"{health['memory_available_gb']:.1f} GB available of {health['memory_total_gb']:.1f} GB",
            delta_color="inverse",
        )
        st.progress(min(health["memory_used_percent"] / 100, 1.0))
    else:
        st.metric("Memory", "Unavailable")
with health_cols[1]:
    if "disk_used_percent" in health:
        st.metric(
            "Disk used (data volume)",
            f"{health['disk_used_percent']:.1f}%",
            f"{health['disk_free_gb']:.1f} GB free of {health['disk_total_gb']:.1f} GB",
            delta_color="inverse",
        )
        st.progress(min(health["disk_used_percent"] / 100, 1.0))
    else:
        st.metric("Disk", "Unavailable")
st.caption(f"Sampled at {health['sampled_at']}")

st.subheader("Agent Performance")
agent_metrics = statistics["agent_metrics"]
if agent_metrics:
    agent_frame = pd.DataFrame(agent_metrics)
    performance_col, table_col = st.columns([3, 2])
    with performance_col:
        st.plotly_chart(
            px.bar(
                agent_frame,
                x="Agent",
                y="Avg duration (s)",
                title="Average agent duration",
                color="Agent",
            ),
            use_container_width=True,
        )
    with table_col:
        st.dataframe(agent_frame, hide_index=True, use_container_width=True)
else:
    st.info("No agent timing metrics have been captured yet.")

st.subheader("Errors and Common Failures")
failures = statistics["common_failures"]
if failures:
    st.plotly_chart(
        px.bar(
            pd.DataFrame(failures),
            x="Failure",
            y="Count",
            title="Most common failure categories",
            color="Failure",
        ),
        use_container_width=True,
    )
else:
    st.info("No failed runs have been recorded.")

st.subheader("Pipeline Runs")
status_counts = statistics["status_counts"]
if status_counts:
    status_frame = pd.DataFrame(
        [{"Status": status, "Runs": count} for status, count in status_counts.items()]
    )
    status_col, overview_col = st.columns([2, 3])
    with status_col:
        st.plotly_chart(
            px.pie(status_frame, names="Status", values="Runs", hole=0.5, title="Run status"),
            use_container_width=True,
        )
    with overview_col:
        st.dataframe(
            pd.DataFrame(statistics["runs"]),
            hide_index=True,
            use_container_width=True,
            height=420,
        )
else:
    st.info("No pipeline runs have been recorded yet.")

st.caption(
    f"Total LLM API calls: {statistics['api_calls']:,} · "
    f"Tokens: {statistics['input_tokens']:,} input / {statistics['output_tokens']:,} output"
)
