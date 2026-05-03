"""3D Print In-Process Monitor Dashboard — Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import FAILURE_COLORS, PrintConfig, PrintFleetGenerator
from monitor import PrintMonitor

st.set_page_config(
    page_title="3D Print Process Monitor",
    page_icon="🖨️",
    layout="wide",
)

ALERT_COLORS = {"OK": "#2ecc71", "WARNING": "#f39c12", "CRITICAL": "#e74c3c"}


# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------

@st.cache_resource
def load_monitor(n_jobs: int) -> tuple[pd.DataFrame, PrintMonitor, object]:
    fleet_df = PrintFleetGenerator(n_jobs=n_jobs).generate()
    monitor = PrintMonitor()
    results = monitor.fit(fleet_df)
    return fleet_df, monitor, results


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.title("🖨️ 3D Print Monitor")
st.sidebar.markdown("Layer-by-layer anomaly detection and failure mode classification for an FDM printer fleet.")

n_jobs = st.sidebar.slider("Fleet size (print jobs)", 10, 50, 30, step=5)
fleet_df, monitor, results = load_monitor(n_jobs)

job_ids = sorted(fleet_df["job_id"].unique())
selected_job = st.sidebar.selectbox("Inspect print job", job_ids)

health_threshold = st.sidebar.slider("Warning threshold (health score)", 20, 80, 55)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Classifier accuracy:** {results.test_accuracy:.1%}")
st.sidebar.markdown(f"**Total layers:** {len(fleet_df):,}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

st.title("3D Print In-Process Monitor")

# Score the selected job
job_df = fleet_df[fleet_df["job_id"] == selected_job].reset_index(drop=True)
scored = monitor.score(job_df)

# Current status
last_scored = scored.dropna(subset=["health_score"])
if len(last_scored) == 0:
    st.warning("Not enough layers to score this job.")
    st.stop()

last_layer = last_scored.iloc[-1]
health = float(last_layer["health_score"])
alert = str(last_layer["alert_level"])
failure = str(last_layer["predicted_failure"])
alert_color = ALERT_COLORS.get(alert, "#95a5a6")

# Status cards
col1, col2, col3, col4 = st.columns(4)
col1.markdown(
    f"<div style='background:{alert_color};padding:14px;border-radius:8px;"
    f"text-align:center;color:white;font-weight:bold'>"
    f"Alert Level<br><span style='font-size:22px'>{alert}</span></div>",
    unsafe_allow_html=True,
)
col2.metric("Health Score", f"{health:.0f} / 100")
col3.metric("Predicted Failure", failure)
col4.metric("Layers Processed", f"{len(scored):,}")

st.markdown("---")

# ------------------------------------------------------------------
# Health score timeline
# ------------------------------------------------------------------

st.subheader("Layer Health Score Timeline")
fig_health = go.Figure()

fig_health.add_trace(go.Scatter(
    x=scored["layer"],
    y=scored["health_score"],
    mode="lines",
    line=dict(color="#3498db", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(52,152,219,0.15)",
    name="Health Score",
))

# Threshold reference line
fig_health.add_hline(
    y=health_threshold,
    line_dash="dash",
    line_color="#e74c3c",
    annotation_text=f"Warning threshold ({health_threshold})",
)

# Anomaly markers
anomaly_rows = scored[scored["zscore_anomaly"] == True]
fig_health.add_trace(go.Scatter(
    x=anomaly_rows["layer"],
    y=anomaly_rows["health_score"],
    mode="markers",
    marker=dict(color="#e74c3c", size=6, symbol="x"),
    name="Z-score anomaly",
))

fig_health.update_layout(
    xaxis_title="Layer Number",
    yaxis_title="Health Score (0–100)",
    yaxis_range=[0, 105],
    height=300,
    margin=dict(l=30, r=10, t=10, b=30),
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig_health, use_container_width=True)

# ------------------------------------------------------------------
# Sensor traces
# ------------------------------------------------------------------

st.subheader("Sensor Traces")

sensor_labels = {
    "nozzle_temp_c":      "Nozzle Temp (°C)",
    "bed_temp_c":         "Bed Temp (°C)",
    "extruder_current_a": "Extruder Current (A)",
    "layer_height_dev_mm":"Layer Height Deviation (mm)",
}

cols = st.columns(2)
for i, (col, label) in enumerate(sensor_labels.items()):
    if col not in scored.columns:
        continue
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scored["layer"], y=scored[col],
        mode="lines", line=dict(width=1.2, color="#2c3e50"),
        name=label,
    ))
    # Highlight anomaly zones
    anom = scored[scored["zscore_anomaly"] == True]
    if len(anom) > 0:
        fig.add_trace(go.Scatter(
            x=anom["layer"], y=anom[col],
            mode="markers",
            marker=dict(color="#e74c3c", size=4),
            name="Anomaly",
        ))
    fig.update_layout(
        title=label,
        xaxis_title="Layer",
        height=220,
        margin=dict(l=30, r=10, t=35, b=30),
        showlegend=False,
    )
    cols[i % 2].plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Failure mode distribution across fleet
# ------------------------------------------------------------------

st.markdown("---")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Fleet — Failure Mode Distribution")
    # Score all jobs (last layer per job)
    all_scored = []
    for jid in job_ids:
        jdf = fleet_df[fleet_df["job_id"] == jid].reset_index(drop=True)
        try:
            s = monitor.score(jdf)
            s = s.dropna(subset=["predicted_failure"])
            if len(s) > 0:
                last = s.iloc[-1]
                all_scored.append({"job_id": jid, "failure": str(last["predicted_failure"]), "health": float(last["health_score"])})
        except Exception:
            pass

    if all_scored:
        fleet_summary = pd.DataFrame(all_scored)
        dist = fleet_summary["failure"].value_counts().reset_index()
        dist.columns = ["Failure Mode", "Count"]
        fig_dist = px.pie(
            dist, values="Count", names="Failure Mode",
            color="Failure Mode", color_discrete_map=FAILURE_COLORS,
        )
        fig_dist.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig_dist, use_container_width=True)

with col6:
    st.subheader("Feature Importance (failure classifier)")
    fi = results.feature_importances
    # Show top 8 features
    fi_top = fi.tail(8)
    fig_fi = px.bar(
        fi_top,
        orientation="h",
        color=fi_top.values,
        color_continuous_scale="Purples",
        labels={"value": "Importance", "index": "Feature"},
    )
    fig_fi.update_layout(
        showlegend=False, coloraxis_showscale=False,
        height=300, margin=dict(t=10),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ------------------------------------------------------------------
# Alert log
# ------------------------------------------------------------------

st.markdown("---")
st.subheader("Alert Log")
alerts = scored[
    (scored["alert_level"] != "OK") & scored["zscore_anomaly"]
][["layer", "predicted_failure", "health_score", "alert_level", "nozzle_temp_c", "extruder_current_a"]].copy()
alerts.columns = ["Layer", "Predicted Failure", "Health Score", "Alert", "Nozzle Temp", "Extruder A"]

if len(alerts) == 0:
    st.success("No alerts for this print job.")
else:
    st.dataframe(alerts.reset_index(drop=True), use_container_width=True, hide_index=True)
