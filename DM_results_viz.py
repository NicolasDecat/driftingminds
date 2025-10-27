#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:50:11 2025

DM: visualisation of DM results

@author: nicolas.decat
"""

import os
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Config ----------

# To add in Streamlit > Manage app > Settings > Secrets
# REDCAP_API_URL = "https://redcap-icm.icm-institute.org/api/"
# REDCAP_API_TOKEN = "641919114F091FC5A1860BCFC53D3947"

REDCAP_API_URL = st.secrets.get("REDCAP_API_URL")
REDCAP_API_TOKEN = st.secrets.get("REDCAP_API_TOKEN")

if not REDCAP_API_URL or not REDCAP_API_TOKEN:
    st.error("Missing REDCap secrets. Set REDCAP_API_URL and REDCAP_API_TOKEN in Streamlit Secrets.")
    st.stop()

# --- Quick connectivity check (fast, no token required) ---
try:
    r = requests.post(REDCAP_API_URL, data={"content": "version", "format": "json"}, timeout=6)
    if r.ok:
        st.caption(f"REDCap API reachable. Version: {r.text.strip()}")
    else:
        st.warning(f"REDCap API responded with HTTP {r.status_code}. Check URL or network.")
except Exception as e:
    st.error(f"Cannot reach REDCap API at {REDCAP_API_URL}. Is it public? Firewall/VPN? {e}")
    st.stop()
    

    

# Define which fields belong to which scales for the radar
SCALES = {
    "Thoughts": ["freq_think_nocontrol", "freq_think_seq_bizarre", "freq_think_seq_ordinary"],
    "Perception":    ["freq_percept_precise",  "freq_percept_real",  "freq_percept_imposed"]
}

# Optional: load anonymized norms to position the participant vs crowd
# def load_norms():
#     try:
#         df = pd.read_csv("norms.csv")   # columns should match your fields or pre-agg scales
#         return df
#     except Exception:
#         return None

# NORMS = load_norms()

# ---------- Helpers ----------
def fetch_by_record_id(record_id: str):
    payload = {
        "token": REDCAP_API_TOKEN,
        "content": "record",
        "format": "json",
        "type": "flat",
        "records[0]": record_id,
        "rawOrLabel": "raw",
        "rawOrLabelHeaders": "raw",
        "exportSurveyFields": "true",
        "exportDataAccessGroups": "false",
    }
    try:
        with st.spinner("Fetching your responses…"):
            r = requests.post(REDCAP_API_URL, data=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None
    except requests.Timeout:
        st.error("REDCap API request timed out. The server might be behind a firewall/VPN.")
    except Exception as e:
        st.exception(e)
    return None

def fetch_by_viz_code(viz_code: str) -> dict | None:
    # Use REDCap "filterLogic" to look up by custom code (if you store it in a field).
    payload = {
        "token": REDCAP_API_TOKEN,
        "content": "record",
        "format": "json",
        "type": "flat",
        "filterLogic": f"[viz_code] = '{viz_code}'",
        "rawOrLabel": "raw",
        "rawOrLabelHeaders": "raw",
        "exportSurveyFields": "true",
        "exportDataAccessGroups": "false",
    }
    r = requests.post(REDCAP_API_URL, data=payload, timeout=15)
    if r.ok:
        data = r.json()
        return data[0] if data else None
    return None

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def scale_means(record: dict) -> pd.Series:
    means = {}
    for scale, items in SCALES.items():
        vals = [safe_float(record.get(f, np.nan)) for f in items]
        vals = [v for v in vals if np.isfinite(v)]
        means[scale] = float(np.mean(vals)) if vals else np.nan
    return pd.Series(means)

def radar_plot(series: pd.Series, title: str):
    labels = list(series.index)
    values = series.values.astype(float)

    # close the loop for radar
    labels += [labels[0]]
    values = np.append(values, values[0])

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    return fig

def bar_plot(series: pd.Series, title: str):
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Mean score")
    ax.set_xlabel("Scale")
    return fig

# ---------- App ----------
st.set_page_config(page_title="Your Sleep Onset Profile")

st.title("Your sleep-onset profile")

# Read query params (?id=123 or ?code=XYZ)
qp = st.experimental_get_query_params()
record_id = qp.get("id", [None])[0]
viz_code  = qp.get("code", [None])[0]

# Fetch data
record = None
if viz_code:
    record = fetch_by_viz_code(viz_code)
elif record_id:
    record = fetch_by_record_id(record_id)

if not record:
    st.error("We couldn’t find your responses. Please return to the survey or contact the study team.")
    st.stop()

# Show raw responses (toggle)
with st.expander("See your raw responses"):
    st.json(record)

# Compute scale means
means = scale_means(record)

# Radar
st.subheader("Your profile (radar)")
fig_radar = radar_plot(means, "Scale means")
st.pyplot(fig_radar)

# Bars
st.subheader("Breakdown by scale")
fig_bar = bar_plot(means, "Scale means")
st.pyplot(fig_bar)

# Optional: you vs crowd (if you maintain a norms table)
# if NORMS is not None and set(SCALES.keys()).issubset(NORMS.columns):
#     st.subheader("How you compare to other participants (optional)")
#     # Build a small table of percentiles per scale
#     comp = {}
#     for s in SCALES.keys():
#         comp[s] = {
#             "your_mean": means[s],
#             "p50": float(np.nanpercentile(NORMS[s], 50)),
#             "p25": float(np.nanpercentile(NORMS[s], 25)),
#             "p75": float(np.nanpercentile(NORMS[s], 75)),
#         }
#     st.dataframe(pd.DataFrame(comp).T)
#     st.caption("Reference values update as more anonymous data accumulates.")
# else:
#     st.info("Group comparison will appear here once we’ve collected enough anonymous data.")
