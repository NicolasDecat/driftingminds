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


#%% Config API ################################################################
###############################################################################


# To add in Streamlit > Manage app > Settings > Secrets
# REDCAP_API_URL = "https://redcap-icm.icm-institute.org/api/"
# REDCAP_API_TOKEN = "641919114F091FC5A1860BCFC53D3947"

# Optional: load anonymized norms to position the participant vs crowd
# def load_norms():
#     try:
#         df = pd.read_csv("norms.csv")   # columns should match your fields or pre-agg scales
#         return df
#     except Exception:
#         return None
# NORMS = load_norms()


REDCAP_API_URL = st.secrets.get("REDCAP_API_URL")
REDCAP_API_TOKEN = st.secrets.get("REDCAP_API_TOKEN")



#%% Retrieve Participant data #################################################
###############################################################################


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



#%% Prepare vizualisation #####################################################
###############################################################################


# ---------- Prepare computation ----------


# Define which fields belong to which scales for the viz
SCALES = {
    "Thoughts": ["freq_think_nocontrol", "freq_think_seq_bizarre", "freq_think_seq_ordinary"],
    "Perception":    ["freq_percept_precise",  "freq_percept_real",  "freq_percept_imposed"]
}

# convert any numeric string to a float
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan
    
# Convert means
def scale_means(record: dict) -> pd.Series:
    means = {}
    for scale, items in SCALES.items():
        vals = [safe_float(record.get(f, np.nan)) for f in items]
        vals = [v for v in vals if np.isfinite(v)]
        means[scale] = float(np.mean(vals)) if vals else np.nan
    return pd.Series(means)


# ---------- App Layout ----------


st.set_page_config(page_title="Your Sleep Onset Profile")

st.title("Your sleep-onset profile")

# Read query params (?id=123)
qp = st.query_params
record_id = qp.get("id")  

# Load participant data
record = None
if record_id:
    record = fetch_by_record_id(record_id)
if not record:
    st.error("We couldn’t find your responses.")
    st.stop()

# Show raw responses as toggle list
with st.expander("See your raw responses"):
    st.json(record)



#%% Vizualisation #############################################################
###############################################################################


import numpy as np
import matplotlib.pyplot as plt

# ---- Values -----------------------------------------------------------------
def get_val(rec, key):
    v = rec.get(key)
    try:
        v = float(v)
    except:
        v = np.nan
    return None if np.isnan(v) else v

anxiety   = get_val(record, "anxiety")            # expects 1–100
creativity = get_val(record, "creativity_trait")  # expects 1–6

if anxiety is None:
    st.warning("Anxiety score not found.")
    st.stop()

# Clamp
anxiety = min(100.0, max(0.0, anxiety))
if creativity is not None:
    creativity = min(6.0, max(1.0, creativity))

# ---- Style ------------------------------------------------------------------
BG      = "#0E1117"
CURVE   = "#7F8894"   # soft grey
MARK_A  = "#22D3EE"   # cyan
MARK_C  = "#7C3AED"   # purple

def plot_minimal(value, xmin, xmax, mu, sigma, color_curve, color_mark, left_label, right_label):
    """Returns a minimal figure with thin Gaussian, baseline + end labels, and a vertical marker."""
    x = np.linspace(xmin, xmax, 400)
    y = np.exp(-0.5*((x - mu)/sigma)**2)
    y /= y.max()

    fig, ax = plt.subplots(figsize=(3.9, 2.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Thin curve & marker
    ax.plot(x, y, color=color_curve, linewidth=1.1)
    ax.axvline(value, color=color_mark, linewidth=2.0, alpha=0.95)

    # Baseline + end labels (subtle)
    ax.axhline(0, color=color_curve, linewidth=0.6, alpha=0.3)
    ax.text(xmin, -0.12, f"{left_label}", color="#9AA3AF", ha="center", va="top", fontsize=9)
    ax.text(xmax, -0.12, f"{right_label}", color="#9AA3AF", ha="center", va="top", fontsize=9)

    # Clean layout
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.18, 1.05)                # extra bottom room for labels
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Tighten paddings so Streamlit doesn't crop labels
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.35)
    return fig

# ---- Streamlit layout: titles OUTSIDE figures (no overlap) ------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div style='color:#E6E6E6;font-weight:600;'>Anxiety</div>", unsafe_allow_html=True)
    figA = plot_minimal(
        value=anxiety, xmin=0, xmax=100, mu=50, sigma=10,          # narrower distribution
        color_curve=CURVE, color_mark=MARK_A,
        left_label="0", right_label="100"
    )
    st.pyplot(figA, use_container_width=False)

with col2:
    st.markdown("<div style='color:#E6E6E6;font-weight:600;'>Creativity</div>", unsafe_allow_html=True)
    if creativity is not None:
        figC = plot_minimal(
            value=creativity, xmin=1, xmax=6, mu=3.5, sigma=1.0,   # centered on 3.5 for 1–6 scale
            color_curve=CURVE, color_mark=MARK_C,
            left_label="1", right_label="6"
        )
        st.pyplot(figC, use_container_width=False)
    else:
        st.caption("No creativity score available.")




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
