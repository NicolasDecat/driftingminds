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

# ---- Retrieve values ----
def get_val(rec, key, lo, hi):
    v = rec.get(key)
    try:
        v = float(v)
    except:
        v = np.nan
    if np.isnan(v):
        return None
    return max(lo, min(hi, v))

anxiety = get_val(record, "anxiety", 1, 100)
creativity = get_val(record, "creativity_trait", 1, 6)
if creativity is not None:
    creativity = np.interp(creativity, [1, 6], [0, 100])  # rescale 1–6 to 0–100

if anxiety is None:
    st.warning("Anxiety score not found.")
    st.stop()

# ---- Style constants ----
BG = "#0E1117"       # background
CURVE = "#777C85"    # crowd curve (soft grey)
MARK = "#22D3EE"     # participant (cyan)
CURVE2 = "#777C85"   # second crowd
MARK2 = "#7C3AED"    # participant (purple)

def plot_trait(value, color_curve, color_mark, title):
    """Return a small clean figure for a single trait."""
    x = np.linspace(0, 100, 400)
    sigma = 10.0
    y = np.exp(-0.5 * ((x - 50) / sigma) ** 2)
    y /= y.max()

    fig, ax = plt.subplots(figsize=(3.8, 2.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # curve + marker
    ax.plot(x, y, color=color_curve, linewidth=1.2)
    ax.axvline(value, color=color_mark, linewidth=2.2, alpha=0.95)

    # horizontal baseline + labels
    ax.axhline(0, color=color_curve, linewidth=0.6, alpha=0.3)
    ax.text(0, -0.08, "0", color="#9AA3AF", ha="center", va="top", fontsize=9)
    ax.text(100, -0.08, "100", color="#9AA3AF", ha="center", va="top", fontsize=9)

    # title above
    ax.text(50, 1.05, title, color="#E6E6E6", ha="center", va="bottom",
            fontsize=12, fontweight="medium")

    # clean layout
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.15, 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig

# ---- Layout in Streamlit ----
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    if anxiety is not None:
        figA = plot_trait(anxiety, CURVE, MARK, "Anxiety")
        st.pyplot(figA, use_container_width=False)

with col2:
    if creativity is not None:
        figC = plot_trait(creativity, CURVE2, MARK2, "Creativity")
        st.pyplot(figC, use_container_width=False)





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
