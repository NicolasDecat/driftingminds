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

# convert numeric string to float
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# ---------- App Layout ----------


st.set_page_config(page_title="Driftind Minds")

st.title("Project Driftind Minds: this is how my mind drifts into sleep")

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
with st.expander("Raw responses"):
    st.json(record)



#%% Vizualisation #############################################################
###############################################################################


#%% Radar (1–6) with PythonCharts-style setup for DM ##########################
import numpy as np
import matplotlib.pyplot as plt

# --- fields (order) + short labels ---
FIELDS = [
    ("degreequest_vividness",       "vivid"),
    ("degreequest_immersiveness",   "immersive"),
    ("degreequest_bizarreness",     "bizarre"),
    ("degreequest_spontaneity",     "spontaneous"),
    ("degreequest_fleetingness",    "fleeting"),
    ("degreequest_emotionality",    "emotional"),
    ("degreequest_sleepiness",      "sleepy"),
]

def as_float(x):
    try: return float(x)
    except: return np.nan

def clamp_1_6(v):
    if np.isnan(v): return np.nan
    return max(1.0, min(6.0, v))

# --- extract 1–6 values ---
vals, labels = [], []
for k, lab in FIELDS:
    v = clamp_1_6(as_float(record.get(k, np.nan)))
    vals.append(v); labels.append(lab)

if all(np.isnan(v) for v in vals):
    st.warning("No dimension scores found.")
    st.stop()

# Keep polygon continuous: fill missing with neutral 3.5
neutral = 3.5
vals_filled = [neutral if np.isnan(v) else v for v in vals]

# Close the loop
num_vars = len(vals_filled)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
values = vals_filled + [vals_filled[0]]
angles_p = angles + angles[:1]  # plotting angles (closed)

# --- Title + Subtitle (Streamlit, outside the figure) ---
st.markdown(
    "<div style='font-size:1.5rem;font-weight:700;'>Project Drifting Minds</div>"
    "<div style='color:#666;margin-top:0.2rem;'>this is how my mind drifts into sleep</div>",
    unsafe_allow_html=True
)

# --- Styling ---
POLY = "#7C3AED"     # opaque purple (fill + outline)
GRID = "#999999"     # circular gridlines
SPINE = "#222222"    # outermost grid (spine)
TICK = "#222222"     # tick label color
LABEL = "#000000"    # spoke label color

# --- Figure (compact) with NO background ---
fig, ax = plt.subplots(figsize=(3.0, 3.0), subplot_kw=dict(polar=True))
fig.patch.set_alpha(0)         # transparent figure
ax.set_facecolor("none")       # transparent axes

# Fix axis orientation (12 o’clock start, clockwise)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Spoke labels placed at angles
ax.set_thetagrids(np.degrees(angles), labels)
# Angle-aware label alignment
for lbl, ang in zip(ax.get_xticklabels(), angles):
    if ang in (0, np.pi):
        lbl.set_horizontalalignment('center')
    elif 0 < ang < np.pi:
        lbl.set_horizontalalignment('left')
    else:
        lbl.set_horizontalalignment('right')
    lbl.set_color(LABEL)  # black spoke labels

# Radial range 1–6 with labeled rings
ax.set_ylim(0, 6)
ax.set_rgrids([1, 2, 3, 4, 5, 6], angle=180/num_vars, color=TICK)  # y-labels centered between first two axes
ax.tick_params(axis='y', labelsize=8, colors=TICK)                 # smaller radial tick labels
ax.grid(color=GRID)                                                # grid color
ax.spines['polar'].set_color(SPINE)                                # outer ring color

# Draw the polygon (opaque purple)
ax.plot(angles_p, values, color=POLY, linewidth=1.2)
ax.fill(angles_p, values, color=POLY, alpha=1.0)  # full opaque fill

# Ensure spokes meet the center (draw over grid for emphasis)
for a in angles:
    ax.plot([a, a], [0, 6], color=GRID, linewidth=0.6, alpha=0.35, zorder=0)

# Tight layout so labels aren't clipped
plt.tight_layout(pad=0.25)

st.pyplot(fig, use_container_width=False)
















