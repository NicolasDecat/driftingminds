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


# ---------- Radar ----------

#%% Final Radar — Project Drifting Minds ######################################
import numpy as np
import matplotlib.pyplot as plt

# --- Titles (centered & large) ---
st.markdown(
    """
    <div style="text-align:center; margin-bottom:0.6rem;">
        <div style="font-size:2rem; font-weight:800;">Project Drifting Minds</div>
        <div style="color:#666; font-size:1rem; margin-top:0.3rem;">
            this is how my mind drifts into sleep
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Fields & labels ---
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

vals, labels = [], []
for k, lab in FIELDS:
    v = clamp_1_6(as_float(record.get(k, np.nan)))
    vals.append(v); labels.append(lab)

if all(np.isnan(v) for v in vals):
    st.warning("No dimension scores found.")
    st.stop()

neutral = 3.5
vals_filled = [neutral if np.isnan(v) else v for v in vals]

num_vars = len(vals_filled)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
values = vals_filled + [vals_filled[0]]
angles_p = angles + angles[:1]

# --- Colors ---
POLY  = "#7C3AED"   # purple
GRID  = "#B0B0B0"
SPINE = "#222222"
TICK  = "#555555"
LABEL = "#000000"

# --- Figure ---
fig, ax = plt.subplots(figsize=(3.0, 3.0), subplot_kw=dict(polar=True))
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# Orientation
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles), labels)

# Labels: smaller + closer
for lbl, ang in zip(ax.get_xticklabels(), angles):
    if ang in (0, np.pi):
        lbl.set_horizontalalignment('center')
    elif 0 < ang < np.pi:
        lbl.set_horizontalalignment('left')
    else:
        lbl.set_horizontalalignment('right')
    lbl.set_color(LABEL)
    lbl.set_fontsize(7.5)
# Bring them closer
ax.tick_params(axis='x', pad=2)

# Radial range 1–6
ax.set_ylim(0, 6)
ax.set_rgrids([1,2,3,4,5,6], angle=180/num_vars, color=TICK)
ax.tick_params(axis='y', labelsize=6.5, colors=TICK, pad=-1)  # smaller & closer radial labels

# Thin grid lines
ax.grid(color=GRID, linewidth=0.4)
ax.spines['polar'].set_color(SPINE)
ax.spines['polar'].set_linewidth(0.6)

# Polygon (thin + semi-transparent purple)
ax.plot(angles_p, values, color=POLY, linewidth=0.8, zorder=3)
ax.fill(angles_p, values, color=POLY, alpha=0.22, zorder=2)

# Spokes (center to edge)
for a in angles:
    ax.plot([a, a], [0, 6], color=GRID, linewidth=0.4, alpha=0.35, zorder=1)

plt.tight_layout(pad=0.25)
st.pyplot(fig, use_container_width=False)


















