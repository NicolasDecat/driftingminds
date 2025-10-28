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


#%% Minimal radar (1–6): vivid, immersive, bizarre, spontaneous, fleeting, emotional, sleepy
import numpy as np
import matplotlib.pyplot as plt

# ---- fields in order + short labels ----
FIELDS = [
    ("degreequest_vividness",       "vivid"),
    ("degreequest_immersiveness",   "immersive"),
    ("degreequest_bizarreness",     "bizarre"),
    ("degreequest_spontaneity",     "spontaneous"),
    ("degreequest_fleetingness",    "fleeting"),
    ("degreequest_emotionality",    "emotional"),
    ("degreequest_sleepiness",      "sleepy"),
]

# ---- helpers ----
def as_float(x):
    try: return float(x)
    except: return np.nan

def clamp_1_6(v):
    if np.isnan(v): return np.nan
    return max(1.0, min(6.0, v))

# ---- extract values ----
vals, labels = [], []
for k, lab in FIELDS:
    v = clamp_1_6(as_float(record.get(k, np.nan)))
    vals.append(v); labels.append(lab)

if all(np.isnan(v) for v in vals):
    st.warning("No dimension scores found.")
    st.stop()

# fill missing with neutral (3.5) to keep polygon continuous
neutral = 3.5
vals_filled = [neutral if np.isnan(v) else v for v in vals]

# close the loop
values = np.array(vals_filled + [vals_filled[0]], dtype=float)
angles = np.linspace(0, 2*np.pi, len(vals_filled), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

# ---- titles (outside the figure) ----
st.markdown(
    "<div style='font-size:1.5rem;font-weight:700;'>Project Drifting Minds</div>"
    "<div style='color:#9AA3AF;margin-top:0.15rem;'>this is how my mind drifts into sleep</div>",
    unsafe_allow_html=True
)

# ---- minimalist style ----
COLOR_RING = "#A0A7B3"   # outer ring & spokes
COLOR_LINE = "#3BC6FF"   # polygon outline
COLOR_FILL = "#3BC6FF"   # polygon fill (transparent)
COLOR_LABEL= "#E6E6E6"

fig = plt.figure(figsize=(5.6, 5.6))
# no background
fig.patch.set_alpha(0)

ax = plt.subplot(111, polar=True)
ax.set_facecolor("none")                      # transparent axes
ax.grid(False)                                # no default grids

# polar orientation
ax.set_theta_offset(np.pi / 2)                # start at top
ax.set_theta_direction(-1)                    # clockwise

# limits and ticks
ax.set_rmin(1.0); ax.set_rmax(6.0)
ax.set_rticks([]); ax.set_thetagrids([])

# single outer ring at r=6
theta = np.linspace(0, 2*np.pi, 512)
ax.plot(theta, np.full_like(theta, 6.0), color=COLOR_RING, linewidth=0.8)

# thin spokes
for a in angles[:-1]:
    ax.plot([a, a], [1.0, 6.0], color=COLOR_RING, linewidth=0.6, alpha=0.35)

# polygon
ax.plot(angles, values, color=COLOR_LINE, linewidth=1.6)
ax.fill(angles, values, color=COLOR_FILL, alpha=0.12, zorder=2)

# point markers (subtle)
ax.scatter(angles[:-1], values[:-1], s=14, color=COLOR_LINE, zorder=3, alpha=0.9)

# labels just outside the ring
for ang, lab in zip(angles[:-1], labels):
    ax.text(ang, 6.22, lab, ha="center", va="center", fontsize=10, color=COLOR_LABEL)

# tighten so Streamlit doesn't crop
plt.tight_layout(pad=0.4)

st.pyplot(fig, use_container_width=False)













