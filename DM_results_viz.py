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


import numpy as np
import matplotlib.pyplot as plt

# --- fields & pretty names (order matters) ---
FIELDS = [
    ("degreequest_vividness",       "vivid"),
    ("degreequest_immersiveness",   "immersive"),
    ("degreequest_bizarreness",     "bizarre"),
    ("degreequest_spontaneity",     "spontaneous"),
    ("degreequest_fleetingness",    "fleeting"),
    ("degreequest_emotionality",    "emotional"),   # special range -3..3
    ("degreequest_sleepiness",      "sleepy"),
]

# --- helpers ---
def as_float(x):
    try:
        return float(x)
    except:
        return np.nan

def clamp_1_6(v):
    if np.isnan(v): return np.nan
    return max(1.0, min(6.0, v))


# --- extract values from REDCap record ---
vals = []
labels = []
for key, lab in FIELDS:
    raw = as_float(record.get(key, np.nan))
    val = clamp_1_6(raw)
    vals.append(val)
    labels.append(lab)

# if everything missing, bail out
if all(np.isnan(v) for v in vals):
    st.warning("No dimension scores found for this participant.")
    st.stop()

# replace missing with NaN-safe mean (or 3.5 fallback) to keep polygon continuous, but fade it out later
mean_val = np.nanmean([v for v in vals if not np.isnan(v)]) if any(~np.isnan(v) for v in vals) else 3.5
vals_filled = [mean_val if np.isnan(v) else v for v in vals]

# close the loop
values = np.array(vals_filled + [vals_filled[0]], dtype=float)
angles = np.linspace(0, 2*np.pi, len(vals_filled), endpoint=False)
angles = np.roll(angles, -1)  # rotate so first label sits near top-left
angles = np.concatenate([angles, [angles[0]]])

# --- minimal style ---
BG      = "#0E1117"
GRID    = "#2B3241"
CURVE   = "#9AA3AF"   # subtle outline
FILL    = "#22D3EE"   # accent

fig = plt.figure(figsize=(5.4, 5.4))
fig.patch.set_facecolor(BG)
ax = plt.subplot(111, polar=True)
ax.set_facecolor(BG)

# polar limits
ax.set_theta_offset(np.pi / 2)   # start at top
ax.set_theta_direction(-1)       # clockwise
ax.set_rmin(1.0); ax.set_rmax(6.0)

# minimal grid (no tick labels)
ax.set_rticks([]); ax.set_thetagrids([])
for r in [2, 4, 6]:
    ax.plot(np.linspace(0, 2*np.pi, 360), np.full(360, r), color=GRID, linewidth=0.6, alpha=0.35)

# spokes (very faint)
for a in angles[:-1]:
    ax.plot([a, a], [1, 6], color=GRID, linewidth=0.6, alpha=0.25)

# polygon
ax.plot(angles, values, color=CURVE, linewidth=1.2)
ax.fill(angles, values, color=FILL, alpha=0.15)
ax.scatter(angles[:-1], values[:-1], s=10, color=FILL, zorder=3)

# labels (outside, minimal)
for ang, lab, v in zip(angles[:-1], labels, vals):
    # position label slightly outside max radius
    r_lab = 6.25
    ha = "center"
    rot = np.degrees(ang)
    # adjust alignment for better readability
    if -90 <= rot <= 90:
        ha = "left"
    elif rot < -90 or rot > 90:
        ha = "right"
    ax.text(ang, r_lab, lab, ha=ha, va="center", fontsize=10, color="#E6E6E6")

# optional: show tiny dots faded for missing values
for ang, orig in zip(angles[:-1], vals):
    if np.isnan(orig):
        ax.scatter([ang], [values[0]], s=10, color="#9AA3AF", alpha=0.2)

# margins so Streamlit doesn't crop
plt.tight_layout(pad=0.6)

st.pyplot(fig, use_container_width=False)












