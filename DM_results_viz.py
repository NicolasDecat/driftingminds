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



# Add vertical space below the radar
st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

#%% Sleep-onset timeline — thin gradient bar, spaced 3-line blocks ############
import re
import numpy as np
import matplotlib.pyplot as plt

# --- Variables lists (unchanged) ---------------------------------------------
FREQ_VARS = [
    "freq_think_ordinary","freq_scenario","freq_negative","freq_absorbed",
    "freq_percept_fleeting","freq_think_bizarre","freq_planning","freq_spectator",
    "freq_ruminate","freq_percept_intense","freq_percept_narrative","freq_percept_ordinary",
    "freq_time_perc_fast","freq_percept_vague","freq_replay","freq_percept_bizarre",
    "freq_emo_intense","freq_percept_continuous","freq_think_nocontrol","freq_percept_dull",
    "freq_emo_neutral","freq_actor","freq_think_seq_bizarre","freq_percept_precise",
    "freq_percept_imposed","freq_hear_env","freq_positive","freq_think_seq_ordinary",
    "freq_percept_real","freq_time_perc_slow","freq_syn","freq_creat"
]
TIME_VARS = [
    "timequest_scenario","timequest_positive","timequest_absorbed","timequest_percept_fleeting",
    "timequest_think_bizarre","timequest_planning","timequest_spectator","timequest_ruminate",
    "timequest_percept_intense","timequest_percept_narrative","timequest_percept_ordinary",
    "timequest_time_perc_fast","timequest_percept_vague","timequest_replay",
    "timequest_percept_bizarre","timequest_emo_intense","timequest_percept_continuous",
    "timequest_think_nocontrol","timequest_percept_dull","timequest_emo_neutral",
    "timequest_actor","timequest_think_seq_bizarre","timequest_percept_precise",
    "timequest_percept_imposed","timequest_hear_env","timequest_negative",
    "timequest_think_seq_ordinary","timequest_percept_real","timequest_time_perc_slow",
    "timequest_syn","timequest_creat"
]

# --- Helpers -----------------------------------------------------------------
def core_name(v): return re.sub(r"^(freq_|timequest_)", "", v)
def as_float(x):
    try: return float(x)
    except: return np.nan
def pretty_label(c): return c.replace("_", " ")

# Build dicts
freq_scores = {core_name(v): as_float(record.get(v, np.nan)) for v in FREQ_VARS}
time_scores = {core_name(v): as_float(record.get(v, np.nan)) for v in TIME_VARS}
common = [c for c in time_scores if c in freq_scores and not np.isnan(time_scores[c]) and not np.isnan(freq_scores[c])]

# Bucket by temporality
groups = {"Early": [], "Middle": [], "Late": []}
for c in common:
    t, f = time_scores[c], freq_scores[c]
    if 1 <= t <= 33:       groups["Early"].append((c, f))
    elif 34 <= t <= 66:    groups["Middle"].append((c, f))
    elif 67 <= t <= 100:   groups["Late"].append((c, f))

# Top-3 by frequency per bucket
top_labels = {k: [pretty_label(c) for c, _ in sorted(v, key=lambda x: x[1], reverse=True)[:3]]
              for k, v in groups.items()}

# --- Draw gradient bar -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 1.25))
fig.patch.set_alpha(0)
ax.set_facecolor("none")
ax.axis("off")

# Geometry
x0, x1 = 0.05, 0.95
y_bar = 0.50
h_bar = 0.16     # half of the previous 0.32 (twice as thin)
seg   = (x1 - x0) / 3.0

# Gradient: light gray (#F2F2F2) → dark purple (#5B21B6)
left_rgb  = np.array([0xF2, 0xF2, 0xF2]) / 255.0
right_rgb = np.array([0x5B, 0x21, 0xB6]) / 255.0
n = 800
grad = np.linspace(0, 1, n)
colors = (left_rgb[None, :] * (1 - grad)[:, None]) + (right_rgb[None, :] * grad[:, None])
# Create an image strip for a smooth gradient bar
grad_img = np.tile(colors[None, :, :], (20, 1, 1))  # 20px tall strip
ax.imshow(
    grad_img,
    extent=(x0, x1, y_bar - h_bar/2, y_bar + h_bar/2),
    origin="lower",
    aspect="auto",
    interpolation="bilinear"
)

# End labels ON the bar, larger but regular (no bold)
ax.text(x0 - 0.012, y_bar, "Awake",  ha="right", va="center", color="#000000", fontsize=12)
ax.text(x1 + 0.012, y_bar, "Asleep", ha="left",  va="center", color="#000000", fontsize=12)

# Segment centers
centers = {
    "Early":  x0 + seg * 0.5,
    "Middle": x0 + seg * 1.5,
    "Late":   x0 + seg * 2.5,
}

# --- Three-line text blocks, with a bit more spacing -------------------------
def draw_stack_block(xc, names):
    block = "\n".join(names[:3]) if names else ""
    ax.text(
        xc,
        y_bar + h_bar/2 + 0.018,   # just above bar
        block,
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#000000",
        linespacing=1.18           # slightly more space between lines
    )

draw_stack_block(centers["Early"],  top_labels.get("Early", []))
draw_stack_block(centers["Middle"], top_labels.get("Middle", []))
draw_stack_block(centers["Late"],   top_labels.get("Late", []))

plt.tight_layout(pad=0.15)
st.pyplot(fig, use_container_width=True)


















