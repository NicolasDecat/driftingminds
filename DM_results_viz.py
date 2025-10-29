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
import re
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

st.set_page_config(page_title="Drifting Minds — Profile", layout="centered")

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

st.image("assets/symbols.png", use_container_width=True) # add

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
# with st.expander("Raw responses"):
#     st.json(record)
    
# --- Titles ---

st.markdown(
    """
    <div style="text-align:center; margin-bottom:2.5rem;">  <!-- more space -->
        <div style="font-size:2rem; font-weight:800;">Drifting Minds Study</div>
        <div style="font-size:1rem; margin-top:0.1rem;">
            This is how my mind drifts into sleep
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



#%% Profile #############################################################
###############################################################################



# ---------- 1) Normalization helpers ------------------------------------------------------------
def _to_float(x):
    try:
        return float(x)
    except:
        return np.nan

def norm_1_6(x):
    x = _to_float(x)
    if np.isnan(x): return np.nan
    return np.clip((x - 1.0) / 5.0, 0.0, 1.0)

def norm_0_100(x):
    x = _to_float(x)
    if np.isnan(x): return np.nan
    return np.clip(x / 100.0, 0.0, 1.0)

def norm_1_100(x):
    x = _to_float(x)
    if np.isnan(x): return np.nan
    return np.clip((x - 1.0) / 99.0, 0.0, 1.0)

def _to_minutes_relaxed(x):
    # numeric already?
    if isinstance(x, (int, float)):
        if 0.0 <= x <= 1.0:
            return None  # already normalized
        return float(x)

    if x is None:
        return np.nan

    s = str(x).strip().lower()
    if s == "" or s in {"na", "n/a", "none"}:
        return np.nan

    # HH:MM pattern
    m = re.match(r"^\s*(\d{1,2})\s*:\s*(\d{1,2})\s*$", s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        return float(hh * 60 + mm)

    # "1h05" or "1h 05m" or "1 h 5" etc.
    m = re.findall(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)?", s)
    if m:
        total = 0.0
        any_unit = False
        for val, unit in m:
            if val == "": 
                continue
            v = float(val)
            if unit in ("h","hr","hrs","hour","hours"):
                total += v * 60.0
                any_unit = True
            elif unit in ("m","min","mins","minute","minutes"):
                total += v
                any_unit = True
        if any_unit:
            return total

    # plain float-ish: "15" / "15.5"
    try:
        v = float(s)
        if 0.0 <= v <= 1.0:
            return None  # already normalized
        return v
    except:
        return np.nan

def norm_latency_auto(x, cap_minutes=60.0):
    # First try relaxed minutes
    mins = _to_minutes_relaxed(x)
    if mins is None:
        # already normalized 0..1
        v = _to_float(x)
        return np.clip(v, 0.0, 1.0)
    if np.isnan(mins):
        return np.nan
    return np.clip(mins / cap_minutes, 0.0, 1.0)


# ---------- 2) Aliases + robust fetch -------------------------------
def _get_first(record, keys):
    """Return the first present, non-empty value for any of the candidate keys."""
    if isinstance(keys, (list, tuple)):
        for k in keys:
            if k in record and record[k] not in (None, "", "NA"):
                return record[k]
        return np.nan
    # single key
    return record.get(keys, np.nan)

# ---------- 2) Define composite dimensions  -------------------------------

DIMENSIONS = {
    "vividness": [
        ("freq_percept_real",      norm_1_6,   1.0, {}),
        ("freq_percept_intense",   norm_1_6,   1.0, {}),
    ],
    "spontaneity": [
        ("freq_think_nocontrol",   norm_1_6,   1.0, {}),
    ],
    "bizarreness": [
        ("freq_percept_bizarre",   norm_1_6,   1.0, {}),
    ],
    "immersion": [
        ("freq_absorbed",          norm_1_6,   1.0, {}),
    ],
    "emotion_pos": [
        ("freq_positive",          norm_1_6,   1.0, {}),
    ],
    "sleep_latency": [
    (["sleep_latency_min","sleep_latency","sleep_latency_minutes",
      "latency_minutes","sleep_onset_latency"],  # aliases
     norm_latency_auto, 1.0, {"cap_minutes": 60.0}),
    ],
    "baseline_anxiety": [
        (["anxiety"], norm_1_100, 1.0, {}),
    ],
}

def composite_scores_from_record(record, dimensions=DIMENSIONS):
    out = {}
    for dim, items in dimensions.items():
        vals, wts = [], []
        for item in items:
            if len(item) == 3:
                field_keys, norm_fn, wt = item
                kwargs = {}
            else:
                field_keys, norm_fn, wt, kwargs = item
            raw = _get_first(record, field_keys)
            try:
                v = norm_fn(raw, **kwargs) if kwargs else norm_fn(raw)
            except TypeError:
                v = norm_fn(raw)
            if not np.isnan(v):
                vals.append(v * wt)
                wts.append(wt)
        out[dim] = (np.sum(vals) / np.sum(wts)) if wts else np.nan
    return out

DIM_KEYS = list(DIMENSIONS.keys())

def vector_from_scores(scores, dim_keys=DIM_KEYS):
    return np.array([scores.get(k, np.nan) for k in dim_keys], dtype=float)

# ---------- 3) Prototype profiles (aligned to DIM_KEYS order) ---------------------
# Order = ["vividness","spontaneity","bizarreness","immersion","emotion_pos","sleep_latency","baseline_anxiety"]
profiles = {
    "Early Dreamer":     [0.90, 0.30, 0.90, 0.80, 0.50, 0.50, 0.50],
    "Letting Go":        [0.70, 0.30, 0.50, 0.60, 0.50, 0.60, 0.50],
    "Pragmatic Thinker": [0.20, 0.10, 0.10, 0.30, 0.50, 0.50, 0.50],
    "Ruminator":         [0.20, 0.10, 0.10, 0.10, 0.20, 0.90, 0.90],
    "Quiet Mind":        [0.20, 0.20, 0.20, 0.20, 0.50, 0.50, 0.50],
}

# ---------- 4) Assignment by nearest prototype ---------------------------------------------------
def _nanaware_distance(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if not np.any(mask):
        return np.inf
    diff = a[mask] - b[mask]
    return np.sqrt(np.sum(diff * diff))

def assign_profile_from_record(record, profiles=profiles):
    scores = composite_scores_from_record(record)
    vec = vector_from_scores(scores)
    best_name, best_dist = None, np.inf
    for name, proto in profiles.items():
        d = _nanaware_distance(vec, proto)
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, scores

# ---------- 5) Streamlit display (name + description only, no duplicate title) --

# Ensure a record is available
if 'record' not in globals():
    st.error("No 'record' dict found. Provide your participant data before profile assignment.")
    st.stop()

# Compute participant’s profile and scores
prof, scores = assign_profile_from_record(record)

# --- Profile descriptions (as before) ---
descriptions = {
    "Early Dreamer": "You tend to drift into sleep through vivid, sensory experiences; colors, sounds, or mini-dreams.",
    "Letting Go": "You start by thinking intentionally, but gradually surrender to spontaneous imagery.",
    "Pragmatic Thinker": "You stay in control. Analytical or practical thoughts until you switch off.",
    "Ruminator": "You tend to replay or analyze things in bed, with longer sleep latency and emotional tension.",
    "Quiet Mind": "You fall asleep effortlessly, with little mental content.",
}

# --- Robust lookup (ignores case + stray spaces) ---
_descriptions_ci = {k.lower(): v for k, v in descriptions.items()}
prof_key = str(prof).strip().lower()
prof_desc = _descriptions_ci.get(prof_key, "")

# --- Render clean horizontal layout: left-aligned title block + right description ---
st.markdown(
    f"""
    <div style="
        display: flex;
        align-items: flex-start;
        justify-content: flex-start;
        gap: 40px;
        margin-top: 30px;
        margin-bottom: 40px;
        flex-wrap: wrap;
    ">
        <!-- Left block: "Your profile is..." + profile name -->
        <div style="text-align: left;">
            <p style="font-size:1rem; margin:0; color:#000000;">Your profile is...</p>
            <h2 style="font-size:2rem; margin:6px 0 0 0;"><strong>{prof}</strong></h2>
        </div>

        <!-- Right block: description -->
        <div style="max-width:420px; text-align:left;">
            <p style="font-size:1.05rem; margin:0; line-height:1.5;">{prof_desc}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)






# --- Helper for nice numeric formatting ---
def _fmt(v, nd=3):
    if v is None:
        return "NA"
    try:
        if np.isnan(v):
            return "NA"
    except TypeError:
        pass
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

# --- Toggle: computation outcomes (no highlights, no radar) ---
with st.expander("See how this was computed"):
    # 1) Dimension scores (0–1)
    dim_rows = []
    for k in DIM_KEYS:
        v = scores.get(k, np.nan)
        dim_rows.append({
            "Dimension": k,
            "Score (0–1)": None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v),
        })
    dim_df = pd.DataFrame(dim_rows)

    st.markdown("**Normalized dimension scores**")
    st.dataframe(
        dim_df,
        hide_index=True,
        use_container_width=True,
    )

    # 2) Prototype fit (distance; lower = closer)
    vec = vector_from_scores(scores)
    dists = [{"Profile": name, "Distance": _nanaware_distance(vec, proto)}
             for name, proto in profiles.items()]
    dist_df = pd.DataFrame(dists).sort_values("Distance")

    st.markdown("**Prototype fit (lower = closer)**")
    st.dataframe(
        dist_df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption("Notes: sleep_latency normalized with cap=60 min; baseline_anxiety normalized from 1–100 → 0–1.")





#%% Comparitive visualisation ################################################
###############################################################################

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load population (N=1000) data
csv_path = os.path.join("assets", "N1000_comparative_viz_ready.csv")
pop_data = pd.read_csv(csv_path)


# --- Shared helpers -----------------------------------------------------------
CAP_MIN = 60.0  # same cap as normalization

def _to_hours_for_plot(x):
    if x is None: return np.nan
    s = str(x).strip()
    if s == "": return np.nan
    if s.endswith("+"):
        try: return float(s[:-1])
        except: return 12.0
    try: return float(s)
    except: return np.nan

# Put BOTH tiles inside columns so they align horizontally
col_left, col_right = st.columns([1, 1], gap="small")

# ========================= LEFT: LATENCY (KDE) ===============================
with col_left:
    # --- Population samples (minutes)
    lat_col = [c for c in pop_data.columns if "sleep_latency" in c.lower()][0]
    raw = pd.to_numeric(pop_data[lat_col], errors="coerce").dropna()
    samples = np.clip(raw.values * CAP_MIN if raw.max() <= 1.5 else raw.values, 0, CAP_MIN)

    # --- Participant: raw (for title) and display (capped) values
    sl_norm = scores.get("sleep_latency", np.nan)
    if np.isnan(sl_norm):
        st.info("No sleep-latency value available for this participant.")
    else:
        sl_raw = _get_first(record, [
            "sleep_latency_min", "sleep_latency", "sleep_latency_minutes",
            "latency_minutes", "sleep_onset_latency"
        ])
        try:
            part_raw_minutes = float(sl_raw)
        except Exception:
            part_raw_minutes = np.nan

        part_display = sl_norm * CAP_MIN if sl_norm <= 1.5 else sl_norm
        part_display = float(np.clip(part_display, 0, CAP_MIN))
        rounded_raw = int(round(part_raw_minutes)) if not np.isnan(part_raw_minutes) else int(round(part_display))

        # KDE
        kde = gaussian_kde(samples, bw_method="scott")
        xs = np.linspace(0, CAP_MIN, 400)
        ys = kde(xs)

        # Plot (compact)
        fig, ax = plt.subplots(figsize=(2.2, 2.4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.fill_between(xs, ys, color="#D9D9D9", alpha=0.8, linewidth=0)
        ax.plot(xs, ys, color="#BBBBBB", linewidth=1)

        # Thin purple marker
        ax.axvline(part_display, color="#7C3AED", lw=0.5)
        ax.scatter([part_display], [kde(part_display)], color="#7C3AED", s=20, zorder=3)

        ax.set_title(f"{rounded_raw} minutes to fall asleep", fontsize=10, pad=6)
        ax.set_xlabel("Time (min)", fontsize=9)
        ax.set_ylabel("Population", fontsize=9)

        # Minimal y-axis
        ax.set_yticks([]); ax.set_yticklabels([])

        # Force ticks 0..60 with final "60+"
        xticks = np.linspace(0, CAP_MIN, 7)  # 0,10,...,60
        ax.set_xticks(xticks)
        xlabels = [str(int(t)) if t < CAP_MIN else "60+" for t in xticks]
        ax.set_xticklabels(xlabels)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", length=0)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

# ====================== RIGHT: DURATION (1..12+, histogram) ==================
with col_right:
    # 1) Find a sleep-duration column
    dur_cols = [c for c in pop_data.columns if c.lower() in (
        "sleep_duration", "sleep_duration_h", "sleep_duration_hours", "total_sleep_time_h"
    )]
    if not dur_cols:
        st.warning("No sleep duration column found in population data.")
    else:
        col = dur_cols[0]
        # Population → hours clipped to [1,12]
        raw_series = pop_data[col].apply(_to_hours_for_plot)
        samples_h = raw_series.astype(float).to_numpy()
        samples_h = samples_h[np.isfinite(samples_h)]
        samples_h = np.clip(samples_h, 1.0, 12.0)

        if samples_h.size == 0:
            st.info("No valid sleep duration values in population data.")
        else:
            # Participant raw -> title and plotting hours
            dur_raw = _get_first(record, [
                "sleep_duration", "sleep_duration_h", "sleep_duration_hours", "total_sleep_time_h"
            ])
            dur_raw_str = str(dur_raw).strip() if dur_raw is not None else ""

            try:
                if dur_raw_str.endswith("+"):
                    part_hours_plot = float(dur_raw_str[:-1])
                    title_str = f"{dur_raw_str} hours of sleep"
                else:
                    part_hours_plot = float(dur_raw_str)
                    title_str = f"{int(round(part_hours_plot))} hours of sleep"
            except:
                part_hours_plot = float(np.nanmedian(samples_h))
                title_str = "Sleep duration"

            part_hours_plot = float(np.clip(part_hours_plot, 1.0, 12.0))

            # 2) Bins: one per hour (1..12), last labeled "12+"
            edges = np.arange(0.5, 12.5 + 1.0, 1.0)  # 0.5..12.5 step 1
            counts, _ = np.histogram(samples_h, bins=edges, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])  # 1..12

            # Highlight participant bin
            highlight_idx = np.digitize(part_hours_plot, edges) - 1
            highlight_idx = np.clip(highlight_idx, 0, len(counts) - 1)

            # Plot (compact)
            fig, ax = plt.subplots(figsize=(2.2, 2.4))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            # Population bars
            ax.bar(
                centers, counts,
                width=edges[1] - edges[0],
                color="#D9D9D9",
                edgecolor="white",
                align="center"
            )
            # Highlight bar
            ax.bar(
                centers[highlight_idx],
                counts[highlight_idx],
                width=edges[1] - edges[0],
                color="#7C3AED",
                edgecolor="white",
                align="center",
                label="Your duration"
            )

            ax.set_title(title_str, fontsize=10, pad=6)
            ax.set_xlabel("Time (h)", fontsize=9)
            ax.set_ylabel("Population", fontsize=9)

            # Remove y ticks
            ax.set_yticks([]); ax.set_yticklabels([])

            # X ticks: 1..12 but only show labels 4..10, and set last to "12+"
            ticks = np.arange(1, 13, 1)
            ax.set_xticks(ticks)
            
            labels = ["" for _ in ticks]  # start with all hidden
            for i in range(4, 11):        # 4..10 inclusive
                labels[i-1] = str(i)
            
            ax.set_xticklabels(labels)
            
            # Clean minimal style
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", length=0)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            




#%% Easy-to-pick vizualisation ################################################
###############################################################################






# --- Fields & labels ---
FIELDS = [
    ("degreequest_vividness",       "vivid"),
    ("degreequest_immersiveness",   "immersive"),
    ("degreequest_bizarreness",     "bizarre"),
    ("degreequest_spontaneity",     "spontaneous"),
    ("degreequest_fleetingness",    "fleeting"),
    ("degreequest_emotionality",    "positive\nemotions"),
    ("degreequest_sleepiness",      "sleepy"),
]

def as_float(x):
    try:
        return float(x)
    except:
        return np.nan

def clamp_1_6(v):
    if np.isnan(v):
        return np.nan
    return max(1.0, min(6.0, v))

vals, labels = [], []
for k, lab in FIELDS:
    v = clamp_1_6(as_float(record.get(k, np.nan)))
    vals.append(v)
    labels.append(lab)

if all(np.isnan(v) for v in vals):
    st.warning("No dimension scores found.")
    st.stop()

neutral = 3.5
vals_filled = [neutral if np.isnan(v) else v for v in vals]

num_vars = len(vals_filled)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
values = vals_filled + [vals_filled[0]]
angles_p = angles + angles[:1]

# --- Colors ---
POLY  = "#7C3AED"
GRID  = "#B0B0B0"
SPINE = "#222222"
TICK  = "#555555"
LABEL = "#000000"

# Scale
s = 1.4

# Left alignment layout
col_left, col_right = st.columns([1.3, 1.7])

# --- Figure ---
fig, ax = plt.subplots(figsize=(3.0 * s, 3.0 * s), subplot_kw=dict(polar=True))
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# Orientation
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles), labels)

# Label placement (scaled fonts)
for lbl, ang in zip(ax.get_xticklabels(), angles):
    if ang in (0, np.pi):
        lbl.set_horizontalalignment("center")
    elif 0 < ang < np.pi:
        lbl.set_horizontalalignment("left")
    else:
        lbl.set_horizontalalignment("right")
    lbl.set_color(LABEL)
    lbl.set_fontsize(8.5 * s)
ax.tick_params(axis="x", pad=int(2.5 * s))

# Radial range 1–6
ax.set_ylim(0, 6)
ax.set_rgrids([1, 2, 3, 4, 5, 6], angle=180 / num_vars, color=TICK)
ax.tick_params(axis="y", labelsize=7.0 * s, colors=TICK, pad=-1)

# Grid lines & spine
ax.grid(color=GRID, linewidth=0.45 * s)
ax.spines["polar"].set_color(SPINE)
ax.spines["polar"].set_linewidth(0.7 * s)

# Polygon
ax.plot(angles_p, values, color=POLY, linewidth=1.0 * s, zorder=3)
ax.fill(angles_p, values, color=POLY, alpha=0.22, zorder=2)

# Spokes (center to edge)
for a in angles:
    ax.plot([a, a], [0, 6], color=GRID, linewidth=0.4 * s, alpha=0.35, zorder=1)

plt.tight_layout(pad=0.3 * s)

with col_left:
    st.pyplot(fig, use_container_width=False)



# ---------- Trajectory plot ----------


# Add vertical space below the horizontal bar
st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

import streamlit as st
from PIL import Image
import os

# Retrieve participant's trajectory value
traj_value = record.get("trajectories")  # expecting 1, 2, 3, or 4

# Convert safely to int
try:
    traj_value = int(traj_value)
except (TypeError, ValueError):
    traj_value = None

# Match image file path
if traj_value in [1, 2, 3, 4]:
    img_path = f"assets/trajectories-0{traj_value}.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        # Display image at half its container width
        st.image(
            img,
            width=400,   # 👈 adjust this number if you want smaller/larger (e.g., 350 or 450)
        )
    else:
        st.warning(f"Image not found: {img_path}")
else:
    st.info("No trajectory information available for this participant.")


# ---------- Timeline plot ----------


# Add vertical space below the radar
st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

import re
import numpy as np
import matplotlib.pyplot as plt

# --- Frequency variable label mapping (with "perceptions") -------------------
CUSTOM_LABELS = {
    "freq_think_ordinary": "thinking logical thoughts",
    "freq_scenario": "imagining scenarios",
    "freq_negative": "feeling negative",
    "freq_absorbed": "feeling absorbed",
    "freq_percept_fleeting": "fleeting perceptions",
    "freq_think_bizarre": "thinking strange things",
    "freq_planning": "planning the day",
    "freq_spectator": "feeling like a spectator",
    "freq_ruminate": "ruminating",
    "freq_percept_intense": "intense perceptions",
    "freq_percept_narrative": "narrative scenes",
    "freq_percept_ordinary": "ordinary perceptions",
    "freq_time_perc_fast": "time feels fast",
    "freq_percept_vague": "vague perceptions",
    "freq_replay": "replaying the day",
    "freq_percept_bizarre": "strange perceptions",
    "freq_emo_intense": "feeling intense emotions",
    "freq_percept_continuous": "continuous perceptions",
    "freq_think_nocontrol": "losing control of thoughts",
    "freq_percept_dull": "dull perceptions",
    "freq_actor": "acting in the scene",
    "freq_think_seq_bizarre": "thinking illogical thoughts",
    "freq_percept_precise": "precise perceptions",
    "freq_percept_imposed": "imposed perceptions",
    "freq_hear_env": "hearing my environment",
    "freq_positive": "feeling positive",
    "freq_think_seq_ordinary": "thinking logical thoughts",
    "freq_percept_real": "perceptions feel real",
    "freq_time_perc_slow": "time feels slow",
    "freq_syn": "experiencing synaesthesia",
    "freq_creat": "feeling creative",
}

# --- Variable lists ----------------------------------------------------------
FREQ_VARS = list(CUSTOM_LABELS.keys())
TIME_VARS = [
    "timequest_scenario","timequest_positive","timequest_absorbed","timequest_percept_fleeting",
    "timequest_think_bizarre","timequest_planning","timequest_spectator","timequest_ruminate",
    "timequest_percept_intense","timequest_percept_narrative","timequest_percept_ordinary",
    "timequest_time_perc_fast","timequest_percept_vague","timequest_replay",
    "timequest_percept_bizarre","timequest_emo_intense","timequest_percept_continuous",
    "timequest_think_nocontrol","timequest_percept_dull",
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

# Build dicts
freq_scores = {core_name(v): as_float(record.get(v, np.nan)) for v in FREQ_VARS}
time_scores = {core_name(v): as_float(record.get(v, np.nan)) for v in TIME_VARS}
common = [c for c in time_scores if c in freq_scores and not np.isnan(time_scores[c]) and not np.isnan(freq_scores[c])]


# ---------- Vertical timeline (10-pt bins, max 2 labels, cleaner ends) -------

# Space below the previous plot
st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

import numpy as np 
import matplotlib.pyplot as plt

# Build core -> (time, freq, label)
cores = []
for c in common:
    t = float(time_scores[c])
    f = float(freq_scores[c])
    lab = CUSTOM_LABELS.get(f"freq_{c}", c.replace("_", " "))
    cores.append((c, t, f, lab))

# 10-point bins
bins = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,90),(91,100)]
bin_centers = [(lo+hi)/2 for (lo,hi) in bins]

# Assign cores to bins
bin_items = {i: [] for i in range(len(bins))}
for c, t, f, lab in cores:
    for i, (lo, hi) in enumerate(bins):
        if lo <= t <= hi:
            bin_items[i].append((c, t, f, lab))
            break

# Per-bin winners: all tied at max freq; skip if max < 3; cap to 2 labels
winners = {i: [] for i in range(len(bins))}
for i, items in bin_items.items():
    if not items:
        continue
    items_sorted = sorted(items, key=lambda x: (-x[2], x[3]))  # freq desc, then label
    top_f = items_sorted[0][2]
    if top_f < 3:
        continue
    tied = [it for it in items_sorted if abs(it[2] - top_f) < 1e-9]
    winners[i] = [it[3] for it in tied[:2]]  # <-- cap to 2 labels max

# ----------------------- Draw -----------------------
fig, ax = plt.subplots(figsize=(3.6, 6.0))
fig.patch.set_alpha(0)
ax.set_facecolor("none")
ax.axis("off")

x_bar = 0.5
bar_half_w = 0.007   # ultra-thin bar stays
y_top, y_bot = 0.92, 0.08

def ty(val):
    # Awake (1) at top, Asleep (100) at bottom
    return y_top - (val - 1) / 99.0 * (y_top - y_bot)

# Gradient (Awake white top → Asleep purple bottom), narrow strip
top_rgb  = np.array([1.0, 1.0, 1.0])
bot_rgb  = np.array([0x5B/255, 0x21/255, 0xB6/255])
n = 900
rows = np.linspace(bot_rgb, top_rgb, n)
grad_img = np.tile(rows[:, None, :], (1, 12, 1))

ax.imshow(
    grad_img,
    extent=(x_bar - bar_half_w, x_bar + bar_half_w, ty(100), ty(1)),
    origin="lower",
    aspect="auto",
    interpolation="bilinear"
)

# End labels (a bit farther from the bar now)
ax.text(x_bar, ty(1)  + 0.035, "Awake",  ha="center", va="bottom", fontsize=11, color="#000000")
ax.text(x_bar, ty(100) - 0.035, "Asleep", ha="center", va="top",    fontsize=11, color="#000000")

# Annotation geometry (keep labels close to bar; very short leader lines)
x_right = x_bar + 0.042
x_left  = x_bar - 0.042
line_w  = 0.15
label_fs = 9.2

for i, center in enumerate(bin_centers):
    labs = winners[i]
    if not labs:
        continue
    y_c = ty(center)

    # vertical offsets with a bit more spacing when there are two labels
    if len(labs) == 1:
        y_positions = [y_c]
    else:
        y_positions = [y_c + 0.02, y_c - 0.02]  # more separation for two labels

    # Alternate sides per bin
    side_right = (i % 2 == 0)

    if side_right:
        ax.plot([x_bar + bar_half_w, x_right - 0.003], [y_positions[0], y_positions[0]],
                color="#000000", linewidth=line_w)
        for yy, text_label in zip(y_positions, labs):
            ax.text(x_right, yy, text_label, ha="left", va="center",
                    fontsize=label_fs, color="#000000", linespacing=1.18)
    else:
        ax.plot([x_bar - bar_half_w, x_left + 0.003], [y_positions[0], y_positions[0]],
                color="#000000", linewidth=line_w)
        for yy, text_label in zip(y_positions, labs):
            ax.text(x_left, yy, text_label, ha="right", va="center",
                    fontsize=label_fs, color="#000000", linespacing=1.18)

plt.tight_layout(pad=0.25)
st.pyplot(fig, use_container_width=True)



# ---------- VVIQ Distribution Plot ---------------------------------------------


# Add vertical space below the radar
st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# --- Compute participant's VVIQ score ---
VVIQ_FIELDS = [
    "quest_a1","quest_a2","quest_a3","quest_a4",
    "quest_b1","quest_b2","quest_b3","quest_b4",
    "quest_c1","quest_c2","quest_c3","quest_c4",
    "quest_d1","quest_d2","quest_d3","quest_d4"
]

def as_float(x):
    try:
        return float(x)
    except:
        return np.nan

vviq_score = sum(as_float(record.get(k, np.nan)) for k in VVIQ_FIELDS if not np.isnan(as_float(record.get(k, np.nan))))

# --- Simulate population distribution (truncated normal) ---
N = 10000
mu, sigma = 61.0, 9.2
low, high = 16, 80
a, b = (low - mu) / sigma, (high - mu) / sigma
samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=N, random_state=42)

# --- Distribution bins ---
bins = np.linspace(low, high, 33)
counts, edges = np.histogram(samples, bins=bins, density=True)
centers = 0.5 * (edges[:-1] + edges[1:])

# --- Find bin that contains participant's score ---
highlight_idx = np.digitize(vviq_score, edges) - 1
highlight_idx = np.clip(highlight_idx, 0, len(counts)-1)

# --- Plot styling ---
fig, ax = plt.subplots(figsize=(6.5, 3.5))
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# Light grey bars for population
ax.bar(centers, counts, width=edges[1]-edges[0], color="#D9D9D9", edgecolor="white")

# Highlight participant’s bin in purple
ax.bar(
    centers[highlight_idx],
    counts[highlight_idx],
    width=edges[1]-edges[0],
    color="#7C3AED",
    edgecolor="white",
    label=f"Your score: {int(vviq_score)}"
)

# Add cutoffs
aphantasia_cut = 32
hyper_cut = 75
ax.axvline(aphantasia_cut, color="#D9D9D9", linestyle="--", linewidth=1)
ax.axvline(hyper_cut, color="#D9D9D9", linestyle="--", linewidth=1)

# --- Add cutoff labels ---
y_text = ax.get_ylim()[1] * 0.92  # vertical placement near top of plot
ax.text(aphantasia_cut - 1.5, y_text, "Aphantasia", color="#888888",
        ha="right", va="center", fontsize=8)
ax.text(hyper_cut + 1.5, y_text, "Hyperphantasia", color="#888888",
        ha="left", va="center", fontsize=8)

# Labels and title
ax.set_title("Vididness for visual imagery during wakefulness (VVIQ)", fontsize=11, pad=10)
ax.set_xlabel("VVIQ score")
ax.set_ylabel("Distribution in the population")

# Move legend to bottom-left
ax.legend(frameon=False, fontsize=8, loc="lower left")

# Clean axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=8)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)











