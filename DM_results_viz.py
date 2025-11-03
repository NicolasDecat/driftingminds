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
    <div style="text-align:center; margin-bottom:0.35rem;">  <!-- more space -->
        <div style="font-size:1.4rem; font-weight:200;">DRIFTING MINDS STUDY</div>
        <div style="font-size:1rem; margin-top:0.1rem;">
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

from textwrap import dedent

# --- Profile header (centered, small lead, big title, centered desc) ---------
from textwrap import dedent

# Fallback if description is missing
if not prof_desc:
    prof_desc = " "

st.markdown(dedent("""
<style>
  .dm-prof-wrap {
    text-align: center;
    margin: 8px auto 10px auto;
    max-width: 820px;          /* keeps line lengths pleasant */
  }
  /* Lead line (small) */
  .dm-prof-lead {
    font-weight: 600;
    font-size: 0.95rem;
    color: #444;
    margin: 0 0 6px 0;
    letter-spacing: 0.2px;
  }
  /* Profile name (very big) */
  .dm-prof-key {
    font-weight: 900;
    /* responsive size: min 28px, prefers 48px, max 64px */
    font-size: clamp(28px, 5vw, 64px);
    line-height: 1.05;
    margin: 0 0 8px 0;
    color: #000;
  }
  /* Description (normal size) */
  .dm-prof-desc {
    color: #111;
    font-size: 1.05rem;
    line-height: 1.55;
    margin: 0 auto;
    max-width: 680px;
  }

  @media (max-width: 640px){
    .dm-prof-lead { font-size: 0.9rem; }
    .dm-prof-desc { font-size: 1rem; }
  }
</style>
"""), unsafe_allow_html=True)

st.markdown(dedent("""
<style>
  .dm-prof-wrap {
    text-align: center;
    margin: 10px auto 14px auto;
    max-width: 820px;
  }

  /* Lead line — now softer and lighter */
  .dm-prof-lead {
    font-weight: 400;          /* lighter */
    font-size: 1rem;
    color: #666;               /* softer gray */
    margin: 0 0 8px 0;
    letter-spacing: 0.3px;
    font-style: italic;        /* gives it a dreamy touch */
  }

  /* Profile name — big and clean */
  .dm-prof-key {
    font-weight: 800;
    font-size: clamp(28px, 5vw, 60px);
    line-height: 1.05;
    margin: 0 0 10px 0;
    color: #000;
  }

  /* Description — normal weight, centered */
  .dm-prof-desc {
    color: #111;
    font-size: 1.05rem;
    line-height: 1.55;
    margin: 0 auto;
    max-width: 680px;
    font-weight: 400;
  }

  @media (max-width: 640px){
    .dm-prof-lead { font-size: 0.95rem; }
    .dm-prof-desc { font-size: 1rem; }
  }
</style>
"""), unsafe_allow_html=True)

st.markdown(dedent(f"""
<div class="dm-prof-wrap" role="group" aria-label="Sleep-onset profile">
  <p class="dm-prof-lead">You drift into sleep like a...</p>
  <h1 class="dm-prof-key">{prof}</h1>
  <p class="dm-prof-desc">{prof_desc}</p>
</div>
"""), unsafe_allow_html=True)





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



#%% Five-Dimension Bars (One-sided amplification) #############################
###############################################################################

# --- Rename the 5 dimensions (DROP-IN REPLACEMENT) ---------------------------
DIM_BAR_CONFIG = {
    "Vivid": {
        "freq_keys": ["freq_percept_intense", "freq_percept_precise", "freq_percept_real"],
        "weight_keys": ["degreequest_vividness", "degreequest_distinctness"],
        "invert_keys": [],
        "weight_mode": "standard",
        "help": "Dull  ↔  Vivid",  # (was: Thoughts ↔ Imagery)
    },
    "Bizarre": {
        "freq_keys": ["freq_think_bizarre", "freq_percept_bizarre", "freq_think_seq_bizarre"],
        "weight_keys": ["degreequest_bizarreness"],
        "invert_keys": [],
        "weight_mode": "standard",
        "help": "Ordinary  ↔  Bizarre",
    },
    "Immersive": {
        "freq_keys": ["freq_absorbed", "freq_actor", "freq_percept_narrative"],
        "weight_keys": ["degreequest_immersiveness"],
        "invert_keys": [],
        "weight_mode": "standard",
        "help": "External-oriented  ↔  Immersive",
    },
    "Spontaneous": {
        "freq_keys": ["freq_percept_imposed", "freq_spectator"],
        "weight_keys": ["degreequest_spontaneity"],
        "invert_keys": [],
        "weight_mode": "standard",
        "help": "Voluntary  ↔  Spontaneous",
    },
    "Emotional": {
        # Base is "positivity": positives kept, negatives/rumination inverted.
        "freq_keys": ["freq_positive", "freq_negative", "freq_ruminate"],
        "weight_keys": ["degreequest_emotionality"],  # 1=very negative, 6=very positive (bipolar)
        "invert_keys": ["freq_negative", "freq_ruminate"],
        "weight_mode": "emotion_bipolar",
        "help": "Negative  ↔  Positive",
    },
}


def _get(record, key, default=np.nan):
    return record.get(key, default)

def _norm16(x):
    try:
        v = float(x)
    except:
        return np.nan
    if np.isnan(v): return np.nan
    return np.clip((v - 1.0) / 5.0, 0.0, 1.0)

def _mean_ignore_nan(arr):
    arr = [a for a in arr if not (isinstance(a, float) and np.isnan(a))]
    return np.nan if len(arr) == 0 else float(np.mean(arr))

def _weight_boost(wvals, mode: str):
    """
    Return a boost term in [0, 0.5] that controls the one-sided amplification.
    - 'standard': average w (0..1), boost = max(0, w - 0.5)  → 0..0.5
    - 'emotion_bipolar': average w (0..1) where 0=very negative, 1=very positive.
       Use intensity relative to neutral 0.5: boost = max(0, 2*abs(w-0.5) - 0.5) → 0..0.5
       (i.e., only strong emotions—positive OR negative—amplify; neutral does not)
    If all weights missing → treat as neutral (0.5) → boost = 0.
    """
    w = _mean_ignore_nan([_norm16(v) for v in wvals])
    if isinstance(w, float) and np.isnan(w):
        w = 0.5  # no weight info → neutral (no amplification)

    if mode == "emotion_bipolar":
        # map distance from neutral to a 0..1 intensity: I = 2*|w-0.5|
        intensity = 2.0 * abs(w - 0.5)   # 0 at 0.5; 1 at 0 or 1
        boost = max(0.0, intensity - 0.5)  # only stronger-than-moderate emotions amplify
    else:
        boost = max(0.0, w - 0.5)

    # both formulas produce a boost in [0, 0.5]
    return float(np.clip(boost, 0.0, 0.5))

def compute_dimension_score(record, cfg, k_bump=0.8):
    # 1) Base from freq_* (normalize 1–6 → 0–1; invert some for Emotion)
    vals = []
    for k in cfg["freq_keys"]:
        v = _norm16(_get(record, k))
        if k in cfg.get("invert_keys", []):
            v = 1.0 - v if not (isinstance(v, float) and np.isnan(v)) else v
        vals.append(v)
    base = _mean_ignore_nan(vals)
    if isinstance(base, float) and np.isnan(base):
        return np.nan

    # 2) Compute boost (0..0.5) based on weight mode
    boost = _weight_boost([_get(record, wk) for wk in cfg["weight_keys"]],
                          cfg.get("weight_mode", "standard"))

    # 3) One-sided additive bump; zero when boost==0.
    #    base*(1-base) softly protects extremes (0 or 1) from huge jumps.
    bump = k_bump * boost * base * (1.0 - base)
    final = float(np.clip(base + bump, 0.0, 1.0))
    return final



# --- Load population (N=1000) -----------------------------------------------
csv_path = os.path.join("assets", "N1000_comparative_viz_ready.csv")
try:
    pop_data = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"Could not load population data at {csv_path}: {e}")
    pop_data = None

# --- Compute participant scores (0..100) -------------------------------------
bars = []
for name, cfg in DIM_BAR_CONFIG.items():
    score01 = compute_dimension_score(record, cfg, k_bump=0.8)
    score100 = None if (isinstance(score01, float) and np.isnan(score01)) else float(score01 * 100.0)
    bars.append({"name": name, "help": cfg["help"], "score": score100})

# --- Recompute SAME scores for all population rows ---------------------------
def compute_population_distributions(df: pd.DataFrame, dim_config: dict, k_bump=0.8):
    if df is None or df.empty:
        return {}
    dist = {name: [] for name in dim_config.keys()}
    for _, row in df.iterrows():
        rec = row.to_dict()
        for name, cfg in dim_config.items():
            s = compute_dimension_score(rec, cfg, k_bump=k_bump)
            if not (isinstance(s, float) and np.isnan(s)):
                dist[name].append(float(np.clip(s * 100.0, 0.0, 100.0)))
    for name in dist:
        arr = np.array(dist[name], dtype=float)
        dist[name] = arr[~np.isnan(arr)]
    return dist

pop_dists = compute_population_distributions(pop_data, DIM_BAR_CONFIG, k_bump=0.8)
pop_medians = {
    k: (float(np.nanmedian(v)) if (isinstance(v, np.ndarray) and v.size) else None)
    for k, v in pop_dists.items()
}

# --- Styling (bars closer; % not bold) ---------------------------------------
from textwrap import dedent
st.markdown(dedent("""
<style>
  /* ===============================
     Drifting Minds — Bars Styling
     =============================== */

  /* NEW wrapper that shifts the whole bars section left */
  .dm2-outer {
    margin-left: -70px !important;  /* tweak -40..-80px to taste */
    width: 100%;
  }
  
  .dm2-row {
  display: grid;
  grid-template-columns: 160px 1fr;  /* label column | bar column */
  column-gap: 8px;                   /* 👈 distance between label and bar */
  align-items: center;
  margin: 10px 0;
}


  .dm2-bars {
    margin-top: 16px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    width: 100%;
    text-align: left;
  }

  .dm2-left {
  display:flex; align-items:center;
  gap:0px;
  width: 160px;            /* pick one value and keep it consistent */
  flex: 0 0 160px;         /* ⟵ match width exactly */
}



  /* Left: label only, narrow so the bars start closer */
  .dm2-left {
  display:flex; align-items:center;
  gap:0px;
  width: 150px;          /* ⟵ was 168px: shifts bars a bit left */
  flex: 0 0 160px;
  }


  .dm2-label {
  font-weight: 800;
  font-size: 1.10rem;
  line-height: 1.05;
  white-space: nowrap;
  letter-spacing: 0.1px;
  position: relative;
  top: -3px;
  text-align: right;
  width: 100%;
  padding-right: 40px;     /* ⟵ keep this as you asked */
  margin: 0;
}


  /* Middle: bar + overlays */
  .dm2-wrap {
  flex: 1 1 auto;
  display:flex; 
  flex-direction:column; 
  gap:4px;
  margin-left: -12px;      /* ⟵ key trick: nudge bars left toward labels */
}


  .dm2-track {
    position: relative;
    width: 100%; height: 14px;
    background: #EDEDED;
    border-radius: 999px;
    overflow: visible;       /* allow overlay labels outside the bar */
  }

  .dm2-fill {
    height: 100%;
    background: linear-gradient(90deg, #CBBEFF 0%, #A18BFF 60%, #7B61FF 100%);
    border-radius: 999px;
    transition: width 600ms ease;
  }

  .dm2-median {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 8px; height: 8px;
    background: #000000;
    border: 1.5px solid #FFFFFF;
    border-radius: 50%;
    pointer-events: none; box-sizing: border-box;
  }

  /* Median label ("world") */
  .dm2-mediantag {
    position: absolute;
    bottom: calc(100% + 2px);      /* default ABOVE the bar */
    transform: translateX(-50%);
    font-size: 0.82rem;
    font-weight: 600;
    color: #000;
    white-space: nowrap;
    pointer-events: none;
    line-height: 1.05;
  }
  .dm2-mediantag.below {
    bottom: auto;
    top: calc(100% + 2px);         /* BELOW the bar when needed */
  }

  /* Purple % label at end of participant bar */
  .dm2-scoretag {
    position: absolute;
    bottom: calc(100% + 2px);
    transform: translateX(-50%);
    font-size: 0.86rem;
    font-weight: 500;
    color: #7B61FF;                /* dark gradient purple */
    white-space: nowrap;
    pointer-events: none;
    line-height: 1.05;
  }
  .dm2-scoretag.below {
    bottom: auto;
    top: calc(100% + 2px);
  }

  /* Anchors under the bar */
  .dm2-anchors {
    display:flex; justify-content:space-between;
    font-size: 0.85rem; color:#666; margin-top: 0;
    line-height: 1;
  }

  /* Mobile tweaks */
  @media (max-width: 640px){
    .dm2-left { width: 148px; flex-basis:148px; }
    .dm2-label { font-size: 1.05rem; top:-2px; padding-right: 6px; }
  }
</style>
"""), unsafe_allow_html=True)
    




# --- Render (labels LEFT; bars RIGHT; "world" on Perception/Vivid; purple % on bar end) ---
st.markdown("<div class='dm2-outer'><div class='dm2-bars'>", unsafe_allow_html=True)

min_fill = 2  # minimal % fill for aesthetic continuity

def _clamp_pct(p, lo=2.0, hi=98.0):
    try: p = float(p)
    except: return lo
    return max(lo, min(hi, p))

for idx, b in enumerate(bars):
    name = b["name"]               # e.g., "Vivid", "Bizarre", ...
    help_txt = b["help"]
    score = b["score"]             # 0..100 or None
    median = pop_medians.get(name, None)  # 0..100 or None

    # Participant bar width and display text
    if score is None or (isinstance(score, float) and np.isnan(score)):
        width = min_fill
        score_txt = "NA"
        width_clamped = _clamp_pct(width)
    else:
        width = int(round(np.clip(score, 0, 100)))
        width = max(width, min_fill)
        score_txt = f"{int(round(score))}%"
        width_clamped = _clamp_pct(width)

    # Median dot position (and clamped for label positioning)
    if median is None or (isinstance(median, float) and np.isnan(median)):
        med_left = None
        med_left_clamped = None
    else:
        med_left = float(np.clip(median, 0, 100))
        med_left_clamped = _clamp_pct(med_left)

    # Anchors
    if isinstance(help_txt, str) and "↔" in help_txt:
        left_anchor, right_anchor = [s.strip() for s in help_txt.split("↔", 1)]
    else:
        left_anchor, right_anchor = "0", "100"

    # Build overlays
    median_html = "" if med_left is None else f"<div class='dm2-median' style='left:{med_left}%;'></div>"

    # Identify Perception/Vivid for 'world' label
    is_perception = (name.lower() in ("perception", "vivid"))

    mediantag_html = ""
    scoretag_html = ""
    
    if is_perception and (med_left_clamped is not None):
        # Determine if the participant score and median dot are close (overlap condition)
        overlap = (
            score_txt != "NA"
            and abs(width_clamped - med_left_clamped) <= 6.0  # you can tweak this threshold
        )
    
        # If overlap → place "world" below the median; otherwise above
        put_world_below = overlap
    
        mediantag_class = "dm2-mediantag below" if put_world_below else "dm2-mediantag"
        mediantag_html = f"<div class='{mediantag_class}' style='left:{med_left_clamped}%;'>world</div>"
    
    # Purple % label above the participant bar end (for all dimensions)
    if score_txt != "NA":
        scoretag_html = f"<div class='dm2-scoretag' style='left:{width_clamped}%;'>{score_txt}</div>"
    
        
            
            
    
    
    row_html = (
        "<div class='dm2-row'>"
          "<div class='dm2-left'>"
            f"<div class='dm2-label'>{name}</div>"
          "</div>"
          "<div class='dm2-wrap'>"
            f"<div class='dm2-track' aria-label='{name} score {score_txt}'>"
              f"<div class='dm2-fill' style='width:{width}%;'></div>"
              f"{median_html}"
              f"{mediantag_html}"
              f"{scoretag_html}"
            "</div>"
            "<div class='dm2-anchors'>"
              f"<span>{left_anchor}</span>"
              f"<span>{right_anchor}</span>"
            "</div>"
          "</div>"
        "</div>"
    )

    st.markdown(row_html, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)














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











