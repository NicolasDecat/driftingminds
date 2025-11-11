#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drifting Minds — Streamlit visualisation of participant results
Maintains the SAME UI/UX as your current app.
"""

# ==============
# Imports
# ==============
import os
import re
import json
import base64
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, truncnorm

# ==============
# App config
# ==============`

st.set_page_config(page_title="Drifting Minds — Profile", layout="centered")

REDCAP_API_URL = st.secrets.get("REDCAP_API_URL")
REDCAP_API_TOKEN = st.secrets.get("REDCAP_API_TOKEN")

# Shareable png starts now
st.markdown('<div id="dm-share-card">', unsafe_allow_html=True)


# ==============
# QR code
# ==============

st.markdown("""
<style>
/* Ensure absolute children are positioned relative to the page content area */
[data-testid="stAppViewContainer"] > .main > div {
  position: relative;
}
</style>
""", unsafe_allow_html=True)

# --- QR code (top-right inside page padding) ---
def _data_uri(path: str) -> str:
    mime = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    with open(path, "rb") as f:
        import base64
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

qr_path = os.path.join("assets", "qr_code_DM.png")
qr_src  = _data_uri(qr_path) if os.path.exists(qr_path) else ""

st.markdown(
    f"""
    <div style="
        position: absolute;
        top: 40px;
        right: 0px;
        text-align: center;
        font-size: 0.9rem;
        color: #000;
        z-index: 1000;
        line-height: 1.05;
    ">
        <img src="{qr_src}" width="88" style="display:block; margin:0 auto 1px auto;" />
        <div style="font-weight:600; margin:0;">Participate!</div>
        <div style="font-size:0.8rem; margin-top:3px;">redcap.link/DriftingMinds</div>
    </div>
    """,
    unsafe_allow_html=True
)



# ==============
# Global constants
# ==============
PURPLE_HEX = "#7C3AED"   # plots/polygons
PURPLE_NAME = "#7A5CFA"  # profile name
CAP_MIN = 60.0           # sleep latency cap (minutes)
ASSETS_CSV = os.path.join("assets", "N1000_comparative_viz_ready.csv")


# ==============
# CSS (single block; exact same visual result)
# ==============
st.markdown("""
<style>
/* Remove Streamlit default padding (multiple selectors for version coverage) */
section.main > div.block-container { padding-top: 0 !important; }
div.block-container { padding-top: 0 !important; }
[data-testid="stAppViewContainer"] > .main > div { padding-top: 0 !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
header[data-testid="stHeader"] { height: 0px; background: transparent; }
header[data-testid="stHeader"]::before { content: none; }

/* Layout primitives */
:root { --dm-max: 820px; }
.dm-center { max-width: var(--dm-max); margin: 0 auto; }

/* Title */
.dm-title {
  font-size: 2.5rem;
  font-weight: 200;
  margin: 0 0 1.25rem 0;  /* tightened below, zero above */
  text-align: center;
}

/* Row (icon + text) stays side-by-side at all widths */
.dm-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;          /* small gap between pictogram and text block */
  margin-bottom: .25rem;  /* ⟵ add this for extra space before bars */
}

/* Icon slightly shifted right (keeps close to right-shifted text) */
.dm-icon {
  width: 140px;
  height: auto;
  flex: 0 0 auto;
  transform: translateX(6rem);  /* mobile/tablet shift */
}

/* Text block (strong right shift) */
.dm-text {
  flex: 1 1 0;
  min-width: 0;
  padding-left: 6rem;           /* matches current look */
}

/* Typography for lead/name/desc (unchanged) */
.dm-lead {
  font-weight: 400;
  font-size: 1rem;
  color: #666;
  margin: 0 0 8px 0;
  letter-spacing: 0.3px;
  font-style: italic;
  text-align: left;
}
.dm-key {
  font-weight: 600;
  font-size: clamp(28px, 5vw, 60px);
  line-height: 1.05;
  margin: 0 0 10px 0;
  color: #7A5CFA;
  text-align: left;
}
.dm-desc {
  color: #111;
  font-size: 1.05rem;
  line-height: 1.55;
  margin: 0;
  max-width: 680px;
  font-weight: 400;
  text-align: left;
}

/* Responsive tweaks */
@media (min-width: 640px) {
  .dm-icon { transform: translateX(8rem); } /* extra shift on desktop */
  .dm-text { padding-left: 8rem; }
}
@media (max-width: 420px) {
  .dm-icon { width: 110px; }
  .dm-row  { gap: 0.9rem; }
}

/* ===============================
   Drifting Minds — Bars Styling
   =============================== */
.dm2-outer { margin-left: -70px !important; width: 100%; }

.dm2-row {
  display: grid;
  grid-template-columns: 160px 1fr;  /* label | bar */
  column-gap: 8px;
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

/* Left column — label only */
.dm2-left { display:flex; align-items:center; gap:0px; width:160px; flex:0 0 160px; }
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
  padding-right: 40px;
  margin: 0;
}

/* Right column — bars + overlays */
.dm2-wrap {
  flex: 1 1 auto;
  display:flex; flex-direction:column; gap:4px;
  margin-left: -12px;   /* nudge bars toward labels */
}

.dm2-track {
  position: relative;
  width: 100%; height: 14px;
  background: #EDEDED;
  border-radius: 999px;
  overflow: visible;
}
.dm2-fill {
  height: 100%;
  background: linear-gradient(90deg, #CBBEFF 0%, #A18BFF 60%, #7B61FF 100%);
  border-radius: 999px;
  transition: width 600ms ease;
}
.dm2-median {
  position: absolute; top: 50%;
  transform: translate(-50%, -50%);
  width: 8px; height: 8px;
  background: #000000;
  border: 1.5px solid #FFFFFF;
  border-radius: 50%;
  pointer-events: none; box-sizing: border-box;
}
.dm2-mediantag {
  position: absolute;
  bottom: calc(100% + 2px);
  transform: translateX(-50%);
  font-size: 0.82rem; font-weight: 600; color: #000;
  white-space: nowrap; pointer-events: none; line-height: 1.05;
}
.dm2-mediantag.below { bottom: auto; top: calc(100% + 2px); }

.dm2-scoretag {
  position: absolute;
  bottom: calc(100% + 2px);
  transform: translateX(-50%);
  font-size: 0.86rem; font-weight: 500; color: #7B61FF;
  white-space: nowrap; pointer-events: none; line-height: 1.05;
}
.dm2-scoretag.below { bottom: auto; top: calc(100% + 2px); }

.dm2-anchors {
  display:flex; justify-content:space-between;
  font-size: 0.85rem; color:#666; margin-top: 0; line-height: 1;
}

@media (max-width: 640px){
  .dm2-left { width: 148px; flex-basis:148px; }
  .dm2-label { font-size: 1.05rem; top:-2px; padding-right: 6px; }
}

/* Reduce default left/right padding for main content */
.block-container {
    padding-left: 5rem !important;
    padding-right: 5rem !important;
    max-width: 1100px !important;   /* optional: widen usable area */
    margin: 0 auto !important;      /* keep centered */
}


</style>


""", unsafe_allow_html=True)

# --- Force light mode globally (no dark mode on any device) ---
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
  background-color: #FFFFFF !important;
  color-scheme: light !important;
}
@media (prefers-color-scheme: dark) {
  html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
    background-color: #FFFFFF !important;
    color: #000000 !important;
  }
}
</style>
""", unsafe_allow_html=True)


# ==============
# Data access
# ==============
def fetch_by_record_id(record_id: str):
    """Fetch a single REDCap record as dict."""
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

# ==============
# Query param → record
# ==============
record = None
record_id = st.query_params.get("id")
if record_id:
    record = fetch_by_record_id(record_id)
if not record:
    st.error("We couldn’t find your responses.")
    st.stop()
    
# Export helpers
from matplotlib import font_manager as _fm

def _dm_register_fonts():
    candidates = [
        ("Inter Regular", "assets/Inter-Regular.ttf"),
        ("Inter Medium",  "assets/Inter-Medium.ttf"),
        ("Inter Bold",    "assets/Inter-Bold.ttf"),
    ]
    for _, path in candidates:
        try:
            _fm.fontManager.addfont(path)
        except Exception:
            pass
_dm_register_fonts()


# ==============
# Normalization helpers
# ==============
def _to_float(x):
    try:
        return float(x)
    except:
        return np.nan
    
def norm_bool(x): # For binary 0 / 1
    try:
        return 1.0 if float(x) >= 0.5 else 0.0
    except:
        return np.nan
    
import re
def norm_eq(x, value):
    """
    Returns 1.0 if x equals `value` (tolerant to floats/strings/newlines),
    else 0.0; NaN -> NaN.
    """
    if x is None:
        return np.nan
    # pull first numeric token if present
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "nan":
            return np.nan
        m = re.search(r'[-+]?\d+(\.\d+)?', s)
        if m:
            try:
                return 1.0 if float(m.group(0)) == float(value) else 0.0
            except:
                return 0.0
        # no number found: fallback to exact string compare
        return 1.0 if s == str(value) else 0.0
    # numeric path
    try:
        return 1.0 if float(x) == float(value) else 0.0
    except:
        return np.nan

    
def norm_1_4(x):
    x = _to_float(x)
    if np.isnan(x): return np.nan
    return np.clip((x - 1.0) / 3.0, 0.0, 1.0)    

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
    """Accepts minutes, 'HH:MM', '1h05', '15', or 0..1 normalized; returns minutes or None if already normalized."""
    if isinstance(x, (int, float)):
        if 0.0 <= x <= 1.0: return None
        return float(x)
    if x is None: return np.nan

    s = str(x).strip().lower()
    if s == "" or s in {"na", "n/a", "none"}: return np.nan

    m = re.match(r"^\s*(\d{1,2})\s*:\s*(\d{1,2})\s*$", s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        return float(hh * 60 + mm)

    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)?", s)
    if parts:
        total = 0.0; any_unit = False
        for val, unit in parts:
            if val == "": continue
            v = float(val)
            if unit in ("h","hr","hrs","hour","hours"): total += v * 60.0; any_unit = True
            elif unit in ("m","min","mins","minute","minutes"): total += v; any_unit = True
        if any_unit: return total

    try:
        v = float(s)
        if 0.0 <= v <= 1.0: return None
        return v
    except:
        return np.nan

def norm_latency_auto(x, cap_minutes=CAP_MIN):
    mins = _to_minutes_relaxed(x)
    if mins is None:
        v = _to_float(x)
        return np.clip(v, 0.0, 1.0)
    if np.isnan(mins): return np.nan
    return np.clip(mins / cap_minutes, 0.0, 1.0)


# ---- Profile dictionary (single source of truth) -----------------------------
# All profiles below use their own 'features'. Each feature is a target in [0..1].
# You can mix 'dim' (composite DIMENSIONS) and 'var' (raw fields + normalizer).

PROFILES = {

    # =====================================================================
    # Dreamweaver
    # =====================================================================
    "Dreamweaver": {
    "features": [
        {"type": "var", "key": ["freq_percept_intense"],     "norm": norm_1_6, "norm_kwargs": {}, "target": 0.70, "weight": 1},
        {"type": "var", "key": ["freq_percept_narrative"],   "norm": norm_1_6, "norm_kwargs": {}, "target": 0.80, "weight": 1},
        {"type": "var", "key": ["freq_absorbed"],            "norm": norm_1_6, "norm_kwargs": {}, "target": 0.70, "weight": 1.0},
        {"type": "var", "key": ["degreequest_vividness"] ,   "norm": norm_1_6, "norm_kwargs": {}, "target": 0.70, "weight": 0.8},  
        {"type": "var", "key": ["degreequest_bizarreness"] , "norm": norm_1_6, "norm_kwargs": {}, "target": 0.70, "weight": 0.8},  
    
    ],
    "description": "You drift into vivid, sensory mini-dreams as you fall asleep.",
    "icon": "seahorse.svg",
    },


    # =====================================================================
    # Switch-Off
    # =====================================================================
    "The Switch-Off": {
        "features": [
            {"type": "var", "key": ["sleep_latency"],                   "norm": norm_latency_auto, "norm_kwargs": {"cap_minutes": CAP_MIN}, "target": 0.05, "weight": 1.2,  "hit_op": "lte"},
            {"type": "var", "key": ["degreequest_sleepiness"],          "norm": norm_1_6, "norm_kwargs": {}, "target": 0.60, "weight": 0.8},
            {"type":"var","key":["trajectories"],                       "norm": norm_eq, "norm_kwargs": {"value": 2}, "target": 1.0, "weight": 1.2}
        ],
        "description": "You fall asleep quickly, especially when you already feel sleepy.",
        "icon": "bear.svg",
    },
    
       
    # =====================================================================
    # Fantasizer
    # =====================================================================
    "Fantasizer": {
    "features": [
        {"type": "var","key": ["freq_scenario"],       "norm": norm_1_6, "norm_kwargs": {}, "target": 0.90, "weight": 1},
        {"type":"var","key":["anytime_20"],            "norm": norm_bool, "target": 1.0, "weight": 1},         
        {"type": "var","key": ["freq_positive"],       "norm": norm_1_6, "norm_kwargs": {},"target": 0.70, "weight": 0.6,
         "only_if": {"key": ["timequest_positive"],    "norm": norm_1_100,"norm_kwargs": {},"op": "between","bounds": [0, 0.50]}
        },
    ],
    "description": "Your mind drifts into imagined stories; vivid, intentional scenarios that feel like daydreams easing you into sleep.",
    "icon": "dolphin.svg",
    },
    
    # =====================================================================
    # Archivist
    # =====================================================================
    "Archivist": {
    "features": [
        {"type": "var", "key": ["freq_replay"],                "norm": norm_1_6, "norm_kwargs": {}, "target": 0.90, "weight": 1.3},
        {"type": "var","key": ["freq_think_ordinary"],         "norm": norm_1_6, "norm_kwargs": {},"target": 0.90, "weight": 1,
         "only_if": {"key": ["timequest_think_ordinary"],      "norm": norm_1_100,"norm_kwargs": {},"op": "between","bounds": [0, 0.50]}
        },
        {"type": "var","key": ["freq_think_seq_ordinary"],     "norm": norm_1_6, "norm_kwargs": {},"target": 0.90, "weight": 1,
         "only_if": {"key": ["timequest_think_seq_ordinary"],  "norm": norm_1_100,"norm_kwargs": {},"op": "between","bounds": [0, 0.50]}
        },
    ],
    "description": "You mentally revisit the past as you fall asleep — replaying moments, conversations, or scenes that linger from the day, like an archivist sorting through memories before rest.",
    "icon": "salmon.svg",
    },
    
    # =====================================================================
    # Ruminator
    # =====================================================================
    "Ruminator": {
        "features": [
            {"type": "var", "key": ["freq_ruminate"],                   "norm": norm_1_6,   "norm_kwargs": {}, "target": 0.80, "weight": 1.3},  
            {"type": "var", "key": ["freq_negative"],                   "norm": norm_1_6,   "norm_kwargs": {}, "target": 0.70, "weight": 0.80},  
            {"type": "var", "key": ["anxiety"],                         "norm": norm_1_100, "norm_kwargs": {}, "target": 0.70, "weight": 0.80},  
            {"type": "var", "key": ["sleep_latency"],                   "norm": norm_latency_auto, "norm_kwargs": {"cap_minutes": CAP_MIN}, "target": 0.90, "weight": 1.1},
            {"type": "var", "key": ["degreequest_emotionality"],        "norm": norm_1_6,   "norm_kwargs": {}, "target": 0.30, "weight": 0.70},  

        ],
        "description": "You replay or analyze the day, with longer latency and tension.",
        "icon": "cow.svg",
    },

    # =====================================================================
    # Creative
    # =====================================================================
    # "Creative": {
    #     "features": [
    #         {"type": "var","key": ["freq_creat"],           "norm": norm_1_6, "norm_kwargs": {}, "target": 0.95, "weight": 1},
    #     ],
    #     "description": "Ideas spark at the edge of sleep — you drift off with creativity alive.",
    #     "icon":        "octopus.svg",
    # },


    # =====================================================================
    # Freewheeler
    # =====================================================================
    "Freewheeler": {
    "features": [
        {"type": "var", "key": ["freq_think_nocontrol"],      "norm": norm_1_6, "norm_kwargs": {}, "target": 0.80, "weight": 1.1},  
        {"type": "var", "key": ["freq_think_bizarre"],        "norm": norm_1_6, "norm_kwargs": {}, "target": 0.70, "weight": 0.8},  
        {"type": "var", "key": ["freq_think_seq_bizarre"],    "norm": norm_1_6, "norm_kwargs": {}, "target": 0.70, "weight": 0.8},  
        {"type": "var", "key": ["degreequest_spontaneity"] ,  "norm": norm_1_6, "norm_kwargs": {}, "target": 0.70, "weight": 1},  
        {"type": "var", "key": ["degreequest_bizarreness"] ,  "norm": norm_1_6, "norm_kwargs": {}, "target": 0.70, "weight": 0.8},  
    ],
    "description": "You start intentional, then let go into spontaneous imagery.",
    "icon": "otter.svg",
    },
        
    # =====================================================================
    # Quiet Mind
    # =====================================================================
    "Quiet Mind": {
        "features": [
            {"type": "var", "key": ["degreequest_vividness"],      "norm": norm_1_6, "norm_kwargs": {}, "target": 0.20, "weight": 0.80},  
            {"type": "var", "key": ["degreequest_distinctness"],   "norm": norm_1_6, "norm_kwargs": {}, "target": 0.20, "weight": 0.80},  
            {"type": "var", "key": ["degreequest_immersiveness"],  "norm": norm_1_6, "norm_kwargs": {}, "target": 0.20, "weight": 0.80},  
            {"type": "var", "key": ["degreequest_bizarreness"],    "norm": norm_1_6, "norm_kwargs": {}, "target": 0.20, "weight": 0.80},  
            {"type": "var", "key": ["degreequest_emotionality"],   "norm": norm_1_6, "norm_kwargs": {}, "target": 0.50, "weight": 0.80},  
        ],
        "description": "You fall asleep with little mental content — soft, quiet onset.",
        "icon": "sloth.svg",
    },
    
    # =====================================================================
    # Radio Tuner
    # =====================================================================
    "Radio Tuner": {
        "features": [
            {"type": "var","key": ["freq_think_ordinary"],      "norm": norm_1_6, "norm_kwargs": {},"target": 0.90, "weight": 1,
             "only_if": {"key": ["timequest_think_ordinary"],   "norm": norm_1_100,"norm_kwargs": {},"op": "between","bounds": [0, 0.50]}
            },
            {"type": "var","key": ["freq_think_bizarre"],       "norm": norm_1_6, "norm_kwargs": {},"target": 0.90, "weight": 1,
             "only_if": {"key": ["timequest_think_bizarre"],   "norm": norm_1_100,"norm_kwargs": {},"op": "between","bounds": [0.51, 1]}
            },
            {"type": "var","key": ["freq_percept_dull"],        "norm": norm_1_6, "norm_kwargs": {},"target": 0.90, "weight": 1,
             "only_if": {"key": ["timequest_percept_dull"],     "norm": norm_1_100,"norm_kwargs": {},"op": "between","bounds": [0, 0.50]}
            },
            {"type": "var","key": ["freq_percept_intense"],     "norm": norm_1_6, "norm_kwargs": {},"target": 0.90, "weight": 1,
             "only_if": {"key": ["timequest_percept_intense"],  "norm": norm_1_100,"norm_kwargs": {},"op": "between","bounds": [0.51, 1]}
            },
            {"type": "var","key": ["freq_hear_env"],            "norm": norm_1_6, "norm_kwargs": {},"target": 0.90, "weight": 1,
             "only_if": {"key": ["timequest_hear_env"],         "norm": norm_1_100,"norm_kwargs": {},"op": "between","bounds": [0, 0.50]}
            },
            {"type": "var","key": ["freq_absorbed"],            "norm": norm_1_6, "norm_kwargs": {},"target": 0.90, "weight": 1,
             "only_if": {"key": ["timequest_absorbed"],         "norm": norm_1_100,"norm_kwargs": {},"op": "between","bounds": [0.51, 1]}
            },
        ],
        "description": "Your mind shifts rapidly between thoughts and sensations — like tuning through mental stations, never lingering on one for long.",
        "icon": "chameleon.svg",
    },
    
    # =====================================================================
    # Strategist
    # =====================================================================
    "Strategist": {
        "features": [
            {"type": "var", "key": ["freq_planning"],              "norm": norm_1_6, "norm_kwargs": {}, "target": 0.90, "weight": 1.2},  
            {"type": "var", "key": ["freq_think_ordinary"],        "norm": norm_1_6, "norm_kwargs": {}, "target": 0.80, "weight": 1.0},  
            {"type": "var", "key": ["freq_think_seq_ordinary"],      "norm": norm_1_6, "norm_kwargs": {}, "target": 0.80, "weight": 1.0},  
        ],
        "description": "You stay in control with practical or analytical thoughts until lights out.",
        "icon": "octopus.svg",
    },
    
    # =====================================================================
    # Sentinelle
    # =====================================================================
    "Sentinelle": {
        "features": [
            {"type": "var", "key": ["anytime_24"],                   "norm": norm_bool, "norm_kwargs": {},"target": 1,"weight": 1.3},
            {"type": "var", "key": ["sleep_latency"],                "norm": norm_latency_auto, "norm_kwargs": {"cap_minutes": CAP_MIN}, "target": 0.4, "weight": 0.3},
            {"type": "var", "key": ["degreequest_immersiveness"],    "norm": norm_1_6, "norm_kwargs": {}, "target": 0.2, "weight": 0.3},  

        
        ],
        "description": "Your mind stays quiet but your ears stay on. Ambient sounds remain present as you drift off slowly, like a sentinel keeping watch.",
        "icon": "merkaat.svg",
    },
    
    # =====================================================================
    # Fragmented Mind
    # =====================================================================
    "Fragmented Mind": {
        "features": [
            {"type": "var", "key": ["freq_percept_fleeting"],      "norm": norm_1_6, "norm_kwargs": {}, "target": 0.8, "weight": 1.2},  
            {"type": "var", "key": ["freq_think_seq_bizarre"],     "norm": norm_1_6, "norm_kwargs": {}, "target": 0.8, "weight": 1.0},  
            {"type": "var", "key": ["degreequest_fleetingness"],   "norm": norm_1_6, "norm_kwargs": {}, "target": 0.8, "weight": 1.2},  
         
        ],
        "description": "Your mind breaks into fleeting fragments. Flashes of images, words, or sensations that appear and vanish before taking shape.",
        "icon": "hummingbird.svg",
    },
    
    # =====================================================================
    # Pragmatic
    # =====================================================================
    "Pragmatic": {
        "features": [
            {"type": "var", "key": ["freq_think_ordinary"],      "norm": norm_1_6, "norm_kwargs": {}, "target": 0.8, "weight": 1},  
            {"type": "var", "key": ["freq_think_bizarre"],       "norm": norm_1_6, "norm_kwargs": {}, "target": 0.2, "weight": 1},  
            {"type": "var", "key": ["freq_think_nocontrol"],     "norm": norm_1_6, "norm_kwargs": {}, "target": 0.2, "weight": 1},  
            {"type": "var", "key": ["freq_think_seq_bizarre"],  "norm": norm_1_6, "norm_kwargs": {}, "target": 0.2, "weight": 1},  
            {"type": "var", "key": ["degreequest_bizarreness"],  "norm": norm_1_6, "norm_kwargs": {}, "target": 0.2, "weight": 1.2},  

        ],
        "description": "Your thoughts stay clear and practical — grounded in everyday logic rather than drifting into the strange or dreamlike.",
        "icon": "ant.svg",
    },
    

    
}




# ==============
# Dimensions & composite scores
# ==============
def _get_first(record, keys):
    """Return the first present, non-empty value for any of the candidate keys."""
    if isinstance(keys, (list, tuple)):
        for k in keys:
            if k in record and record[k] not in (None, "", "NA"):
                return record[k]
        return np.nan
    return record.get(keys, np.nan)

# === Conditional helpers ======================================================
def _eval_condition(record, cond: dict) -> bool:
    """
    Evaluate a single condition against the record.
    cond supports:
      - key: str | [str,...]
      - norm: callable (optional)
      - norm_kwargs: dict (optional)
      - op: "between" | "gte" | "lte" | "gt" | "lt" | "eq" | "in"  (default: "between")
      - bounds: [lo, hi]  (for op="between", inclusive)
      - value: float       (for gte/lte/gt/lt/eq)
      - values: list       (for op="in")
    Values are compared on the normalized scale if 'norm' is provided,
    otherwise raw floats are used.
    """
    keys = cond.get("key")
    keys = keys if isinstance(keys, (list, tuple)) else [keys]
    raw = _get_first(record, keys)

    norm_fn = cond.get("norm")
    kwargs = cond.get("norm_kwargs", {}) or {}

    # Compute comparable value v (normalized if norm given, else raw float)
    if norm_fn is None:
        v = _to_float(raw)
    else:
        try:
            v = norm_fn(raw, **kwargs)
        except TypeError:
            v = norm_fn(raw)

    if v is None:
        return False
    try:
        if np.isnan(v):
            return False
    except Exception:
        pass

    op = (cond.get("op") or "between").lower()
    if op == "between":
        lo, hi = cond.get("bounds", [0.0, 1.0])
        return (float(v) >= float(lo)) and (float(v) <= float(hi))
    if op == "gte": return float(v) >= float(cond["value"])
    if op == "lte": return float(v) <= float(cond["value"])
    if op == "gt":  return float(v) >  float(cond["value"])
    if op == "lt":  return float(v) <  float(cond["value"])
    if op == "eq":  return abs(float(v) - float(cond["value"])) < 1e-9
    if op == "in":
        return float(v) in set(float(x) for x in cond.get("values", []))
    return False


def _conditions_met(record, feat: dict) -> bool:
    """Support three shapes: only_if (single), only_if_all (AND), only_if_any (OR)."""
    if "only_if" in feat and not _eval_condition(record, feat["only_if"]):
        return False

    for c in feat.get("only_if_all", []) or []:
        if not _eval_condition(record, c):
            return False

    any_list = feat.get("only_if_any")
    if any_list:
        if not any(_eval_condition(record, c) for c in any_list):
            return False

    return True

def _shortcircuit_or_value(record, feat):
    """
    If any/all OR-conditions are met, return an override value for the feature.
    Returns None if no OR condition fires.
    Supported keys on the feature:
      - or_if_any: [cond, ...]   # OR over these conditions
      - or_if_all: [cond, ...]   # AND over these conditions
      - or_value: float          # value to return if fired (defaults to feature['target'])
    """
    # OR (any)
    any_list = feat.get("or_if_any") or []
    if any_list and any(_eval_condition(record, c) for c in any_list):
        return float(feat.get("or_value", feat.get("target", 1.0)))

    # OR (all) – i.e., fire if ALL are true
    all_list = feat.get("or_if_all") or []
    if all_list and all(_eval_condition(record, c) for c in all_list):
        return float(feat.get("or_value", feat.get("target", 1.0)))

    return None


def _feature_value_from_record(record, scores_unused, feat):
    if feat.get("type") != "var":
        return np.nan

    if not _conditions_met(record, feat):
        return np.nan

    # --- NEW: short-circuit OR that *provides a value* even if the main var is low/missing
    sc = _shortcircuit_or_value(record, feat)
    if sc is not None:
        return np.clip(sc, 0.0, 1.0)

    # normal path (uses the declared variable)
    keys = feat["key"] if isinstance(feat["key"], (list, tuple)) else [feat["key"]]
    raw = _get_first(record, keys)

    norm_fn = feat.get("norm")
    kwargs = feat.get("norm_kwargs", {}) or {}

    if norm_fn is None:
        v = _to_float(raw)
        if np.isnan(v): return np.nan
        return np.clip(v, 0.0, 1.0)
    try:
        return norm_fn(raw, **kwargs)
    except TypeError:
        return norm_fn(raw)


def _weighted_nanaware_distance(values, targets, weights):
    a = np.asarray(values, float)
    b = np.asarray(targets, float)
    w = np.asarray(weights, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if not np.any(mask):
        return np.inf
    d = a[mask] - b[mask]
    return np.sqrt(np.sum(w[mask] * d * d))

# --- AND-ish coherence helpers --------------------------------------
def _passes_only_if(rec, cond):
    if not cond:
        return True
    v = _get_first(rec, cond["key"])
    n = cond["norm"](v, **cond.get("norm_kwargs", {})) if cond.get("norm") else v
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return False
    op = cond.get("op", "between")
    if op == "between":
        lo, hi = cond["bounds"]
        return (float(n) >= lo) and (float(n) <= hi)
    if op == "eq":
        return float(n) == float(cond["value"])
    return False

def _feature_hit(rec, f, value, tol=0.98):
    """
    Returns 1 if feature 'hits' its target, 0 if not, None if ineligible.
    hit_op:
      - "gte" (default): value >= target * tol
      - "lte":           value <= target / tol
    """
    if not _passes_only_if(rec, f.get("only_if")):
        return None
    tgt = float(f.get("target", np.nan))
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0
        op = f.get("hit_op", "gte").lower()
        v = float(value)
        if op == "lte":
            return 1 if v <= (tgt / tol) else 0
        else:  # "gte"
            return 1 if v >= (tgt * tol) else 0
    except:
        return 0




def assign_profile_from_record(record):
    """
    For each profile, compute a weighted distance using only 'var' features.
    Returns (best_profile_name, {}).
    """
    scores = {}  # kept for compatibility with caller
    best_name, best_dist = None, np.inf

    # AND-ish knobs:
    K_RATIO = 0.6   # need ~60% of eligible criteria to be met
    GAMMA   = 1.5   # >1 increases penalty steepness

    for name, cfg in PROFILES.items():
        feats = cfg.get("features", [])
        if not feats:
            continue

        vals, targs, wts = [], [], []

        # --- collect values AND count how many criteria are hit simultaneously
        hits, eligible = 0, 0
        tmp_vals = []  # keep raw values to evaluate hits
        for f in feats:
            v   = _feature_value_from_record(record, scores, f)  # already normalized by your code
            tgt = float(f.get("target", np.nan))
            wt  = float(f.get("weight", 1.0))
            tmp_vals.append((f, v))
            vals.append(v); targs.append(tgt); wts.append(wt)

        # compute hit count (respecting only_if)
        for f, v in tmp_vals:
            h = _feature_hit(record, f, v)
            if h is None:
                continue           # not eligible due to only_if
            eligible += 1
            hits     += h

        # --- raw distance
        d = _weighted_nanaware_distance(vals, targs, wts)

        # --- AND-ish penalty: if not enough criteria are met, inflate distance
        if eligible > 0:
            K = max(1, int(np.ceil(K_RATIO * eligible)))
            if hits < K:
                # penalty grows as hits fall short of K
                shortfall = max(K - hits, 0)
                # multiplicative inflation; e.g., misses double/triple distance depending on GAMMA
                penalty = ((K / max(hits, 1)) ** GAMMA)  # if hits=0, uses 1 to avoid div/0
                d *= penalty

        # keep best
        if d < best_dist:
            best_name, best_dist = name, d

    return best_name, scores



# ==============
# Title + Profile header (icon + text)
# ==============
def _data_uri(path: str) -> str:
    mime = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

# Title
st.markdown("""
<div class="dm-center">
  <div class="dm-title">DRIFTING MINDS STUDY</div>
</div>
""", unsafe_allow_html=True)


# Assign profile + get text/icon
prof_name, scores = assign_profile_from_record(record)
prof_cfg = PROFILES.get(prof_name, {})
prof_desc = prof_cfg.get("description", "")
icon_file = prof_cfg.get("icon")
icon_path = f"assets/{icon_file}" if icon_file else None
has_icon = bool(icon_path and os.path.exists(icon_path))

# --- Add population percentage at the end of the profile description ---
POP_PERC = {
    "Dreamweaver": 4,
    "The Switch-Off": 14,
    "Fantasizer": 5,
    "Archivist": 9,
    "Ruminator": 6,
    "Freewheeler": 7,
    "Quiet Mind": 8,
    "Radio Tuner": 9,
    "Strategist": 7,
    "Sentinelle": 10,
    "Fragmented Mind": 5,
    "Pragmatic": 15
}

perc_val = POP_PERC.get(prof_name, 0)
prof_desc_ext = f"""{prof_desc}<br><span style='display:block; margin-top:2px; font-size:1rem; color:#222;'>
{prof_name}s represent {perc_val}% of the population.</span>"""

# --- Render profile header ---
icon_src = _data_uri(icon_path) if has_icon else ""
st.markdown(f"""
<div class="dm-center">
  <div class="dm-row">
    {'<img class="dm-icon" src="'+icon_src+'" alt="profile icon"/>' if has_icon else ''}
    <div class="dm-text">
      <p class="dm-lead">You drift into sleep like a</p>
      <div class="dm-key">{prof_name}</div>
      <p class="dm-desc">{prof_desc_ext}</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ==============
# Helper (formatting)
# ==============
def _fmt(v, nd=3):
    if v is None: return "NA"
    try:
        if np.isnan(v): return "NA"
    except TypeError:
        pass
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

# ==============
# Five-Dimension Bars (One-sided amplification)
# ==============
DIM_BAR_CONFIG = {
    "Vivid": {
        "freq_keys": ["freq_percept_intense", "freq_percept_precise", "freq_percept_real"],
        "weight_keys": ["degreequest_vividness"],
        "invert_keys": [],
        "weight_mode": "standard",
        "help": "Dull  ↔  Vivid",
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
        "freq_keys": ["freq_positive", "freq_negative", "freq_ruminate"],
        "weight_keys": ["degreequest_emotionality"],  # 1=very negative, 6=very positive
        "invert_keys": ["freq_negative", "freq_ruminate"],
        "weight_mode": "emotion_bipolar",
        "help": "Negative  ↔  Positive",
    },
}

def _get(record, key, default=np.nan): return record.get(key, default)

def _norm16(x):
    try: v = float(x)
    except: return np.nan
    if np.isnan(v): return np.nan
    return np.clip((v - 1.0) / 5.0, 0.0, 1.0)

def _mean_ignore_nan(arr):
    arr = [a for a in arr if not (isinstance(a, float) and np.isnan(a))]
    return np.nan if len(arr) == 0 else float(np.mean(arr))

def _weight_boost(wvals, mode: str):
    w = _mean_ignore_nan([_norm16(v) for v in wvals])
    if isinstance(w, float) and np.isnan(w): w = 0.5
    if mode == "emotion_bipolar":
        intensity = 2.0 * abs(w - 0.5)       # 0..1
        boost = max(0.0, intensity - 0.5)    # 0..0.5
    else:
        boost = max(0.0, w - 0.5)            # 0..0.5
    return float(np.clip(boost, 0.0, 0.5))

def compute_dimension_score(record, cfg, k_bump=0.8):
    vals = []
    for k in cfg["freq_keys"]:
        v = _norm16(_get(record, k))
        if k in cfg.get("invert_keys", []):
            v = 1.0 - v if not (isinstance(v, float) and np.isnan(v)) else v
        vals.append(v)
    base = _mean_ignore_nan(vals)
    if isinstance(base, float) and np.isnan(base): return np.nan

    boost = _weight_boost([_get(record, wk) for wk in cfg["weight_keys"]],
                          cfg.get("weight_mode", "standard"))
    bump = k_bump * boost * base * (1.0 - base)
    return float(np.clip(base + bump, 0.0, 1.0))

# Load population once (for bars + later plots)
try:
    pop_data = pd.read_csv(ASSETS_CSV)
except Exception as e:
    st.error(f"Could not load population data at {ASSETS_CSV}: {e}")
    pop_data = None

bars = []
for name, cfg in DIM_BAR_CONFIG.items():
    s01 = compute_dimension_score(record, cfg, k_bump=0.8)
    s100 = None if (isinstance(s01, float) and np.isnan(s01)) else float(s01 * 100.0)
    bars.append({"name": name, "help": cfg["help"], "score": s100})

def compute_population_distributions(df: pd.DataFrame, dim_config: dict, k_bump=0.8):
    if df is None or df.empty: return {}
    dist = {name: [] for name in dim_config.keys()}
    for _, row in df.iterrows():
        rec = row.to_dict()
        for nm, cfg in dim_config.items():
            s = compute_dimension_score(rec, cfg, k_bump=k_bump)
            if not (isinstance(s, float) and np.isnan(s)):
                dist[nm].append(float(np.clip(s * 100.0, 0.0, 100.0)))
    for nm in dist:
        arr = np.array(dist[nm], dtype=float)
        dist[nm] = arr[~np.isnan(arr)]
    return dist

pop_dists = compute_population_distributions(pop_data, DIM_BAR_CONFIG, k_bump=0.8)
pop_medians = {k: (float(np.nanmedian(v)) if (isinstance(v, np.ndarray) and v.size) else None)
               for k, v in pop_dists.items()}

# Render bars (unchanged visuals)
st.markdown("<div class='dm2-outer'><div class='dm2-bars'>", unsafe_allow_html=True)

def _clamp_pct(p, lo=2.0, hi=98.0):
    try: p = float(p)
    except: return lo
    return max(lo, min(hi, p))

min_fill = 2  # minimal % fill for aesthetic continuity

for b in bars:
    name = b["name"]
    help_txt = b["help"]
    score = b["score"]
    median = pop_medians.get(name, None)

    if score is None or (isinstance(score, float) and np.isnan(score)):
        width = min_fill; score_txt = "NA"; width_clamped = _clamp_pct(width)
    else:
        width = int(round(np.clip(score, 0, 100))); width = max(width, min_fill)
        score_txt = f"{int(round(score))}%"; width_clamped = _clamp_pct(width)

    if median is None or (isinstance(median, float) and np.isnan(median)):
        med_left = None; med_left_clamped = None
    else:
        med_left = float(np.clip(median, 0, 100)); med_left_clamped = _clamp_pct(med_left)

    if isinstance(help_txt, str) and "↔" in help_txt:
        left_anchor, right_anchor = [s.strip() for s in help_txt.split("↔", 1)]
    else:
        left_anchor, right_anchor = "0", "100"

    median_html = "" if med_left is None else f"<div class='dm2-median' style='left:{med_left}%;'></div>"

    is_perception = (name.lower() in ("perception", "vivid"))
    mediantag_html = ""
    if is_perception and (med_left_clamped is not None):
        overlap = (score_txt != "NA" and abs(width_clamped - med_left_clamped) <= 6.0)
        mediantag_class = "dm2-mediantag below" if overlap else "dm2-mediantag"
        mediantag_html = f"<div class='{mediantag_class}' style='left:{med_left_clamped}%;'>world average</div>"

    scoretag_html = "" if score_txt == "NA" else f"<div class='dm2-scoretag' style='left:{width_clamped}%;'>{score_txt}</div>"

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




# === Build exportable HTML mirror of the title + icon/text + 5 bars ===
# Title
export_title_html = """
<div class="dm-center">
  <div class="dm-title">DRIFTING MINDS STUDY</div>
</div>
"""

# Header (icon + text)
export_icon_html = f'<img class="dm-icon" src="{icon_src}" alt="profile icon"/>' if has_icon else ""

export_header_html = f"""
<div class="dm-center">
  <div class="dm-row">
    {export_icon_html}
    <div class="dm-text">
      <p class="dm-lead">You drift into sleep like a</p>
      <div class="dm-key">{prof_name}</div>
      <p class="dm-desc">{prof_desc_ext or "&nbsp;"}</p>
    </div>
  </div>
</div>
"""


# Bars (same values you computed above)
export_bars_html = ["<div class='dm2-outer'><div class='dm2-bars'>"]
for b in bars:
    name = b["name"]; help_txt = b["help"]; score = b["score"]
    median = pop_medians.get(name, None)

    if score is None or (isinstance(score, float) and np.isnan(score)):
        width = 2; score_txt = "NA"; width_clamped = 2.0
    else:
        width = int(round(np.clip(score, 0, 100))); width = max(width, 2)
        score_txt = f"{int(round(score))}%"
        width_clamped = max(2.0, min(98.0, float(width)))

    if median is None or (isinstance(median, float) and np.isnan(median)):
        median_html = ""; med_left_clamped = None
    else:
        med_left = float(np.clip(median, 0, 100))
        med_left_clamped = max(2.0, min(98.0, med_left))
        median_html = f"<div class='dm2-median' style='left:{med_left_clamped}%;'></div>"

    if isinstance(help_txt, str) and "↔" in help_txt:
        left_anchor, right_anchor = [s.strip() for s in help_txt.split("↔", 1)]
    else:
        left_anchor, right_anchor = "0", "100"

    # 'world' tag logic for Vivid (keep identical to page)
    mediantag_html = ""
    if name.lower() in ("perception", "vivid") and (med_left_clamped is not None):
        overlap = (score_txt != "NA" and abs(width_clamped - med_left_clamped) <= 6.0)
        mediantag_class = "dm2-mediantag below" if overlap else "dm2-mediantag"
        mediantag_html = f"<div class='{mediantag_class}' style='left:{med_left_clamped}%;'>world average</div>"

    scoretag_html = "" if score_txt == "NA" else f"<div class='dm2-scoretag' style='left:{width_clamped}%;'>{score_txt}</div>"

    export_bars_html.append(
        "<div class='dm2-row'>"
          "<div class='dm2-left'>"
            f"<div class='dm2-label'>{name}</div>"
          "</div>"
          "<div class='dm2-wrap'>"
            f"<div class='dm2-track' aria-label='{name} score {score_txt}'>"
              f"<div class='dm2-fill' style='width:{width}%;'></div>"
              f"{median_html}{mediantag_html}{scoretag_html}"
            "</div>"
            "<div class='dm2-anchors'>"
              f"<span>{left_anchor}</span><span>{right_anchor}</span>"
            "</div>"
          "</div>"
        "</div>"
    )
export_bars_html.append("</div></div>")
export_bars_html = "\n".join(export_bars_html)

# Full HTML we will snapshot inside the component
DM_SHARE_HTML = f"""
<div style='position:relative;'>
  <div style='position:absolute; top:5px; right:0px; text-align:center; font-size:0.9rem; color:#000; line-height:1.05;'>
    <img src="{qr_src}" width="88" style="display:block; margin:0 auto 1px auto;" />
    <div style="font-weight:600; font-size:0.6rem; margin:0;">Participate!</div>
    <div style="font-size:0.6rem; margin-top:3px;">redcap.link/DriftingMinds</div>
  </div>
  {export_title_html}
  {export_header_html}
  {export_bars_html}
</div>
"""


# Subset of your CSS needed for the mirror (copied from your big CSS)
DM_SHARE_CSS = r"""
<style>
/* Match Streamlit's native font (Inter fallback stack) */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

body, .dm-center, .dm-row, .dm-text, .dm-title, .dm-lead, .dm-key, .dm-desc,
.dm2-label, .dm2-anchors, .dm2-scoretag, .dm2-mediantag {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
:root { --dm-max: 820px; }
.dm-center { max-width: var(--dm-max); margin: 0 auto; }
.dm-title { font-size: 2.5rem; font-weight: 200; margin: 0 0 1.25rem 0; text-align: center; }
.dm-row { display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: .25rem; }
.dm-icon { width: 140px; height: auto; flex: 0 0 auto; transform: translateX(6rem); }
.dm-text { flex: 1 1 0; min-width: 0; padding-left: 6rem; }
.dm-lead { font-weight: 400; font-size: 1rem; color: #666; margin: 0 0 8px 0; letter-spacing: .3px; font-style: italic; text-align: left; }
.dm-key { font-weight: 600; font-size: clamp(28px, 5vw, 60px); line-height: 1.05; margin: 0 0 10px 0; color: #7A5CFA; text-align: left; }
.dm-desc { color: #111; font-size: 1.05rem; line-height: 1.55; margin: 0; max-width: 680px; font-weight: 400; text-align: left; }

.dm2-outer { margin-left: -70px !important; width: 100%; }
.dm2-bars { margin-top: 16px; display: flex; flex-direction: column; align-items: flex-start; width: 100%; text-align: left; }
.dm2-row { display: grid; grid-template-columns: 160px 1fr; column-gap: 8px; align-items: center; margin: 10px 0; }
.dm2-left { display:flex; align-items:center; gap:0px; width:160px; flex:0 0 160px; }
.dm2-label { font-weight: 800; font-size: 1.10rem; line-height: 1.05; white-space: nowrap; letter-spacing: .1px; position: relative; top: -3px; text-align: right; width: 100%; padding-right: 40px; margin: 0; }
.dm2-wrap { flex: 1 1 auto; display:flex; flex-direction:column; gap:4px; margin-left: -12px; }
.dm2-track { position: relative; width: 100%; height: 14px; background: #EDEDED; border-radius: 999px; overflow: visible; }
.dm2-fill { height: 100%; background: linear-gradient(90deg, #CBBEFF 0%, #A18BFF 60%, #7B61FF 100%); border-radius: 999px; }
.dm2-median { position: absolute; top: 50%; transform: translate(-50%, -50%); width: 8px; height: 8px; background: #000; border: 1.5px solid #FFF; border-radius: 50%; pointer-events: none; box-sizing: border-box; }
.dm2-mediantag { position: absolute; bottom: calc(100% + 2px); transform: translateX(-50%); font-size: .82rem; font-weight: 600; color: #000; white-space: nowrap; line-height: 1.05; }
.dm2-mediantag.below { bottom: auto; top: calc(100% + 2px); }
.dm2-scoretag { position: absolute; bottom: calc(100% + 2px); transform: translateX(-50%); font-size: .86rem; font-weight: 500; color: #7B61FF; white-space: nowrap; line-height: 1.05; }
.dm2-scoretag.below { bottom: auto; top: calc(100% + 2px); }
.dm2-anchors { display:flex; justify-content:space-between; font-size: .85rem; color:#666; margin-top: 0; line-height: 1; }

/* ===== Export-only overrides: keep the mirror aligned and full-width ===== */
#export-root { 
  width: 820px;                 
  /* Add top/right/bottom padding for breathing room */
  padding: 40px 60px 30px 60px;   /* top 40, right 50, bottom 40, left 70 */
  box-sizing: border-box;
  background: #ffffff;
  border-radius: 8px;             /* optional – gives a softer PNG edge */
}

/* Remove layout nudges that were useful on-page but distort the export */
#export-root .dm2-outer { 
  margin-left: 0 !important;
  margin-top: 36px;          /* ⟵ adds breathing room between profile text and bars */
}
#export-root .dm2-wrap  { margin-left: 0; }

/* Ensure the grid allocates full horizontal space for the bar column */
#export-root .dm2-row {
  grid-template-columns: 160px auto !important; /* labels fixed, bars expand */
  width: 100%;
}

/* Force the bar itself to use all available width within its row */
#export-root .dm2-track {
  width: 100% !important;
}

/* Keep header row tidy and aligned */
#export-root .dm-row  { justify-content: flex-start; gap: 16px; }
#export-root .dm-icon { transform: none; margin: 0; }
#export-root .dm-text { padding-left: 0; }

/* Spacing tweaks for readability */
#export-root .dm-title {
  letter-spacing: 0.2px;
  margin-bottom: 30px;            /* ⟵ more space below title */
}
#export-root .dm-lead {
  margin-top: 18px;               /* ⟵ more gap below the title line */
}
#export-root .dm-desc  { max-width: 680px; }


              
</style>
"""

import streamlit.components.v1 as components

# --- Note (left) + Download button (right) -----------------------------------
left_note, right_btn = st.columns([7, 3], gap="small")

with left_note:
    st.markdown(
        """
        <div style="
            max-width:720px;
            margin:14px 0 0 0;
            text-align:justify;
            text-justify:inter-word;
            font-size:0.82rem;
            color:#444;
            line-height:1.2;
        ">
          <p style="margin:0;">
            <strong>Vivid</strong>: brightness or contrast of your imagery, or the loudness of what you hear. 
            <strong>Bizarre</strong>: how unusual or unrealistic the content feels. 
            <strong>Immersive</strong>: how deeply absorbed you are in your mental content. 
            <strong>Spontaneous</strong>: how much the content comes to you on its own, without deliberate control. 
            <strong>Emotional</strong>: how strongly you felt emotions.<br>
            <span style="font-size:0.6rem;">⚫️</span>    = "world average”: represents the average scores from 1,000 people worldwide.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with right_btn:
    # keep the button in its own iframe so JS works; align it to the right
    components.html(
    f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
{DM_SHARE_CSS}
<style>
  body {{ margin:0; background:#fff; }}
  .wrap {{
    display:flex; justify-content:flex-end; align-items:flex-start;
    gap:8px; padding-top:14px;
  }}
  .bar {{
    display:inline-block; padding:9px 14px; border:none; border-radius:8px;
    font-size:14px; cursor:pointer; background:#000; color:#fff;
    box-shadow:0 2px 6px rgba(0,0,0,.08); transition:background .2s ease;
  }}
  .bar:hover {{ background:#222; }}
  .bar:active {{ background:#444; }}
  /* Hidden export mirror */
  #export-root {{ position:fixed; left:-10000px; top:0; width:820px; background:#fff; }}
</style>
</head>
<body data-rec="{record_id or ''}">
  <div class="wrap">
    <button id="dmshot" class="bar">⬇️ Download</button>
    <button id="copylink" class="bar">🔗 Copy link</button>
  </div>

  <div id="export-root">{DM_SHARE_HTML}</div>

  <script src="https://cdn.jsdelivr.net/npm/dom-to-image-more@3.4.0/dist/dom-to-image-more.min.js"></script>
  <script>
  (function() {{
    const dlBtn   = document.getElementById('dmshot');
    const copyBtn = document.getElementById('copylink');
    const root    = document.getElementById('export-root');
    const recId   = document.body.dataset.rec || '';

    // Build a share URL from the PARENT page (not the iframe)
    function buildShareUrl() {{
      let href = '';
      try {{
        // Prefer parent (Streamlit host page)
        href = (window.parent && window.parent.location) ? window.parent.location.href : window.location.href;
      }} catch(e) {{
        // Cross-origin fallback: iframe URL
        href = window.location.href;
      }}
      try {{
        const u = new URL(href);
        if (recId) u.searchParams.set('id', recId);
        return u.toString();
      }} catch(e) {{
        return href; // very old browsers fallback
      }}
    }}

    // --- COPY LINK ---
    copyBtn.addEventListener('click', async () => {{
      const share = buildShareUrl();
      try {{
        await navigator.clipboard.writeText(share);
        copyBtn.textContent = '✅ Copied!';
        setTimeout(() => copyBtn.textContent = '🔗 Copy link', 1500);
      }} catch (e) {{
        // Fallback (selection trick)
        const ta = document.createElement('textarea');
        ta.value = share;
        ta.style.position='fixed'; ta.style.left='-9999px';
        document.body.appendChild(ta); ta.select();
        try {{ document.execCommand('copy'); }} catch(_) {{}}
        ta.remove();
      }}
    }});

    // --- DOWNLOAD PNG ---
    async function downloadBlob(blob, name) {{
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = name;
      document.body.appendChild(a); a.click();
      setTimeout(() => {{ URL.revokeObjectURL(url); a.remove(); }}, 400);
    }}

    async function capture() {{
      try {{
        await new Promise(r => requestAnimationFrame(r));
        const rect = root.getBoundingClientRect();
        const w = Math.ceil(rect.width), h = Math.ceil(rect.height);
        const blob = await window.domtoimage.toBlob(root, {{
          width: w, height: h, bgcolor: '#ffffff', quality: 1, cacheBust: true
        }});
        if (!blob || !blob.size) throw new Error('empty blob');
        await downloadBlob(blob, 'drifting_minds_profile.png');
      }} catch (e) {{
        console.error(e);
        alert('Capture failed. Try refreshing the page or a different browser.');
      }}
    }}
    dlBtn.addEventListener('click', capture);
  }})();
  </script>
</body>
</html>
    """,
    height=70
)




          
          
          
# Lock-in axes rectangles (left, bottom, width, height) so x-axes align perfectly
AX_POS_YOU   = [0.14, 0.24, 0.82, 0.66]   # used by You: imagery/creativity/anxiety
AX_POS_SLEEP = [0.14, 0.24, 0.82, 0.66]   # used by Your sleep: latency/duration



# ==============
# "You" — Imagery · Creativity · Anxiety (final alignment + clean title line)
# ==============

# --- Centered title for "You" (thinner black line) -----------------
st.markdown(
    """
    <div class="dm-center" style="max-width:960px; margin:18px auto 10px;">
      <div style="display:flex; align-items:center; gap:18px;">
        <div style="height:1px; background:#000; flex:1;"></div>
        <div style="flex:0; font-weight:600; font-size:1.35rem; letter-spacing:0.2px;">YOU</div>
        <div style="height:1px; background:#000; flex:1;"></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Color setup -------------------------------------------------------------
_HL = None
for name in ["PURPLE_HEX", "DM_PURPLE", "SLEEP_COLOR", "COLOR_PURPLE", "ACCENT_PURPLE"]:
    if name in globals():
        _HL = globals()[name]
        break
if not _HL:
    _HL = "#6F45FF"

def _hex_to_rgb_tuple(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
HL_RGB = _hex_to_rgb_tuple(_HL)

# --- Helpers -----------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def _mini_hist(ax, counts, edges, highlight_idx, title, bar_width_factor=0.95):
    centers = 0.5 * (edges[:-1] + edges[1:])
    width   = (edges[1] - edges[0]) * bar_width_factor

    # population bars
    ax.bar(centers, counts, width=width, color="#D9D9D9", edgecolor="white", align="center")
    # participant bin
    if 0 <= highlight_idx < len(counts):
        ax.bar(centers[highlight_idx], counts[highlight_idx], width=width,
               color=HL_RGB, edgecolor="white", align="center")

    # title
    ax.set_title(title, fontsize=8, pad=6, color="#222222")

    # x-axis baseline and labels
    ax.spines["bottom"].set_linewidth(0.3)
    ax.set_xlabel("")
    ax.set_xticks([])

    # remove y
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.margins(y=0)


def _col_values(df, colname):
    if df is None or df.empty or (colname not in df.columns):
        return np.array([])
    return pd.to_numeric(df[colname], errors="coerce").to_numpy()

def _participant_value(rec, key):
    try:
        return float(str(rec.get(key, np.nan)).strip())
    except Exception:
        return np.nan

# --- Data prep ---------------------------------------------------------------

# 1) Imagery (VVIQ)
try:
    vviq_score
except NameError:
    VVIQ_FIELDS = [
        "quest_a1","quest_a2","quest_a3","quest_a4",
        "quest_b1","quest_b2","quest_b3","quest_b4",
        "quest_c1","quest_c2","quest_c3","quest_c4",
        "quest_d1","quest_d2","quest_d3","quest_d4"
    ]
    vviq_vals  = [float(record.get(k, np.nan)) if pd.notna(record.get(k, np.nan)) else np.nan for k in VVIQ_FIELDS]
    vviq_score = sum(v for v in vviq_vals if np.isfinite(v))

N = 8000; mu, sigma = 61.0, 9.2; low, high = 30, 80
a, b = (low - mu) / sigma, (high - mu) / sigma
vviq_samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=N, random_state=42)
vviq_edges   = np.linspace(low, high, 26)
vviq_counts, _ = np.histogram(vviq_samples, bins=vviq_edges, density=True)
vviq_hidx = int(np.clip(np.digitize(vviq_score, vviq_edges) - 1, 0, len(vviq_counts)-1))


# 2) Creativity 1–6
cre_vals  = _col_values(pop_data, "creativity_trait")
cre_edges = np.arange(0.5, 6.5 + 1.0, 1.0)
cre_counts, _ = np.histogram(cre_vals, bins=cre_edges, density=True) if cre_vals.size else (np.array([]), cre_edges)
cre_part  = _participant_value(record, "creativity_trait")
cre_hidx  = int(np.clip(np.digitize(cre_part, cre_edges) - 1, 0, len(cre_counts)-1)) if cre_counts.size else 0

# 3) Anxiety 1–100, 5-point bins
anx_vals  = _col_values(pop_data, "anxiety")
anx_edges = np.arange(0.5, 100.5 + 5, 5)
anx_counts, _ = np.histogram(anx_vals, bins=anx_edges, density=True) if anx_vals.size else (np.array([]), anx_edges)
anx_part  = _participant_value(record, "anxiety")
anx_hidx  = int(np.clip(np.digitize(anx_part, anx_edges) - 1, 0, len(anx_counts)-1)) if anx_counts.size else 0

# --- Display side-by-side ----------------------------------------------------
# imagery slightly taller before; now tuned so x-axes align exactly
FIGSIZE_IMAGERY  = (2.4, 2.60)
FIGSIZE_STANDARD = (2.4, 2.60)

c1, c2, c3 = st.columns(3, gap="small")

with c1:
    fig, ax = plt.subplots(figsize=FIGSIZE_IMAGERY)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # --- Rebuild histogram so x starts at 16 ---------------------------------
    vviq_edges = np.linspace(16, 80, 22)  # 16 → 80
    vviq_counts, _ = np.histogram(vviq_samples, bins=vviq_edges, density=True)
    vviq_hidx = int(np.clip(np.digitize(vviq_score, vviq_edges) - 1, 0, len(vviq_counts) - 1))

    # --- Plot imagery histogram ----------------------------------------------
    _mini_hist(ax, vviq_counts, vviq_edges, vviq_hidx,
               f"Your visual imagery at wake: {int(round(vviq_score))}")

    # --- Custom x-axis labels -------------------------------------------------
    ax.text(0.00, -0.05, "low (16)",   transform=ax.transAxes,
            ha="left",  va="top", fontsize=7.5)
    ax.text(1.00, -0.05, "high (80)", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5)
    
    # --- Optional vertical marker for very low imagery (<30) -----------------
    if vviq_score < 35:
        x_line = vviq_score
        # short vertical segment (20% of current y max)
        y_max = ax.get_ylim()[1]
        ax.vlines(x_line, 0, y_max * 0.2, color=PURPLE_HEX, lw=1.2)
    

    # --- In-axes minimalist legend (left, mid-height) ------------------------
    x0 = 0.02
    y_top = 0.63
    y_gap = 0.085
    box_size = 0.038

    # you (purple square)
    ax.add_patch(plt.Rectangle((x0, y_top - box_size / 2),
                               box_size, box_size,
                               transform=ax.transAxes,
                               color=PURPLE_HEX, lw=0))
    ax.text(x0 + 0.05, y_top, "you",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7.5, color=PURPLE_HEX)

    # world (gray square)
    ax.add_patch(plt.Rectangle((x0, y_top - y_gap - box_size / 2),
                               box_size, box_size,
                               transform=ax.transAxes,
                               color="#D9D9D9", lw=0))
    ax.text(x0 + 0.05, y_top - y_gap, "world",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7.5, color="#444444")

    # maintain alignment
    ax.set_position(AX_POS_YOU)
    st.pyplot(fig, use_container_width=False)





with c2:
    if not cre_counts.size:
        st.info("Population data for creativity unavailable.")
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        _mini_hist(ax, cre_counts, cre_edges, cre_hidx,
           f"Your level of creativity: {int(round(cre_part))}")
        # Replace default x-labels
        ax.text(0.0, -0.05, "low (1)",  transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5)
        ax.text(1.0, -0.05, "high (6)", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5)
        ax.set_position(AX_POS_YOU)  # ← lock baseline
        st.pyplot(fig, use_container_width=False)


with c3:
    if not anx_counts.size:
        st.info("Population data for anxiety unavailable.")
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        _mini_hist(ax, anx_counts, anx_edges, anx_hidx,
           f"Your level of anxiety: {int(round(anx_part))}")
        # Replace default x-labels
        ax.text(0.0, -0.05, "low (1)",  transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5)
        ax.text(1.0, -0.05, "high (100)", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5)
        ax.set_position(AX_POS_YOU)  # ← lock baseline
        st.pyplot(fig, use_container_width=False)

# --- Explanatory note below the three histograms ----------------------------
st.markdown(
    """
    <div style="
        max-width:820px;
        margin:14px 0 0 0;   /* no horizontal centering, flush left */
        text-align:left;
        font-size:0.82rem;
        color:#444;
        line-height:1.45;
    ">
      <em>
        Grey info ("world”) represents data from 1,000 people worldwide.<br>
        Left graph: Vividness of visual imagery when awake (VVIQ score).  Middle and right: self-rated creativity and anxiety over the past year.
      </em>
    </div>
    """,
    unsafe_allow_html=True
)




# ==============
# "Your sleep" — Latency · Duration · Chronotype & Dream recall (final visual alignment)
# ==============

# --- Centered title for "Your sleep" (thinner black line, one line text) -----
st.markdown(
    """
    <div class="dm-center" style="max-width:1020px; margin:28px auto 16px;">
      <div style="display:flex; align-items:center; gap:20px;">
        <div style="height:1px; background:#000; flex:0.5;"></div>
        <div style="flex:0; font-weight: 600; font-size:1.35rem; letter-spacing:0.2px; white-space:nowrap;">
          YOUR SLEEP
        </div>
        <div style="height:1px; background:#000; flex:0.5;"></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Three-column layout -----------------------------------------------------
col_left, col_mid, col_right = st.columns(3, gap="small")

from scipy.stats import gaussian_kde

# =============================================================================
# LEFT: Sleep latency KDE (capped line height)
# =============================================================================
with col_left:
    if pop_data is None or pop_data.empty:
        st.info("Population data unavailable.")
    else:
        lat_cols = [c for c in pop_data.columns if "sleep_latency" in c.lower()]
        if not lat_cols:
            st.info("No sleep latency column in population data.")
        else:
            lat_col = lat_cols[0]
            raw = pd.to_numeric(pop_data[lat_col], errors="coerce").dropna()
            if raw.empty:
                st.info("No valid population sleep-latency values.")
            else:
                samples = np.clip(raw.values * CAP_MIN if raw.max() <= 1.5 else raw.values, 0, CAP_MIN)
                raw_sl = _get_first(record, ["sleep_latency"])
                sl_norm = norm_latency_auto(raw_sl, cap_minutes=CAP_MIN)
                if np.isnan(sl_norm):
                    st.info("No sleep-latency value available for this participant.")
                else:
                    try:
                        part_raw_minutes = float(str(raw_sl).strip())
                    except Exception:
                        part_raw_minutes = np.nan
                    part_display = sl_norm * CAP_MIN if sl_norm <= 1.5 else float(sl_norm)
                    part_display = float(np.clip(part_display, 0, CAP_MIN))
                    rounded_raw = int(round(part_raw_minutes)) if np.isfinite(part_raw_minutes) else int(round(part_display))

                    kde = gaussian_kde(samples, bw_method="scott")
                    xs = np.linspace(0, CAP_MIN, 400)
                    ys = kde(xs)

                    from matplotlib.ticker import MaxNLocator
                    with plt.rc_context({
                        "axes.facecolor": "none",
                        "axes.edgecolor": "#000000",
                        "axes.linewidth": 0.3,
                        "xtick.color": "#333333",
                        "ytick.color": "#333333",
                        "font.size": 7.5,
                    }):
                        fig, ax = plt.subplots(figsize=(2.2, 2.52))
                        fig.patch.set_alpha(0.0)
                        ax.set_facecolor("none")
                        
                        # Match typography style of latency plot
                        ax.tick_params(axis="x", labelsize=7.5, labelcolor="#333333")
                        for label in ax.get_xticklabels():
                            label.set_fontweight("regular")    # ensure non-bold tick labels
                        ax.set_xlabel("hours", fontsize=7.5, color="#333333", fontweight="regular")

                        # KDE area
                        ax.fill_between(xs, ys, color="#e6e6e6", linewidth=0)

                        # Participant marker (line capped to KDE height)
                        y_part = float(kde(part_display))
                        ax.vlines(part_display, 0, y_part, lw=0.8, color="#222222")
                        ax.scatter([part_display], [y_part], s=28, zorder=3,
                                   color=PURPLE_HEX, edgecolors="none")

                        # Titles & labels
                        ax.set_title(f"You fall asleep in {rounded_raw} minutes", fontsize=8, pad=6, color="#222222")
                        ax.set_xlabel("minutes", fontsize=7.5, color="#333333")

                        # Remove y-axis
                        ax.set_ylabel("")
                        ax.get_yaxis().set_visible(False)
                        for side in ("left", "right", "top"):
                            ax.spines[side].set_visible(False)
                            ax.spines["bottom"].set_linewidth(0.3)   # thinner x-axis

                            
                        # --- Add legend (right side, mid-height) ---------------------------------
                        x0 = 0.72     # further to the right inside axes (0–1 in Axes coords)
                        y_top = 0.73  # vertical position for first label
                        y_gap = 0.085
                        size = 0.038  # symbol size (same scale as imagery legend)
                        
                        # "you" — purple circle
                        circle = plt.Circle((x0 + size/2, y_top), size/2,
                                            transform=ax.transAxes, color=PURPLE_HEX, lw=0)
                        ax.add_patch(circle)
                        ax.text(x0 + 0.05, y_top, "you", transform=ax.transAxes,
                                ha="left", va="center", fontsize=7.5, color=PURPLE_HEX)
                        
                        # "world" — gray square below
                        ax.add_patch(plt.Rectangle((x0, y_top - y_gap - size / 2),
                                                   size, size,
                                                   transform=ax.transAxes,
                                                   color="#D9D9D9", lw=0))
                        ax.text(x0 + 0.05, y_top - y_gap, "world",
                                transform=ax.transAxes, ha="left", va="center",
                                fontsize=7.5, color="#444444")


                        xticks = np.linspace(0, CAP_MIN, 7)
                        ax.set_xticks(xticks)
                        xlabels = [str(int(t)) if t < CAP_MIN else "60+" for t in xticks]
                        ax.set_xticklabels(xlabels)
                        ax.tick_params(axis="x", labelsize=8, color="#333333")
                        plt.tight_layout()
                        plt.tight_layout()
                        ax.set_position(AX_POS_SLEEP)  # ← lock baseline
                        st.pyplot(fig, use_container_width=False)

# =============================================================================
# MIDDLE: Sleep duration histogram (perfectly aligned baseline)
# =============================================================================
with col_mid:
    if pop_data is None or pop_data.empty:
        st.info("Population data unavailable.")
    else:
        dur_cols = [c for c in pop_data.columns if c.lower() in (
            "sleep_duration", "sleep_duration_h", "sleep_duration_hours", "total_sleep_time_h"
        )]
        if not dur_cols:
            st.warning("No sleep duration column found in population data.")
        else:
            col = dur_cols[0]

            def _to_hours_for_plot(x):
                if x is None: return np.nan
                s = str(x).strip()
                if s == "": return np.nan
                if s.endswith("+"):
                    try: return float(s[:-1])
                    except: return 12.0
                try: return float(s)
                except: return np.nan

            raw_series = pop_data[col].apply(_to_hours_for_plot)
            samples_h = raw_series.astype(float).to_numpy()
            samples_h = samples_h[np.isfinite(samples_h)]
            samples_h = np.clip(samples_h, 1.0, 12.0)

            if samples_h.size == 0:
                st.info("No valid sleep duration values in population data.")
            else:
                dur_raw = _get_first(record, [
                    "sleep_duration", "sleep_duration_h", "sleep_duration_hours", "total_sleep_time_h"
                ])
                dur_raw_str = str(dur_raw).strip() if dur_raw is not None else ""
                try:
                    if dur_raw_str.endswith("+"):
                        part_hours_plot = float(dur_raw_str[:-1])
                        title_str = f"You sleep on average {dur_raw_str} hours"
                    else:
                        part_hours_plot = float(dur_raw_str)
                        title_str = f"You sleep {int(round(part_hours_plot))} hours on average"
                except:
                    part_hours_plot = float(np.nanmedian(samples_h))
                    title_str = "Your sleep duration"

                part_hours_plot = float(np.clip(part_hours_plot, 1.0, 12.0))
                edges = np.arange(0.5, 12.5 + 1.0, 1.0)
                counts, _ = np.histogram(samples_h, bins=edges, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                highlight_idx = np.digitize(part_hours_plot, edges) - 1
                highlight_idx = np.clip(highlight_idx, 0, len(counts) - 1)

                # ✅ Slightly adjusted figure height (2.52) for perfect x-axis alignment
                fig, ax = plt.subplots(figsize=(2.2, 2.52))
                fig.patch.set_alpha(0)
                ax.set_facecolor("none")
                ax.bar(centers, counts, width=edges[1]-edges[0],
                       color="#D9D9D9", edgecolor="white", align="center")
                ax.bar(centers[highlight_idx], counts[highlight_idx],
                       width=edges[1]-edges[0], color=PURPLE_HEX,
                       edgecolor="white", align="center")
                ax.set_title(title_str, fontsize=8, pad=6, color="#222222")
                ax.set_xlabel("hours", fontsize=7.5)

                # Remove y-axis
                ax.set_ylabel("")
                ax.get_yaxis().set_visible(False)
                for side in ("left", "right", "top"):
                    ax.spines[side].set_visible(False)
                    ax.spines["bottom"].set_linewidth(0.3)   # thinner x-axis


                ticks = np.arange(1, 13, 1)
                ax.set_xticks(ticks)
                labels = ["" for _ in ticks]
                for i in range(4, 11):
                    labels[i-1] = str(i)
                ax.set_xticklabels(labels)
                ax.tick_params(axis="x", labelsize=7.5)
                for label in ax.get_xticklabels():
                    label.set_fontweight("normal")  # ensure not bold`
                    label.set_fontfamily("Inter")  # force same font as rest of UI

                plt.tight_layout()
                ax.set_position(AX_POS_SLEEP)  # ← lock baseline
                st.pyplot(fig, use_container_width=False)



# =============================================================================
# RIGHT: Chronotype + Dream recall (pushed further down)
# =============================================================================
with col_right:
    raw_chrono = record.get("chronotype", None)
    raw_recall = record.get("dream_recall", None)

    def _to_int(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    chronotype_val = _to_int(raw_chrono)
    dreamrec_val   = _to_int(raw_recall)

    CHRONO_LBL = {
        1: "Morning type",
        2: "Evening type",
        3: "No preference",
        4: "None",
    }
    DREAMRECALL_LBL = {
        1: "Less than once a month",
        2: "1 to 2 times a month",
        3: "Once a week",
        4: "Several times a week",
        5: "Every day",
    }

    chrono_txt = CHRONO_LBL.get(chronotype_val, "—")
    recall_txt = DREAMRECALL_LBL.get(dreamrec_val, "—")

    # Larger spacer (really more down)
    st.markdown("<div style='height:90px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="dm-center" style="max-width:320px; margin:0 auto;">
          <div style="
              border:1px solid rgba(0,0,0,0.15);
              border-radius:12px;
              padding:16px 18px;
              background:rgba(255,255,255,0.02);
            ">
            <div style="font-size:0.95rem; line-height:1.6; color:#111;">
              <div><span style="opacity:0.8;">Chronotype:</span> <strong>{chrono_txt}</strong></div>
              <div><span style="opacity:0.8;">Dream recall:</span> <strong>{recall_txt}</strong></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )



# ==============
# Your experience — Section header + 3-column layout (image left, radar middle)
# ==============
st.markdown(
    """
    <div class="dm-center" style="max-width:1020px; margin:28px auto 32px;">
      <div style="display:flex; align-items:center; gap:24px;">
        <div style="height:1px; background:#000; flex:1;"></div>
        <div style="flex:0; font-weight:600; font-size:1.35rem; letter-spacing:0.2px; white-space:nowrap;">
          YOUR EXPERIENCE
        </div>
        <div style="height:1px; background:#000; flex:1;"></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# 3-column scaffold
exp_left, exp_mid, exp_right = st.columns(3, gap="small")

# ------------
# LEFT: Trajectory image (trajectories-01..04.png based on record["trajectories"])
# ------------
def _safe_int(val):
    try:
        return int(str(val).strip())
    except Exception:
        return None

traj_val = _safe_int(record.get("trajectories"))
traj_map = {
    1: "trajectories-01.png",
    2: "trajectories-02.png",
    3: "trajectories-03.png",
    4: "trajectories-04.png",
}
img_name = traj_map.get(traj_val)

with exp_left:
    # mini-title to match histo titles
    st.markdown("<div style='font-size:18px; color:#222; text-align:center; margin:2px 0 6px 0;'>Your trajectory</div>", unsafe_allow_html=True)

    if img_name:
        img_path = os.path.join("assets", img_name)
        try:
            st.image(img_path, use_container_width=True)
        except Exception:
            st.info("Trajectory image not found.")
    else:
        st.info("No trajectory selected.")

# ------------
# MIDDLE: Radar
# ------------
FIELDS = [
    ("degreequest_vividness",       "vivid"),
    ("degreequest_immersiveness",   "immersive"),
    ("degreequest_bizarreness",     "bizarre"),
    ("degreequest_spontaneity",     "spontaneous"),
    ("degreequest_fleetingness",    "fleeting"),
    ("degreequest_emotionality",    "positive\nemotions"),
    ("degreequest_sleepiness",      "sleepy"),
]

def _as_float_or_nan(x):
    try:
        v = float(x)
        return np.nan if np.isnan(v) else v
    except Exception:
        return np.nan

# Pull values from current participant record (1..6 scale expected)
vals = [_as_float_or_nan(record.get(k)) for k, _ in FIELDS]
labels = [lab for _, lab in FIELDS]

# If all missing, default to zeros so the chart still renders
if np.all(np.isnan(vals)):
    vals_filled = [0.0] * len(vals)
else:
    mean_val = float(np.nanmean(vals))
    vals_filled = [mean_val if np.isnan(v) else v for v in vals]

# Close the loop for polar plot
values = vals_filled + [vals_filled[0]]
num_vars = len(vals_filled)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles_p = angles + angles[:1]

# Visual style (kept identical to your app’s radar)
POLY, GRID, SPINE, TICK, LABEL = PURPLE_HEX, "#B0B0B0", "#222222", "#555555", "#000000"
s = 1.4  # global scale used in your originals

with exp_mid:
    fig, ax = plt.subplots(figsize=(3.0 * s, 3.0 * s), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    
    ax.set_title("Intensity of your experience", fontsize=20, pad=56, color="#222")
    ax.title.set_y(1.03)


    # Orientation and labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels)

    # Fine-tune label alignment
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

    # Radial settings
    ax.set_ylim(0, 6)
    ax.set_rgrids([1, 2, 3, 4, 5, 6], angle=180 / num_vars, color=TICK)
    ax.tick_params(axis="y", labelsize=7.0 * s, colors=TICK, pad=-1)

    # Grid & spine
    ax.grid(color=GRID, linewidth=0.45 * s)
    ax.spines["polar"].set_color(SPINE)
    ax.spines["polar"].set_linewidth(0.7 * s)

    # Data polygon
    ax.plot(angles_p, values, color=POLY, linewidth=1.0 * s, zorder=3)
    ax.fill(angles_p, values, color=POLY, alpha=0.22, zorder=2)

    # Light spokes
    for a in angles:
        ax.plot([a, a], [0, 6], color=GRID, linewidth=0.4 * s, alpha=0.35, zorder=1)

    plt.tight_layout(pad=0.3 * s)
    st.pyplot(fig, use_container_width=False)

# RIGHT column (placeholder for future content)
# with exp_right:
#     st.markdown("&nbsp;")

    
    
# ==============
# Horizontal timeline (3 bins) — goes in RIGHT column of "Your experience"
# ==============
st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

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

def _core_name(v): 
    return re.sub(r"^(freq_|timequest_)", "", v)

def _as_float(x): 
    try: 
        return float(x)
    except: 
        return np.nan

# Pair time (1..100) with frequency (1..6) per concept, and keep only complete pairs
freq_scores = {_core_name(v): _as_float(record.get(v, np.nan)) for v in FREQ_VARS}
time_scores = {_core_name(v): _as_float(record.get(v, np.nan)) for v in TIME_VARS}
common = [c for c in time_scores if c in freq_scores and not np.isnan(time_scores[c]) and not np.isnan(freq_scores[c])]

# Build core tuples: (concept_key, time, freq, label)
cores = []
for c in common:
    t = float(time_scores[c]); f = float(freq_scores[c])
    lab = CUSTOM_LABELS.get(f"freq_{c}", c.replace("_", " "))
    cores.append((c, t, f, lab))

# --- 2 bins across 1..100 (1–50, 51–100)
bins = [(1.0, 50.0), (51.0, 100.0)]

# Assign items to bins
bin_items = {i: [] for i in range(len(bins))}
for c, t, f, lab in cores:
    for i, (lo, hi) in enumerate(bins):
        if lo <= t <= hi:
            bin_items[i].append((c, t, f, lab))
            break

# Winners per bin: top 3 by frequency (break ties by label for determinism)
winners = {0: [], 1: []}
for i, items in bin_items.items():
    if not items:
        continue
    items_sorted = sorted(items, key=lambda x: (-x[2], x[3]))  # highest freq, then label
    winners[i] = [it[3] for it in items_sorted[:3]]

# --- Plot (horizontal bar with L→R gradient: Awake → Asleep)
with exp_right:
    
    st.markdown(
    "<div style='font-size:18px; color:#222; text-align:center; margin-bottom:6px;'>Dynamics of your experience</div>",
    unsafe_allow_html=True
    )
    
    # keep it slightly lowered on the page
    st.markdown("<div style='height:1px;'></div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.axis("off")
    


    # Geometry
    y_bar = 0.50
    bar_half_h = 0.08
    x_left, x_right = 0.14, 0.86

    def tx(val):  # map 1..100 → x in [x_left, x_right]
        return x_left + (val - 1.0) / 99.0 * (x_right - x_left)

    # Bar gradient
    left_rgb = np.array([1.0, 1.0, 1.0])
    right_rgb = np.array([0x5B/255, 0x21/255, 0xB6/255])
    n = 1200
    cols = np.linspace(left_rgb, right_rgb, n)
    grad_img = np.tile(cols[None, :, :], (12, 1, 1))
    ax.imshow(
        grad_img,
        extent=(tx(1), tx(100), y_bar - bar_half_h, y_bar + bar_half_h),
        origin="lower",
        aspect="auto",
        interpolation="bilinear"
    )

    # In-bar end labels
    end_fs = 22
    ax.text(tx(6),   y_bar, "Awake",  ha="left",  va="center",
            fontsize=end_fs, color="#000000")
    ax.text(tx(94),  y_bar, "Asleep", ha="right", va="center",
            fontsize=end_fs, color="#FFFFFF")

        # --- Single-stem stacked labels per bin ---
    # --- Single-stem stacked labels per bin ---
    label_fs = 15.0   # smaller text for both lists
    stem_lw = 1.6
    row_gap = 0.05    # keep same spacing between each item
    
    # Bin 0 (1–50): one stem around x≈33, labels above the bar (stacked downward)
    top_anchor_x = tx(33.0)
    top_base_y   = y_bar + bar_half_h + 0.22   # farther above the bar
    top_positions = [top_base_y,
                     top_base_y - row_gap,
                     top_base_y - 2 * row_gap]
    
    if winners[0]:
        nearest_top = top_positions[-1]  # closest to the bar
        ax.plot([top_anchor_x, top_anchor_x],
                [y_bar + bar_half_h, nearest_top - 0.018],
                color="#000000", linewidth=stem_lw)
        for yy, text_label in zip(top_positions, winners[0]):
            ax.text(top_anchor_x, yy, text_label, ha="center", va="bottom",
                    fontsize=label_fs, color="#000000", linespacing=1.12)

    
    # Bin 1 (51–100): one stem around x≈66, labels below the bar (stacked downward)
    bot_anchor_x = tx(66.0)
    bot_base_y   = y_bar - bar_half_h - 0.22   # farther below the bar
    bot_positions = [bot_base_y,
                     bot_base_y + row_gap,
                     bot_base_y + 2 * row_gap]
    
    if winners[1]:
        nearest_bot = bot_positions[-1]  # closest to the bar
        ax.plot([bot_anchor_x, bot_anchor_x],
                [y_bar - bar_half_h, nearest_bot + 0.018],
                color="#000000", linewidth=stem_lw)
        for yy, text_label in zip(bot_positions, winners[1]):
            ax.text(bot_anchor_x, yy, text_label,
                    ha="center", va="top",
                    fontsize=label_fs, color="#000000", linespacing=1.15)


    plt.tight_layout(pad=0.25)
    st.pyplot(fig, use_container_width=False)


# --- Explanatory note below "Your Experience" -----------------------------------
st.markdown(
    """
    <div style="
        max-width:740px;
        margin:-20px 0 0 0;
        text-align:left;
        font-size:0.82rem;
        color:#444;
        line-height:1.35;
    ">
      <em>
        Left graph: your own trajectory selection.  
        Middle graph: your self-rated intensity scores of your typical mental content (1 = low, 6 = high).
        Right graph: mental content that most often emerged early vs. late as you fall asleep.
      </em>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============
# ALL DMs (embedded, centered)
# ==============
import base64

with open("assets/all_DMs.png", "rb") as f:
    all_dms_b64 = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 50px 0;
    ">
        <img src="data:image/png;base64,{all_dms_b64}" 
             style="max-width: 70%; height: auto; border-radius: 8px;">
    </div>
    """,
    unsafe_allow_html=True
)


# ==============
# Disclaimer
# ==============

st.markdown(
    """
    <div style="max-width:740px; margin:10px 0 0 0; text-align:left; font-size:0.8rem; color:#444;">
      <em>
        These results are automatically generated from your responses to the Drifting Minds questionnaire.  
        They are meant for research and self-reflection only, not as medical or diagnostic advice.  
        For any questions, contact <a href="mailto:driftingminds@icm-institute.org" style="color:#7C3AED; text-decoration:none;">driftingminds@icm-institute.org</a>.
      </em>
    </div>
    """,
    unsafe_allow_html=True
)




# ==============
# Profile distribution across the N=1000 population
# ==============
st.markdown("<div style='height:120px;'></div>", unsafe_allow_html=True)

if pop_data is None or pop_data.empty:
    st.info("Population data unavailable.")
else:
    # Assign a best profile to each participant
    prof_names = []
    for _, row in pop_data.iterrows():
        name, _ = assign_profile_from_record(row.to_dict())
        prof_names.append(name if name is not None else "Unassigned")

    # Tally counts in the order of your PROFILES dict for readability
    order = list(PROFILES.keys())
    ser = pd.Series(prof_names, name="profile")
    counts = ser.value_counts().reindex(order, fill_value=0)

    total_n = int(counts.sum())
    perc = (counts / max(total_n, 1) * 100).round(1)
    dist_df = pd.DataFrame({
        "profile": counts.index,
        "count": counts.values,
        "percent": perc.values
    }).reset_index(drop=True)

    # --- Issue checks (keep diagnostic feedback)
    zero_profiles = dist_df.loc[dist_df["count"] == 0, "profile"].tolist()
    over_profiles = dist_df.loc[dist_df["percent"] > 30.0, "profile"].tolist()

    if zero_profiles:
        st.error("Profiles with **0%** representation: " + ", ".join(zero_profiles))
    if over_profiles:
        st.warning("Profiles above **30%** of the sample: " + ", ".join(over_profiles))

    # --- Bar chart (clean, minimal, consistent look)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    x = np.arange(len(dist_df))
    ax.bar(x, dist_df["percent"].values, width=0.6, color=PURPLE_HEX, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(dist_df["profile"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Participants (%)", fontsize=10)
    ax.set_title(f"Profile distribution (N = {total_n})", fontsize=11, pad=8)

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels above bars
    for xi, p in zip(x, dist_df["percent"].values):
        ax.text(xi, p + 0.8, f"{p:.1f}%", ha="center", va="bottom", fontsize=8, color="#111")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


