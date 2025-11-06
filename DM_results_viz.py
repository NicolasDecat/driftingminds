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
# ==============
st.set_page_config(page_title="Drifting Minds — Profile", layout="centered")

REDCAP_API_URL = st.secrets.get("REDCAP_API_URL")
REDCAP_API_TOKEN = st.secrets.get("REDCAP_API_TOKEN")

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
        "icon": "ant.svg",
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

# Icon + text row (side-by-side; exact same spacing/shifts)
icon_src = _data_uri(icon_path) if has_icon else ""
st.markdown(f"""
<div class="dm-center">
  <div class="dm-row">
    {'<img class="dm-icon" src="'+icon_src+'" alt="profile icon"/>' if has_icon else ''}
    <div class="dm-text">
      <p class="dm-lead">You drift into sleep like a</p>
      <div class="dm-key">{prof_name}</div>
      <p class="dm-desc">{prof_desc or "&nbsp;"}</p>
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
        "weight_keys": ["degreequest_vividness", "degreequest_distinctness"],
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
        mediantag_html = f"<div class='{mediantag_class}' style='left:{med_left_clamped}%;'>world</div>"

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


# ==============
# "You" — Imagery · Creativity · Anxiety (three compact histos)
# ==============

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
st.subheader("You")

def _mini_hist(ax, counts, edges, highlight_idx, title, xlabel):
    centers = 0.5 * (edges[:-1] + edges[1:])
    width   = edges[1] - edges[0]
    ax.bar(centers, counts, width=width, color="#D9D9D9", edgecolor="white", align="center")
    ax.bar(centers[highlight_idx], counts[highlight_idx], width=width,
           color=PURPLE_HEX, edgecolor="white", align="center")
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Population", fontsize=9)
    ax.set_yticks([]); ax.set_yticklabels([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8); ax.tick_params(axis="y", length=0)

def _prep_bins(values: np.ndarray):
    """Pick sensible bins: discrete 1–6 if appropriate, else 0–100 by 10s, else auto."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    # Discrete 1..6
    if 1.0 <= vmin <= 6.0 and vmax <= 6.0 and np.unique(values).size <= 6:
        edges = np.arange(0.5, 6.5 + 1.0, 1.0)
        return edges, "Score (1–6)"
    # Percent/0..100
    if 0.0 <= vmin and vmax <= 100.0:
        edges = np.arange(-0.5, 100.5 + 10.0, 10.0)
        return edges, "Score (0–100)"
    # Fallback
    edges = np.linspace(vmin, vmax, 16)
    return edges, "Score"

def _col_values(df, colname):
    if df is None or df.empty or (colname not in df.columns):
        return np.array([])
    return pd.to_numeric(df[colname], errors="coerce").to_numpy()

def _participant_value(rec, key):
    try:
        return float(str(rec.get(key, np.nan)).strip())
    except Exception:
        return np.nan

c1, c2, c3 = st.columns(3, gap="small")

# --- 1) Imagery (VVIQ) — compact version
with c1:
    # reuse vviq_score if already computed above; otherwise compute quickly here
    try:
        vviq_score  # exists from previous block
    except NameError:
        VVIQ_FIELDS = [
            "quest_a1","quest_a2","quest_a3","quest_a4",
            "quest_b1","quest_b2","quest_b3","quest_b4",
            "quest_c1","quest_c2","quest_c3","quest_c4",
            "quest_d1","quest_d2","quest_d3","quest_d4"
        ]
        vviq_vals  = [float(record.get(k, np.nan)) if pd.notna(record.get(k, np.nan)) else np.nan for k in VVIQ_FIELDS]
        vviq_score = sum(v for v in vviq_vals if np.isfinite(v))
    # synthetic normative distribution (same params as your big plot)
    from scipy.stats import truncnorm
    N = 8000; mu, sigma = 61.0, 9.2; low, high = 16, 80
    a, b = (low - mu) / sigma, (high - mu) / sigma
    vviq_samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=N, random_state=42)

    bins = np.linspace(low, high, 33)
    counts, edges = np.histogram(vviq_samples, bins=bins, density=True)
    hidx = np.digitize(vviq_score, edges) - 1
    hidx = np.clip(hidx, 0, len(counts)-1)

    fig, ax = plt.subplots(figsize=(2.4, 2.5))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    _mini_hist(ax, counts, edges, hidx, "Imagery (VVIQ)", "Score")
    st.pyplot(fig, use_container_width=False)

# --- 2) Creativity
with c2:
    colname = "creativity_trait"
    vals = _col_values(pop_data, colname)
    if vals.size == 0:
        st.info("Population data for creativity unavailable.")
    else:
        edges, xlabel = _prep_bins(vals)
        counts, _ = np.histogram(vals, bins=edges, density=True)
        part_val = _participant_value(record, colname)
        if not np.isfinite(part_val):  # try alt keys if needed
            part_val = _participant_value(record, "creativity_trait")
        hidx = np.digitize(part_val, edges) - 1
        hidx = int(np.clip(hidx, 0, len(counts)-1))

        fig, ax = plt.subplots(figsize=(2.4, 2.5))
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        _mini_hist(ax, counts, edges, hidx, "Creativity", xlabel)
        # tidy x ticks for discrete 1..6
        if np.allclose(np.diff(edges), 1.0) and edges.min() <= 1 and edges.max() >= 6.5:
            ax.set_xticks(np.arange(1, 7, 1)); ax.set_xticklabels([str(i) for i in range(1, 7)])
        st.pyplot(fig, use_container_width=False)

# --- 3) Anxiety
with c3:
    colname = "anxiety"
    vals = _col_values(pop_data, colname)
    if vals.size == 0:
        st.info("Population data for anxiety unavailable.")
    else:
        edges, xlabel = _prep_bins(vals)
        counts, _ = np.histogram(vals, bins=edges, density=True)
        part_val = _participant_value(record, colname)
        hidx = np.digitize(part_val, edges) - 1
        hidx = int(np.clip(hidx, 0, len(counts)-1))

        fig, ax = plt.subplots(figsize=(2.4, 2.5))
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        _mini_hist(ax, counts, edges, hidx, "Anxiety", xlabel)
        # tidy x ticks for discrete 1..6
        if np.allclose(np.diff(edges), 1.0) and edges.min() <= 1 and edges.max() >= 6.5:
            ax.set_xticks(np.arange(1, 7, 1)); ax.set_xticklabels([str(i) for i in range(1, 7)])
        st.pyplot(fig, use_container_width=False)





# ==============
# Comparative visualisation (Latency KDE + Duration histogram)
# ==============
# LEFT: Latency (KDE)
col_left, col_right = st.columns([1, 1], gap="small")

from scipy.stats import gaussian_kde  # ensure this import exists

with col_left:
    if pop_data is None or pop_data.empty:
        st.info("Population data unavailable.")
    else:
        # find a sleep_latency-like column in population
        lat_cols = [c for c in pop_data.columns if "sleep_latency" in c.lower()]
        if not lat_cols:
            st.info("No sleep latency column in population data.")
        else:
            lat_col = lat_cols[0]
            raw = pd.to_numeric(pop_data[lat_col], errors="coerce").dropna()
            if raw.empty:
                st.info("No valid population sleep-latency values.")
            else:
                # minutes: if population stored 0..1, scale by CAP_MIN; else keep as minutes
                samples = np.clip(raw.values * CAP_MIN if raw.max() <= 1.5 else raw.values, 0, CAP_MIN)

                # participant value
                raw_sl = _get_first(record, ["sleep_latency"])
                sl_norm = norm_latency_auto(raw_sl, cap_minutes=CAP_MIN)
                if np.isnan(sl_norm):
                    st.info("No sleep-latency value available for this participant.")
                else:
                    # minutes to display (cap to 60)
                    try:
                        part_raw_minutes = float(str(raw_sl).strip())
                    except Exception:
                        part_raw_minutes = np.nan
                    part_display = sl_norm * CAP_MIN if sl_norm <= 1.5 else float(sl_norm)
                    part_display = float(np.clip(part_display, 0, CAP_MIN))
                    rounded_raw = int(round(part_raw_minutes)) if np.isfinite(part_raw_minutes) else int(round(part_display))

                    # KDE
                    kde = gaussian_kde(samples, bw_method="scott")
                    xs = np.linspace(0, CAP_MIN, 400)
                    ys = kde(xs)

                    from matplotlib.ticker import MaxNLocator
                    with plt.rc_context({
                        "axes.facecolor": "none",
                        "axes.edgecolor": "#000000",
                        "axes.linewidth": 0.8,
                        "xtick.color": "#333333",
                        "ytick.color": "#333333",
                        "xtick.major.width": 0.8,
                        "ytick.major.width": 0.8,
                        "font.size": 9,
                    }):
                        fig, ax = plt.subplots(figsize=(2.2, 2.4))
                        fig.patch.set_alpha(0.0)
                        ax.set_facecolor("none")

                        # KDE area + outline
                        ax.fill_between(xs, ys, color="#e6e6e6", linewidth=0)

                        # Participant marker (thin line + purple dot)
                        ax.axvline(part_display, lw=0.8, color="#222222")
                        ax.scatter(
                            [part_display],
                            [kde(part_display)],
                            s=28,
                            zorder=3,
                            color=PURPLE_HEX,   # uses same purple as duration bar
                            edgecolors="none"   # removes any stroke
                        )

                        # Title & labels
                        ax.set_title(f"{rounded_raw} minutes to fall asleep", fontsize=10, pad=6, color="#222222")
                        ax.set_xlabel("Time (min)", fontsize=9, color="#333333")
                        ax.set_ylabel("Population", fontsize=9, color="#333333")

                        # Ticks / spines
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
                        ax.set_yticks([])  # minimalist look

                        xticks = np.linspace(0, CAP_MIN, 7)
                        ax.set_xticks(xticks)
                        xlabels = [str(int(t)) if t < CAP_MIN else "60+" for t in xticks]
                        ax.set_xticklabels(xlabels)

                        for side in ("top", "right"):
                            ax.spines[side].set_visible(False)

                        ax.tick_params(axis="x", labelsize=8)
                        ax.tick_params(axis="y", length=0)

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=False)


# RIGHT: Duration (histogram)
with col_right:
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
                        title_str = f"{dur_raw_str} hours of sleep"
                    else:
                        part_hours_plot = float(dur_raw_str)
                        title_str = f"{int(round(part_hours_plot))} hours of sleep"
                except:
                    part_hours_plot = float(np.nanmedian(samples_h))
                    title_str = "Sleep duration"

                part_hours_plot = float(np.clip(part_hours_plot, 1.0, 12.0))
                edges = np.arange(0.5, 12.5 + 1.0, 1.0)
                counts, _ = np.histogram(samples_h, bins=edges, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])

                highlight_idx = np.digitize(part_hours_plot, edges) - 1
                highlight_idx = np.clip(highlight_idx, 0, len(counts) - 1)

                fig, ax = plt.subplots(figsize=(2.2, 2.4))
                fig.patch.set_alpha(0); ax.set_facecolor("none")
                ax.bar(centers, counts, width=edges[1]-edges[0], color="#D9D9D9", edgecolor="white", align="center")
                ax.bar(centers[highlight_idx], counts[highlight_idx],
                       width=edges[1]-edges[0], color=PURPLE_HEX, edgecolor="white",
                       align="center", label="Your duration")
                ax.set_title(title_str, fontsize=10, pad=6)
                ax.set_xlabel("Time (h)", fontsize=9); ax.set_ylabel("Population", fontsize=9)
                ax.set_yticks([]); ax.set_yticklabels([])
                ticks = np.arange(1, 13, 1); ax.set_xticks(ticks)
                labels = ["" for _ in ticks]
                for i in range(4, 11): labels[i-1] = str(i)  # show 4..10
                ax.set_xticklabels(labels)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                ax.tick_params(axis="x", labelsize=8); ax.tick_params(axis="y", length=0)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)

# ==============
# Easy-to-pick radar (unchanged visuals)
# ==============
st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

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
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
values = vals_filled + [vals_filled[0]]
angles_p = angles + angles[:1]

POLY, GRID, SPINE, TICK, LABEL = PURPLE_HEX, "#B0B0B0", "#222222", "#555555", "#000000"
s = 1.4
col_left, col_right = st.columns([1.3, 1.7])

fig, ax = plt.subplots(figsize=(3.0 * s, 3.0 * s), subplot_kw=dict(polar=True))
fig.patch.set_alpha(0); ax.set_facecolor("none")
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles), labels)

for lbl, ang in zip(ax.get_xticklabels(), angles):
    if ang in (0, np.pi): lbl.set_horizontalalignment("center")
    elif 0 < ang < np.pi: lbl.set_horizontalalignment("left")
    else: lbl.set_horizontalalignment("right")
    lbl.set_color(LABEL); lbl.set_fontsize(8.5 * s)
ax.tick_params(axis="x", pad=int(2.5 * s))

ax.set_ylim(0, 6)
ax.set_rgrids([1, 2, 3, 4, 5, 6], angle=180 / num_vars, color=TICK)
ax.tick_params(axis="y", labelsize=7.0 * s, colors=TICK, pad=-1)
ax.grid(color=GRID, linewidth=0.45 * s)
ax.spines["polar"].set_color(SPINE); ax.spines["polar"].set_linewidth(0.7 * s)
ax.plot(angles_p, values, color=POLY, linewidth=1.0 * s, zorder=3)
ax.fill(angles_p, values, color=POLY, alpha=0.22, zorder=2)
for a in angles:
    ax.plot([a, a], [0, 6], color=GRID, linewidth=0.4 * s, alpha=0.35, zorder=1)

plt.tight_layout(pad=0.3 * s)
with col_left:
    st.pyplot(fig, use_container_width=False)

# ==============
# Vertical timeline (unchanged visuals)
# ==============
st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

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

def _core_name(v): return re.sub(r"^(freq_|timequest_)", "", v)
def _as_float(x): 
    try: return float(x)
    except: return np.nan

freq_scores = {_core_name(v): _as_float(record.get(v, np.nan)) for v in FREQ_VARS}
time_scores = {_core_name(v): _as_float(record.get(v, np.nan)) for v in TIME_VARS}
common = [c for c in time_scores if c in freq_scores and not np.isnan(time_scores[c]) and not np.isnan(freq_scores[c])]

# Build core tuples
cores = []
for c in common:
    t = float(time_scores[c]); f = float(freq_scores[c])
    lab = CUSTOM_LABELS.get(f"freq_{c}", c.replace("_", " "))
    cores.append((c, t, f, lab))

bins = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,90),(91,100)]
bin_centers = [(lo+hi)/2 for (lo,hi) in bins]
bin_items = {i: [] for i in range(len(bins))}
for c, t, f, lab in cores:
    for i, (lo, hi) in enumerate(bins):
        if lo <= t <= hi:
            bin_items[i].append((c, t, f, lab)); break

winners = {i: [] for i in range(len(bins))}
for i, items in bin_items.items():
    if not items: continue
    items_sorted = sorted(items, key=lambda x: (-x[2], x[3]))
    top_f = items_sorted[0][2]
    if top_f < 3: continue
    tied = [it for it in items_sorted if abs(it[2] - top_f) < 1e-9]
    winners[i] = [it[3] for it in tied[:2]]

fig, ax = plt.subplots(figsize=(3.6, 6.0))
fig.patch.set_alpha(0); ax.set_facecolor("none"); ax.axis("off")
x_bar = 0.5; bar_half_w = 0.007; y_top, y_bot = 0.92, 0.08
def ty(val): return y_top - (val - 1) / 99.0 * (y_top - y_bot)

top_rgb  = np.array([1.0, 1.0, 1.0])
bot_rgb  = np.array([0x5B/255, 0x21/255, 0xB6/255])
n = 900
rows = np.linspace(bot_rgb, top_rgb, n)
grad_img = np.tile(rows[:, None, :], (1, 12, 1))
ax.imshow(grad_img, extent=(x_bar - bar_half_w, x_bar + bar_half_w, ty(100), ty(1)),
          origin="lower", aspect="auto", interpolation="bilinear")

ax.text(x_bar, ty(1)  + 0.035, "Awake",  ha="center", va="bottom", fontsize=11, color="#000000")
ax.text(x_bar, ty(100) - 0.035, "Asleep", ha="center", va="top",    fontsize=11, color="#000000")

x_right = x_bar + 0.042; x_left = x_bar - 0.042
line_w = 0.15; label_fs = 9.2

for i, center in enumerate(bin_centers):
    labs = winners[i]
    if not labs: continue
    y_c = ty(center)
    y_positions = [y_c] if len(labs) == 1 else [y_c + 0.02, y_c - 0.02]
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

# ==============
# VVIQ distribution (unchanged visuals)
# ==============
st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

VVIQ_FIELDS = [
    "quest_a1","quest_a2","quest_a3","quest_a4",
    "quest_b1","quest_b2","quest_b3","quest_b4",
    "quest_c1","quest_c2","quest_c3","quest_c4",
    "quest_d1","quest_d2","quest_d3","quest_d4"
]

def as_float_vviq(x):
    try: return float(x)
    except: return np.nan

vviq_vals = [as_float_vviq(record.get(k, np.nan)) for k in VVIQ_FIELDS]
vviq_score = sum(v for v in vviq_vals if not np.isnan(v))

N = 10000; mu, sigma = 61.0, 9.2; low, high = 16, 80
a, b = (low - mu) / sigma, (high - mu) / sigma
samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=N, random_state=42)

bins = np.linspace(low, high, 33)
counts, edges = np.histogram(samples, bins=bins, density=True)
centers = 0.5 * (edges[:-1] + edges[1:])

highlight_idx = np.digitize(vviq_score, edges) - 1
highlight_idx = np.clip(highlight_idx, 0, len(counts)-1)

fig, ax = plt.subplots(figsize=(6.5, 3.5))
fig.patch.set_alpha(0); ax.set_facecolor("none")
ax.bar(centers, counts, width=edges[1]-edges[0], color="#D9D9D9", edgecolor="white")
ax.bar(centers[highlight_idx], counts[highlight_idx],
       width=edges[1]-edges[0], color=PURPLE_HEX, edgecolor="white",
       label=f"Your score: {int(vviq_score)}")

aphantasia_cut, hyper_cut = 32, 75
ax.axvline(aphantasia_cut, color="#D9D9D9", linestyle="--", linewidth=1)
ax.axvline(hyper_cut, color="#D9D9D9", linestyle="--", linewidth=1)
y_text = ax.get_ylim()[1] * 0.92
ax.text(aphantasia_cut - 1.5, y_text, "Aphantasia", color="#888888", ha="right", va="center", fontsize=8)
ax.text(hyper_cut + 1.5, y_text, "Hyperphantasia", color="#888888", ha="left", va="center", fontsize=8)

ax.set_title("Vididness for visual imagery during wakefulness (VVIQ)", fontsize=11, pad=10)
ax.set_xlabel("VVIQ score"); ax.set_ylabel("Distribution in the population")
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=8)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)



# ==============
# Profile distribution across the N=1000 population
# ==============
st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
st.subheader("How common is each profile in the population?")

if pop_data is None or pop_data.empty:
    st.info("Population data unavailable.")
else:
    # Assign a best profile to each participant (uses your PROFILES + assigner)
    prof_names = []
    for _, row in pop_data.iterrows():
        name, _ = assign_profile_from_record(row.to_dict())
        prof_names.append(name if name is not None else "Unassigned")

    # Tally counts in the order of your PROFILES dict for readability
    order = list(PROFILES.keys())
    ser = pd.Series(prof_names, name="profile")
    counts = ser.value_counts().reindex(order, fill_value=0)

    # Build tidy table
    total_n = int(counts.sum())
    perc = (counts / max(total_n, 1) * 100).round(1)
    dist_df = (
        pd.DataFrame({"profile": counts.index, "count": counts.values, "percent": perc.values})
        .reset_index(drop=True)
    )

    # --- Issue checks
    zero_profiles = dist_df.loc[dist_df["count"] == 0, "profile"].tolist()
    over_profiles  = dist_df.loc[dist_df["percent"] > 30.0, "profile"].tolist()

    if zero_profiles:
        st.error("Profiles with **0%** representation: " + ", ".join(zero_profiles))
    if over_profiles:
        st.warning("Profiles above **30%** of the sample: " + ", ".join(over_profiles))

    # --- Bar chart (consistent minimalist look + your purple)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    fig.patch.set_alpha(0); ax.set_facecolor("none")

    x = np.arange(len(dist_df))
    ax.bar(x, dist_df["percent"].values, width=0.6, color=PURPLE_HEX, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(dist_df["profile"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Participants (%)", fontsize=10)
    ax.set_title(f"Profile distribution (N = {total_n})", fontsize=11, pad=8)

    # Light grid, clean frame
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Add value labels on top
    for xi, p in zip(x, dist_df["percent"].values):
        ax.text(xi, p + 0.8, f"{p:.1f}%", ha="center", va="bottom", fontsize=8, color="#111")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Show the table (sorted by percent descending for quick scan)
    dist_show = dist_df.sort_values("percent", ascending=False, ignore_index=True)
    st.dataframe(dist_show, use_container_width=True)
    st.caption("Heuristic checks: aim for no empty categories and avoid overly dominant ones (>30%).")


