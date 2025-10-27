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


import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# ---- Design system ----------------------------------------------------------
PALETTE = {
    "bg": "#0E1117",
    "card": "#151A23",
    "text": "#E6E6E6",
    "muted": "#9AA3AF",
    "accent1": "#7C3AED",   # purple
    "accent2": "#22D3EE",   # cyan
    "accent3": "#F59E0B",   # amber
    "accent4": "#34D399",   # green
    "bar_bg": "#2B3241",
}

plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 200,
    "axes.edgecolor": PALETTE["card"],
    "axes.facecolor": PALETTE["card"],
    "figure.facecolor": PALETTE["card"],
    "text.color": PALETTE["text"]
})

# ---- Utils ------------------------------------------------------------------
def get_num(record, key, cast=float):
    v = record.get(key, None)
    try:
        if v is None or v == "":
            return None
        return cast(v)
    except Exception:
        return None

def normalized(v, vmin, vmax):
    if v is None:
        return None
    return max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))

def decode_chronotype(x):
    mapping = {"1":"Morning type","2":"Evening type","3":"Intermediate type","4":"I don't know",
               1:"Morning type",2:"Evening type",3:"Intermediate type",4:"I don't know"}
    return mapping.get(x, "—")

def chronotype_position(label):
    return {"Morning type":0.1,"Intermediate type":0.5,"Evening type":0.9,"I don't know":0.5}.get(label, 0.5)

def decode_dream_recall(x):
    mapping = {"1":"Less than once/month","2":"1–2 times/month","3":"Once/week",
               "4":"Several times/week","5":"Every day",
               1:"Less than once/month",2:"1–2 times/month",3:"Once/week",4:"Several times/week",5:"Every day"}
    return mapping.get(x, "—")

def dream_pos(label):
    order = ["Less than once/month","1–2 times/month","Once/week","Several times/week","Every day"]
    return 0.0 if label not in order else (order.index(label) / 4)

# ---- Drawing primitives -----------------------------------------------------
def draw_value_bar(ax, y, value01, left_label, value_text, anchors=("Low","High"),
                   color=PALETTE["accent1"], h=0.16):
    import matplotlib.patches as patches
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    bar_x, bar_w = 0.12, 0.76; r = h/2

    # background
    bg = patches.FancyBboxPatch((bar_x, y - h/2), bar_w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=0, facecolor=PALETTE["bar_bg"])
    ax.add_patch(bg)

    # fill
    if value01 is not None:
        fg = patches.FancyBboxPatch((bar_x, y - h/2), max(0.001, bar_w*value01), h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    linewidth=0, facecolor=color)
        ax.add_patch(fg)

    # labels
    ax.text(0.08, y, left_label, va="center", ha="right", fontsize=12, fontweight="bold", color=PALETTE["text"])
    ax.text(bar_x + bar_w + 0.015, y, value_text if value_text else "—",
            va="center", ha="left", fontsize=11, color=PALETTE["muted"])

    # anchors under bar
    ax.text(bar_x, y - h*1.2, anchors[0], va="top", ha="left", fontsize=9, color=PALETTE["muted"])
    ax.text(bar_x+bar_w, y - h*1.2, anchors[1], va="top", ha="right", fontsize=9, color=PALETTE["muted"])

def draw_tick_slider(ax, y, pos01, left_label, ticks, color=PALETTE["accent3"], h=0.16):
    import matplotlib.patches as patches
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    bar_x, bar_w = 0.12, 0.76; r = h/2

    bg = patches.FancyBboxPatch((bar_x, y - h/2), bar_w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=0, facecolor=PALETTE["bar_bg"])
    ax.add_patch(bg)

    if pos01 is not None:
        cx = bar_x + bar_w*pos01
        ax.add_patch(patches.Circle((cx, y), radius=h*0.42, facecolor=color, edgecolor="none"))

    ax.text(0.08, y, left_label, va="center", ha="right", fontsize=12, fontweight="bold", color=PALETTE["text"])
    # ticks
    n = len(ticks)
    for i, t in enumerate(ticks):
        tx = bar_x + bar_w*(i/(n-1) if n>1 else 0.5)
        ax.text(tx, y - h*1.2, t, va="top", ha="center", fontsize=9, color=PALETTE["muted"])

# ---- Extract participant values --------------------------------------------
anxiety = get_num(record, "anxiety", float)                      # 1–100
creativity = get_num(record, "creativity_trait", float)          # 1–6
chronotype_label = decode_chronotype(record.get("chronotype"))
chronotype_pos = chronotype_position(chronotype_label)

sleep_latency_min = get_num(record, "sleep_latency", float)      # minutes
sleep_duration_h = get_num(record, "sleep_duration", float)      # hours
sleep_quality = get_num(record, "subj_sleep_quality", float)     # 1–6
dream_label = decode_dream_recall(record.get("dream_recall"))
dream_pos01 = dream_pos(dream_label)

# normalize for bars (for fill only)
anxiety01    = normalized(anxiety, 1, 100)
creativity01 = normalized(creativity, 1, 6)
quality01    = normalized(sleep_quality, 1, 6)

# human-readable value texts on the right (no cryptic %)
anx_text = f"{int(round(anxiety))}/100" if anxiety is not None else None
crea_text = f"{int(round(creativity))}/6" if creativity is not None else None
qual_text = f"{int(round(sleep_quality))}/6" if sleep_quality is not None else None

# ---- Streamlit layout -------------------------------------------------------
st.markdown(
    "<h2 style='margin-bottom:0.4rem;'>Your Sleep-Onset Profile</h2>"
    "<p style='color:#9AA3AF;margin-top:0;'>A quick snapshot of your traits and sleep patterns.</p>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

# Personality
with col1:
    fig1, ax1 = plt.subplots(figsize=(6.6, 4.0))
    fig1.patch.set_facecolor(PALETTE["card"]); ax1.set_facecolor(PALETTE["card"]); ax1.axis("off")
    ax1.text(0.06, 0.95, "🧭 Personality", fontsize=14, fontweight="bold", ha="left", va="top", color=PALETTE["text"])
    ax1.text(0.06, 0.89, "Self-reported tendencies", fontsize=10, ha="left", va="top", color=PALETTE["muted"])

    y0 = 0.72
    draw_value_bar(ax1, y0,             anxiety01,    "Anxiety",    anx_text, anchors=("Low","High"),  color=PALETTE["accent1"])
    draw_value_bar(ax1, y0 - 0.28,      creativity01, "Creativity", crea_text, anchors=("Lower","Higher"), color=PALETTE["accent2"])
    draw_tick_slider(ax1, y0 - 0.56,    chronotype_pos, "Chronotype",
                     ["Morning","Intermediate","Evening"], color=PALETTE["accent3"])

    st.pyplot(fig1, use_container_width=True, clear_figure=True)
    buf1 = BytesIO(); fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight", facecolor=fig1.get_facecolor())
    plt.close(fig1)

# Sleep
with col2:
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.0))
    fig2.patch.set_facecolor(PALETTE["card"]); ax2.set_facecolor(PALETTE["card"]); ax2.axis("off")
    ax2.text(0.06, 0.95, "🌙 Sleep", fontsize=14, fontweight="bold", ha="left", va="top", color=PALETTE["text"])
    ax2.text(0.06, 0.89, "Recent sleep patterns", fontsize=10, ha="left", va="top", color=PALETTE["muted"])

    # key metrics
    lat_txt = f"{int(round(sleep_latency_min))} min" if sleep_latency_min is not None else "—"
    dur_txt = f"{sleep_duration_h:.1f} h" if sleep_duration_h is not None else "—"
    ax2.text(0.12, 0.80, "Sleep latency", fontsize=10, color=PALETTE["muted"])
    ax2.text(0.12, 0.74, lat_txt, fontsize=16, fontweight="bold", color=PALETTE["text"])
    ax2.text(0.58, 0.80, "Sleep duration", fontsize=10, color=PALETTE["muted"])
    ax2.text(0.58, 0.74, dur_txt, fontsize=16, fontweight="bold", color=PALETTE["text"])
    ax2.text(0.58, 0.68, "Tip: 7–9 h is typical for adults", fontsize=9, color=PALETTE["muted"])

    # quality stays a bar (1–6), dream recall becomes a discrete slider
    draw_value_bar(ax2, 0.46, quality01, "Sleep quality", qual_text, anchors=("Poor","Great"), color=PALETTE["accent4"])
    draw_tick_slider(ax2, 0.20, dream_pos01, "Dream recall",
                     ["<1/mo","1–2/mo","1/wk","Several/wk","Daily"], color=PALETTE["accent1"])

    st.pyplot(fig2, use_container_width=True, clear_figure=True)
    buf2 = BytesIO(); fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)

# ---- Build a single shareable PNG ------------------------------------------
img1 = Image.open(BytesIO(buf1.getvalue()))
img2 = Image.open(BytesIO(buf2.getvalue()))
w = max(img1.width, img2.width)
def pad_to_width(im, width, bg=tuple(int(PALETTE["card"].lstrip("#")[i:i+2], 16) for i in (0,2,4))):
    if im.width == width: return im
    out = Image.new("RGB", (width, im.height), bg); out.paste(im, ((width-im.width)//2, 0)); return out
img1 = pad_to_width(img1, w); img2 = pad_to_width(img2, w)
stacked = Image.new("RGB", (w, img1.height + img2.height + 28),
                    tuple(int(PALETTE["card"].lstrip("#")[i:i+2], 16) for i in (0,2,4)))
stacked.paste(img1, (0, 0)); stacked.paste(img2, (0, img1.height + 28))
out_buf = BytesIO(); stacked.save(out_buf, format="PNG")

st.download_button("Download your profile (PNG)", data=out_buf.getvalue(),
                   file_name=f"drifting-minds_profile_{record_id}.png", mime="image/png")

st.caption("These visuals are for illustration only and don’t constitute medical advice.")







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
