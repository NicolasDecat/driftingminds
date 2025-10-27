#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 15:20:39 2025

DM: visualisation of DM results (pending API token -> demo)

@author: nicolas.decat
"""

import os
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

REDCAP_API_URL = st.secrets.get("https://redcap-icm.icm-institute.org/api/")
REDCAP_API_TOKEN = st.secrets.get("641919114F091FC5A1860BCFC53D3947")

DEMO = st.query_params.get("demo", ["0"])[0] == "1"
# DEMO = True


if (not REDCAP_API_URL or not REDCAP_API_TOKEN) and DEMO:
    st.title("Your sleep-onset profile — demo")
    demo_record = {
        "freq_think_nocontrol": 4, "freq_think_seq_bizarre": 3, "freq_think_seq_ordinary": 2,
        "freq_percept_precise": 5, "freq_percept_real": 4, "freq_percept_imposed": 1
    }
    # …then reuse your scale_means / plots on demo_record and return
    record = demo_record
else:
    if not REDCAP_API_URL or not REDCAP_API_TOKEN:
        st.warning("Secrets missing and no demo mode. Set secrets or append `?demo=1` to the URL.")
        st.stop()
        
# ---------- Demo visuals ----------

st.caption("Demo data only — replace with real REDCap once the token is set.")

# 1) Define which questions belong to each scale (axes of the radar)
SCALES = {
    "Thoughts":    ["freq_think_nocontrol", "freq_think_seq_bizarre", "freq_think_seq_ordinary"],
    "Perception":  ["freq_percept_precise", "freq_percept_real", "freq_percept_imposed"],
}

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def scale_means(rec: dict) -> pd.Series:
    means = {}
    for scale, items in SCALES.items():
        vals = [safe_float(rec.get(f, np.nan)) for f in items]
        vals = [v for v in vals if np.isfinite(v)]
        means[scale] = float(np.mean(vals)) if vals else np.nan
    return pd.Series(means)

# 2) Compute means for the radar
means = scale_means(record)

# 3) RADAR (matplotlib)
def radar_plot(series: pd.Series, title: str):
    labels = list(series.index)
    values = series.values.astype(float)

    # close the loop
    labels += [labels[0]]
    values = np.append(values, values[0])

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    return fig

st.subheader("Your profile (radar)")
fig_radar = radar_plot(means, "Scale means")
st.pyplot(fig_radar)

# 4) BAR chart (matplotlib)
def bar_plot(series: pd.Series, title: str):
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Mean score")
    ax.set_xlabel("Scale")
    return fig

st.subheader("Breakdown by scale")
fig_bar = bar_plot(means, "Scale means")
st.pyplot(fig_bar)

# 5) (Optional) You vs. crowd — fake distribution for demo
st.subheader("How you compare to others (demo)")
# create a fake distribution around 1..5 Likert with some noise
rng = np.random.default_rng(42)
dist = pd.DataFrame({
    "Thoughts":   rng.normal(loc=3.0, scale=0.8, size=300).clip(1, 5),
    "Perception": rng.normal(loc=3.2, scale=0.7, size=300).clip(1, 5),
})

# show percentiles and your point
stats = dist.describe(percentiles=[0.25, 0.5, 0.75]).loc[["25%", "50%", "75%"]]
st.write("Percentiles (demo):")
st.dataframe(stats)

# simple scatter “cloudpoint” per scale (jittered)
for scale in means.index:
    st.write(f"Distribution for **{scale}** (demo)")
    x = rng.normal(0, 0.1, size=len(dist))  # jitter
    fig, ax = plt.subplots()
    ax.scatter(x, dist[scale], s=10, alpha=0.4)
    ax.scatter([0], [means[scale]], s=120, marker="x")  # your value
    ax.set_xticks([])
    ax.set_ylabel("Score (1–5)")
    ax.set_title(scale)
    st.pyplot(fig)

# Show raw responses (useful to debug field mapping)
with st.expander("See your raw responses (demo)"):
    st.json(record)