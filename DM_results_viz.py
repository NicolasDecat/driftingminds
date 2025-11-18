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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
        top: 160px;
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
ASSETS_CSV = os.path.join("assets", "N1400_comparative_viz_ready.csv")


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


# =====================
# Mobile-only layout: balanced margins + no cropping + left-aligned header
# =====================
st.markdown("""
<style>
@media (max-width: 640px){

  /* 1) Use real mobile viewport width; prevent horizontal scroll/crop */
  html, body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;        /* use % not 100vw to avoid scrollbar crop */
    overflow-x: hidden !important;
    box-sizing: border-box !important;
  }

  /* 2) Streamlit page container: add comfy side padding (margins) */
  [data-testid="stAppViewContainer"] > .main,
  section.main > div.block-container,
  .main .block-container,
  div.block-container {
    padding-left: 16px !important;  /* ← your margin-on-mobile */
    padding-right: 16px !important; /* → your margin-on-mobile */
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    width: 100% !important;         /* avoid 100vw to prevent crop */
    box-sizing: border-box !important;
  }

  /* 3) Any column/row wrappers: no extra inner margins that fight the padding */
  [data-testid="stVerticalBlock"],
  [data-testid="stHorizontalBlock"],
  [data-testid="column"] {
    margin: 0 !important;
    padding: 0 !important;
  }
  [data-testid="stVerticalBlock"] > div,
  [data-testid="stHorizontalBlock"] > div,
  [data-testid="column"] > div {
    margin: 0 !important;
    padding: 0 !important;
  }

  /* 4) Your own centered wrapper: don't clamp width on phones, inherit padding */
  .dm-center {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;          /* padding is handled by the container above */
    box-sizing: border-box !important;
  }

  /* 5) Profile header: truly left aligned *within* the 16px mobile padding */
  .dm-row{
    justify-content: flex-start !important;
    align-items: flex-start !important;
    gap: 10px !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
  }
  .dm-icon{
    transform: none !important;
    width: 84px !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  .dm-text{
    margin: 0 !important;
    padding: 0 !important;
    text-align: left !important;
    width: calc(100% - 84px) !important;
    box-sizing: border-box !important;
  }
  .dm-key{
    text-align: left !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
    font-size: clamp(24px, 8vw, 36px) !important;
  }
  .dm-desc{
    text-align: left !important;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
  }

  /* 6) Make media responsive inside padded layout */
  [data-testid="stImage"] img, img {
    max-width: 100% !important;
    height: auto !important;
  }
  
  /* A) Shift the whole bars section slightly left within the 16px page padding */
  /* 0 = align with page padding; negative pulls it a bit closer to the edge. */
  .dm2-outer{
    margin-left: -8px !important;   /* try -8px first; if you want even more, use -12px or -16px */
  }

  /* B) Make the LABELS themselves start on the left (not right-aligned) */
  .dm2-left{
    width: 140px !important;        /* a bit narrower on phones */
    flex-basis: 140px !important;
    padding-left: 0 !important;
    margin-left: 0 !important;
  }
  .dm2-label{
    text-align: left !important;    /* <— key change: place label text at the left */
    padding-right: 8px !important;  /* small gap before the bar (was 40px) */
    margin: 0 !important;
  }

  /* C) Bars wrapper: remove the extra nudge so bars sit closer to labels */
  .dm2-wrap{
    margin-left: 0 !important;      /* was -12px on desktop */
  }
  
  /* D) Bars: pull them further left on phones */
  .dm2-bar {
    margin-left: 0 !important;      /* cancel any previous offset */
    padding-left: 0 !important;     /* cancel any previous padding */
  }

  /* If the bars are wrapped in .dm2-wrap, adjust that too */
  .dm2-wrap {
    margin-left: -12px !important;   /* bring bars closer to the label column */
    padding-left: 0 !important;
  }

  /* If each individual bar container (.dm2-item or similar) has padding, reset it */
  .dm2-item {
    margin-left: 0 !important;
    padding-left: 0 !important;
  }

  /* Optional: tighten bar section a bit overall for small phones */
  @media (max-width: 420px){
    .dm2-wrap { margin-left: -10px !important; }
  }
  
  /* Narrower label column + smaller gap between label and bar */
 .dm2-row {
   grid-template-columns: 120px 1fr !important;  /* was 160px 1fr */
   column-gap: 4px !important;                   /* was 8px */
   width: 100% !important;
 }

 /* Match the left column to 120px */
 .dm2-left {
   width: 120px !important;
   flex-basis: 120px !important;
 }

 /* Remove the big spacer between label and bar */
 .dm2-label {
   padding-right: 4px !important;  /* was 40px; we keep a tiny gap */
   text-align: left !important;    /* keep labels left-aligned on mobile */
 }

 /* Ensure the bars themselves don't add any extra left offset */
 .dm2-wrap { margin-left: 0 !important; padding-left: 0 !important; }
 .dm2-bar, .dm2-track { margin-left: 0 !important; padding-left: 0 !important; }
  
 }

/* Tiny phones: slightly smaller icon/name */
@media (max-width: 420px){
  .dm-icon{ width:72px !important; }
  .dm-key{ font-size: clamp(22px, 7.2vw, 32px) !important; }
 }

/* --- MOBILE (<=640px): stacked, left, smaller --- */
@media (max-width: 640px){
  .dm-title {
    text-align: left !important;
    margin: 6px 0 14px 0 !important;
    padding: 0 !important;
    line-height: 1.1 !important;
  }
  /* If you used spans for a 2-line title, stack them on phones */
  .dm-title-main,
  .dm-title-sub {
    display: block !important;
  }
  .dm-title-main {
  font-size: clamp(18px, 5vw, 22px) !important;   /* much smaller */
  font-weight: 300 !important;
  margin-bottom: 2px !important;
}

.dm-title-sub {
  font-size: clamp(15px, 4.5vw, 18px) !important; /* smaller too */
  font-weight: 300 !important;
  margin: 0 !important;
}
}

/* --- DESKTOP (>=641px): restore original centered, single line --- */
@media (min-width: 641px){
  .dm-title { 
    text-align: center !important; 
    margin-left: auto !important;   /* ensure visual centering even with container padding */
    margin-right: auto !important;
  }
  /* If spans exist, show them inline so it’s one line on desktop */
  .dm-title-main,
  .dm-title-sub { 
    display: inline !important; 
    margin: 0 !important; 
  }
}
</style>
""", unsafe_allow_html=True)

# =====================
# FINAL mobile-only title size override (wins the cascade)
# =====================
st.markdown("""
<style>
@media screen and (max-width: 640px){
  /* If your title is a single element (no spans), this alone is enough */
  .dm-title.dm-title{
    font-size: clamp(18px, 5.5vw, 22px) !important;
    line-height: 1.15 !important;
    margin: 6px 0 12px !important;
    text-align: left !important;  /* keep your mobile left align */
  }

  /* If you used the two-span markup, these also apply (harmless if spans don't exist) */
  .dm-title-main.dm-title-main{
    font-size: clamp(18px, 5.5vw, 22px) !important;
    font-weight: 300 !important;
    margin-bottom: 2px !important;
  }
  .dm-title-sub.dm-title-sub{
    font-size: clamp(15px, 4.8vw, 18px) !important;
    font-weight: 300 !important;
    margin: 0 !important;
  }
  /* Add more top margin before each main section header */
  div.dm-center:has(> div > div:contains("YOU")) {
    margin-top: 48px !important;
  }
  div.dm-center:has(> div > div:contains("YOUR SLEEP")) {
    margin-top: 52px !important;
  }
  div.dm-center:has(> div > div:contains("YOUR EXPERIENCE")) {
    margin-top: 56px !important;
  }
  /* Make "Dynamics of your experience" subtitle larger on mobile */
  div[data-testid="stMarkdownContainer"] > div:has(> div:contains("Dynamics of your experience")),
  div:has(> div:contains("Dynamics of your experience")) {
    font-size: 20px !important;      /* increase size */
    line-height: 1.3 !important;
    font-weight: 600 !important;     /* slightly bolder */
    letter-spacing: 0.2px;
  }
  
}
</style>
""", unsafe_allow_html=True)

# Change dynamics title size
st.markdown("""
<style>
/* Default (desktop) — slightly bigger than before */
.dm-subtitle-dynamics{
  font-size: 20px !important;   /* up from 18px */
  line-height: 1.3 !important;
  font-weight: 400 !important;  /* semi-bold */
  letter-spacing: 0.1px;
}

/* Mobile tweak — even bigger but lighter */
@media (max-width: 640px){
  .dm-subtitle-dynamics{
    font-size: 30px !important;   /* larger for phone readability */
    line-height: 1.28 !important;
    font-weight: 400 !important;  /* slightly lighter than desktop */
    letter-spacing: 0.15px;
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Default (desktop) — slightly larger, same style as Dynamics */
.dm-subtitle-trajectory{
  font-size: 20px !important;   /* up from 18px */
  line-height: 1.3 !important;
  font-weight: 400 !important;
  letter-spacing: 0.1px;
}

/* Mobile — larger, but lighter weight for visual balance */
@media (max-width: 640px){
  .dm-subtitle-trajectory{
    font-size: 24px !important;
    line-height: 1.28 !important;
    font-weight: 400 !important;
    letter-spacing: 0.15px;
  }
}
</style>
""", unsafe_allow_html=True)

# add space after profile desc
st.markdown("""
<style>
/* Add space after profile description (applies to all devices) */
.dm-desc {
  margin-bottom: 24px !important;  /* extra gap below the profile text */
}

/* Optional: slightly more breathing room on mobile */
@media (max-width: 640px) {
  .dm-desc {
    margin-bottom: 36px !important;
  }
}
</style>
""", unsafe_allow_html=True)

# Page starts higher up
st.markdown("""
<style>
/* --- Move QR code further down so main content can sit higher --- */
div[style*="qr_code_DM.png"],
div[style*="Participate!"] {
  top: 200px !important;   /* was 60px — move it lower */
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
@media (max-width: 640px) {
  /* Remove top padding/margin from the main container on mobile */
  [data-testid="stAppViewContainer"] > .main,
  section.main > div.block-container,
  .main .block-container,
  div.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
  }
  
  /* Also remove any top margin from the first element */
  [data-testid="stVerticalBlock"]:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }
  
  /* Lift the title up on mobile */
  .dm-title {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }
  
  /* Lift the QR code up on mobile */
  div[style*="Participate!"] {
    top: 80px !important;  /* Adjust this value - try 10px, 20px, 30px */
  }
}
</style>
""", unsafe_allow_html=True)


# =====================
# MOBILE-ONLY SPACERS (extra gaps before sections)
# =====================
st.markdown("""
<style>
@media (max-width: 640px){
  .dm-spacer-you    { height: 16px; }
  .dm-spacer-sleep  { height: 56px; }
  .dm-spacer-exp    { height: 56px; }
}
@media (min-width: 641px){
  .dm-spacer-you,
  .dm-spacer-sleep,
  .dm-spacer-exp { height: 0; }
}
</style>
""", unsafe_allow_html=True)

# =====================
# MOBILE: add extra vertical space *after* the title (below it)
# =====================
st.markdown("""
<style>
@media screen and (max-width: 640px){
  /* Add extra top margin to the first section after the title */
  .dm-row:first-of-type {
    margin-top: 46px !important;   /* pushes down "You drift into sleep like a" + icon + name + desc */
  }
}
</style>
""", unsafe_allow_html=True)

# =====================
# MOBILE: lift the title section (force override on its wrapper)
# =====================
st.markdown("""
<style>
@media screen and (max-width: 640px){

  /* Move the title upward by reducing the block container’s padding */
  div.block-container:first-of-type,
  [data-testid="stVerticalBlock"]:has(.dm-title),
  [data-testid="stVerticalBlock"] .dm-title {
    margin-top: -50px !important;     /* pull up the block containing the title */
    padding-top: 0 !important;
  }

  /* Also make sure the title element itself doesn’t get constrained */
  .dm-title {
    margin-top: 0 !important;         /* reset inner margin so container shift works cleanly */
  }
}
</style>
""", unsafe_allow_html=True)

# =====================
# DESKTOP: precisely lift the title container (not the first block)
# =====================
st.markdown("""
<style>
@media (min-width: 641px){

  /* Instead, directly target the title's parent container */
  [data-testid="stVerticalBlock"]:has(.dm-title) {
    margin-top: -10px !important;  /* Adjust this value as needed */
    padding-top: 0 !important;
  }

  /* Ensure the title itself doesn't add spacing back */
  .dm-title {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }
  /* 1) Lift the ACTUAL block container that holds the title */
  section.main div.block-container:has(.dm-title),
  [data-testid="stVerticalBlock"]:has(.dm-title) {
    margin-top: -120px !important;   /* tweak: -40 / -56 / -72 as you prefer */
    padding-top: 0 !important;
  }

  /* 2) Ensure the title itself doesn't add spacing back */
  .dm-title {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* --- Move QR code up on mobile only --- */
@media (max-width: 640px) {
  div[style*="position: absolute"][style*="Participate"] {
    top: 20px !important;   /* ← Adjust this: try 20px, 30px, 40px, 50px */
  }
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

# =====================================================
# Language-based normalization of questionnaire columns
# =====================================================

# Make a local mutable copy
record = dict(record)

# Detect which questionnaire version was completed
q_base = record.get("questionnaire_complete")
q_fr   = record.get("questionnaire_fr_complete")
q_sp   = record.get("questionnaire_en_complete")   # Spanish version

def _strip_suffix_keep_first(rec, suffix, n_keep=5):
    """
    Keep the first n_keep columns as-is, and for all other columns
    that end with the given suffix, remove the suffix.

    Example:
      freq_mindwandering_fr -> freq_mindwandering
    """
    new_rec = {}

    # Keep the first n_keep keys exactly as they are
    keys = list(rec.keys())
    for k in keys[:n_keep]:
        new_rec[k] = rec[k]

    # Add all suffixed questionnaire variables with suffix removed
    for k, v in rec.items():
        if k.endswith(suffix):
            base = k[:-len(suffix)]
            new_rec[base] = v

    return new_rec

# Apply your rules:
# - If questionnaire_complete = 2 → English, keep as is
# - If questionnaire_fr_complete = 2 → keep first 5 cols + *_fr → strip _fr
# - If questionnaire_en_complete = 2 → keep first 5 cols + *_en → strip _en
if q_fr == "2" or q_fr == 2:
    record = _strip_suffix_keep_first(record, "_fr")

elif q_sp == "2" or q_sp == 2:
    record = _strip_suffix_keep_first(record, "_en")

else:
    # Default: original English version (questionnaire_complete = 2)
    # do nothing, keep the original column names
    pass


# =====================================================
# Language detection
# =====================================================
if q_fr in ("2", 2):
    LANG = "fr"
elif q_sp in ("2", 2):
    LANG = "es"     # Spanish version uses _en suffix → language = ES
else:
    LANG = "en"



# =====================================================
# Translation dictionary
# =====================================================
TEXT = {
    # -----------------------
    # SECTION HEADERS
    # -----------------------
    "DRIFTING MINDS STUDY": {
        "en": "DRIFTING MINDS STUDY",
        "fr": "ÉTUDE DRIFTING MINDS",
        "es": "ESTUDIO DRIFTING MINDS",
    },
    "YOU": {
        "en": "YOU",
        "fr": "VOUS",
        "es": "TÚ",
    },
    "YOUR SLEEP": {
        "en": "YOUR SLEEP",
        "fr": "VOTRE SOMMEIL",
        "es": "TU SUEÑO",
    },
    "YOUR EXPERIENCE": {
        "en": "YOUR EXPERIENCE",
        "fr": "VOTRE EXPÉRIENCE",
        "es": "TU EXPERIENCIA",
    },
    "Dynamics of your experience": {
        "en": "Dynamics of your experience",
        "fr": "Dynamique de votre expérience",
        "es": "Dinámica de tu experiencia",
    },
    "Intensity of your experience": {
        "en": "Intensity of your experience",
        "fr": "Intensité de votre expérience",
        "es": "Intensidad de tu experiencia",
    },
    "Your trajectory": {
       "en": "Your trajectory",
       "fr": "Votre trajectoire",
       "es": "Tu trayectoria",
   },

    # -----------------------
    # AXIS LABELS
    # -----------------------
    "hours": {"en": "hours", "fr": "heures", "es": "horas"},
    "minutes": {"en": "minutes", "fr": "minutes", "es": "minutos"},
    "low": {"en": "low", "fr": "faible", "es": "bajo"},
    "high": {"en": "high", "fr": "élevé", "es": "alto"},
    "Matching strength": {
        "en": "Matching strength",
        "fr": "Niveau de correspondance",
        "es": "Nivel de correspondencia",
    },

    # -----------------------
    # PROFILE HEADER
    # -----------------------
    "You drift into sleep like a": {
        "en": "You drift into sleep like a",
        "fr": "Vous glissez dans le sommeil comme un(e)",
        "es": "Te deslizas hacia el sueño como un(a)",
    },

    
        
    # -----------------------
    # POPULATION STATS LINE
    # -----------------------
    "{name}s represent {perc}% of the population.": {
        "en": "{name}s represent {perc}% of the population.",
        "fr": "{name} représente {perc}% de la population.",
        "es": "{name} representa el {perc}% de la población.",
    },

    # -----------------------
    # RADAR LABELS
    # -----------------------
    "vivid": {"en": "vivid", "fr": "vive", "es": "vívida"},
    "immersive": {"en": "immersive", "fr": "immersive", "es": "inmersiva"},
    "bizarre": {"en": "bizarre", "fr": "bizarre", "es": "extraña"},
    "spontaneous": {"en": "spontaneous", "fr": "spontanée", "es": "espontánea"},
    "fleeting": {"en": "fleeting", "fr": "fugace", "es": "fugaz"},
    "positive\nemotions": {
        "en": "positive\nemotions",
        "fr": "émotions\npositives",
        "es": "emociones\npositivas",
    },
    "sleepy": {"en": "sleepy", "fr": "somnolent(e)", "es": "adormilado/a"},

    # -----------------------
    # HISTOGRAM TITLES
    # -----------------------
    "Your visual imagery at wake: {val}": {
        "en": "Your visual imagery at wake: {val}",
        "fr": "Votre imagerie visuelle à l’éveil : {val}",
        "es": "Tu imaginación visual al despertar: {val}",
    },
    "Your self-rated creativity: {val}": {
        "en": "Your self-rated creativity: {val}",
        "fr": "Votre créativité auto-évaluée : {val}",
        "es": "Tu creatividad autoevaluada: {val}",
    },
    "Your self-rated anxiety: {val}": {
        "en": "Your self-rated anxiety: {val}",
        "fr": "Votre anxiété auto-évaluée : {val}",
        "es": "Tu ansiedad autoevaluada: {val}",
    },
    
    # -----------------------
   # Timeline axis labels
   # -----------------------
   "LBL_Awake": {
       "en": "Awake",
       "fr": "Éveillé",
       "es": "Despierto",
   },
   "LBL_Asleep": {
       "en": "Asleep",
       "fr": "Endormi",
       "es": "Dormido",
   },

    # -----------------------
    # INFO / ERROR MESSAGES
    # -----------------------
    "Population data for creativity unavailable.": {
        "en": "Population data for creativity unavailable.",
        "fr": "Les données de population pour la créativité ne sont pas disponibles.",
        "es": "Los datos de población sobre creatividad no están disponibles.",
    },
    "Population data for anxiety unavailable.": {
        "en": "Population data for anxiety unavailable.",
        "fr": "Les données de population pour l’anxiété ne sont pas disponibles.",
        "es": "Los datos de población sobre ansiedad no están disponibles.",
    },
    "Population data unavailable.": {
        "en": "Population data unavailable.",
        "fr": "Les données de population ne sont pas disponibles.",
        "es": "Los datos de población no están disponibles.",
    },
    "Could not compute profile likelihoods for this record.": {
        "en": "Could not compute profile likelihoods for this record.",
        "fr": "Impossible de calculer les probabilités de profil pour ce participant.",
        "es": "No se pudieron calcular las probabilidades de perfil para este participante.",
    },
    
    # -----------------------
   # LEGEND / SMALL LABELS
   # -----------------------
   "you": {
       "en": "you",
       "fr": "vous",
       "es": "tú",
   },
   "world": {
       "en": "world",
       "fr": "monde",
       "es": "mundo",
   },
   
   "WORLD_AVERAGE_TAG": {
        "en": "world average",
        "fr": "moyenne mondiale",
        "es": "promedio mundial",
    },
   
   "no content": {
    "en": "no content",
    "fr": "aucun contenu",
    "es": "sin contenido",
},

   # -----------------------
   # DIMENSION BAR NAMES
   # -----------------------
   "Vivid":       {"en": "Vivid",       "fr": "Vif",              "es": "Vívido"},
   "Bizarre":     {"en": "Bizarre",     "fr": "Bizarre",          "es": "Extraño"},
   "Immersive":   {"en": "Immersive",   "fr": "Immersif",         "es": "Inmersivo"},
   "Spontaneous": {"en": "Spontaneous", "fr": "Spontané",         "es": "Espontáneo"},
   "Emotional":   {"en": "Emotional",   "fr": "Émotionnel",       "es": "Emocional"},

   # Anchors for the horizontal bars
   "Dull":           {"en": "Dull",           "fr": "Terne",           "es": "Apagado"},
   "Vivid_anchor":   {"en": "Vivid",          "fr": "Vif",             "es": "Vívido"},
   "Ordinary":       {"en": "Ordinary",       "fr": "Ordinaire",       "es": "Ordinario"},
   "Bizarre_anchor": {"en": "Bizarre",        "fr": "Bizarre",         "es": "Extraño"},
   "External-oriented": {
       "en": "External-oriented", "fr": "Tourné vers l'extérieur", "es": "Orientado al exterior",
   },
   "Immersive_anchor": {
       "en": "Immersive", "fr": "Immersif", "es": "Inmersivo",
   },
   "Voluntary": {"en": "Voluntary", "fr": "Volontaire", "es": "Voluntario"},
   "Spontaneous_anchor": {
       "en": "Spontaneous", "fr": "Spontané", "es": "Espontáneo",
   },
   "Negative": {"en": "Negative", "fr": "Négatif", "es": "Negativo"},
   "Positive": {"en": "Positive", "fr": "Positif", "es": "Positivo"},
   
   # -----------------------
   # Notes bars
   # -----------------------
    "EXPERIENCE_DIM_NOTES_HTML": {
     "en": (
         '<strong>Vivid</strong>: brightness or contrast of your imagery, '
         'or the loudness of what you hear. '
         '<strong>Bizarre</strong>: how unusual or unrealistic the content feels. '
         '<strong>Immersive</strong>: how deeply absorbed you are in your mental content. '
         '<strong>Spontaneous</strong>: how much the content comes to you on its own, '
         'without deliberate control. '
         '<strong>Emotional</strong>: how strongly you felt emotions.<br>'
         '<span style="font-size:0.6rem; margin-right:6px;">⚫️</span>'
         '= world average: represents the average scores from 1,400 people worldwide.'
     ),
     "fr": (
         '<strong>Vif</strong> : luminosité ou contraste de vos images mentales, '
         'ou intensité de ce que vous entendez. '
         '<strong>Bizarre</strong> : à quel point le contenu vous paraît inhabituel ou irréaliste. '
         '<strong>Immersif</strong> : à quel point vous êtes absorbé(e) par ce qui se passe dans votre esprit. '
         '<strong>Spontané</strong> : à quel point le contenu vient tout seul, sans contrôle volontaire. '
         '<strong>Émotionnel</strong> : à quel point vous avez ressenti des émotions.<br>'
         '<span style="font-size:0.6rem; margin-right:6px;">⚫️</span>'
         '= moyenne mondiale : correspond à la moyenne des scores de 1 400 personnes dans le monde.'
     ),
     "es": (
         '<strong>Vívido</strong>: brillo o contraste de tus imágenes mentales, '
         'o intensidad de lo que oyes. '
         '<strong>Bizarro</strong>: cuán inusual o poco realista se siente el contenido. '
         '<strong>Inmersivo</strong>: hasta qué punto estás profundamente absorbido/a '
         'por tu contenido mental. '
         '<strong>Espontáneo</strong>: hasta qué punto el contenido aparece por sí solo, '
         'sin control deliberado. '
         '<strong>Emocional</strong>: cuán intensamente sentiste emociones.<br>'
         '<span style="font-size:0.6rem; margin-right:6px;">⚫️</span>'
         '= media mundial: representa la media de las puntuaciones de 1.400 personas en todo el mundo.'
     ),
 },

   # -----------------------
   # Buttons
   # -----------------------
    "DOWNLOAD_BUTTON": {
            "en": "⬇️ Download",
            "fr": "⬇️ Télécharger",
            "es": "⬇️ Descargar",
        },
        "COPY_LINK_BUTTON": {
            "en": "🔗 Copy link",
            "fr": "🔗 Copier le lien",
            "es": "🔗 Copiar el enlace",
        },
        "COPY_LINK_COPIED": {
            "en": "✅ Copied!",
            "fr": "✅ Copié !",
            "es": "✅ Copiado",
        },

   # -----------------------
   # YOUR SLEEP TITLES
   # -----------------------
   "You fall asleep in {val} minutes": {
       "en": "You fall asleep in {val} minutes",
       "fr": "Vous vous endormez en {val} minutes",
       "es": "Te duermes en {val} minutos",
   },
   "You sleep {val} hours on average": {
       "en": "You sleep {val} hours on average",
       "fr": "Vous dormez en moyenne {val} heures",
       "es": "Duermes una media de {val} horas",
   },
   "Your sleep duration": {
       "en": "Your sleep duration",
       "fr": "La durée de votre sommeil",
       "es": "La duración de tu sueño",
   },

   # Chronotype titles
   "You are a morning type": {
       "en": "You are a morning type",
       "fr": "Vous êtes de type matinal",
       "es": "Eres de tipo matutino",
   },
   "You are an evening type": {
       "en": "You are an evening type",
       "fr": "Vous êtes de type vespéral",
       "es": "Eres de tipo vespertino",
   },
   "You have no chronotype": {
       "en": "You have no preference",
       "fr": "Vous n'avez pas de chronotype",
       "es": "No tienes cronotipo",
   },
   "Chronotype": {
       "en": "Chronotype",
       "fr": "Chronotype",
       "es": "Cronotipo",
   },

   # Dream recall titles
   "You recall your dreams\nless than once a month": {
       "en": "You recall your dreams\nless than once a month",
       "fr": "Vous vous souvenez de vos rêves\nmoins d'une fois par mois",
       "es": "Recuerdas tus sueños\nmenos de una vez al mes",
   },
   "You recall your dreams\nonce or twice a month": {
       "en": "You recall your dreams\nonce or twice a month",
       "fr": "Vous vous souvenez de vos rêves\nune à deux fois par mois",
       "es": "Recuerdas tus sueños\nuna o dos veces al mes",
   },
   "You recall your dreams\nonce a week": {
       "en": "You recall your dreams\nnce a week",
       "fr": "Vous vous souvenez de vos rêves\nune fois par semaine",
       "es": "Recuerdas tus sueños\nuna vez por semana",
   },
   "You recall your dreams\nseveral times a week": {
       "en": "You recall your dreams\nseveral times a week",
       "fr": "Vous vous souvenez de vos rêves\nplusieurs fois par semaine",
       "es": "Recuerdas tus sueños\nvarias veces por semana",
   },
   "You recall your dreams\nevery day": {
       "en": "You recall your dreams every day",
       "fr": "Vous vous souvenez de vos rêves\ntous les jours",
       "es": "Recuerdas tus sueños\ntodos los días",
   },
   "Dream recall": {
       "en": "Dream recall",
       "fr": "Rappel de rêves",
       "es": "Recuerdo de sueños",
   },

   # Chronotype / recall tick labels
   "morning": {"en": "morning", "fr": "matin", "es": "mañana"},
   "evening": {"en": "evening", "fr": "soir", "es": "tarde"},
   "no type": {"en": "no type", "fr": "aucun type", "es": "sin tipo"},

   "<1/month": {"en": "<1/month", "fr": "<1/mois", "es": "<1/mes"},
   "1-2/month": {"en": "1–2/month", "fr": "1–2/mois", "es": "1–2/mes"},
   "1/week": {"en": "1/week", "fr": "1/semaine", "es": "1/semana"},
   "several/week": {"en": "several/week", "fr": "plusieurs/sem.", "es": "varias/sem."},
   "every day": {"en": "every day", "fr": "tous les jours", "es": "tous les jours"
    },
   
   # Notes
    "YOU_SECTION_NOTE": {
     "en": 'Grey info ("world”) represents data from 1,400 people worldwide.<br>'
           'Left graph: vividness of visual imagery when awake (VVIQ score). '
           'Middle and right: self-rated creativity and anxiety over the past year.',
     "fr": 'La partie grise ("monde") représente les données de 1 400 personnes dans le monde.<br>'
           'Graphique de gauche : vivacité des images mentales à l’éveil (score VVIQ). '
           'Au centre et à droite : créativité et anxiété auto-évaluées sur l’année écoulée.',
     "es": 'La parte gris ("mundo") representa los datos de 1.400 personas en todo el mundo.<br>'
          'Gráfico de la izquierda: viveza de las imágenes mentales en vigilia (puntuación VVIQ). '
          'En el centro y la derecha: creatividad y ansiedad autoevaluadas durante el último año.',
 },
    
    # -----------------------
    # PROFILE LIKELIHOOD BAR (12 profiles)
    # -----------------------
    "How much you match each profile": {
        "en": "How much you match each profile",
        "fr": "À quel point vous correspondez à chaque profil",
        "es": "Cuánto encajas con cada perfil",
    },
    "PROFILE_BARS_NOTE": {
        "en": (
            "Each bar shows how closely your answers align with each profile, "
            "from your strongest match (left) to your weakest (right)."
        ),
        "fr": (
            "Chaque barre montre à quel point vos réponses s’alignent avec chaque profil, "
            "du profil qui vous correspond le plus (à gauche) au moins marqué (à droite)."
        ),
        "es": (
            "Cada barra muestra hasta qué punto tus respuestas se alinean con cada perfil, "
            "desde el perfil que más te corresponde (izquierda) hasta el más débil (derecha)."
        ),
    },
    
    # -----------------------
    # Your experience - notes 
    # -----------------------
    "YOUR_EXPERIENCE_LAYOUT_NOTE": {
        "en": (
            "Left graph: your own trajectory selection. "
            "Middle graph: your self-rated intensity scores of your typical mental content (1 = low, 6 = high). "
            "Right graph: mental content that most often emerged early vs. late as you fall asleep."
        ),
        "fr": (
            "Graphique de gauche : la trajectoire que vous avez choisie. "
            "Graphique central : vos scores d’intensité pour votre contenu mental typique (1 = faible, 6 = élevé). "
            "Graphique de droite : les contenus mentaux qui apparaissent le plus souvent au début vs à la fin de l’endormissement."
        ),
        "es": (
            "Gráfico de la izquierda: tu propia selección de trayectoria. "
            "Gráfico central: tus puntuaciones de intensidad para tu contenido mental típico (1 = bajo, 6 = alto). "
            "Gráfico de la derecha: el contenido mental que aparece con más frecuencia al principio vs al final de quedarte dormido."
        ),
    },
    
    # -----------------------
    # PROFILE NAMES (display)
    # -----------------------
    "Dreamweaver": {
    "en": "Dreamweaver",
    "fr": "Tisseur de songes",
    "es": "Tejedor(a) de sueños",
    },
    "Quick Diver": {
        "en": "Quick Diver",
        "fr": "Plongeur éclair",
        "es": "Buceador(a) fulminante",
    },
    "Fantasizer": {
        "en": "Fantasizer",
        "fr": "Scénariste",
        "es": "Guionista",
    },
    "Archivist": {
        "en": "Archivist",
        "fr": "Archiviste",
        "es": "Archivista",
    },
    "Worrier": {
        "en": "Worrier",
        "fr": "Esprit en remous",
        "es": "Mente en agitación",
    },
    "Freewheeler": {
        "en": "Freewheeler",
        "fr": "Lâche-prise",
        "es": "Soltador(a)",
    },
    "Quiet Mind": {
        "en": "Quiet Mind",
        "fr": "Esprit silencieux",
        "es": "Mente silenciosa",
    },
    "Radio Tuner": {
        "en": "Radio Tuner",
        "fr": "Zappeur mental",
        "es": "Sintonizador(a) mental",
    },
    "Strategist": {
        "en": "Strategist",
        "fr": "Stratège",
        "es": "Estratega",
    },
    "Sentinel": {
        "en": "Sentinel",
        "fr": "Sentinelle",
        "es": "Centinela",
    },
    "Fragmented Mind": {
        "en": "Fragmented Mind",
        "fr": "Esprit morcelé",
        "es": "Mente fragmentada",
    },
    "Pragmatic": {
        "en": "Pragmatic",
        "fr": "Pragmatique",
        "es": "Pragmático/a",
    },
        
    # -----------------------
    # PROFILE DESCRIPTIONS (display)
    # -----------------------
    "PROFILE_DESC_DREAMWEAVER": {
        "en": "You drift into vivid, sensory mini-dreams as you fall asleep.",
        "fr": "Vous glissez vers le sommeil dans de petites mini-rêveries vives et sensorielles.",
        "es": "Te deslizas hacia el sueño a través de pequeños mini-sueños vívidos y sensoriales.",
    },
    "PROFILE_DESC_QUICK_DIVER": {
        "en": "You fall asleep quickly, especially when you already feel sleepy.",
        "fr": "Vous vous endormez rapidement, surtout lorsque vous vous sentez déjà somnolent(e).",
        "es": "Te duermes rápido, especialmente cuando ya te sientes somnoliento/a.",
    },
    "PROFILE_DESC_FANTASIZER": {
        "en": "You tend to drift into constructed scenes or stories, with rich imagery and absorption.",
        "fr": "Vous dérivez vers des scènes ou des histoires construites, avec des images riches et une forte immersion.",
        "es": "Tiendes a entrar en escenas o historias construidas, con imágenes ricas y una fuerte inmersión.",
    },
    "PROFILE_DESC_ARCHIVIST": {
        "en": "You mentally revisit past events as you fall asleep, like an archivist sorting through memories.",
        "fr": "Vous revisitez mentalement des événements passés en vous endormant, comme un(e) archiviste triant ses souvenirs.",
        "es": "Revisitas mentalmente hechos pasados al quedarte dormido/a, como un archivero ordenando recuerdos.",
    },
    "PROFILE_DESC_WORRIER": {
        "en": "Your mind keeps turning over worries or emotionally charged thoughts before sleep.",
        "fr": "Votre esprit ressasse des inquiétudes ou des pensées chargées d’émotion avant de s’endormir.",
        "es": "Tu mente sigue dando vueltas a preocupaciones o pensamientos cargados de emoción antes de dormirte.",
    },
    "PROFILE_DESC_FREEWHEELER": {
        "en": "You start with intentional thoughts, then let go into more spontaneous imagery.",
        "fr": "Vous commencez avec des pensées intentionnelles, puis vous lâchez prise vers des images plus spontanées.",
        "es": "Empiezas con pensamientos intencionales y luego te dejas llevar hacia imágenes más espontáneas.",
    },
    "PROFILE_DESC_QUIET_MIND": {
        "en": "Your mental field stays relatively quiet at lights-out—less vivid, less odd, less all-absorbing.",
        "fr": "Votre paysage mental reste relativement calme à l’extinction des lumières — moins vif, moins étrange, moins absorbant.",
        "es": "Tu paisaje mental se mantiene relativamente tranquilo al apagar la luz: menos vivo, menos extraño y menos absorbente.",
    },
    "PROFILE_DESC_RADIO_TUNER": {
        "en": "Your mind shifts rapidly between thoughts, images, and sensations, like tuning through mental stations.",
        "fr": "Votre esprit passe rapidement d’une pensée, image ou sensation à l’autre, comme si vous faisiez défiler des stations mentales.",
        "es": "Tu mente cambia rápidamente entre pensamientos, imágenes y sensaciones, como si fueras pasando por estaciones mentales.",
    },
    "PROFILE_DESC_STRATEGIST": {
        "en": "Your thoughts stay clear and structured, focused on plans and everyday logic.",
        "fr": "Vos pensées restent claires et structurées, centrées sur vos plans et une logique du quotidien.",
        "es": "Tus pensamientos se mantienen claros y estructurados, centrados en tus planes y en la lógica cotidiana.",
    },
    "PROFILE_DESC_SENTINEL": {
        "en": "Your mind stays mostly quiet but your attention monitors the environment, like a sentinel keeping watch.",
        "fr": "Votre esprit reste plutôt calme mais une partie de vous surveille l’environnement, comme une sentinelle qui veille.",
        "es": "Tu mente permanece bastante tranquila, pero una parte de ti vigila el entorno, como una centinela en guardia.",
    },
    "PROFILE_DESC_FRAGMENTED_MIND": {
        "en": "Your experience breaks into fleeting fragments of thoughts, images or sensations that vanish quickly.",
        "fr": "Votre expérience se fragmente en brèves pensées, images ou sensations qui disparaissent très vite.",
        "es": "Tu experiencia se fragmenta en breves pensamientos, imágenes o sensaciones que desaparecen rápidamente.",
    },
    "PROFILE_DESC_PRAGMATIC": {
        "en": "Your thoughts stay practical and grounded in everyday concerns rather than drifting into the dreamlike.",
        "fr": "Vos pensées restent pratiques et ancrées dans le quotidien plutôt que de dériver vers le monde onirique.",
        "es": "Tus pensamientos se mantienen prácticos y anclados en lo cotidiano, en lugar de derivar hacia lo onírico.",
    },
    
    # -----------------------
    # Timeline 
    # -----------------------
    "LBL_freq_think_ordinary": {
        "en": "thinking logical thoughts",
        "fr": "penser de façon logique",
        "es": "pensar de forma lógica",
    },
    "LBL_freq_scenario": {
        "en": "imagining scenarios",
        "fr": "imaginer des scénarios",
        "es": "imaginar escenarios",
    },
    "LBL_freq_negative": {
        "en": "feeling negative",
        "fr": "ressentir du négatif",
        "es": "sentirse negativo",
    },
    "LBL_freq_absorbed": {
        "en": "feeling absorbed",
        "fr": "se sentir absorbé",
        "es": "sentirse absorbido",
    },
    "LBL_freq_percept_fleeting": {
        "en": "fleeting perceptions",
        "fr": "perceptions fugaces",
        "es": "percepciones fugaces",
    },
    "LBL_freq_think_bizarre": {
        "en": "thinking strange things",
        "fr": "penser des choses étranges",
        "es": "pensar cosas extrañas",
    },
    "LBL_freq_planning": {
        "en": "planning the day",
        "fr": "planifier sa journée",
        "es": "planificar el día",
    },
    "LBL_freq_spectator": {
        "en": "feeling like a spectator",
        "fr": "se sentir spectateur",
        "es": "sentirse espectador",
    },
    "LBL_freq_ruminate": {
        "en": "ruminating",
        "fr": "ruminer",
        "es": "rumiar",
    },
    "LBL_freq_percept_intense": {
        "en": "intense perceptions",
        "fr": "perceptions intenses",
        "es": "percepciones intensas",
    },
    "LBL_freq_percept_narrative": {
        "en": "narrative scenes",
        "fr": "scènes narratives",
        "es": "escenas narrativas",
    },
    "LBL_freq_percept_ordinary": {
        "en": "ordinary perceptions",
        "fr": "perceptions ordinaires",
        "es": "percepciones ordinarias",
    },
    "LBL_freq_time_perc_fast": {
        "en": "time feels fast",
        "fr": "le temps semble rapide",
        "es": "el tiempo parece ir rápido",
    },
    "LBL_freq_percept_vague": {
        "en": "vague perceptions",
        "fr": "perceptions vagues",
        "es": "percepciones vagas",
    },
    "LBL_freq_replay": {
        "en": "replaying the day",
        "fr": "rejouer sa journée",
        "es": "repasar el día",
    },
    "LBL_freq_percept_bizarre": {
        "en": "strange perceptions",
        "fr": "perceptions étranges",
        "es": "percepciones extrañas",
    },
    "LBL_freq_emo_intense": {
        "en": "feeling intense emotions",
        "fr": "ressentir des émotions intenses",
        "es": "sentir emociones intensas",
    },
    "LBL_freq_percept_continuous": {
        "en": "continuous perceptions",
        "fr": "perceptions continues",
        "es": "percepciones continuas",
    },
    "LBL_freq_think_nocontrol": {
        "en": "losing control of thoughts",
        "fr": "perdre le contrôle de ses pensées",
        "es": "perder el control de los pensamientos",
    },
    "LBL_freq_percept_dull": {
        "en": "dull perceptions",
        "fr": "perceptions ternes",
        "es": "percepciones apagadas",
    },
    "LBL_freq_actor": {
        "en": "acting in the scene",
        "fr": "agir dans la scène",
        "es": "actuar en la escena",
    },
    "LBL_freq_think_seq_bizarre": {
        "en": "thinking illogical thoughts",
        "fr": "penser de façon illogique",
        "es": "pensar de forma ilógica",
    },
    "LBL_freq_percept_precise": {
        "en": "precise perceptions",
        "fr": "perceptions précises",
        "es": "percepciones precisas",
    },
    "LBL_freq_percept_imposed": {
        "en": "imposed perceptions",
        "fr": "perceptions imposées",
        "es": "percepciones impuestas",
    },
    "LBL_freq_hear_env": {
        "en": "hearing my environment",
        "fr": "entendre mon environnement",
        "es": "oír mi entorno",
    },
    "LBL_freq_positive": {
        "en": "feeling positive",
        "fr": "se sentir positif",
        "es": "sentirse positivo",
    },
    "LBL_freq_think_seq_ordinary": {
        "en": "thinking logical thoughts",
        "fr": "penser de façon logique",
        "es": "pensar de forma lógica",
    },
    "LBL_freq_percept_real": {
        "en": "perceptions feel real",
        "fr": "les perceptions semblent réelles",
        "es": "las percepciones parecen reales",
    },
    "LBL_freq_time_perc_slow": {
        "en": "time feels slow",
        "fr": "le temps semble lent",
        "es": "el tiempo parece lento",
    },
    "LBL_freq_syn": {
        "en": "experiencing synaesthesia",
        "fr": "vivre une synesthésie",
        "es": "experimentar sinestesia",
    },
    "LBL_freq_creat": {
        "en": "feeling creative",
        "fr": "se sentir créatif",
        "es": "sentirse creativo",
    },




    "DISCLAIMER_NOTE": {
    "en": (
        'These results are automatically generated from your responses to the '
        'Drifting Minds questionnaire.<br>'
        'They are meant for research and self-reflection only — not as medical or diagnostic advice.<br>'
        'For any questions, contact '
        '<a href="mailto:driftingminds@icm-institute.org" '
        'style="color:#7C3AED; text-decoration:none;">driftingminds@icm-institute.org</a>.<br>'
        '<em>Researcher of the study: Nicolas Decat, Paris Brain Institute</em>.'
    ),
    "fr": (
        'Ces résultats sont générés automatiquement à partir de vos réponses au '
        'questionnaire Drifting Minds.<br>'
        'Ils sont destinés à la recherche et à la réflexion personnelle uniquement. '
        "ils ne constituent ni un avis médical ni un outil de diagnostic.<br>"
        'Pour toute question, contactez '
        '<a href="mailto:driftingminds@icm-institute.org" '
        'style="color:#7C3AED; text-decoration:none;">driftingminds@icm-institute.org</a>.<br>'
        '<em>Chercheur de l’étude : Nicolas Decat, Institut du Cerveau</em>.'
    ),
    "es": (
        'Estos resultados se generan automáticamente a partir de tus respuestas al '
        'cuestionario Drifting Minds.<br>'
        'Están pensados solo para la investigación y la reflexión personal. '
        'No constituyen consejo médico ni una herramienta de diagnóstico.<br>'
        'Para cualquier pregunta, contacta con '
        '<a href="mailto:driftingminds@icm-institute.org" '
        'style="color:#7C3AED; text-decoration:none;">driftingminds@icm-institute.org</a>.<br>'
        '<em>Investigador del estudio : Nicolas Decat, Paris Brain Institute</em>.'
    ),
},
        
    

}


# =====================================================
# translation function
# =====================================================
def tr(key, **kwargs):
    """Translate a display string into the participant's language."""
    entry = TEXT.get(key)
    if entry is None:
        return key.format(**kwargs) if kwargs else key
    text = entry.get(LANG, entry.get("en", key))
    return text.format(**kwargs) if kwargs else text





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
    # Quick Diver
    # =====================================================================
    "Quick Diver": {
        "features": [
            {"type": "var", "key": ["sleep_latency"],                   "norm": norm_latency_auto, "norm_kwargs": {"cap_minutes": CAP_MIN}, "target": 0.05, "weight": 1.2,  "hit_op": "lte"},
            {"type": "var", "key": ["degreequest_sleepiness"],          "norm": norm_1_6, "norm_kwargs": {}, "target": 0.60, "weight": 0.8},
            {"type":"var","key":["trajectories"],                       "norm": norm_eq, "norm_kwargs": {"value": 2}, "target": 1.0, "weight": 1.2}
        ],
        "description": "You fall asleep quickly, especially when you already feel sleepy.",
        "icon": "dolphin.svg",
    },
    
       
    # =====================================================================
    # Fantasizer
    # =====================================================================
    "Fantasizer": {
        "features": [
            # Core: frequent scenario-building
            {"type": "var", "key": ["freq_scenario"],
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.80, "weight": 1.4},
    
            # Support: can slip into scenarios anytime (percentage / propensity)
            {"type": "var", "key": ["anytime_20"],
             "norm": norm_1_100, "norm_kwargs": {}, "target": 0.60, "weight": 0.9},
    
            # Support: vivid imagery
            {"type": "var", "key": ["degreequest_vividness"],
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.65, "weight": 0.9},
    
            # Support: immersive absorption
            {"type": "var", "key": ["degreequest_immersive"],
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.60, "weight": 0.8},
    
            # Optional flavor: a touch of bizarreness helps but isn’t required
            {"type": "var", "key": ["degreequest_bizarre"],
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.55, "weight": 0.5},
    ],
    "description": "You tend to drift into constructed scenes or stories, with richer imagery and absorption.",
    "icon": "monkey.svg",
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
    "icon": "elephant.svg",
    },
    
    # =====================================================================
    # Worrier
    # =====================================================================
    "Worrier": {
        "features": [
            # Core — frequent ruminative thoughts
            {"type": "var", "key": ["freq_ruminate"], 
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.85, "weight": 1.4},
    
            # Often accompanied by negative tone or worry
            {"type": "var", "key": ["freq_negative"], 
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.70, "weight": 0.9},
    
            # Higher trait anxiety or tension
            {"type": "var", "key": ["anxiety"], 
             "norm": norm_1_100, "norm_kwargs": {}, "target": 0.70, "weight": 0.9},
    
            # Longer sleep latency — mind keeps spinning before switching off
            {"type": "var", "key": ["sleep_latency"], 
             "norm": norm_latency_auto, "norm_kwargs": {"cap_minutes": CAP_MIN}, "target": 0.35, "weight": 1.1},
    
            # Emotional involvement (often high)
            {"type": "var", "key": ["degreequest_emotionality"], 
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.65, "weight": 0.8},
    
            # Low spontaneity helps differentiate from Freewheeler
            {"type": "var", "key": ["degreequest_spontaneity"], 
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.40, "weight": 0.6},
    ],
    "description": "Your mind keeps turning over thoughts, often with tension or analysis before sleep. Longer latency and emotional charge reflect a tendency to ruminate.",
    "icon": "sheep.svg",
    },


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
            # Low intensity across core dimensions (note the lte ops)
            {"type": "var", "key": ["degreequest_vividness"],
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.25, "weight": 1.0, "hit_op": "lte"},
    
            {"type": "var", "key": ["degreequest_bizarre"],
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.25, "weight": 1.0, "hit_op": "lte"},
    
            {"type": "var", "key": ["degreequest_immersive"],
             "norm": norm_1_6, "norm_kwargs": {}, "target": 0.30, "weight": 0.9, "hit_op": "lte"},
    
            # Often settles a bit faster (soft cue)
            {"type": "var", "key": ["sleep_latency"],
             "norm": norm_latency_auto, "norm_kwargs": {"cap_minutes": CAP_MIN},
             "target": 0.18, "weight": 0.5, "hit_op": "lte"},
    ],
    "description": "Your mental field stays relatively quiet at lights-out—less vivid, less odd, less all-absorbing.",
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
    # Sentinel
    # =====================================================================
    "Sentinel": {
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
    a = np.asarray(values, float); b = np.asarray(targets, float); w = np.asarray(weights, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if not np.any(mask):
        return np.inf
    d2 = np.sum(w[mask] * (a[mask] - b[mask])**2)
    W  = np.sum(w[mask])
    # Root-mean-weighted-square error (comparable across profiles)
    return np.sqrt(d2 / max(W, 1e-9))


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

def _eligible_for_hit(record, feat):
    """
    Return True if the feature should count toward the AND-penalty
    (i.e., same gating you use when computing the feature value).
    """
    # single condition
    if "only_if" in feat and not _eval_condition(record, feat["only_if"]):
        return False

    # all-of conditions
    if feat.get("only_if_all"):
        for c in feat["only_if_all"]:
            if not _eval_condition(record, c):
                return False

    # any-of conditions
    if feat.get("only_if_any"):
        if not any(_eval_condition(record, c) for c in feat["only_if_any"]):
            return False

    return True


def _feature_hit(record, feat, value, tol=0.95):
    """
    Returns:
      1  → eligible + hit
      0  → eligible + miss
      None → ineligible (doesn't count toward K nor against it)
    """
    # 2a) same gating as value computation
    if not _eligible_for_hit(record, feat):
        return None

    # 2b) missing value = ineligible (don’t punish missing data)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    tgt = float(feat.get("target", float("nan")))
    op  = (feat.get("hit_op") or "gte").lower()
    v   = float(value)

    if op == "lte":
        return 1 if v <= (tgt / tol) else 0
    else:  # default: gte
        return 1 if v >= (tgt * tol) else 0



def assign_profile_from_record(record):
    """
    For each profile, compute a weighted distance using only 'var' features.
    Returns (best_profile_name, {}).
    """
    scores = {}  # kept for compatibility with caller
    best_name, best_dist, best_r = None, np.inf, -1.0

    # AND-ish knobs:
    K_RATIO = 0.20   # need ~60% of eligible criteria to be met
    GAMMA   = 0.8    # >1 increases penalty steepness
    CAP     = 3.0    # max multiplicative penalty

    for name, cfg in PROFILES.items():
        feats = cfg.get("features", [])
        if not feats:
            continue
        
        # NEW — pre-filter by must/veto
        if any(not _eval_guard(record, r) for r in cfg.get("must", [])):   # fails any must → skip
            continue
        if any(_eval_guard(record, r) for r in cfg.get("veto", [])):       # triggers any veto → skip
            continue

        vals, targs, wts = [], [], []
        hits, eligible = 0, 0
        tmp_vals = []

        for f in feats:
            # 1️⃣ Compute the feature value (as you already did)
            v   = _feature_value_from_record(record, scores, f)
            tgt = float(f.get("target", np.nan))
            wt  = float(f.get("weight", 1.0))
            vals.append(v)
            targs.append(tgt)
            wts.append(wt)
            tmp_vals.append((f, v))

            # 2️⃣ New eligibility-aware hit logic
            h = _feature_hit(record, f, v)   # ← uses the new version below
            if h is not None:
                eligible += 1
                hits += int(h)

        # --- per-feature distance (RMSE) so fewer-criteria profiles aren't advantaged
        d = _weighted_nanaware_distance(vals, targs, wts)

        # --- AND-ish: smooth proportional penalty based on fraction hit, capped
        if eligible > 0:
            r = hits / float(eligible)
            if r < K_RATIO:
                penalty = (K_RATIO / max(r, 1e-6)) ** GAMMA
                d *= min(penalty, CAP)

        # --- keep best — break near-ties by favoring higher hit ratio
        r = (hits / float(eligible)) if eligible > 0 else 0.0
        EPS = 1e-6
        if (d + EPS) < best_dist or (abs(d - best_dist) <= EPS and r > best_r):
            best_name, best_dist, best_r = name, d, r

    return best_name, scores


def compute_profile_distances(record):
    """
    Compute the same AND-ish, weighted distance used in assign_profile_from_record,
    but return a dict of distances for ALL eligible profiles.
    """
    scores = {}
    dists = {}

    # Must match assign_profile_from_record
    K_RATIO = 0.20
    GAMMA   = 0.8
    CAP     = 3.0

    for name, cfg in PROFILES.items():
        feats = cfg.get("features", [])
        if not feats:
            continue

        # Apply must/veto filtering to match assign_profile_from_record
        if any(not _eval_guard(record, r) for r in cfg.get("must", [])):
            continue
        if any(_eval_guard(record, r) for r in cfg.get("veto", [])):
            continue

        vals, targs, wts = [], [], []
        hits, eligible = 0, 0
        tmp_vals = []

        for f in feats:
            v   = _feature_value_from_record(record, scores, f)
            tgt = float(f.get("target", np.nan))
            wt  = float(f.get("weight", 1.0))
            vals.append(v)
            targs.append(tgt)
            wts.append(wt)
            tmp_vals.append((f, v))

            # Same eligibility-aware hit logic
            h = _feature_hit(record, f, v)
            if h is not None:
                eligible += 1
                hits += int(h)

        # Same distance as in assign_profile_from_record
        d = _weighted_nanaware_distance(vals, targs, wts)

        if eligible > 0:
            r = hits / float(eligible)
            if r < K_RATIO:
                penalty = (K_RATIO / max(r, 1e-6)) ** GAMMA
                d *= min(penalty, CAP)

        dists[name] = d

    return dists




# ---- MUST/VETO guard evaluation --------------------------------------------
def _eval_guard(record, rule):
    """
    Evaluates a guard 'rule' that can be:
      - a single condition dict compatible with your _eval_condition()
      - {"all": [cond1, cond2, ...]}  -> AND
      - {"any": [cond1, cond2, ...]}  -> OR
    Returns True/False.
    """
    if rule is None:
        return True
    if "all" in rule:
        return all(_eval_condition(record, c) for c in rule["all"])
    if "any" in rule:
        return any(_eval_condition(record, c) for c in rule["any"])
    # single condition
    return _eval_condition(record, rule)


# ==============
# Title + Profile header (icon + text)
# ==============
def _data_uri(path: str) -> str:
    mime = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

# Title
st.markdown(f"""
<div class="dm-center">
    <div class="dm-title">{tr("DRIFTING MINDS STUDY")}</div>
</div>
""", unsafe_allow_html=True)



# Assign profile + get text/icon
prof_name, scores = assign_profile_from_record(record)
prof_cfg = PROFILES.get(prof_name, {})
icon_file = prof_cfg.get("icon")
icon_path = f"assets/{icon_file}" if icon_file else None
has_icon = bool(icon_path and os.path.exists(icon_path))

# Profile description via translation keys
PROFILE_TEXT_KEYS = {
    "Dreamweaver": "PROFILE_DESC_DREAMWEAVER",
    "Quick Diver": "PROFILE_DESC_QUICK_DIVER",
    "Fantasizer": "PROFILE_DESC_FANTASIZER",
    "Archivist": "PROFILE_DESC_ARCHIVIST",
    "Worrier": "PROFILE_DESC_WORRIER",
    "Freewheeler": "PROFILE_DESC_FREEWHEELER",
    "Quiet Mind": "PROFILE_DESC_QUIET_MIND",
    "Radio Tuner": "PROFILE_DESC_RADIO_TUNER",
    "Strategist": "PROFILE_DESC_STRATEGIST",
    "Sentinel": "PROFILE_DESC_SENTINEL",
    "Fragmented Mind": "PROFILE_DESC_FRAGMENTED_MIND",
    "Pragmatic": "PROFILE_DESC_PRAGMATIC",
}


desc_key = PROFILE_TEXT_KEYS.get(prof_name, "")
prof_desc = tr(desc_key) if desc_key else ""

POP_PERC = {
    "Dreamweaver": 4,
    "Quick Diver": 11,
    "Fantasizer": 18,
    "Archivist": 5,
    "Worrier": 6,
    "Freewheeler": 8,
    "Quiet Mind": 15,
    "Radio Tuner": 4,
    "Strategist": 7,
    "Sentinel": 4,
    "Fragmented Mind": 5,
    "Pragmatic": 13,
}

perc_val = POP_PERC.get(prof_name, 0)
lead_txt = tr("You drift into sleep like a")

# Display name translated (keeps English if no entry)
prof_name_disp = tr(prof_name)

pop_line = tr(
    "{name}s represent {perc}% of the population.",
    name=prof_name_disp,
    perc=perc_val,
)

prof_desc_ext = (
    f"{prof_desc}<br>"
    f"<span style='display:block; margin-top:2px; font-size:1rem; color:#222;'>"
    f"{pop_line}</span>"
)
# --- Render profile header ---
icon_src = _data_uri(icon_path) if has_icon else ""

st.markdown(f"""
<div class="dm-center">
  <div class="dm-row">
    {'<img class="dm-icon" src="'+icon_src+'" alt="profile icon"/>' if has_icon else ''}
    <div class="dm-text">
      <p class="dm-lead">{lead_txt}</p>
      <div class="dm-key">{prof_name_disp}</div>
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
world_tag = tr("WORLD_AVERAGE_TAG")   # ⟵ ajoute cette ligne une fois avant la boucle

for b in bars:
    name = b["name"]
    display_name = tr(name)
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
            raw_left, raw_right = [s.strip() for s in help_txt.split("↔", 1)]
            left_anchor  = tr(raw_left if raw_left != "Vivid" else "Vivid_anchor")
            right_anchor = tr(
                raw_right if raw_right not in {"Vivid", "Bizarre", "Immersive", "Spontaneous"}
                else raw_right + "_anchor"
                )
        else:
            left_anchor, right_anchor = "0", "100"


    median_html = "" if med_left is None else f"<div class='dm2-median' style='left:{med_left}%;'></div>"

    is_perception = (name.lower() in ("perception", "vivid"))
    mediantag_html = ""
    if is_perception and (med_left_clamped is not None):
        overlap = (score_txt != "NA" and abs(width_clamped - med_left_clamped) <= 9.0)
        mediantag_class = "dm2-mediantag below" if overlap else "dm2-mediantag"
        mediantag_html = (
            f"<div class='{mediantag_class}' style='left:{med_left_clamped}%;'>{world_tag}</div>"
        )

    scoretag_html = "" if score_txt == "NA" else f"<div class='dm2-scoretag' style='left:{width_clamped}%;'>{score_txt}</div>"

    row_html = (
        "<div class='dm2-row'>"
          "<div class='dm2-left'>"
            f"<div class='dm2-label'>{display_name}</div>"
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
export_title_html = f"""
<div class="dm-center">
  <div class="dm-title">{tr("DRIFTING MINDS STUDY")}</div>
</div>
"""

# Header (icon + text)
export_icon_html = f'<img class="dm-icon" src="{icon_src}" alt="profile icon"/>' if has_icon else ""

export_header_html = f"""
<div class="dm-center">
  <div class="dm-row">
    {export_icon_html}
    <div class="dm-text">
      <p class="dm-lead">{lead_txt}</p>
      <div class="dm-key">{prof_name_disp}</div>
      <p class="dm-desc">{prof_desc_ext or "&nbsp;"}</p>
    </div>
  </div>
</div>
"""



# Bars (same values you computed above)
export_bars_html = ["<div class='dm2-outer'><div class='dm2-bars'>"]
world_tag = tr("WORLD_AVERAGE_TAG")   # ⟵ ajoute ici aussi

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
        mediantag_html = (
            f"<div class='{mediantag_class}' style='left:{med_left_clamped}%;'>{world_tag}</div>"
        )

    scoretag_html = "" if score_txt == "NA" else f"<div class='dm2-scoretag' style='left:{width_clamped}%;'>{score_txt}</div>"

    export_bars_html.append(
        "<div class='dm2-row'>"
          "<div class='dm2-left'>"
            f"<div class='dm2-label'>{tr(name)}</div>"
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
        f"""
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
            {tr("EXPERIENCE_DIM_NOTES_HTML")}
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )



with right_btn:
    
  download_label = tr("DOWNLOAD_BUTTON")
  copy_label = tr("COPY_LINK_BUTTON")
  copied_label = tr("COPY_LINK_COPIED")  
  
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

  /* Export root: fixed width card, in-viewport but hidden (so iOS paints it) */
  #export-root {{
    position: fixed;
    left: 0; top: 0;
    width: 820px;
    background: #fff;
    visibility: hidden;     /* shown only during capture */
    pointer-events: none;
    box-sizing: border-box;
    z-index: -1;
  }}

  /* Prevent any image from blowing up the layout */
  #export-root img {{
    max-width: 100%;
    height: auto;
  }}
</style>
</head>
<body data-rec="{record_id or ''}">
  <div class="wrap">
    <button id="dmshot"  class="bar">{download_label}</button>
    <button id="copylink" class="bar">{copy_label}</button>
  </div>

  <!-- Hidden, fully-opaque export mirror (toggled visible only during capture) -->
  <div id="export-root">{DM_SHARE_HTML}</div>

  <script src="https://cdn.jsdelivr.net/npm/dom-to-image-more@3.4.0/dist/dom-to-image-more.min.js"></script>
  <script>
  (function() {{
    const dlBtn   = document.getElementById('dmshot');
    const copyBtn = document.getElementById('copylink');
    const root    = document.getElementById('export-root');
    const recId   = document.body.dataset.rec || '';

    function buildShareUrl() {{
      let href = '';
      try {{ href = (window.parent && window.parent.location) ? window.parent.location.href : window.location.href; }}
      catch(e) {{ href = window.location.href; }}
      try {{
        const u = new URL(href);
        if (recId) u.searchParams.set('id', recId);
        return u.toString();
      }} catch(e) {{ return href; }}
    }}

    copyBtn.addEventListener('click', async () => {{
      const share = buildShareUrl();
      try {{
        await navigator.clipboard.writeText(share);
        const prev = copyBtn.textContent;
        copyBtn.textContent = "{copied_label}";
        setTimeout(() => copyBtn.textContent = prev, 1500);
      }} catch (e) {{
        const ta = document.createElement('textarea');
        ta.value = share;
        ta.style.position='fixed'; ta.style.left='-9999px';
        document.body.appendChild(ta); ta.select();
        try {{ document.execCommand('copy'); }} catch(_) {{}}
        ta.remove();
      }}
    }});

    async function ensureImagesReady(node) {{
      const imgs = Array.from(node.querySelectorAll('img'));
      if (!imgs.length) return;
      await Promise.all(imgs.map(img => {{
        if (img.complete && img.naturalWidth > 0) {{
          if (typeof img.decode === 'function') {{
            return img.decode().catch(() => new Promise(r => setTimeout(r, 60)));
          }}
          return Promise.resolve();
        }}
        if (typeof img.decode === 'function') {{
          return img.decode().catch(() => new Promise(r => setTimeout(r, 120)));
        }}
        return new Promise(res => {{
          const done = () => {{ img.removeEventListener('load', done); img.removeEventListener('error', done); res(); }};
          img.addEventListener('load', done, {{ once:true }});
          img.addEventListener('error', done, {{ once:true }});
        }});
      }}));
    }}

    // Rasterize <img> to PNG using the *rendered* size to avoid layout blowups
    async function rasterizeImages(node) {{
      const imgs = Array.from(node.querySelectorAll('img'));
      if (!imgs.length) return;

      const loadImage = (src) => new Promise((resolve) => {{
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = () => resolve(null);
        im.src = src;
      }});

      await Promise.all(imgs.map(async (img) => {{
        // Always prefer the rendered (CSS) size
        let rect = img.getBoundingClientRect();
        let w = Math.round(rect.width);
        let h = Math.round(rect.height);

        // Fallback to natural size if rect is 0 (rare, but safe)
        if (!w || !h) {{
          w = img.naturalWidth  || img.width  || 0;
          h = img.naturalHeight || img.height || 0;
        }}
        if (!w || !h) return;

        const src = img.currentSrc || img.src;
        const bitmap = await loadImage(src);
        if (!bitmap) return;

        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        const ctx = c.getContext('2d');
        try {{
          ctx.drawImage(bitmap, 0, 0, w, h);
          const data = c.toDataURL('image/png');

          // Lock the rendered dimensions so layout remains identical
          img.style.width  = w + 'px';
          img.style.height = h + 'px';

          img.setAttribute('src', data);
        }} catch(_){{
          /* ignore draw errors, keep original src */
        }}
      }}));
    }}

    async function downloadBlob(blob, name) {{
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = name;
      document.body.appendChild(a); a.click();
      setTimeout(() => {{ URL.revokeObjectURL(url); a.remove(); }}, 400);
    }}

    async function capture() {{
      try {{
        // Let layout settle
        await new Promise(r => requestAnimationFrame(r));
        await new Promise(r => requestAnimationFrame(r));

        // Make export root visible (must be visible to measure/rasterize correctly)
        const prevVis = root.style.visibility;
        root.style.visibility = 'visible';
        void root.offsetHeight;

        // Ensure images are decoded, then rasterize at rendered size
        await ensureImagesReady(root);
        await rasterizeImages(root);

        // Measure the final card size *as rendered*
        const rect = root.getBoundingClientRect();
        const w = Math.max(1, Math.round(rect.width));
        const h = Math.max(1, Math.round(rect.height));

        // Snapshot without extra scaling to avoid oversized outputs
        const blob = await window.domtoimage.toBlob(root, {{
          width:  w,
          height: h,
          bgcolor: '#ffffff',
          quality: 1,
          cacheBust: true
        }});

        root.style.visibility = prevVis;

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

st.markdown('<div class="dm-spacer-you"></div>', unsafe_allow_html=True)

# --- Centered title for "You" (thinner black line) -----------------
st.markdown(
    f"""
    <div class="dm-center" style="max-width:960px; margin:18px auto 10px;">
      <div style="display:flex; align-items:center; gap:18px;">
        <div style="height:1px; background:#000; flex:1;"></div>
        <div class="dm-section-title" style="font-size:28px; font-weight:600;">
            {tr("YOU")}
        </div>
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

# 3) Anxiety 1–100, 10-point bins  ← fewer bins
anx_vals  = _col_values(pop_data, "anxiety")
anx_edges = np.arange(0.5, 100.5 + 10, 10)  # 0.5 → 110.5, step 10 → ~10 bins
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
    _mini_hist(
        ax,
        vviq_counts,
        vviq_edges,
        vviq_hidx,
        tr("Your visual imagery at wake: {val}", val=int(round(vviq_score)))
    )

    # --- Custom x-axis labels -------------------------------------------------
    ax.text(0.00, -0.05, f"{tr('low')} (16)",   transform=ax.transAxes,
        ha="left",  va="top", fontsize=7.5)
    ax.text(1.00, -0.05, f"{tr('high')} (80)", transform=ax.transAxes,
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
    ax.text(x0 + 0.05, y_top, tr("you"),
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7.5, color=PURPLE_HEX)

    # world (gray square)
    ax.add_patch(plt.Rectangle((x0, y_top - y_gap - box_size / 2),
                               box_size, box_size,
                               transform=ax.transAxes,
                               color="#D9D9D9", lw=0))
    ax.text(x0 + 0.05, y_top - y_gap, tr("world"),
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7.5, color="#444444")

    # maintain alignment
    ax.set_position(AX_POS_YOU)
    st.pyplot(fig, use_container_width=False)





with c2:
    if not cre_counts.size:
        st.info(tr("Population data for creativity unavailable."))
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        _mini_hist(
            ax,
            cre_counts,
            cre_edges,
            cre_hidx,
            tr("Your self-rated creativity: {val}", val=int(round(cre_part)))
        )
        # Replace default x-labels
        ax.text(0.0, -0.05, f"{tr('low')} (1)",  transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5)
        ax.text(1.0, -0.05, f"{tr('high')} (6)", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5)
        ax.set_position(AX_POS_YOU)  # ← lock baseline
        st.pyplot(fig, use_container_width=False)


with c3:
    if not anx_counts.size:
        st.info(tr("Population data for anxiety unavailable."))
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        
        _mini_hist(
            ax,
            anx_counts,
            anx_edges,
            anx_hidx,
            tr("Your self-rated anxiety: {val}", val=int(round(anx_part)))
        )
        # Replace default x-labels
        ax.text(0.0, -0.05, f"{tr('low')} (1)",  transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5)
        ax.text(1.0, -0.05, f"{tr('high')} (100)", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5)
        ax.set_position(AX_POS_YOU)  # ← lock baseline
        st.pyplot(fig, use_container_width=False)

# --- Explanatory note below the three histograms ----------------------------
st.markdown(
    f"""
    <div style="
        max-width:820px;
        margin:14px 0 0 0;
        text-align:left;
        font-size:0.82rem;
        color:#444;
        line-height:1.45;
    ">
      <em>{tr("YOU_SECTION_NOTE")}</em>
    </div>
    """,
    unsafe_allow_html=True
)





# ==============
# "Your sleep" — Latency · Duration · Chronotype & Dream recall (final visual alignment)
# ==============

st.markdown('<div class="dm-spacer-sleep"></div>', unsafe_allow_html=True)

# --- Centered title for "Your sleep" (thinner black line, one line text) -----
st.markdown(
    f"""
    <div class="dm-center" style="max-width:1020px; margin:28px auto 16px;">
      <div style="display:flex; align-items:center; gap:20px;">
        <div style="height:1px; background:#000; flex:0.5;"></div>
        <div class="dm-section-title" style="font-size:28px; font-weight:600;">
          {tr("YOUR SLEEP")}
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
        st.info(tr("Population data unavailable."))
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
                        ax.set_title(
                            tr("You fall asleep in {val} minutes", val=rounded_raw),
                            fontsize=8, pad=6, color="#222222"
                        )                        
                        ax.set_xlabel(tr("minutes"), fontsize=7.5, color="#333333")

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
                        ax.text(x0 + 0.05, y_top, tr("you"), transform=ax.transAxes,
                                ha="left", va="center", fontsize=7.5, color=PURPLE_HEX)
                        
                        # "world" — gray square below
                        ax.add_patch(plt.Rectangle((x0, y_top - y_gap - size / 2),
                                                   size, size,
                                                   transform=ax.transAxes,
                                                   color="#D9D9D9", lw=0))
                        ax.text(x0 + 0.05, y_top - y_gap, tr("world"),
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
        st.info(tr("Population data unavailable."))
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
                        title_str = tr("You sleep {val} hours on average", val=dur_raw_str)
                    else:
                        part_hours_plot = float(dur_raw_str)
                        title_str = tr("You sleep {val} hours on average", val=int(round(part_hours_plot)))
                except:
                    part_hours_plot = float(np.nanmedian(samples_h))
                    title_str = tr("Your sleep duration")

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
                ax.set_xlabel(tr("hours"), fontsize=7.5)

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
# RIGHT: Chronotype + Dream recall — population bars with your bin in purple
# =============================================================================
with col_right:
    if pop_data is None or pop_data.empty:
        st.info(tr("Population data unavailable."))
    else:
        def _to_int(x):
            try:
                return int(str(x).strip())
            except Exception:
                return None

        raw_chrono = record.get("chronotype", None)
        raw_recall = record.get("dream_recall", None)

        chronotype_val    = _to_int(raw_chrono)
        dreamrec_val_raw  = _to_int(raw_recall)

        CHRONO_LBL = {
            1: "morning",
            2: "evening",
            3: "no type",
        }
        DREAMRECALL_LBL = {
            1: "<1/month",
            2: "1-2/month",
            3: "1/week",
            4: "several/week",
            5: "every day",
        }

        # ---------------------------------------------------------------------
        # Population distributions
        # ---------------------------------------------------------------------
        # Chronotype: 1–3
        chrono_series = pd.to_numeric(pop_data.get("chronotype"), errors="coerce")
        chrono_counts = (
            chrono_series.value_counts(dropna=True)
            .reindex([1, 2, 3], fill_value=0)
            .astype(float)
        )
        chrono_x = np.arange(1, 4)

        # Dream recall: full 1–5 scale
        recall_raw = pd.to_numeric(pop_data.get("dream_recall"), errors="coerce")
        recall_counts = (
            recall_raw.value_counts(dropna=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .astype(float)
        )
        recall_x = np.arange(1, 6)

        # Normalize
        if chrono_counts.sum() > 0:
            chrono_counts /= chrono_counts.sum()
        if recall_counts.sum() > 0:
            recall_counts /= recall_counts.sum()

        # ---------------------------------------------------------------------
        # FIGURE: slightly less flat than before
        # ---------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=(2.6, 2.1)  # was 2.6 → a bit taller
        )
        fig.patch.set_alpha(0)

        def _style_cat_axis(ax):
            ax.set_facecolor("none")
            ax.set_ylabel("")
            ax.get_yaxis().set_visible(False)
            ax.spines["bottom"].set_linewidth(0.3)
            for side in ("left", "right", "top"):
                ax.spines[side].set_visible(False)
            ax.margins(y=0.08)

        width = 0.85

        # ---------------------------------------------------------------------
        # CHRONOTYPE
        # ---------------------------------------------------------------------
        _style_cat_axis(ax1)

        ax1.bar(
            chrono_x, chrono_counts.values,
            width=width, color="#D9D9D9",
            edgecolor="white", align="center"
        )

        if chronotype_val in [1, 2, 3]:
            idx = chronotype_val - 1
            ax1.bar(
                chrono_x[idx], chrono_counts.values[idx],
                width=width, color=HL_RGB,
                edgecolor="white", align="center"
            )

        # Dynamic title based on participant’s type
        if chronotype_val == 1:
            chrono_title = tr("You are a morning type")
        elif chronotype_val == 2:
            chrono_title = tr("You are an evening type")
        elif chronotype_val == 3:
            chrono_title = tr("You have no chronotype")
        else:
            chrono_title = tr("Chronotype")
        
        ax1.set_title(chrono_title, fontsize=8.5, pad=7, color="#222222")


        ax1.set_xticks(chrono_x)
        ax1.set_xticklabels(
            [tr("morning"), tr("evening"), tr("no type")],
            fontsize=8,
            rotation=0,
            ha="center",
        )

        # ---------------------------------------------------------------------
        # DREAM RECALL
        # ---------------------------------------------------------------------
        _style_cat_axis(ax2)

        ax2.bar(
            recall_x, recall_counts.values,
            width=width, color="#D9D9D9",
            edgecolor="white", align="center"
        )

        if dreamrec_val_raw in [1, 2, 3, 4, 5]:
            idx = dreamrec_val_raw - 1  # 0-based index for bar position
            ax2.bar(
                recall_x[idx], recall_counts.values[idx],
                width=width, color=HL_RGB,
                edgecolor="white", align="center"
            )

        # Dynamic title based on participant’s recall frequency
        if dreamrec_val_raw == 1:
            dr_title = tr("You recall your dreams\nless than once a month")
        elif dreamrec_val_raw == 2:
            dr_title = tr("You recall your dreams\nonce or twice a month")
        elif dreamrec_val_raw == 3:
            dr_title = tr("You recall your dreams\nonce a week")
        elif dreamrec_val_raw == 4:
            dr_title = tr("You recall your dreams\nseveral times a week")
        elif dreamrec_val_raw == 5:
            dr_title = tr("You recall your dreams\nevery day")
        else:
            dr_title = tr("Dream recall")


        ax2.set_title(
            dr_title,
            fontsize=8.5,      # slightly reduced
            pad=7,
            color="#222222",
        )

        ax2.set_xticks(recall_x)
        ax2.set_xticklabels(
            [
                tr("<1/month"),
                tr("1-2/month"),
                tr("1/week"),
                tr("several/week"),
                tr("every day"),
            ],
            fontsize=8,
            rotation=18,
            ha="right",
        )

        # ---------------------------------------------------------------------
        # SPACING BETWEEN SUBPLOTS
        # ---------------------------------------------------------------------
        fig.subplots_adjust(
            hspace=1.20,
            left=0.20,
            right=0.98,
            bottom=0.08,
            top=0.98,
        )

        st.pyplot(fig, use_container_width=False)




st.markdown('<div class="dm-spacer-exp"></div>', unsafe_allow_html=True)

# ==============
# Your experience — Section header + 3-column layout (image left, radar middle)
# ==============
st.markdown(
    f"""
    <div class="dm-center" style="max-width:1020px; margin:28px auto 32px;">
      <div style="display:flex; align-items:center; gap:24px;">
        <div style="height:1px; background:#000; flex:1;"></div>
        <div class="dm-section-title" style="font-size:28px; font-weight:600;">
          {tr("YOUR EXPERIENCE")}
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

# Base filenames (English)
base_traj_map = {
    1: "trajectories-01.png",
    2: "trajectories-02.png",
    3: "trajectories-03.png",
    4: "trajectories-04.png",
}

# Adjust filenames depending on detected language
traj_map = {}
for k, fname in base_traj_map.items():
    root, ext = os.path.splitext(fname)  # e.g., "trajectories-01", ".png"

    if LANG == "fr":
        # trajectories-01 → trajectories_fr-01
        root = root.replace("trajectories-", "trajectories_fr-")
    elif LANG == "es":
        # trajectories-01 → trajectories_en-01  (per your naming)
        root = root.replace("trajectories-", "trajectories_en-")

    traj_map[k] = root + ext

img_name = traj_map.get(traj_val)


with exp_left:
    # mini-title to match histo titles
    st.markdown(
    f"<div class='dm-subtitle-trajectory' style='color:#222; text-align:center; margin:2px 0 6px 0;'>{tr('Your trajectory')}</div>",
    unsafe_allow_html=True
    )
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
labels = [tr(lab) for _, lab in FIELDS]

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
    
    ax.set_title(tr("Intensity of your experience"), fontsize=20, pad=56, color="#222")
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
    key: tr(f"LBL_{key}")
    for key in [
        "freq_think_ordinary",
        "freq_scenario",
        "freq_negative",
        "freq_absorbed",
        "freq_percept_fleeting",
        "freq_think_bizarre",
        "freq_planning",
        "freq_spectator",
        "freq_ruminate",
        "freq_percept_intense",
        "freq_percept_narrative",
        "freq_percept_ordinary",
        "freq_time_perc_fast",
        "freq_percept_vague",
        "freq_replay",
        "freq_percept_bizarre",
        "freq_emo_intense",
        "freq_percept_continuous",
        "freq_think_nocontrol",
        "freq_percept_dull",
        "freq_actor",
        "freq_think_seq_bizarre",
        "freq_percept_precise",
        "freq_percept_imposed",
        "freq_hear_env",
        "freq_positive",
        "freq_think_seq_ordinary",
        "freq_percept_real",
        "freq_time_perc_slow",
        "freq_syn",
        "freq_creat",
    ]
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
# If a bin is completely empty → display "no content"
for i in winners:
    if len(winners[i]) == 0:
        winners[i] = [tr("no content")]

# --- Plot (horizontal bar with L→R gradient: Awake → Asleep)
with exp_right:
    
    st.markdown(
    f"<div class='dm-subtitle-dynamics' style='color:#222; text-align:center; margin-bottom:6px;'>{tr('Dynamics of your experience')}</div>",
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
    ax.text(tx(6),   y_bar, tr("LBL_Awake"),  ha="left",  va="center",
        fontsize=end_fs, color="#000000")
    ax.text(tx(94),  y_bar, tr("LBL_Asleep"), ha="right", va="center",
            fontsize=end_fs, color="#FFFFFF")


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
        labels0 = winners[0]
        # Use only as many positions as labels, starting from the one closest to the bar
        pos0 = top_positions[-len(labels0):]  # 1 label → [closest]; 2 → [mid, closest]; 3 → all
        nearest_top = min(pos0)  # numerically closest to the bar

        ax.plot(
            [top_anchor_x, top_anchor_x],
            [y_bar + bar_half_h, nearest_top - 0.018],
            color="#000000",
            linewidth=stem_lw,
        )

        for yy, text_label in zip(pos0, labels0):
            ax.text(
                top_anchor_x,
                yy,
                text_label,
                ha="center",
                va="bottom",
                fontsize=label_fs,
                color="#000000",
                linespacing=1.12,
            )

    # Bin 1 (51–100): one stem around x≈66, labels below the bar (stacked upward)
    bot_anchor_x = tx(66.0)
    bot_base_y   = y_bar - bar_half_h - 0.22   # farther below the bar
    bot_positions = [bot_base_y,
                     bot_base_y + row_gap,
                     bot_base_y + 2 * row_gap]

    if winners[1]:
        labels1 = winners[1]
        # Use only as many positions as labels, starting from the one closest to the bar
        pos1 = bot_positions[-len(labels1):]
        nearest_bot = max(pos1)  # numerically closest to the bar here

        ax.plot(
            [bot_anchor_x, bot_anchor_x],
            [y_bar - bar_half_h, nearest_bot + 0.018],
            color="#000000",
            linewidth=stem_lw,
        )

        for yy, text_label in zip(pos1, labels1):
            ax.text(
                bot_anchor_x,
                yy,
                text_label,
                ha="center",
                va="top",
                fontsize=label_fs,
                color="#000000",
                linespacing=1.15,
            )


    plt.tight_layout(pad=0.25)
    st.pyplot(fig, use_container_width=False)


# --- Explanatory note below "Your Experience" -----------------------------------
st.markdown(
    f"""
    <div style="
        max-width:740px;
        margin:-20px 0 0 0;
        text-align:left;
        font-size:0.82rem;
        color:#444;
        line-height:1.35;
    ">
      <em>
        {tr("YOUR_EXPERIENCE_LAYOUT_NOTE")}
      </em>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============
# Profile likelihoods for this participant (all 12 profiles)
# ==============
st.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)

try:
    profile_dists = compute_profile_distances(record)

    if profile_dists:
        # Fixed order (definition order) for reproducibility
        prof_names = list(PROFILES.keys())
        d_arr = np.array([profile_dists.get(n, np.inf) for n in prof_names], dtype=float)

        # Convert distances → similarities → normalized likelihoods (0–1, sum = 1)
        finite_mask = np.isfinite(d_arr)
        if not finite_mask.any():
            sims = np.ones_like(d_arr) / len(d_arr)
        else:
            sims = np.zeros_like(d_arr)
            sims[finite_mask] = 1.0 / (1.0 + d_arr[finite_mask])
            total = sims.sum()
            if total <= 0:
                sims = np.ones_like(d_arr) / len(d_arr)
            else:
                sims /= total

        lik_pct = sims * 100.0

        # Sort profiles from most to least likely (left → right)
        order = np.argsort(-lik_pct)
        names_sorted = [prof_names[i] for i in order]
        lik_sorted   = lik_pct[order]
        names_sorted_disp = [tr(name) for name in names_sorted]

                # --- Plot (gradient + icons, clean axis) ---
        fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=200)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        x = np.arange(len(names_sorted))

        # Gradient from dark purple (left) → light purple (right)
        dark_rgb  = np.array([0x7C/255, 0x62/255, 0xFF/255])   # #7C62FF
        light_rgb = np.array([0xC9/255, 0xBD/255, 0xFF/255])   # #C9BDFF             # very light lilac

        if len(names_sorted) > 1:
            cols = np.linspace(dark_rgb, light_rgb, len(names_sorted))
        else:
            cols = np.array([dark_rgb])

        bar_colors = [tuple(c) for c in cols]

        ax.bar(x, lik_sorted, width=0.6, color=bar_colors, edgecolor="white")

        # X labels
        ax.set_xticks(x)
        ax.set_xticklabels(names_sorted_disp, rotation=20, ha="right", fontsize=9)

        # Y axis: custom low/high scaling (no numbers)
        min_lik = float(np.nanmin(lik_sorted)) if len(lik_sorted) else 0.0

        if min_lik < 1.0:
            y_min = 0.0
        else:
            y_min = max(min_lik - 1.0, 0.0)

        y_max = float(np.nanmax(lik_sorted)) if len(lik_sorted) else 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0  # safety

        y_max = y_max * 1.08   # small headroom above tallest bar
        ax.set_ylim(y_min, y_max)

        # Only "low" (min) and "high" (max) as y-axis labels
        ax.set_yticks([y_min, y_max])
        ax.set_yticklabels([tr("low"), tr("high")], fontsize=8)
        ax.set_ylabel(tr("Matching strength"), fontsize=8, labelpad=2)
        ax.set_title(tr("How much you match each profile"), fontsize=11, pad=8)

        # No grid, minimal frame
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --- Icons above bars ------------------------------------------------
        for xi, name, p in zip(x, names_sorted, lik_sorted):
            icon_file = PROFILES.get(name, {}).get("icon")
            img_path = None

            if icon_file:
                base, ext = os.path.splitext(icon_file)
                # Prefer PNG if available (better for matplotlib), otherwise try the given file
                candidates = [
                    os.path.join("assets", base + ".png"),
                    os.path.join("assets", icon_file),
                ]
                for cpath in candidates:
                    # Skip SVGs for matplotlib; keep code safe if only SVG exists
                    if os.path.exists(cpath) and not cpath.lower().endswith(".svg"):
                        img_path = cpath
                        break

            if img_path:
                try:
                    img = plt.imread(img_path)
                    im = OffsetImage(img, zoom=0.08)  # tweak zoom if icons too big/small

                    # Place icon a bit above the bar
                    icon_y = p + (y_max - y_min) * 0.02
                    ab = AnnotationBbox(im, (xi, icon_y), frameon=False)
                    ax.add_artist(ab)
                except Exception:
                    # Fail silently if an icon can't be loaded
                    pass

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)


        # Short explanatory line below
        st.markdown(
            f"""
            <div style="
                max-width:740px;
                margin:6px 0 0 0;
                font-size:0.82rem;
                color:#444;
                line-height:1.35;
            ">
              <em>{tr("PROFILE_BARS_NOTE")}</em>
            </div>
            """,
            unsafe_allow_html=True
        )

except Exception as e:
    st.warning("Could not compute profile likelihoods for this record.")




# ==============
# ALL DMs (embedded, centered)
# ==============
import base64

# Choose correct image depending on language
if LANG == "fr":
    final_img_path = "assets/all_DMs_fr.png"
elif LANG == "es":
    final_img_path = "assets/all_DMs_en.png"
else:
    final_img_path = "assets/all_DMs.png"

# Load and encode
with open(final_img_path, "rb") as f:
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
             style="max-width: 100%; height: auto; border-radius: 8px;">
    </div>
    """,
    unsafe_allow_html=True
)


# ==============
# Disclaimer
# ==============

st.markdown(
    f"""
    <div style="
        max-width:740px;
        margin: 0 auto 0 auto;    /* auto centers horizontally */
        text-align:center;          /* centers text */
        font-size:0.85rem;
        color:#444;
        line-height:1.6;
    ">
      <em>{tr("DISCLAIMER_NOTE")}</em>
    </div>
    """,
    unsafe_allow_html=True
)




