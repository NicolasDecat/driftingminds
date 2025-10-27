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

REDCAP_API_URL = st.secrets.get("REDCAP_API_URL")
REDCAP_API_TOKEN = st.secrets.get("REDCAP_API_TOKEN")

DEMO = st.query_params.get("demo", ["0"])[0] == "1"
DEMO = True


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