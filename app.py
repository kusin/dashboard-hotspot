# library ui-dashboard
import streamlit as st;

# library manipulation dataset
import pandas as pd;

# library manipulation array
import numpy as np;

# library data visualization
import plotly.express as px;

# set config ui-dasboard streamlit
st.set_page_config(
    page_title="My Dasboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://www.github.com/kusin",
        "Report a bug": "https://www.github.com/kusin",
        "About": "### Copyright 2022 all rights reserved by Aryajaya Alamsyah"
    }
);

# --------------------------------------------------------------------------------------- #
# data acquisition ---------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------- #

# container-header-fuild
st.markdown("## Prediction hotspot with enso and rainfall factor used LSTM-RNN");
st.markdown("- reference dataset from https://firms.modaps.eosdis.nasa.gov/download/");

