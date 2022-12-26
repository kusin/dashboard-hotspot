# library ui-dashboard
import streamlit as st;

# library manipulation dataset
import pandas as pd;

# library manipulation array
import numpy as np;

# library data visualization
import plotly.express as px;

# import function any file .py
from func_acq_data import *;

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
# load dataset
dataset = pd.read_csv("dataset/fire_archive_M-C61_317736.csv");


# container-header
with st.container():
    st.markdown("## Prediction hotspot with enso and rainfall factor used LSTM-RNN and GRU-RNN");
    st.write("The impact of enso on increasing the number of hotspots");


# container-summary of impact enso to hotspot
with st.container():
    
    # define col-4 row-1
    col1, col2, col3, col4 = st.columns(4);
    
    # col year 2006
    col1.metric(
        label="Year 2006",
        value="154,985 point", 
        delta="0,00%"
    );
    
    # col year 2009
    col2.metric(
        label="Year 2009",
        value="102344 point", 
        delta="0,00%"
    );
    
    # col year 2015
    col3.metric(
        label="Year 2015",
        value="178,578 point", 
        delta="0,00%"
    );
    
    # col year 2019
    col4.metric(
        label="Year 2019",
        value="90,475 point", 
        delta="0,00%"
    );


# container-hotspot-indonesia
with st.container():
    
    # call dataset karhutla indonesia
    df_karhutla = func_karhutla();
    
    # set sub-header
    st.write("Data visualization hotspot indonesia");
    
    # define columns with col-2 row-1
    col1, col2 = st.columns(2);
    
    # columns col-1
    with col1:
        
        # with pie-plot
        st.plotly_chart(
            px.pie(
                df_karhutla, 
                values="area", 
                names="year", 
                color="year",
                hole=0.5,
                color_discrete_sequence=["#9FC9C9", "#B4DBD4", "#F2F2C2", "#EEE8AE", "#F7D05E", "#9FC9C9"]
            ),
            use_container_width=True
        );
    
    # columns col-2
    with col2:
        
        # with bar-plot
        st.plotly_chart(
            px.bar(
                df_karhutla,
                x="year",
                y="area",
                color_discrete_sequence=["#ffbf40"],
                text_auto=".2s"
            ),
            use_container_width=True
        );


# container-hotspot-indonesia with eda
with st.container():
    
    # set sub-header
    st.write("Data visualization hotspot indonesia");
    
    # set tab-index
    tab1, tab2 = st.tabs(["with line-plot", "with bar-plot"]);
    
    # tab-index-1
    with tab1:
        
        # call dataset hotspot indonesia
        df_hospot = func_agg(dataset, "acq_date", "hotspot", "D");
        
        # set time-series-plot
        st.plotly_chart(
            px.line(
                df_hospot,
                x=df_hospot.index,
                y="hotspot",
                color_discrete_sequence=["#3457D5"]
            ),
            use_container_width=True
        );
    
    # tab-index-2
    with tab2:
        
        # call dataset hotspot indonesia
        df_hospot = func_agg(dataset, "acq_date", "hotspot", "Y");
        
        # set barplot
        st.plotly_chart(
            px.bar(
                df_hospot,
                x=df_hospot.index,
                y="hotspot",
                color_discrete_sequence=["#3457D5"],
                text_auto=".2s"
            ),
            use_container_width=True
        );
    
    
    
