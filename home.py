# import library streamlit
import streamlit as st;
from streamlit_extras import add_vertical_space as avs;

# import library manipulation dataset
import pandas as pd;
import numpy as np;

# import method from other files
from class_dataset import *;
from class_visualization import *;
from statsmodels.tsa.stattools import adfuller

# --------------------------------------------------------------- #
# -- Main Function ---------------------------------------------- #
# --------------------------------------------------------------- #
if __name__ == "__main__":
    
    # --------------------------------------------------------------- #
    # -- setting configuration -------------------------------------- #
    # --------------------------------------------------------------- #
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

    # --------------------------------------------------------------- #
    # -- container-wrapper ------------------------------------------ #
    # --------------------------------------------------------------- #
    with st.container():

        # load dataset
        df = Dataset.load_data();

        # container-header
        with st.container():
            st.markdown("<h1 style='color:#9AC66C; text-align:center;'>Hotspot predictions with algorithm LSTM-RNN</h1>",unsafe_allow_html=True);
            avs.add_vertical_space(4);


        # container-data-acquisition
        with st.container():

            # label data-acquisition
            st.info("1. Data Acquisition");

            # dataset hostpot
            st.dataframe(df, use_container_width=True);


        # container-eda
        with st.container():
            
            # label eda
            st.info("2. Exploratory Data Analysis");
            
            # data visualization
            tab1, tab2, tab3, tab4 = st.tabs(["Hotspot Sumsel", "Rainfall", "SST Nina 3.4", "Index SOI"]);
            tab1.pyplot(
                Visualization.time_series(
                    df["date"], df["hotspot"], "blue",
                    "Hotspot Sumsel years 2001 - 2020"
                ), use_container_width=True
            );
            tab2.pyplot(
                Visualization.time_series(
                    df["date"], df["rainfall"], "blue",
                    "Rainfall Sumsel years 2001 - 2020"
                ), use_container_width=True
            );
            tab3.pyplot(
                Visualization.time_series(
                    df["date"], df["sst"], "blue",
                    "SST Nina 3.4"
                ), use_container_width=True
            );
            tab4.pyplot(
                Visualization.time_series(
                    df["date"], df["soi"], "blue",
                    "Index SOI"
                ), use_container_width=True
            );
        
            # acf and pacf
            tab1, tab2 = st.tabs(["ACF plot", "PACF plot"]);
            


        # container-pre-processing
        with st.container():
            
            # label eda
            st.info("3. Data Pre-processing");
            st.write(adfuller(df["hotspot"]));
