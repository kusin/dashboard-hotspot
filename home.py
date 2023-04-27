# import library streamlit
import streamlit as st;
from streamlit_extras import add_vertical_space as avs;

# import library manipulation dataset
import pandas as pd;
import numpy as np;
from bokeh.plotting import figure
from bokeh.io import output_file, show

# import method from other files
from class_dataset import *;
from class_visualization import *;

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

            col1, col2= st.columns(2, gap="large");
            col1.plotly_chart(
                Visualization.time_series(
                    df["date"],
                    df["hotspot"],
                    "Hotspot Sumatera Selatan years 2001 - 2020",
                    "#70C4A5"
                ), use_container_width=True
            );
            col2.plotly_chart(
                Visualization.time_series(
                    df["date"],
                    df["rainfall"],
                    "Rainfall Sumatera Selatan years 2001 - 2020",
                    "#70C4A5"
                ), use_container_width=True
            );
            col1.plotly_chart(
                Visualization.time_series(
                    df["date"],
                    df["sst"],
                    "SST Nina 3.4",
                    "#70C4A5"
                ), use_container_width=True
            );
            col2.plotly_chart(
                Visualization.time_series(
                    df["date"],
                    df["soi"],
                    "Index SOI",
                    "#70C4A5"
                ), use_container_width=True
            );

        # container-pre-processing
        with st.container():
            
            # label eda
            st.info("2. Data Pre-processing");