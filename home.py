# import library streamlit
import streamlit as st;
from streamlit_extras import add_vertical_space as avs

# import library manipulation dataset
import pandas as pd;
import numpy as np;

# import method from other files
from class_dataset import *;


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
            st.info("1. Data Acquisition");
        
        # container-data-acquisition
        with st.container():
            
            # define columns with col-4 row-1
            col1, col2, col3, col4, col5 = st.columns(5);
            col1.metric(
                label="Year 2020", value="999 point", delta="0.00%"  
            );
            col2.metric(
                label="Year 2019", value="999 point", delta="0.00%"  
            );
            col3.metric(
                label="Year 2018", value="999 point", delta="0.00%"  
            );
            col4.metric(
                label="Year 2017", value="999 point", delta="0.00%"  
            );
            col5.metric(
                label="Year 2016", value="999 point", delta="0.00%"  
            );
        
            st.dataframe(df, use_container_width=True);

        # container-eda
        with st.container():
            st.info("2. Exploratory Data Analysis");

