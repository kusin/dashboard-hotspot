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
from func_plot import *;

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
dataset = df_country();

# container-sidebar
# with st.sidebar:
#     st.header("My Dashboard");
#     st.selectbox("Select page", options=["data visualization", "hotspot prediction"]);
    
# container-header
with st.container():
    st.markdown("## Prediction hotspot with enso and rainfall factor used LSTM-RNN and GRU-RNN");

# container-karhutla-indonesia
with st.container():
    
    # set sub-header
    st.subheader("Data visualization karhutla indonesia 2014 - 2022");
    
    # df_karhutla = pd.read_excel("D:/dataset/luas-karhutla-provinsi-2018-2022.xlsx");
    # st.dataframe(df_karhutla, use_container_width=True);

    # call dataset karhutla indonesia
    df_karhutla = func_karhutla();

    # define columns with col-2 row-1
    col1, col2 = st.columns(2);
    
    # columns col-1
    with col1:
        # with pie-plot
        st.plotly_chart(
            plot_pie(df_karhutla, "area", "year", 0.5, ["#9FC9C9", "#B4DBD4", "#F2F2C2", "#EEE8AE", "#F7D05E", "#9FC9C9"]),
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

# container-summary hotspot
with st.container():
    
    # set sub-header
    st.subheader("Data visualization hotspot indonesia");
    
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

# container-hotspot-indonesia with eda
with st.container():
    
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

# container-hotspot-south-sumatra with eda
with st.container():

    # call dataset hotspot province
    df_hospot = df_province();
    
    # set sub-header
    st.subheader("Data visualization hotspot south sumatra");
    st.write("hotspot south sumatra with enso and rainfall factor");
    
    # define col-5 row-1
    col1, col2, col3, col4, col5 = st.columns(5);
    
    # col year 2016
    col1.metric(
        label="Year 2016",
        value="999 point", 
        delta="0,00%"
    );
    
    # col year 2017
    col2.metric(
        label="Year 2017",
        value="999 point", 
        delta="0,00%"
    );
    
    # col year 2018
    col3.metric(
        label="Year 2018",
        value="999 point", 
        delta="0,00%"
    );
    
    # col year 2019
    col4.metric(
        label="Year 2019",
        value="999 point", 
        delta="0,00%"
    );
    
    # col year 2020
    col5.metric(
        label="Year 2020",
        value="999 point", 
        delta="0,00%"
    );
    
    # set tab-index
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["with line-plot", "with bar-plot", "sst nina 3.4", "soi index", "oni index", "rainfall"]);
    
    # tab-index-1 hotspot 
    with tab1:    
        # plot line hotspot sumsel
        st.plotly_chart(
            plot_line(df_hospot, "date", "hotspot", "#3457D5"), use_container_width=True
        );
    
    # tab-index-2 hotspot
    with tab2:
        # plot bar hotspot sumsel
        st.plotly_chart(
            plot_bar(df_hospot, "date", "hotspot", "#3457D5"), use_container_width=True
        );
    
    # tab-index-3 sst nina 3.4
    with tab3:    
        # plot line sst nina 3.4
        st.plotly_chart(
            plot_line(df_hospot, "date", "sst", "#3457D5"), use_container_width=True
        );
    
    # tab-index-4 soi index
    with tab4:    
        # plot line soi index
        st.plotly_chart(
            plot_line(df_hospot, "date", "soi", "#3457D5"), use_container_width=True
        );
    
    # tab-index-5 oni index
    with tab5:    
        # plot line oni index
        st.plotly_chart(
            plot_line(df_hospot, "date", "oni", "#3457D5"), use_container_width=True
        );
    
    # tab-index-6 rainfall
    with tab6:    
        # plot line rainfall
        st.plotly_chart(
            plot_line(df_hospot, "date", "rainfall", "#3457D5"), use_container_width=True
        );
    





