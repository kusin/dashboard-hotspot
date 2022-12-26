# library ui-dashboard
import streamlit as st;

# library manipulation dataset
import pandas as pd;

# library manipulation array
import numpy as np;

# library data visualization
import plotly.express as px;


# func load dataset Indonesia country
@st.cache(allow_output_mutation=True)
def df_country():
    
    # return value
    return pd.read_csv("dataset/fire_archive_M-C61_317736.csv");


# func load dataset province South Sumatra
@st.cache(allow_output_mutation=True)
def df_province():
    
    # return value
    return pd.read_excel("dataset/dataset.xlsx", sheet_name="dataset", engine="openpyxl");


# func aggregate dataset hotspot
@st.cache(allow_output_mutation=True)
def func_agg(data, nm_group="", nm_index="", nm_agg=""):
    
    # groupby data by columns and count each rows
    data = data.groupby([nm_group]).size().reset_index(name=nm_index);
    
    # set index datetime
    data = data.set_index(pd.to_datetime(data[nm_group]));

    # resample mountly
    data = data.resample(nm_agg).sum();
    
    #return value
    return data;


# func load dataset karhutla
@st.cache(allow_output_mutation=True)
def func_karhutla():
    
    # the data karhutla 2014 - 2019
    data={
        "year" : ["2014", "2015", "2016", "2017", "2018", "2019"],
        "area" : ["44411", "2611411", "438363", "165484", "529267", "1649258"]
    };
    
    # convert data to dataframe
    data = pd.DataFrame(data);
    
    #return value
    return data;

