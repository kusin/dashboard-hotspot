# import library
import streamlit as st; 
import pandas as pd;
import numpy as np;

# main class
class Dataset:

    @st.cache_data
    def load_data():

        # load dataset
        df = pd.read_csv("dataset/dataset.csv");

        # return values
        return df;

    @st.cache_data
    def agg_year(df):

        # aggregate by year pandas
        hasil = df.set_index('date').resample('Y')["hotspot"].sum().to_frame();
        
        # return values
        return hasil;