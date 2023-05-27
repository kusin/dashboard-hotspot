# import library streamlit
import streamlit as st;
from streamlit_extras import add_vertical_space as avs;

# import library manipulation dataset
import pandas as pd;
import numpy as np;
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

# import method from other files
from class_dataset import *;
from class_visualization import *;
from class_pre_processing import *;
from arch.unitroot import ADF
from arch.unitroot import PhillipsPerron
from arch.unitroot import KPSS

# ------------------
# library manipulation dataset
import pandas as pd
from pandas import concat
from pandas import DataFrame
from pandas import read_csv
from pandas import read_excel

# library manipulation array
import numpy as np
from numpy import concatenate
from numpy import array

# library configuration date and time
import time
from datetime import datetime

# library data visualization
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib import pyplot
from matplotlib import pyplot as plt

# library analysis acf and pacf
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

# library normalize data with max-min algorithm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# library algorithm lstm-rnn with keras
import tensorflow as tf
from tensorflow.keras import models
from keras.models import Sequential
from keras.layers import RNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.optimizers import Adam, Adamax, RMSprop, SGD
from keras.layers import LeakyReLU

# Early stoping
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# library evaluation model
from math import sqrt
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error




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

            # set margin 2
            avs.add_vertical_space(2);
        

        # container-eda
        with st.container():
            
            # label eda
            st.info("2. Exploratory Data Analysis");
            
            # data visualization
            col1, col2= st.columns(2, gap="large");
            col1.plotly_chart(
                Visualization.time_series(
                    dataX=df["date"],
                    dataY=df["hotspot"],
                    title="Hotspot Sumsel years 2001 - 2020",
                    color="blue"
                ),
                use_container_width=True
            );
            col2.plotly_chart(
                Visualization.time_series(
                    dataX=df["date"],
                    dataY=df["rainfall"],
                    title="Rainfall Sumsel years 2001 - 2020",
                    color="blue"
                ),
                use_container_width=True
            );
            col1.plotly_chart(
                Visualization.time_series(
                    dataX=df["date"],
                    dataY=df["sst"],
                    title="SST Nina 3.4",
                    color="blue"
                ),
                use_container_width=True
            );
            col2.plotly_chart(
                Visualization.time_series(
                    dataX=df["date"],
                    dataY=df["soi"],
                    title="Index SOI",
                    color="blue"
                ),
                use_container_width=True
            );

            # stationarity test
            col1, col2, col3= st.columns(3, gap="medium");
            col1.text(ADF(df["hotspot"], lags=15));
            col2.text(PhillipsPerron(df["hotspot"], lags=15));
            col3.text(KPSS(df["hotspot"], lags=15));

            # set margin 2
            avs.add_vertical_space(2);
        
        # /. end container-eda
                    
        
        # data preprocessing
        with st.container():
            
            # label data preprocessing
            st.info("3. Data Preprocessing");

            # 1. feature selection
            df_sumsel = df.filter(["hotspot"]);
            df_sumsel = df_sumsel.values;

            # 2. normalized min-max
            df_sumsel = PreProcessing.normalization(df_sumsel);

            # 3. splitting data
            train_size, test_size = PreProcessing.splitting(df_sumsel, 0.80, 0.20);

            temp_train = pd.concat([
                pd.DataFrame(df.iloc[0:len(train_size),0:1], columns=["date"], index=list(range(0,192))),
                pd.DataFrame(np.array(train_size), columns=["train"], index=list(range(0,192))),
            ], axis=1);

            temp_test = pd.concat([
                pd.DataFrame(df.iloc[len(train_size):len(df),0], columns=['date'], index=list(range(192, 240))),
                pd.DataFrame(np.array(test_size), columns=['test'], index=list(range(192, 240))),
            ], axis=1);
            
            st.plotly_chart(
                Visualization.splitting(
                    # train_size
                    dataX1=temp_train["date"],
                    dataY1=temp_train["train"],

                    # test_size
                    dataX2=temp_test["date"],
                    dataY2=temp_test["test"],
                    
                    # other
                    color1="blue",
                    color2="red",
                    title="The result of data preprocessing"
                ),
                use_container_width=True
            ); 
        
        # data preprocessing
        with st.container():
            # Design network
            model = Sequential()

            # First LSTM layer with Dropout regularisation
            model.add(
                LSTM(
                    units=10,
                    activation="selu",
                    input_shape=(trainX.shape[1], 1)
                )
            )
            model.add(Dropout(0.20))

            # The output layer
            model.add(Dense(1))

            # Compiling model the LSTM-RNN
            model.compile(
                optimizer='sgd',
                loss='mae',
                metrics=[
                    tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.RootMeanSquaredError()
                ]
            )