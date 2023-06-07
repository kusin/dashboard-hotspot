# import library streamlit
import streamlit as st;
from streamlit_extras import add_vertical_space as avs;

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

# import method from other files
from class_dataset import *;
from class_visualization import *;
from class_pre_processing import *;
from arch.unitroot import ADF
from arch.unitroot import PhillipsPerron
from arch.unitroot import KPSS

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
            train, test = PreProcessing.splitting(df_sumsel, 0.80, 0.20);

            temp_normalized = pd.concat([
                pd.DataFrame(
                    np.array(df["hotspot"]),
                    columns=["date"]
                ),
                pd.DataFrame(
                    df_sumsel,
                    columns=["normalized"]
                ),
            ], axis=1);

            temp_train = pd.concat([
                pd.DataFrame(
                    df.iloc[0:len(train),0:1],
                    columns=["date"],
                    index=list(range(0,192))
                ),
                pd.DataFrame(
                    np.array(train),
                    columns=["train"], 
                    index=list(range(0,192))
                ),
            ], axis=1);

            temp_test = pd.concat([
                pd.DataFrame(
                    df.iloc[len(train):len(df),0],
                    columns=['date'],
                    index=list(range(192,240))
                ),
                pd.DataFrame(
                    np.array(test),
                    columns=['test'],
                    index=list(range(192,240))
                ),
            ], axis=1);
            
            col1, col2= st.columns(2, gap="large");
            col2.plotly_chart(
                Visualization.time_series(
                    dataX=temp_normalized["date"],
                    dataY=temp_normalized["normalized"],
                    title="Index SOI",
                    color="blue"
                ),
                use_container_width=True
            );
            col2.plotly_chart(
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
        

        # data supervised learning
        with st.container():
            
            # set look back -1
            look_back = 1

            # set supervised learning for data train
            trainX, trainY = PreProcessing.supervised_learning(train, look_back)

            # set supervised learning for data test
            testX, testY = PreProcessing.supervised_learning(test, look_back)


            # reshape data train
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

            # reshape data test
            testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


        # # modeling LSTM-RNN
        with st.container():
            # label data preprocessing
            st.info("4. Modeling LSTM-RNN");
            
            col1, col2= st.columns(2, gap="large");

            col1.selectbox(
                "Activation function", 
                ("tanh", "sigmoid", "relu", "selu", "elu", "softplus")
            );

            col1.selectbox(
                "Optimizers", 
                ("adam", "adamax", "rmsprop", "sgd")
            );

            col1.selectbox(
                "Dropout", 
                (0.00, 0.10, 0.15, 0.20, 0.25)
            );

            col1.selectbox(
                "Batch Size", 
                (4, 8, 16, 32, 64)
            );

            col1.selectbox(
                "Epoch", 
                (2000, 4000)
            );

            st.button("Button")

            

        #     # label data preprocessing
        #     st.info("4. Modeling LSTM-RNN");

        #     # Design network
        #     model = Sequential()

        #     # First LSTM layer with Dropout regularisation
        #     model.add(
        #         LSTM(
        #             units=10,
        #             activation="selu",
        #             input_shape=(trainX.shape[1], 1)
        #         )
        #     )
        #     model.add(Dropout(0.20))

        #     # The output layer
        #     model.add(Dense(1))

        #     # Compiling model the LSTM-RNN
        #     model.compile(
        #         optimizer='sgd',
        #         loss='mae',
        #         metrics=[
        #             tf.keras.metrics.MeanAbsoluteError(),
        #             tf.keras.metrics.MeanSquaredError(),
        #             tf.keras.metrics.RootMeanSquaredError()
        #         ]
        #     )

        #     # fit network
        #     history = model.fit(
        #         trainX, trainY, epochs=2000, batch_size=8,
        #         validation_data=(testX, testY),
        #         verbose=1, shuffle=False
        #     )

        #     # 5. make predictions
        #     predictions = model.predict(testX, verbose=0)
        #     print(predictions[:, 0])

        #     # generate urutan data sesuai panjang datanya
        #     x = pd.date_range(start="2017-01-01", periods=len(testY), freq='MS')

        #     # membuat frame
        #     fig, ax = plt.subplots(figsize = (10,5))

        #     # membuat time series plot
        #     ax.plot(x, testY, color="tab:blue", label="actual data", linewidth=2.5)
        #     ax.plot(x, predictions, color="tab:red", label="prediction data", linewidth=2.5)

        #     # membuat label-label
        #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        #     ax.legend(loc="best")
        #     ax.grid(True)

        #     st.pyplot(fig)