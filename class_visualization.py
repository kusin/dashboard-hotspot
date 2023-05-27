# declaration of library
import pandas as pd;
import numpy as np;
import plotly.express as px;
import plotly.graph_objects as go;
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

# define class visualization
class Visualization:

    # property visualization
    x = "";
    y = "";

    def time_series(dataX, dataY, title, color):
        
        # define a new figure
        fig = go.Figure();

        # add plot time series
        fig.add_trace(
            go.Scatter(
                x=dataX, y=dataY, mode='lines', line_color=color,
            )
        );

        # update plot
        fig.update_layout({
            "title": title
        });

        # return values
        return fig;
    
    def splitting(dataX1, dataY1, dataX2, dataY2, color1, color2, title):
        
        # define a new figure
        fig = go.Figure();

        # add plot time series
        fig.add_trace(
            go.Scatter(
                x=dataX1, y=dataY1, mode='lines', line_color=color1, line_width=2.5
            )
        );

        fig.add_trace(
            go.Scatter(
                x=dataX2, y=dataY2, mode='lines', line_color=color2, line_width=2.5
            )
        );

        # update plot
        fig.update_layout({
            "title": title,
            "showlegend": False
        });

        # return values
        return fig;
