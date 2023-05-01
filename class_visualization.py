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

    def time_series(dataX, dataY, color, label):
        
        # define a new figure
        fig = plt.figure(figsize=(20,6));

        # make a time series plot
        plt.plot(dataX, dataY, color=color, label=label, linewidth=2);

        # make are labels
        plt.legend(loc="best");
        plt.grid(True);

        # return values
        return fig;
