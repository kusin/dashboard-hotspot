# declaration of library
import pandas as pd;
import numpy as np;
import plotly.express as px;
import plotly.graph_objects as go;


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
                x=dataX, y=dataY, mode='lines', line_color=color, line_width=3
            )
        );

        # update plot
        fig.update_layout(title_text=title);

        # return values
        return fig;
