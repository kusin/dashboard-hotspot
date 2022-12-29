# library ui-dashboard
import streamlit as st;

# library manipulation dataset
import pandas as pd;

# library manipulation array
import numpy as np;

# library data visualization
import plotly.express as px;
import plotly.graph_objects as go

# func plot line with plotly graph objects
def plot_line(data, value_x, value_y, nm_color):
    
    # show plot
    fig = go.Figure(
        data = [
            go.Scatter(x=data[value_x], y=data[value_y], line_color=nm_color)
        ]
    );
    
    # return value
    return fig;
            
# func plot  pie with plotly graph objects
def plot_pie(data, nm_values, nm_names, nm_hole, nm_colors):
    
    # show plot
    fig = go.Figure(
        data = [
            go.Pie(values=data[nm_values], labels=data[nm_names], hole=nm_hole)
        ]
    );
    
    # update custom
    fig.update_traces(marker=dict(colors=nm_colors));
    
    # return value
    return fig;

# func plot bar with plotly graph objects
def plot_bar(data, value_x, value_y, nm_color):
    
    # show plot
    fig = go.Figure(
        data = [
            go.Bar(x=data[value_x], y=data[value_y])
        ]
    );
    
    # return value
    return fig;