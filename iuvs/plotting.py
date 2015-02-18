"""some plotting utilities for IUVS"""
import plotly.plotly as py
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import ceil
from plotly.graph_objs import Heatmap, Scatter, Histogram, Data
import plotly.tools as tls


def get_pie_plot(df, col, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    df[col].value_counts().plot(kind='pie', ax=ax, title=col)
    if ax is None:
        return fig


def plot_pie_overview(df, cols, title, ncols=3):
    nrows = int(ceil(len(cols) / ncols))
    scaler = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*scaler,
                                                                nrows*scaler))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        get_pie_plot(df, col, ax=ax)
    fig.suptitle(title, fontsize=20)
    return fig


def produce_pie_plots(df, folder):
    items = 'MCP_HV XUV'.split()
    for item in items:
        fig = get_pie_plot(df, item)
        path = os.path.join(folder, '{}__pie.png'.format(item))
        plt.savefig(path, dpi=100)
        plt.close(fig)


def make_plotly_multiplot(img, spatial=None, spectral=None, title='No title'):
    if spatial is None:
        spatial = img.shape[0]//2
    if spectral is None:
        spectral = img.shape[1]//2

    prof1 = img[spatial]
    prof2 = img[:, spectral]
    trace1 = Heatmap(z=np.flipud(img))
    trace2 = Scatter(x=prof2, name='spatial profile',
                     xaxis='x2', yaxis='y2')
    trace3 = Scatter(y=prof1, name='spectral profile',
                     xaxis='x3', yaxis='y3',
                     showlegend=False)

    trace4 = Histogram(x=img.ravel(), name='image histogram',
                       xaxis='x4', yaxis='y4',
                       showlegend=False)

    data = Data([trace1, trace2, trace3, trace4])

    fig = tls.make_subplots(rows=2, cols=2, show)
    fig['data'] += data
    fig['layout'].update(title=title)
    return fig

