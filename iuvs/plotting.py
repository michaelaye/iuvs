"""some plotting utilities for IUVS"""
import matplotlib.pyplot as plt
import os
from numpy import ceil


def get_pie_plot(df, col, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    df[col].value_counts().plot(kind='pie', ax=ax, title=col)
    if ax is None:
        return fig


def plot_pie_overview(df, cols, ncols=3):
    nrows = int(ceil(len(cols) / ncols))
    scaler = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*scaler,
                                                                nrows*scaler))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        get_pie_plot(df, col, ax=ax)
    return fig


def produce_pie_plots(df, folder):
    items = 'MCP_HV XUV'.split()
    for item in items:
        fig = get_pie_plot(df, item)
        path = os.path.join(folder, '{}__pie.png'.format(item))
        plt.savefig(path, dpi=100)
        plt.close(fig)
