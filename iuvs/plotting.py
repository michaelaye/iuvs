"""some plotting utilities for IUVS"""
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import ceil
from plotly.graph_objs import Heatmap, Scatter, Histogram, Data,\
    Annotation, Annotations, Font, Marker
import plotly.tools as tls
from . import scaling


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


def make_plotly_multiplot(img, spatial=None, spectral=None, title='No title',
                          width=None, height=None, zmin=None, zmax=None):
    if spatial is None:
        spatial = img.shape[0]//2
    if spectral is None:
        spectral = img.shape[1]//2

    prof1 = img[spatial]
    prof2 = img[:, spectral]

    if zmin is None:
        p2, p98 = np.percentile(img, (2, 98))
    else:
        p2 = zmin
        p98 = zmax
    lowhist, p99 = np.percentile(img, (0.05, 99))
    trace1 = Heatmap(z=np.flipud(img), zmin=p2, zmax=p98, zauto=False)
    trace2 = Scatter(x=prof2,
                     name='spatial profile',
                     xaxis='x2',
                     yaxis='y2',
                     showlegend=False,
                     mode='lines+markers')
    trace3 = Scatter(y=prof1,
                     name='spectral profile',
                     xaxis='x3',
                     yaxis='y3',
                     showlegend=False,
                     mode='lines+markers')

    tohist = img[:, :50]
    tohist = tohist[tohist < p99]
    # tohist = tohist[tohist > lowhist]
    trace4 = Histogram(x=tohist.ravel(),
                       name='image histogram',
                       xaxis='x4',
                       yaxis='y4',
                       showlegend=False,
                       marker=Marker(
                            color='blue',
                            opacity=0.5)
                       )
    p50 = np.percentile(img, 50)
    annotation = Annotation(x=0.95, y=0.4, xref='paper', yref='paper',
                            text="Mean: {:.1f}<br>STD: {:.1f}".
                                 format(tohist.mean(), tohist.std()),
                            showarrow=False,
                            font=Font(
                                size=16,
                                color='black'))

    data = Data([trace1, trace2, trace3, trace4])

    fig = tls.make_subplots(rows=2, cols=2, print_grid=False)
    if width is not None:
        fig['layout']['autosize'] = False
        fig['layout']['width'] = width
        fig['layout']['height'] = height
    fig['layout']['annotations'] = Annotations([annotation])
    fig['data'] += data
    fig['layout'].update(title=title)
    return fig


class L1BImageOperator(object):
    """Execute string based operations on contained images"""
    def __init__(self, l1b):
        super().__init__()
        self.l1b = l1b
        cube = self.l1b.detector_raw
        self.dark1 = self.l1b.detector_dark[1]
        self.dark2 = self.l1b.detector_dark[2]
        self.rawfirst = cube[0]
        self.rawlast = cube[-1]

    def get_images_from_statements(self, list_of_statements):
        dataitems = []
        for item in list_of_statements:
            if '-' in item:
                item1, item2 = item.split('-')
                data = getattr(self, item1) - getattr(self, item2)
            else:
                data = getattr(self, item)
            dataitems.append(data)
        return dataitems


