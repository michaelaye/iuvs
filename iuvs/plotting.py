"""some plotting utilities for IUVS"""
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import ceil
# try:
#     from plotly.graph_objs import Heatmap, Scatter, Histogram, Data,\
#         Annotation, Annotations, Font, Marker
#     import plotly.tools as tls
# except ImportError:
#     print("Can't import plotly")
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
from .scaling import poly_fitting


def calc_4_to_3(width):
    height = width * 3 / 4
    return (width, height)


def get_pie_plot(df, col, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    df[col].value_counts().plot(kind='pie', ax=ax, title=col)
    if ax is None:
        return fig


def plot_pie_overview(df, cols, title, ncols=3):
    nrows = int(ceil(len(cols) / ncols))
    scaler = 3
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


# def make_plotly_multiplot(img, spatial=None, spectral=None, title='No title',
#                           width=None, height=None, zmin=None, zmax=None):
#     if spatial is None:
#         spatial = img.shape[0]//2
#     if spectral is None:
#         spectral = img.shape[1]//2

#     prof1 = img[spatial]
#     prof2 = img[:, spectral]

#     if zmin is None:
#         p2, p98 = np.percentile(img, (2, 98))
#     else:
#         p2 = zmin
#         p98 = zmax
#     lowhist, p99 = np.percentile(img, (0.05, 99))
#     trace1 = Heatmap(z=np.flipud(img), zmin=p2, zmax=p98, zauto=False)
#     trace2 = Scatter(x=prof2,
#                      name='spatial profile',
#                      xaxis='x2',
#                      yaxis='y2',
#                      showlegend=False,
#                      mode='lines+markers')
#     trace3 = Scatter(y=prof1,
#                      name='spectral profile',
#                      xaxis='x3',
#                      yaxis='y3',
#                      showlegend=False,
#                      mode='lines+markers')

#     tohist = img[:, :50]
#     tohist = tohist[tohist < p99]
#     # tohist = tohist[tohist > lowhist]
#     trace4 = Histogram(x=tohist.ravel(),
#                        name='image histogram',
#                        xaxis='x4',
#                        yaxis='y4',
#                        showlegend=False,
#                        marker=Marker(
#                             color='blue',
#                             opacity=0.5)
#                        )
#     annotation = Annotation(x=0.95, y=0.4, xref='paper', yref='paper',
#                             text="Mean: {:.1f}<br>STD: {:.1f}".
#                                  format(tohist.mean(), tohist.std()),
#                             showarrow=False,
#                             font=Font(
#                                 size=16,
#                                 color='black'))

#     data = Data([trace1, trace2, trace3, trace4])

#     fig = tls.make_subplots(rows=2, cols=2, print_grid=False)
#     if width is not None:
#         fig['layout']['autosize'] = False
#         fig['layout']['width'] = width
#         fig['layout']['height'] = height
#     fig['layout']['annotations'] = Annotations([annotation])
#     fig['data'] += data
#     fig['layout'].update(title=title)
#     return fig


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


def make_spectogram_anim(l1b, data_attr, fps):
    """Create animation out of `data_attr` spectrograms.

    Parameters
    ==========
    l1b: <io.L1BReader>
    data_attr: <string>
        Name of attribute of the io.L1BReader object that should be
        animated.
    fps: <int>
        Frame rate of resulting animation

    Returns
    =======
    Produces an .mp4 and .mp4 in current directory, with name
    `data_attr`.mp4/.gif
    """
    data = getattr(l1b, data_attr)
    fig_mpl, ax = plt.subplots(1, figsize=(10, 6), facecolor='white')
    xx = l1b.wavelengths[0]

    def zz(x):
        return np.log(data[x])

    ax.set_title(data_attr)
    ax.set_xlim(xx[0], xx[-1])
    ax.axis('off')
    im = ax.imshow(zz(0), cmap='binary')

    duration = len(data) / fps

    def make_frame_mpl(t):
        index = int(t*fps)
        im.set_data(zz(index))  # <= Update the curve
        ax.set_title('Spectogram {} out of {}'.format(index+1, len(data)))
        return mplfig_to_npimage(fig_mpl)  # RGB image of the figure

    animation = mpy.VideoClip(make_frame_mpl, duration=duration)
    animation.write_videofile(data_attr+".mp4", fps=fps)
    animation.write_gif(data_attr+'.gif', fps=fps)


def make_line_profile_anim(l1b, data_attr, fps, spatial=None):
    """Create animation out of `data_attr` line profiles at `spatial` pixel.

    Parameters
    ==========
    l1b: <io.L1BReader>
    data_attr: <string>
        Name of attribute of the io.L1BReader object that should be
        animated.
    fps: <int>
        Framerate of resulting animation
    spatial: <int>
        Pixel index location for profile. Default is central pixel of
        spatial axis.

    Returns
    =======
    Produces an .mp4 and .mp4 in current directory, with name
    `data_attr`.mp4/.gif
    """
    data = getattr(l1b, data_attr)
    if spatial is None:
        spatial = data.shape[1]//2
    fig_mpl, ax = plt.subplots(1, figsize=(10, 6), facecolor='white')
    xx = l1b.wavelengths[0]

    def zz(x):
        return data[x][spatial]

    ax.set_title(data_attr)
    ax.set_xlim(xx[0], xx[-1])
    line, = ax.semilogy(xx, zz(0), lw=3)

    duration = len(data) / fps

    def make_frame_mpl(t):
        index = int(t*fps)
        line.set_ydata(zz(index))
        ax.set_title('t: {}, Profile {} at spatial {} out of {}'
                     .format(t, index+1, spatial, len(data)))
        return mplfig_to_npimage(fig_mpl)

    animation = mpy.VideoClip(make_frame_mpl, duration=duration)
    animation.write_videofile(data_attr+'_profiles.mp4', fps=fps)
    animation.write_gif(data_attr+'_profiles.gif', fps=fps)


def plot_profiles(l1b, spatialslice, spectralslice, integration):
    # sharing x axes
    fig, axes = plt.subplots(nrows=4, sharex=True)
    axes = axes.ravel()
    light, dark = l1b.get_light_and_dark(integration)

    spatial = light.shape[0]//2
    # Raw profile
    l1b.plot_raw_profile(integration, ax=axes[0])
    # Dark profile
    l1b.plot_dark_profile(integration, ax=axes[1])

    # fitting
    fitted_dark = poly_fitting(l1b, integration, spatialslice, spectralslice)
    sub = light - fitted_dark
    min_, max_ = np.percentile(sub, (2, 92))

    # old subtraction from L1B file
    l1b.plot_some_profile('detector_background_subtracted', integration,
                          spatial=spatial, ax=axes[2], scale=True)
    axes[2].set_ylim(min_, max_)
    axes[2].set_title('subtracted from L1B file, y-axis same as last profile')

    # remove first axis xlabels to make plot less messy
    for ax in axes[0:3]:
        ax.set_xlabel('')

    # New subtraction
    axes[3].plot(l1b.wavelengths[integration], sub[spatial])
    axes[3].set_ylim(min_, max_)
    axes[3].set_title('subtracted with fitted dark')
    axes[3].set_xlabel('Wavelength [nm]')

    fig.suptitle("{}\nSlice: [{}:{}, {}:{}]\n"
                 .format(l1b.plottitle,
                         spatialslice.start,
                         spatialslice.stop,
                         spectralslice.start,
                         spectralslice.stop),
                 fontsize=14)
    fig.subplots_adjust(top=0.85)
#     fig.tight_layout()
    fig.savefig('plots/'+l1b.plotfname+'_2.png', dpi=150)
    return sub
