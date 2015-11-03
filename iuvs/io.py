import datetime as dt
import os
import socket
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from scipy.ndimage.filters import generic_filter

from .exceptions import DimensionsError, PathNotReadableError

host = socket.gethostname()
home = Path(os.environ['HOME'])
HOME = home

if host.startswith('maven-iuvs-itf'):
    analysis_out = home / 'to_keep'
else:
    analysis_out = home / 'data' / 'iuvs' / 'to_keep'

mycmap = 'cubehelix'
plotfolder = HOME / 'plots'
outputfolder = HOME / 'output'

sys_byteorder = ('>', '<')[sys.byteorder == 'little']


def env_path(env):
    """Return root path depending on `env`.

    Parameters
    ----------
    env : {'stage', 'production'}

    Returns
    -------
    path
        pathilb.Path
    """
    host = socket.gethostname()
    if host.startswith('maven-iuvs-itf'):
        path = Path('/maven_iuvs/{}/products'.format(env))
    elif host.startswith('test-machine'):
        path = Path('/abc')
    else:
        path = Path(os.environ['HOME']) / 'Dropbox' / 'data' / 'iuvs'
    return path


def convert_big_endian(data):
    try:
        if data.dtype.byteorder not in ('=', sys_byteorder):
            data = data.byteswap().newbyteorder(sys_byteorder)
    except AttributeError:  # when it's boolean e.g.
        pass
    return data


def get_data_path(level, env='stage'):
    """Return data path for given `level`.

    Some shortcuts for making interactive analysis faster.

    Parameters
    ----------
    level : {'l0', 'l1a', 'l1b', 'hk'}
        shorter string key to look up the longer subdir's names.
    }
    env : {'stage', 'production'}, optional
        Switch to decide between production or staging environment.
        Default: stage.
    """
    levelstring = dict(l0='level0', l1a='level1a', l1b='level1b',
                       hk='housekeeping/level1a')
    path = env_path(env) / levelstring[level]
    return path


def get_filenames(level, pattern=None, env='stage', ext='.fits.gz',
                  iterator=True):
    """return iterator (default) or list of filenames for given pattern and environment.

    Parameters
    ----------
    level : {'l0', 'l1a', 'l1b', 'hk'}
        dict key to look up the respective subdir name in `get_data_path`.
    pattern : str, optional
        globbing pattern for `Path.glob()`
    env : {'stage', 'production'}, optional
        Switch to decide between production or staging environment.
        Default: stage.
    ext : str, optional
        Extension for filtering what files to find. Usually '.fits.gz'
    iterator : bool
        Switch between returning iterator (default) or list.

    Returns
    -------
    list or iterator
        List or Iterator of filenames found.
    """
    if pattern is None:
        pattern = '*'
    else:
        pattern = '*' + pattern + '*'
    path = get_data_path(level, env)
    if not os.access(str(path), os.R_OK):
        raise PathNotReadableError(path)
    result = map(str, path.glob(pattern + ext))
    return result if iterator else list(result)


def l1a_filenames(pattern=None, **kwargs):
    """Search for L1A filenames with patterns.

    Parameters
    ----------
    pattern : str
        will be bracketed with '*', so needs to be correct in itself.
        For example "mode080-fuv" but not "mode080fuv".
    kwargs : dict
        To provide to `get_filenames`

    Examples
    --------
    `pattern` = "mode080-fuv"
    but not
    `pattern` = "mode080fuv"
    as that pattern does not exist.
    """
    return get_filenames('l1a', pattern=pattern, **kwargs)


def l1b_filenames(pattern=None, **kwargs):
    """Search for L1B filenames with patterns.

    Parameters
    ----------
    pattern : str
        will be bracketed with '*', so needs to be correct in itself.
        For example "mode080-fuv" but not "mode080fuv".
    kwargs : dict
        To provide to `get_filenames`

    Examples
    --------
    `pattern` = "mode080-fuv"
    but not
    `pattern` = "mode080fuv"
    as that pattern does not exist.
    """
    return get_filenames('l1b', pattern=pattern, **kwargs)


def l0_filenames(pattern=None, **kwargs):
    """Search for L1B filenames with patterns.

    Parameters
    ----------
    pattern : str
        will be bracketed with '*', so needs to be correct in itself.
        For example "mode080-fuv" but not "mode080fuv".
    kwargs : dict
        To provide to `get_filenames`

    Examples
    --------
    `pattern` = "mode080-fuv"
    but not
    `pattern` = "mode080fuv"
    as that pattern does not exist.
    """
    return get_filenames('l0', pattern=pattern, **kwargs)


def l1a_darks(darktype=''):
    searchpattern = darktype + 'dark*.fits.gz'
    print("Searching for", searchpattern)
    return l1a_filenames(searchpattern)


def image_stats(data):
    return pd.Series(data.ravel()).describe()


def get_filename_df(level, env='stage', pattern=None):
    """Return pandas.DataFrame with filename data.

    Parameters
    ----------
    level : {'l0', 'l1a', 'l1b', 'hk'}
        dict key to look up the respective subdir name in `get_data_path`.
    env : {'stage', 'production'}, optional
        Switch to decide between production or staging environment.
        Default: stage.
    pattern : str
        will be bracketed with '*', so needs to be correct in itself.
        For example "mode080-fuv" but not "mode080fuv".
        Provided to `get_filenames`.

    Returns
    -------
    pandas.DataFrame
        Indexed by time, if possible, sorted.
    """
    fnames = get_filenames(level, env=env, pattern=pattern)
    Filename = ScienceFilename
    iuvs_fnames = []
    for fname in fnames:
        if not level == 'hk':
            iuvs_fnames.append(ScienceFilename(fname))
        else:
            iuvs_fnames.append(HKFilename(fname))
    df = pd.DataFrame([fname.as_series() for fname in iuvs_fnames])
    if level != 'hk':
        df['channel'] = df.channel.astype('category')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    # next line filters for newest revisions
    return df[df.p.isin(df.groupby('obs_id', sort=False)['p'].max())]


def get_current_hk_fnames(env='stage'):
    "return only the latest revisions of filenames per observation_id."
    df = get_filename_df('hk', env=env)
    return df.p


def get_current_science_fnames(level, pattern=None, env='stage'):
    "return only the latest revisions of filenames per observation_id."
    df = get_filename_df(level, pattern=pattern, env=env)
    return df.p


def get_header_df(hdu, drop_comment=True):
    """Take a FITS HDU, convert to DataFrame.

    And on the way:
    fix it,drop COMMENT and KERNEL

    Parameters
    ----------
    hdu : FITS header unit
        The HDU to extract a header dataframe from
    drop_comment : bool
        To control if the comment and kernel lines from the header should be dropped.
        Default: True. No errors are raised when those fields do not exist.
    """
    hdu.verify('silentfix')
    header = hdu.header
    d = {}
    for key in set(header.keys()):
        if drop_comment and key == 'COMMENT':
            continue
        data = header[key]
        d[key] = convert_big_endian(data)
    df = pd.DataFrame(d, index=[0])
    return df.drop('COMMENT KERNEL'.split(), axis=1, errors='ignore') if drop_comment else df


def save_to_hdf(df, fname, output_subdir=None):
    """Save temporary HDF file in output folder for later concatenation.

    By default the product is stored in HOME/output.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to save
    fname : string
        The product filename that was used to create this dataframe to save.
        The saving filename for the HDF file will be auto-determined from that.
    output_subdir : str
        String to determine a subfolder inside HOME/output where this data
        should be stored instead of just HOME/output

    """
    path = Path(fname)
    newfname = path.with_suffix('.h5').name
    folderpath = HOME / 'output'
    if output_subdir:
        folderpath = folderpath / output_subdir
    path = folderpath / newfname
    df.to_hdf(str(path), 'df', format='t')
    return str(path)


class Filename(object):

    def __init__(self, fname):
        self.p = Path(fname)
        self.root = self.p.parent
        self.basename = self.p.name
        self.tokens = self.basename.split('_')
        self.mission, self.instrument = self.tokens[:2]

    def as_series(self):
        return pd.Series(self.__dict__)


class ScienceFilename(Filename):

    def __init__(self, fname):
        super(ScienceFilename, self).__init__(fname)
        tokens = self.tokens
        self.level = tokens[2]
        self.phase = tokens[3]
        self.timestr, self.version = tokens[4:6]
        self.revision = tokens[6].split('.')[0]
        phasetokens = self.phase.split('-')
        if len(phasetokens) == 4:
            self.phase, self.cycle_orbit, self.mode, self.channel = phasetokens
        elif len(phasetokens) == 3:
            self.phase, self.cycle_orbit, self.channel = phasetokens
            self.mode = 'N/A'
        else:
            self.phase, self.channel = phasetokens
            self.mode = 'N/A'
            self.cycle_orbit = 'N/A'
        self.time = dt.datetime.strptime(self.timestr,
                                         '%Y%m%dT%H%M%S')
        self.version_string = self.version + self.revision
        self.obs_id = '_'.join(self.basename.split('_')[:5])
        if self.cycle_orbit.startswith('orbit'):
            self.orbit = float(self.cycle_orbit[5:])
        else:
            self.orbit = np.nan

    def __eq__(self, other):
        weak_equality = ['mission', 'instrument', 'level', 'phase', 'timestr']
        strong_equality = ['version', 'revision']
        weak = True
        strong = True
        for attr in weak_equality:
            # if any attribute is different, weak get's set to False
            weak = weak and (getattr(self, attr) == getattr(other, attr))
        for attr in strong_equality:
            strong = strong and (getattr(self, attr) == getattr(other, attr))
        if weak and strong:
            return True
        elif weak:
            return 0
        else:
            return False

    def formatter(self, itemstr):
        return "{}: {}\n".format(itemstr.capitalize(), getattr(self, itemstr))

    def __repr__(self):
        items = ('basename mission instrument level phase cycle_orbit mode channel'
                 ' version revision time'.split())
        s = ''
        for item in items:
            s += self.formatter(item)
        return s

    def __str__(self):
        return self.__repr__()


class HKFilename(Filename):

    def __init__(self, fname):
        super(HKFilename, self).__init__(fname)
        tokens = self.tokens
        self.kind = tokens[2]
        self.level = tokens[3]
        self.datestring = tokens[4]
        self.version = tokens[5].split('.')[0]
        self.obs_id = '_'.join(self.basename.split('_')[:5])
        year, month, day = tokens[4][:4], tokens[4][4:6], tokens[4][6:8]
        self.time = dt.datetime(int(year), int(month), int(day))


class FitsBinTable(object):

    """Convert a binary Fits table to a pandas table.

    Attributes
    ==========
    header: links to the header of the provided HDU
    data: contains the pandas DataFrame with the HDU.data
    """

    def __init__(self, hdu):
        self.header = hdu.header
        self.data = pd.DataFrame(hdu.data).T


def iuvs_utc_to_dtime(utcstring):
    "Convert the IUVS UTC string to a dtime object."
    cleaned = utcstring[:-3] + '0UTC'
    time = dt.datetime.strptime(cleaned, '%Y/%j %b %d %H:%M:%S.%f%Z')
    return time


def set_spec_vmax_vmin(log, inspec, vmax, vmin):
    if log:
        spec = np.log10(inspec)
        vmax = 2.5 if vmax is None else vmax
        vmin = -3.0 if vmin is None else vmin
    else:
        spec = inspec
        vmax = 10 if vmax is None else vmax
        vmin = 0 if vmin is None else vmin
    return spec, vmax, vmin


def do_labels(ax, title='', set_extent=None):
    ax.set_title(title)
    if set_extent is True:
        xlabel = 'Wavelength [nm]'
    elif set_extent is False:
        xlabel = 'Spectral bins'
    else:
        xlabel = 'set_extent not specified in do_labels'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Spatial pixels')


def plot_colorbar(im, ax, log):
    cb = plt.colorbar(im, ax=ax)
    label = 'log(DN/s)' if log else 'DN/s'
    cb.set_label(label, fontsize=14, rotation=0)


def plot_hist(ax, spec):
    in_axes = inset_axes(ax, width="20%", height="20%",
                         loc=2)
    in_axes.hist(spec.ravel(), bins=20, normed=True)
    plt.setp(in_axes.get_xticklabels(), visible=False)
    plt.setp(in_axes.get_yticklabels(), visible=False)
    in_axes.grid('off')


class ScienceFitsFile(object):

    def __init__(self, fname):
        """Base class for L1A/B Reader.

        Input:
            fname: needs to be absolute complete path. (To be tested.)
        """
        if type(fname) == list:
            fname = fname[0]
        self.fname = fname
        self.iuvsfname = Filename(fname)
        self.hdulist = fits.open(self.fname)

    def get_real_binnings(self, dim):
        binning = getattr(self, 'Binning')
        widths = binning[dim+'BINWIDTH']
        transmits = binning[dim+'BINTRANSMIT'].astype(bool)
        return widths[transmits]

    @property
    def spabins(self):
        return self.get_real_binnings('SPA')[:, np.newaxis]

    @property
    def n_unique_spabins(self):
        return np.unique(self.spabins).size

    @property
    def spebins(self):
        return self.get_real_binnings('SPE')[np.newaxis, :]

    @property
    def n_unique_spebins(self):
        return np.unique(self.spebins).size

    @property
    def scaling_factor(self):
        """Return factor to get DN/s.

        Because the binning returns just summed up values, one must
        also include the binning as a scaling factor, not only the
        integration time.
        """
        bin_scale = self.spabins * self.spebins
        return bin_scale * self.int_time

    def __repr__(self):
        s = "Filename: {}\n".format(self.p.name)
        s += "Environment: {}\n".format(self.env)
        s += "n_dims: {}\n".format(self.n_dims)
        s += "spatial: {}\n".format(self.spatial_size)
        s += "spectral: {}".format(self.spectral_size)
        return s

    @property
    def n_dims(self):
        return self.img_header['NAXIS']

    @property
    def n_integrations(self):
        return int(getattr(self, 'Engineering').get_value(0, 'NUMBER'))

    @property
    def primary_img_dn_s(self):
        return (self.img / self.scaling_factor) + 0.00001

    @property
    def spatial_size(self):
        return self.img_header['NAXIS2']

    @property
    def spectral_size(self):
        return self.img_header['NAXIS1']

    @property
    def int_time(self):
        return self.img_header['INT_TIME']

    @property
    def wavelengths(self):
        return getattr(self, 'Observation')['WAVELENGTH'][0]

    @property
    def img_header(self):
        imgdata = self.hdulist[0]
        return imgdata.header

    @property
    def img(self):
        return self.hdulist[0].data

    @property
    def scaled_img(self):
        return self.img / self.scaling_factor

    @property
    def plotfname(self):
        return os.path.basename(self.fname)[12:-16]

    @property
    def plottitle(self):
        title = "{fname}, INT_TIME: {int}".format(fname=self.plotfname,
                                                  int=self.int_time)
        return title

    @property
    def capture(self):
        string = self.img_header['CAPTURE']
        return iuvs_utc_to_dtime(string)

    @property
    def integration_times(self):
        "Convert times from Integration table to pandas TimeSeries"
        return getattr(self, 'Integration').loc['UTC'].map(iuvs_utc_to_dtime)

    def get_integration(self, data_attr, integration):
        data = getattr(self, data_attr)
        if data.ndim == 3:
            if integration is None:
                print("More than 1 integration present.\n"
                      "Need to provide integration index.")
                return
            else:
                spec = data[integration]
        else:
            spec = data
        return spec

    def get_n_data_attr(self, data_attr):
        data = getattr(self, data_attr)
        if data.ndim == 3:
            return data.shape[0]
        else:
            return 1

    def plot_some_spectrogram(self, inspec, title, ax=None, cmap=None,
                              cbar=True, log=False, showaxis=True,
                              min_=None, max_=None, set_extent=None,
                              draw_rectangle=True, vmin=None, vmax=None,
                              **kwargs):
        plot_hist = kwargs.pop('plot_hist', False)
        savename = kwargs.pop('savename', False)

        spec, vmax, vmin = set_spec_vmax_vmin(log, inspec, vmax, vmin)
        cmap = mycmap if cmap is None else cmap

        if ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self.plottitle, fontsize=16)
        try:
            waves = self.wavelengths[0]
        except IndexError:
            waves = self.wavelengths
        if set_extent:
            im = ax.imshow(spec, cmap=cmap,
                           extent=(waves[0], waves[-1], len(spec), 0),
                           vmin=vmin,
                           vmax=vmax,
                           aspect='auto',
                           **kwargs)
        else:
            im = ax.imshow(spec, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect='auto', **kwargs)

        do_labels(ax, title=title, set_extent=set_extent)

        if not showaxis:
            ax.grid('off')
        if cbar:
            plot_colorbar(im, ax, log)

        # rectangle
        if draw_rectangle:
            ax.add_patch(get_rectangle(inspec))

        # inset histogram
        if plot_hist:
            plot_hist(ax, spec)

        if savename:
            ax.get_figure().savefig(savename, dpi=100)

        self.current_ax = ax
        self.current_spec = spec

        return ax

    def plot_some_profile(self, data_attr, integration,
                          spatial=None, ax=None, scale=False,
                          log=False, spa_average=False, title=None,
                          **kwargs):
        plot_hist = kwargs.pop('plot_hist', False)
        savename = kwargs.pop('savename', False)
        spec = self.get_integration(data_attr, integration)
        nints = self.get_n_data_attr(data_attr)
        if scale:
            spec = spec / self.scaling_factor
        if spatial is None:
            # if no spatial bin given, take the middle one
            spatial = self.spatial_size // 2

        if title is None:
            if not spa_average:
                title = ("Profile of {} at spatial: {}, integration {} of {}"
                         .format(data_attr, spatial, integration, nints))
            else:
                title = ("Profile of {}, spatial mean. Integration {} of {}"
                         .format(data_attr, integration, nints))
        if ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self.plottitle, fontsize=12)
        if log:
            func = ax.semilogy
        else:
            func = ax.plot

        if spa_average:
            data = spec.mean(axis=0)
        else:
            data = spec[spatial]

        func(self.wavelengths[spatial], data, **kwargs)

        ax.set_xlim((self.wavelengths[spatial][0],
                     self.wavelengths[spatial][-1]))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Wavelength [nm]")
        if log:
            ax.set_ylabel("log(DN/s)")
        else:
            ax.set_ylabel('DN/s')

        if plot_hist:
            in_axes = inset_axes(ax, width="20%", height="20%",
                                 loc=2)
            in_axes.hist(spec.ravel(), bins=20, normed=True, log=True)
            plt.setp(in_axes.get_xticklabels(), visible=False)
            plt.setp(in_axes.get_yticklabels(), visible=False)
            in_axes.grid('off')

        if savename:
            ax.get_figure().savefig(savename, dpi=100)

        return ax

    def plot_img_spectrogram(self,
                             integration=None, ax=None,
                             cmap=None, cbar=True, log=True, **kwargs):
        spec = self.get_integration('img', integration)
        title = ("Primary spectrogram, integration {} out of {}"
                 .format(integration, self.n_integrations))
        return self.plot_some_spectrogram(spec, title,
                                          ax, cmap, cbar, log, **kwargs)

    def plot_img_profile(self, integration=None, ax=None, log=True,
                         **kwargs):

        return self.plot_some_profile('img', integration, ax=ax,
                                      **kwargs)

    def image_stats(self):
        return image_stats(self.img)


def fits_table_to_dataframe(hdu):
    d = {}
    for col in hdu.columns:
        data = hdu.data[col.name]
        d[col.name] = convert_big_endian(data)
    return pd.DataFrame(d)


class L1AReader(ScienceFitsFile):

    """For Level1a"""

    works_with_dataframes = [
        'Integration',
        'Engineering',
    ]

    level = 'l1a'

    def __init__(self, fname, env='production'):
        # fix relative paths
        self.env = env
        fname = Path(fname)
        if not fname.is_absolute():
            fname = get_data_path(self.level, env) / fname
        self.p = fname
        # call super init
        super(L1AReader, self).__init__(str(fname))

        if self.spectral_size == 1024 and self.spatial_size == 1024:
            warnings.warn("\nNot loading HDU data due to performance issue.\n"
                          "Identified with 'Observation' HDU so far, working with other\n"
                          "data should be fine.")
        else:
            for hdu in self.hdulist[1:]:
                name = hdu.header['EXTNAME']
                setattr(self, name + '_header', hdu.header)

                if name in self.works_with_dataframes:
                    setattr(self, name, fits_table_to_dataframe(hdu))
                else:
                    setattr(self, name, hdu.data)
        # check for error case with binning table not found:
        if self.n_dims == 2 and self.n_integrations > 1:
            raise DimensionsError('n_dims == 2 with n_integrations > 1')


class L1BReader(ScienceFitsFile):

    """For Level1B"""

    level = 'l1b'
    works_with_dataframes = ['DarkIntegration',
                             'DarkEngineering',
                             'background_light_source',
                             'Integration',
                             'Engineering']

    def __init__(self, fname, env='stage'):

        # fix relative path
        if not os.path.isabs(fname):
            fname = get_data_path(self.level, env) / fname

        # call super init
        super(L1BReader, self).__init__(fname)
        for hdu in self.hdulist[1:]:
            name = hdu.header['EXTNAME']
            setattr(self, name + '_header', hdu.header)
            if name in self.works_with_dataframes:
                setattr(self, name, pd.DataFrame(hdu.data))
            else:
                setattr(self, hdu.header['EXTNAME'], hdu.data)
        self.darks_interpolated = getattr(self, 'background_dark')

    @property
    def dark_det_temps(self):
        return getattr(self, 'Dark_Integration')['DET_TEMP_C']

    @property
    def dark_case_temps(self):
        return getattr(self, 'Dark_Integration')['CASE_TEMP_C']

    @property
    def dark_times(self):
        try:
            utcs = getattr(self, 'DarkIntegration')['UTC']
        except AttributeError:
            utcs = getattr(self, 'Dark_Integration')['UTC']
        times = []
        for utc in utcs:
            times.append(iuvs_utc_to_dtime(utc))
        return pd.TimeSeries(times)

    @property
    def n_darks(self):
        return getattr(self, 'detector_dark').shape[0]

    @property
    def raw_dn_s(self):
        return (getattr(self, 'detector_raw') / self.scaling_factor) + 0.001

    @property
    def dark_dn_s(self):
        return (getattr(self, 'detector_dark') / self.scaling_factor) + 0.001

    @property
    def dds_dn_s(self):
        try:
            dds = getattr(self, 'detector_dark_subtracted')
        except AttributeError:
            dds = getattr(self, 'detector_background_subtracted')
        return (dds / self.scaling_factor) + 0.001

    def plot_raw_spectrogram(self, integration=None, ax=None,
                             cmap=None, cbar=True, log=False,
                             set_extent=True,
                             **kwargs):
        if integration is None:
            integration = -1
        spec = self.get_integration('raw_dn_s', integration)
        title = ("Raw light spectrogram, integration {} out of {}"
                 .format(integration, self.n_integrations))
        return self.plot_some_spectrogram(spec, title, ax,
                                          cmap, cbar, log,
                                          set_extent=set_extent,
                                          **kwargs)

    def plot_dark_spectrogram(self, integration=None, ax=None,
                              cmap=None, cbar=True, log=False,
                              **kwargs):
        dark = self.get_integration('dark_dn_s', integration)
        title = ("Dark spectogram, integration {} out of {}"
                 .format(integration, self.n_darks))
        return self.plot_some_spectrogram(dark, title, ax,
                                          cmap, cbar, log, **kwargs)

    def plot_raw_overview(self, integration=None, imglog=True,
                          save_token=None, spatial=None, proflog=True,
                          img_plot_hist=False, prof_plot_hist=False,
                          **kwargs):
        if integration is None:
            integration = -1
        "Plot overview of spectrogram and profile at index `integration`."
        fig, axes = plt.subplots(nrows=2, sharex=False)
        fig.suptitle(self.plottitle, fontsize=16)

        # spectrogram
        ax = self.plot_raw_spectrogram(integration, ax=axes[0],
                                       cbar=False, log=imglog,
                                       set_extent=False, plot_hist=img_plot_hist,
                                       **kwargs)

        # profile
        self.plot_raw_profile(integration, ax=axes[1], log=proflog,
                              spatial=spatial, plot_hist=prof_plot_hist)

        # colorbar
        im = ax.get_images()[0]  # pylint: disable=no-member
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=0.1)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # cb = plt.colorbar(im, cax=cbar_ax)
        cb = plt.colorbar(im, ax=axes.ravel().tolist())
        # cb = plt.colorbar(im, ax=axes[0])
        if imglog:
            label = '  log(DN/s)'
        else:
            label = '  DN/s'
        cb.set_label(label, fontsize=13, rotation=0)

        if save_token is not None:
            fname = "{}_{}.png".format(self.plotfname,
                                       save_token)
            fig.savefig(os.path.join(str(plotfolder), fname), dpi=150)

    def plot_mean_values(self, item):
        fig, ax = plt.subplots()
        fig.suptitle(self.plottitle)
        ax.plot(getattr(self, item).mean(axis=(1, 2)))
        ax.set_xlabel("Integration number")
        ax.set_ylabel("DN / s")
        ax.set_title("Mean {} over observation (i.e. L1B file)".format(item))
        savename = os.path.join(str(plotfolder),
                                self.plotfname + 'mean_{}.png'.format(item))
        plt.savefig(savename, dpi=120)

    def plot_mean_raw_values(self):
        self.plot_mean_values('raw_dn_s')

    def plot_mean_dds_values(self):
        self.plot_mean_values('dds_dn_s')

    def plot_dark_spectrograms(self):
        fig, axes = plt.subplots(nrows=self.n_darks, sharex=True)
        fig.suptitle(self.plottitle)
        for i, ax in zip(range(self.n_darks), axes):
            self.plot_dark_spectrogram(integration=i, ax=ax)
            if i < self.n_darks - 1:
                ax.set_xlabel('')
        savename = os.path.join(str(plotfolder), self.plotfname + '_dark_spectograms.png')
        plt.savefig(savename, dpi=150)

    def plot_dark_histograms(self, save=False):
        fig, ax = plt.subplots()
        for i, dark in enumerate(self.dark_dn_s):
            ax.hist(dark.ravel(), 100, log=True,
                    label="dark{}".format(i), alpha=0.5)
        plt.legend()
        fig.suptitle(self.plottitle)
        ax.set_title('Dark histograms, DN / s')
        if save:
            savename = os.path.join(str(plotfolder), self.plotfname + '_dark_histograms.png')
            plt.savefig(savename, dpi=150)

    def find_scaling_window(self, spec):
        self.spa_slice, self.spe_slice = find_scaling_window(spec)
        return self.spa_slice, self.spe_slice

    def plot_raw_profile(self, integration=-1, ax=None, log=None,
                         spatial=None, **kwargs):
        return self.plot_some_profile('raw_dn_s', integration,
                                      ax=ax, log=log, spatial=spatial,
                                      **kwargs)

    def plot_dark_profile(self, integration=-1, ax=None, log=None):
        return self.plot_some_profile('dark_dn_s', integration,
                                      ax=ax, log=log)

    def plot_dds_profile(self, integration=-1, ax=None, log=None):
        return self.plot_some_profile('dds_dn_s', integration,
                                      ax=ax, log=log)

    def get_light_and_dark(self, integration):
        light = self.get_integration('raw_dn_s', integration)
        dark = self.get_integration('dark_dn_s', integration)
        return light, dark

    def show_all_darks(self):
        fig, axes = plt.subplots(nrows=self.n_darks, sharex=True)
        for ax, i_dark in zip(axes.ravel(), range(self.n_darks)):
            self.plot_dark_spectrogram(i_dark, ax=ax)
        for ax in axes[:-1]:
            ax.set_xlabel('')

    def profile_all_darks(self):
        fig, axes = plt.subplots()
        for i, dark in enumerate(self.dark_dn_s):
            axes.plot(self.wavelengths[0],
                      dark.mean(axis=0), label=i, lw=2)
        axes.legend(loc='best')
        axes.set_xlim((self.wavelengths[0][0],
                       self.wavelengths[0][-1]))
        axes.set_xlabel("Wavelength [nm]")
        axes.set_ylabel("DN / s")
        axes.set_title("{}\nSpatial mean profiles of darks."
                       .format(self.plottitle),
                       fontsize=14)

    def plot_spectrum(self, data, integration, spatial=None, ax=None, scale=False,
                      log=False, spa_average=False, title=None,
                      **kwargs):
        savename = kwargs.pop('savename', False)
        spec = data[integration]
        if spatial is None:
            # if no spatial bin given, take the middle one
            spatial = self.spatial_size // 2

        if ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self.plottitle, fontsize=12)
        if log:
            func = ax.semilogy
        else:
            func = ax.plot

        if spa_average:
            data = spec.mean(axis=0)
        else:
            data = spec[spatial]

        func(self.wavelengths[spatial], data, **kwargs)

        ax.set_xlim((self.wavelengths[spatial][0],
                     self.wavelengths[spatial][-1]))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Wavelength [nm]")
        if log:
            ax.set_ylabel("log(DN/s)")
        else:
            ax.set_ylabel('DN/s')

        if savename:
            ax.get_figure().savefig(savename, dpi=100)

        return ax


def get_rectangle(spectogram):
    spa_slice, spe_slice = find_scaling_window(spectogram)
    xy = spe_slice.start - 0.5, spa_slice.start - 0.5
    width = spe_slice.stop - spe_slice.start
    height = spa_slice.stop - spa_slice.start
    return Rectangle(xy, width, height, fill=False, color='white',
                     lw=2)


def find_scaling_window(to_filter, size=None):
    if size is None:
        x = max(to_filter.shape[0] // 5, 2)
        y = max(to_filter.shape[1] // 10, 1)
        size = (x, y)
    filtered = generic_filter(to_filter, np.median, size=size,
                              mode='constant', cval=to_filter.max() * 100)
    min_spa, min_spe = np.unravel_index(filtered.argmin(), to_filter.shape)
    spa1 = min_spa - size[0] // 2
    if spa1 < 0:
        spa1 = 0
    spa2 = spa1 + size[0]
    if spa2 > to_filter.shape[0]:
        spa1 = to_filter.shape[0] - size[0]
        spa2 = to_filter.shape[0]
    spe1 = min_spe - size[1] // 2
    if spe1 < 0:
        spe1 = 0
    spe2 = spe1 + size[1]
    if spe2 > to_filter.shape[1]:
        spe1 = to_filter.shape[1] - size[1]
        spe2 = to_filter.shape[1]
    spa_slice = slice(spa1, spa2)
    spe_slice = slice(spe1, spe2)
    return (spa_slice, spe_slice)


def check_scaling_window_finder(l1b, integration):
    to_filter = l1b.get_integration('raw_dn_s', integration)
    x = max(to_filter.shape[0] // 10, 1)
    y = max(to_filter.shape[1] // 10, 1)
    size = (x, y)
    print("Img shape:", to_filter.shape)
    print("Kernel size:", size)

    filtered = generic_filter(to_filter, np.std, size=size,
                              mode='constant', cval=to_filter.max() * 100)
    min_spa, min_spe = np.unravel_index(filtered.argmin(), to_filter.shape)
    print("Minimum:", filtered.min())
    print("Minimum coords", min_spa, min_spe)

    spa1 = min_spa - size[0] // 2
    if spa1 < 0:
        spa1 = 0
    spa2 = spa1 + size[0]
    if spa2 > to_filter.shape[0]:
        spa1 = to_filter.shape[0] - size[0]
        spa2 = to_filter.shape[0]
    print("Spatial:", spa1, spa2)

    spe1 = min_spe - size[1] // 2
    if spe1 < 0:
        spe1 = 0
    spe2 = spe1 + size[1]
    if spe2 > to_filter.shape[1]:
        spe1 = to_filter.shape[1] - size[1]
        spe2 = to_filter.shape[1]
    print("Spectral:", spe1, spe2)

    fig, axes = plt.subplots(nrows=3)
    axes[0].imshow(np.log(to_filter), cmap=mycmap)
    axes[0].add_patch(get_rectangle(to_filter))
    axes[1].imshow(np.log(filtered), cmap=mycmap, vmax=0.1)
    axes[1].add_patch(get_rectangle(to_filter))
    axes[2].hist(filtered[~np.isnan(filtered)].ravel(), bins=100)


def some_file(level, pattern):
    try:
        fname = get_filenames(level, pattern=pattern, iterator=False)[0]
    except IndexError:
        print("No {} files found.".format(level))
    return fname


def some_l1a(pattern=None):
    return L1AReader(some_file('l1a', pattern))


def some_l1b(pattern=None):
    return L1BReader(some_file('l1b', pattern))
