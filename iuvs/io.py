import datetime as dt
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import socket
import numpy as np
from scipy.ndimage.filters import generic_filter
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

host = socket.gethostname()
home = Path(os.environ['HOME'])
HOME = home

if host.startswith('maven-iuvs-itf'):
    stage = Path('/maven_iuvs/stage/products')
    production = Path('/maven_iuvs/production/products')
else:
    stage = home / 'data' / 'iuvs'
    production = home / 'data' / 'iuvs'

stagelevel1apath = stage / 'level1a'
stagelevel1bpath = stage / 'level1b'
productionlevel1apath = production / 'level1a'
productionlevel1bpath = production / 'level1b'

mycmap = 'cubehelix'
plotfolder = HOME / 'plots'


def get_filenames(level, pattern=None, stage=True, ext='.fits.gz',
                  iterator=True):
    if pattern is None:
        pattern = '*'
    else:
        pattern = '*' + pattern + '*'
    if level == 'l1a':
        if stage:
            path = stagelevel1apath
        else:
            path = productionlevel1apath
    else:
        if stage:
            path = stagelevel1bpath
        else:
            path = productionlevel1bpath
    if iterator:
        return iter([str(i) for i in path.glob(pattern + ext)])
    else:
        return [str(i) for i in list(path.glob(pattern + ext))]


def l1a_filenames(pattern=None, **kwargs):
    """Search for L1A filenames with patterns.

    <pattern> will be bracketed with '*', so needs to be correct in itself.
    For example "mode080-fuv" but not "mode080fuv".

    kwargs
    ======
        stage: False/True (gives production files if False). Default: True
        ext: Extension is by default .fits.gz Use this to search for other
             files like .txt
    """
    return get_filenames('l1a', pattern=pattern, **kwargs)


def l1b_filenames(pattern=None, **kwargs):
    """Search for L1B filenames with patterns.

    <pattern> will be bracketed with '*', so needs to be correct in itself.
    For example "mode080-fuv" but not "mode080fuv".

    kwargs
    ======
        stage: False/True (gives production files if False). Default: True
        ext: Extension is by default .fits.gz Use this to search for other
             files like .txt
    """
    return get_filenames('l1b', pattern=pattern, **kwargs)


def l1a_darks(darktype=''):
    searchpattern = '*' + darktype + 'dark*.fits.gz'
    print("Searching for", searchpattern)
    return l1a_filenames(searchpattern)


def image_stats(data):
    return pd.Series(data.ravel()).describe()


def get_l1a_filename_stats():
    fnames = l1a_filenames()
    iuvs_fnames = []
    exceptions = []
    for fname in fnames:
        try:
            iuvs_fnames.append(Filename(fname))
        except Exception:
            exceptions.append(fname)
            continue
    s = pd.Series(iuvs_fnames)
    df = pd.DataFrame()
    for item in 'phase cycle mode channel time level version revision'.split():
        df[item] = s.map(lambda x: getattr(x, item))
    df['channel'] = df.channel.astype('category')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df


def get_header_df(hdu, drop_comment=True):
    """Take a FITS HDU, convert to DataFrame.

    And on the way:
    fix it,drop COMMENT and KERNEL
    """
    hdu.verify('silentfix')
    header = hdu.header
    df = pd.DataFrame(header.values(),
                      index=header.keys())
    return df.drop('COMMENT KERNEL'.split()) if drop_comment else df


def save_to_hdf(df, fname, output_subdir=None):
    """Save temporary HDF file in output folder for later concatenation."""
    if os.path.isabs(fname):
        basename = os.path.basename(fname)
    else:
        basename = fname
    newfname = os.path.splitext(basename)[0] + '.h5'
    path = HOME / 'output'
    if output_subdir:
        path = path / output_subdir
    path = path / newfname
    df.to_hdf(str(path), 'df', format='t')
    return str(path)


class Filename:

    def __init__(self, fname):
        try:
            self.root = os.path.dirname(fname)
            self.basename = os.path.basename(fname)
        except AttributeError:
            # happens if fname is a PosixPath
            self.root = str(fname.parent)
            self.basename = fname.name
        tokens = self.basename.split('_')
        self.mission, self.instrument = tokens[:2]
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


class FitsBinTable:

    def __init__(self, hdu):
        self.header = hdu.header
        self.data = pd.DataFrame(hdu.data).T


class FitsFile:

    def __init__(self, fname):
        """fname needs to be absolute complete path. """
        if type(fname) == list:
            fname = fname[0]
        self.fname = fname
        self.hdulist = fits.open(fname)

    @property
    def ndims(self):
        return self.img_header['NAXIS']

    @property
    def n_integrations(self):
        if self.ndims == 2:
            return 1
        else:
            return self.img_header['NAXIS3']

    @property
    def scaling_factor(self):
        """Return factor to get DN/s.

        Because the binning returns just summed up values, one must
        also include the binning as a scaling factor, not only the
        integration time.
        """
        bin_scale = self.img_header['SPA_SIZE'] * self.img_header['SPE_SIZE']
        return bin_scale * self.int_time

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
        return self.Observation.field(18)[0]

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
        import datetime as dt
        cleaned = string[:-3]+'0'
        time = dt.datetime.strptime(cleaned, '%Y/%j %b %d %H:%M:%S.%f')
        return time

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

    def plot_some_spectrogram(self, inspec, title, ax=None, cmap=None,
                              cbar=True, log=False, showaxis=True,
                              min_=None, max_=None, set_extent=None,
                              draw_rectangle=True, vmin=None, vmax=None,
                              **kwargs):
        plot_hist = kwargs.pop('plot_hist', False)
        savename = kwargs.pop('savename', False)
        if log:
            spec = np.log10(inspec)
            vmax = 2.5
            vmin = -3.0
        else:
            spec = inspec
        if cmap is None:
            cmap = mycmap

        if ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self.plottitle, fontsize=16)
        try:
            waves = self.wavelengths[0]
        except IndexError:
            waves = self.wavelengths
        if set_extent:
            im = ax.imshow(spec,
                           cmap=cmap,
                           extent=(waves[0], waves[-1],
                                   len(spec), 0),
                           vmin=vmin, vmax=vmax,
                           **kwargs)
        else:
            im = ax.imshow(spec, cmap=cmap, vmin=vmin, vmax=vmax,
                           **kwargs)
        ax.set_title(title)
        if set_extent:
            ax.set_xlabel("Wavelength [nm]")
        else:
            ax.set_xlabel("Spectral bins")
        ax.set_ylabel('Spatial pixels')
        if not showaxis:
            ax.grid('off')
        if cbar:
            cb = plt.colorbar(im, ax=ax)
            if log:
                label = 'log(DN/s)'
            else:
                label = 'DN/s'
            cb.set_label(label, fontsize=14, rotation=0)
        self.current_ax = ax
        self.current_spec = spec

        # rectangle
        if draw_rectangle:
            ax.add_patch(get_rectangle(inspec))

        # inset histogram
        if plot_hist:
            in_axes = inset_axes(ax, width="20%", height="20%",
                                 loc=2)
            in_axes.hist(spec.ravel(), bins=20, normed=True)
            plt.setp(in_axes.get_xticklabels(), visible=False)
            plt.setp(in_axes.get_yticklabels(), visible=False)
            in_axes.grid('off')

        if savename:
            ax.get_figure().savefig(savename, dpi=100)
        return ax

    def plot_some_profile(self, data_attr, integration,
                          spatial=None, ax=None, scale=False,
                          log=False, spa_average=False, title=None,
                          **kwargs):
        plot_hist = kwargs.pop('plot_hist', False)
        savename = kwargs.pop('savename', False)
        spec = self.get_integration(data_attr, integration)
        if scale:
            spec = spec / self.scaling_factor
        if spatial is None:
            # if no spatial bin given, take the middle one
            spatial = self.spatial_size//2

        if title is not None:
            if not spa_average:
                title = ("Profile of {} at spatial: {}, integration {} of {}"
                         .format(data_attr, spatial, integration,
                                 self.n_integrations))
            else:
                title = ("Profile of {}, spatial mean. Integration {} of {}"
                         .format(data_attr, integration, self.n_integrations))
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


class L1AReader(FitsFile):

    """For Level1a"""

    works_with_dataframes = [
        'Integration',
        'Engineering',
        ]

    def __init__(self, fname, stage=True):

        # fix relative paths
        if not os.path.isabs(fname):
            if stage:
                fname = str(stagelevel1apath / fname)
            else:
                fname = str(productionlevel1apath / fname)

        # call super init
        super().__init__(fname)
        for hdu in self.hdulist[1:]:
            name = hdu.header['EXTNAME']
            setattr(self, name+'_header', hdu.header)
            if name in self.works_with_dataframes:
                setattr(self, name, pd.DataFrame(hdu.data).T)
            else:
                setattr(self, hdu.header['EXTNAME'], hdu.data)


class L1BReader(FitsFile):

    """For Level1B"""

    works_with_dataframes = [
        'DarkIntegration',
        'DarkEngineering',
        'background_light_source',
        'Integration',
        'Engineering']

    def __init__(self, fname, stage=True):

        # fix relative path
        if not os.path.isabs(fname):
            if stage:
                fname = str(stagelevel1bpath / fname)
            else:
                fname = str(productionlevel1bpath / fname)

        # call super init
        super().__init__(fname)
        for hdu in self.hdulist[1:]:
            name = hdu.header['EXTNAME']
            setattr(self, name+'_header', hdu.header)
            if name in self.works_with_dataframes:
                setattr(self, name, pd.DataFrame(hdu.data).T)
            else:
                setattr(self, hdu.header['EXTNAME'], hdu.data)
        self.darks_interpolated = self.background_dark

    @property
    def dark_det_temps(self):
        return self.Dark_Integration.field(7)

    @property
    def dark_case_temps(self):
        return self.Dark_Integration.field(8)

    @property
    def dark_times(self):
        utcs = self.Dark_Integration.field(2)
        times = []
        for utc in utcs:
            times.append(self.parse_UTC_string(utc))
        return times

    def parse_UTC_string(self, s):
        fmt = '%Y/%j %b %d %H:%M:%S%Z'
        s1, s2 = s.split('.')
        t = dt.datetime.strptime(s1+'UTC', fmt)
        tdelta = dt.timedelta(microseconds=int(s2[:-3]))
        return t+tdelta

    @property
    def n_darks(self):
        return self.detector_dark.shape[0]

    @property
    def raw_dn_s(self):
        return (self.detector_raw / self.scaling_factor) + 0.001

    @property
    def dark_dn_s(self):
        return (self.detector_dark / self.scaling_factor) + 0.001

    @property
    def dds_dn_s(self):
        return (self.detector_dark_subtracted / self.scaling_factor)+0.001

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
        im = ax.get_images()[0]
        cb = plt.colorbar(im, ax=axes.tolist())
        if imglog:
            label = 'log(DN/s)'
        else:
            label = 'DN/s'
        cb.set_label(label, fontsize=14, rotation=0)

        # fig.tight_layout()
        # fig.subplots_adjust(top=0.9, bottom=0.1, right=0.75)
        if save_token is not None:
            fname = "{}_{}.png".format(self.plotfname,
                                       save_token)
            fig.savefig(os.path.join(str(plotfolder), fname), dpi=150)

        return fig

    def find_scaling_window(self, spec):
        self.spa_slice, self.spe_slice = find_scaling_window(spec)
        return self.spa_slice, self.spe_slice

    def plot_raw_profile(self, integration=None, ax=None, log=None,
                         spatial=None, **kwargs):
        if integration is None:
            integration = -1
        return self.plot_some_profile('raw_dn_s', integration,
                                      ax=ax, log=log, spatial=spatial,
                                      **kwargs)

    def plot_dark_profile(self, integration, ax=None, log=None):
        return self.plot_some_profile('dark_dn_s', integration,
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


def get_rectangle(spectogram):
    spa_slice, spe_slice = find_scaling_window(spectogram)
    xy = spe_slice.start-0.5, spa_slice.start-0.5
    width = spe_slice.stop - spe_slice.start
    height = spa_slice.stop - spa_slice.start
    return Rectangle(xy, width, height, fill=False)


def find_scaling_window(to_filter, size=None):
    if size is None:
        x = max(to_filter.shape[0]//5, 2)
        y = max(to_filter.shape[1]//10, 1)
        size = (x, y)
    filtered = generic_filter(to_filter, np.median, size=size,
                              mode='constant', cval=to_filter.max()*100)
    min_spa, min_spe = np.unravel_index(filtered.argmin(), to_filter.shape)
    spa1 = min_spa - size[0]//2
    if spa1 < 0:
        spa1 = 0
    spa2 = spa1 + size[0]
    if spa2 > to_filter.shape[0]:
        spa1 = to_filter.shape[0] - size[0]
        spa2 = to_filter.shape[0]
    spe1 = min_spe - size[1]//2
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
    x = max(to_filter.shape[0]//10, 1)
    y = max(to_filter.shape[1]//10, 1)
    size = (x, y)
    print("Img shape:", to_filter.shape)
    print("Kernel size:", size)

    filtered = generic_filter(to_filter, np.std, size=size,
                              mode='constant', cval=to_filter.max()*100)
    min_spa, min_spe = np.unravel_index(filtered.argmin(), to_filter.shape)
    print("Minimum:", filtered.min())
    print("Minimum coords", min_spa, min_spe)

    spa1 = min_spa - size[0]//2
    if spa1 < 0:
        spa1 = 0
    spa2 = spa1 + size[0]
    if spa2 > to_filter.shape[0]:
        spa1 = to_filter.shape[0] - size[0]
        spa2 = to_filter.shape[0]
    print("Spatial:", spa1, spa2)

    spe1 = min_spe - size[1]//2
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


def some_l1a(pattern=''):
    try:
        fname = l1a_filenames(pattern, iterator=False)[0]
    except IndexError:
        print("No L1A files found.")
        return
    return L1AReader(fname)


def some_l1b(pattern=''):
    try:
        fname = l1b_filenames(pattern, iterator=False)[0]
    except IndexError:
        print("No L1B files found.")
        return
    return L1BReader(fname)
