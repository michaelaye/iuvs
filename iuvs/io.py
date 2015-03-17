import datetime as dt
from astropy.io import fits
import gzip
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import socket
from iuvs import scaling
import numpy as np


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

mycmap = 'YlGnBu_r'
plotfolder = HOME / 'Dropbox/src/iuvs/notebooks/plots'


def get_filenames(level, pattern=None, stage=True, ext='.fits.gz'):
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
    return [str(i) for i in list(path.glob(pattern + ext))]


def l1a_filenames(pattern=None, **kwargs):
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

    def image_stats(self):
        return image_stats(self.img)


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
    def scaling_factor(self):
        """Return factor to get DN/s.

        Because the binning returns just summed up values, one must
        also include the binning as a scaling factor.
        """
        bin_scale = self.img_header['SPA_SIZE'] * self.img_header['SPE_SIZE']
        return bin_scale * self.int_time

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
            if integration is not None:
                print("Data is only 2D, no integration index required.")
            spec = data
        return spec

    def plot_some_spectrogram(self, spec, title, ax=None, cmap=None,
                              cbar=True, log=False, showaxis=True,
                              min_=None, max_=None, set_extent=None,
                              **kwargs):
        if log:
            spec = np.log10(spec)
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
            im = ax.imshow(spec, cmap=cmap, extent=(waves[0], waves[-1],
                                                    len(spec), 0),
                           **kwargs)
        else:
            im = ax.imshow(spec, cmap=cmap, **kwargs)
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
        return ax

    def plot_some_profile(self, data_attr, integration,
                          spatial=None, ax=None, scale=False,
                          log=False, **kwargs):
        spec = self.get_integration(data_attr, integration)
        if scale:
            spec = spec / self.scaling_factor
        if spatial is None:
            # if no spatial bin given, take the middle one
            spatial = self.img.shape[1]//2

        title = ("Profile of {} at spatial: {}, integration {} of {}"
                 .format(data_attr, spatial, integration,
                         'length of array to fill in'))


        if ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self.plottitle, fontsize=16)
        if log:
            func = ax.semilogy
        else:
            func = ax.plot

        func(self.wavelengths[spatial], spec[spatial], **kwargs)

        ax.set_xlim((self.wavelengths[spatial][0],
                     self.wavelengths[spatial][-1]))
        ax.set_title(title)
        ax.set_xlabel("Wavelength [nm]")
        if log:
            ax.set_ylabel("log(DN/s)")
        else:
            ax.set_ylabel('DN/s')
        return ax

    def plot_img_spectrogram(self,
                             integration=None, ax=None,
                             cmap=None, cbar=True, log=True):
        spec = self.get_integration('img', integration)
        title = ("Primary spectrogram, integration {} out of {}"
                 .format(integration, 'length of array to fill in'))
        return self.plot_some_spectrogram(spec, title,
                                          ax, cmap, cbar, log)


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
    def scaled_raw(self):
        return self.detector_raw / self.scaling_factor

    @property
    def scaled_dark(self):
        return self.detector_dark / self.scaling_factor

    def plot_raw_spectrogram(self, integration=None, ax=None,
                             cmap=None, cbar=True, log=False,
                             **kwargs):
        spec = self.get_integration('scaled_raw', integration)
        title = ("Raw light spectrogram, integration {} out of {}"
                 .format(integration, 'length of array'))
        return self.plot_some_spectrogram(spec, title, ax,
                                          cmap, cbar, log, **kwargs)

    def plot_dark_spectogram(self, integration=None, ax=None,
                             cmap=None, cbar=True, log=False,
                             **kwargs):
        dark = self.get_integration('scaled_dark', integration)
        title = ("Dark spectogram, integration {} out of {}"
                 .format(integration, 'length of array'))
        return self.plot_some_spectrogram(dark, title, ax,
                                          cmap, cbar, log, **kwargs)

    def plot_raw_overview(self, integration=None, log=False,
                          save_token=None):
        "Plot overview of spectrogram and profile at index `integration`."
        fig, axes = plt.subplots(nrows=2, sharex=False)
        fig.subplots_adjust(top=0.9, bottom=0.1)
        fig.suptitle(self.plottitle, fontsize=16)
        ax = self.plot_raw_spectrogram(integration, ax=axes[0],
                                       cbar=False, log=log,
                                       set_extent=False)
        self.plot_some_profile('scaled_raw', integration,
                               ax=axes[1], log=log)
        im = ax.get_images()[0]
        cb = plt.colorbar(im, ax=axes.tolist())
        if log:
            label = 'log(DN/s)'
        else:
            label = 'DN/s'
        cb.set_label(label, fontsize=14, rotation=0)
        if save_token is not None:
            fname = "{}_{}.png".format(self.plotfname,
                                       save_token)
            fig.savefig(os.path.join(str(plotfolder), fname), dpi=150)

    def plot_raw_profile(self, integration, ax=None):
        return self.plot_some_profile('scaled_raw', integration,
                                      ax=ax)

    def plot_dark_profile(self, integration, ax=None):
        return self.plot_some_profile('scaled_dark', integration,
                                      ax=ax)

    def get_light_and_dark(self, integration):
        light = self.get_integration('scaled_raw', integration)
        dark = self.get_integration('scaled_dark', integration)
        return light, dark


class KindHeader(fits.Header):

    """FITS header with the 'kind' card."""

    def __init__(self, kind='original dark'):
        super().__init__()
        self.set('kind', kind, comment='The kind of image')


class PrimHeader(KindHeader):

    """FITS primary header with a name card."""

    def __init__(self):
        super().__init__()
        self.set('name', 'dark1')


class FittedHeader(KindHeader):

    """FITS header with a kind and a rank card."""

    def __init__(self, rank):
        super().__init__('fitted dark')
        comment = "The degree of polynom used for the scaling."
        self.set('rank', rank, comment=comment)
        self.add_comment("The rank is '-1' for 'Additive' fitting, '0' is "
                         "for 'Multiplicative' fitting without additive "
                         "offset. For all ranks larger than 0 it is "
                         "equivalent to the degree of the polynomial fit.")


class DarkWriter:

    """Manages the creation of FITS file for dark analysis results.
    """

    def __init__(self, outfname, dark1, dark2, clobber=False):
        """Initialize DarkWriter.

        Parameters
        ==========
            outfname: Filename of fits file to write
            dark1, dark2: numpy.array of dark images
            clobber: Boolean to control if to overwrite existing fits file
                Default: False
        """
        self.outfname = outfname
        self.clobber = clobber
        header = PrimHeader()
        hdu = fits.PrimaryHDU(dark1, header=header)
        hdulist = fits.HDUList([hdu])
        header = KindHeader()
        hdu = fits.ImageHDU(dark2, header=header, name='dark2')
        hdulist.append(hdu)
        self.hdulist = hdulist

    def append_polyfitted(self, scaler):
        """Append a polynomial fitted dark to the fits file.

        Parameters
        ==========

            polyscaler: type of scaling.Polyscaler

        """
        if type(scaler) == scaling.PolyScaler:
            rank = scaler.rank
        elif type(scaler) == scaling.MultScaler:
            rank = 0
        elif type(scaler) == scaling.AddScaler:
            rank = -1
        else:
            rank = -99
        # create fits header with rank and kind card
        header = FittedHeader(rank)
        # add coefficienst card
        header['coeffs'] = str(list(scaler.p))
        header.add_comment('The coefficients are listed highest rank first.')
        # add stddev card
        header.set('stddev', scaler.residual.std(),
                   'Standard deviation of residual')
        hdu = fits.ImageHDU(scaler.residual, header=header,
                            name='rank{}'.format(rank))
        self.hdulist.append(hdu)

    def write(self):
        self.hdulist.writeto(self.outfname, clobber=self.clobber)


def some_l1a():
    try:
        fname = l1a_filenames()[0]
    except IndexError:
        print("No L1A files found.")
        return
    return L1AReader(fname)


def some_l1b():
    try:
        fname = l1b_filenames()[0]
    except IndexError:
        print("No L1B files found.")
        return
    return L1BReader(fname)
