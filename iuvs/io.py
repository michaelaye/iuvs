import datetime as dt
from astropy.io import fits
import gzip
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import socket

host = socket.gethostname()
home = Path(os.environ['HOME'])

if host.startswith('maven-iuvs-itf'):
    products = Path('/maven_iuvs/stage/products')
else:
    products = home / 'data' / 'iuvs'

level1apath = products / 'level1a'
level1bpath = products / 'level1b'


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

    def image_stats(self):
        return image_stats(self.img)


class FitsBinTable:

    def __init__(self, hdu):
        self.header = hdu.header
        self.data = pd.DataFrame(hdu.data).T


class IUVS_Data:

    @property
    def img_header(self):
        imgdata = self.hdulist[0]
        return imgdata.header

    @property
    def img(self):
        return self.hdulist[0].data

    def plot_img_data(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()  # figsize=(8, 6))
        ax.imshow(self.img)
        ax.set_title("{channel}, {phase}, {int}"
                     .format(channel=self.fname.channel,
                             phase=self.fname.phase,
                             int=self.img_header['INT_TIME']))
        return ax

    @property
    def capture(self):
        string = self.img_header['CAPTURE']
        import datetime as dt
        cleaned = string[:-3]+'0'
        time = dt.datetime.strptime(cleaned, '%Y/%j %b %d %H:%M:%S.%f')
        return time



class L1AReader(IUVS_Data):

    """For Level1a"""

    def __init__(self, fname):
        infile = gzip.open(fname, 'rb')
        self.fname = fname
        self.hdulist = fits.open(infile)
        self.integration = FitsBinTable(self.hdulist[1])
        self.engineering = FitsBinTable(self.hdulist[2])
        self.binning = self.hdulist[3]
        self.pixelgeo = self.hdulist[4]
        self.spacecraftgeo = self.hdulist[5]
        self.observation = self.hdulist[6]


class L1BReader(IUVS_Data):

    """For Level1B"""

    works_with_dataframes = [
        'DarkIntegration',
        'DarkEngineering',
        'background_light_source',
        'Integration',
        'Engineering']

    def __init__(self, fname):
        infile = gzip.open(fname, 'rb')
        self.fname = os.path.basename(fname)
        self.hdulist = fits.open(infile)
        for hdu in self.hdulist[1:]:
            name = hdu.header['EXTNAME']
            setattr(self, name+'_header', hdu.header)
            if name in self.works_with_dataframes:
                setattr(self, name, pd.DataFrame(hdu.data).T)
            else:
                setattr(self, hdu.header['EXTNAME'], hdu.data)
        self.darks_interpolated = self.background_dark

    def plot_img_data(self, ax=None):
        time = self.capture
        if ax is None:
            fig, ax = plt.subplots()  # figsize=(8, 6))
        ax.imshow(self.img)
        ax.set_title("{xuv}, {time}".format(time=time.isoformat(),
                                            xuv=self.img_header['XUV']))
        return ax


def l1a_filenames():
    return [str(i) for i in level1apath.glob('*.fits.gz')]


def l1a_darks(darktype=''):
    searchpattern = '*' + darktype + 'dark*.fits.gz'
    print("Searching for", searchpattern)
    return level1apath.glob('*'+darktype+'dark*.fits.gz')


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
