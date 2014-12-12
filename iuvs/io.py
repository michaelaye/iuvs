import datetime as dt
from astropy.io import fits
import gzip
import matplotlib.pyplot as plt
import pandas as pd
import os


class IUVS_Filename:
    def __init__(self, fname):
        tokens = fname.split('_')
        self.mission, self.instrument = tokens[:2]
        self.level = tokens[2]
        self.phase = tokens[3]
        self.timestr, self.version = tokens[4:6]
        self.revision = tokens[6].split('.')[0]
        phasetokens = self.phase.split('-')
        self.phase, self.cycle, self.mode, self.channel = phasetokens
        self.time = dt.datetime.strptime(self.timestr,
                                         '%Y%m%dT%H%M%S')


class FitsBinTable:
    def __init__(self, hdu):
        self.header = hdu.header
        self.data = pd.DataFrame(hdu.data).T


class IUVSReader:
    """For Level1a"""
    def __init__(self, fname):
        infile = gzip.open(fname, 'rb')
        self.fname = os.path.basename(fname)
        self.hdulist = fits.open(infile)
        self.integration = FitsBinTable(self.hdulist[1])
        self.engineering = FitsBinTable(self.hdulist[2])
        self.binning = self.hdulist[3]
        self.pixelgeo = self.hdulist[4]
        self.spacecraftgeo = self.hdulist[5]
        self.observation = self.hdulist[6]

    @property
    def img_header(self):
        imgdata = self.hdulist[0]
        return imgdata.header

    @property
    def img(self):
        return self.hdulist[0].data

    @property
    def capture(self):
        string = self.img_header['CAPTURE']
        import datetime as dt
        cleaned = string[:-3]+'0'
        time = dt.datetime.strptime(cleaned, '%Y/%j %b %d %H:%M:%S.%f')
        return time

    def plot_img_data(self):
        time = self.capture
        fig, ax = plt.subplots()  # figsize=(8, 6))
        ax.imshow(self.img)
        ax.set_title("{xuv}, {time}".format(time=time.isoformat(),
                                            xuv=self.img_header['XUV']))
        return ax
