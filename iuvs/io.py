import datetime as dt
from astropy.io import fits
import gzip
import matplotlib.pyplot as plt


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


class IUVSReader:
    """For Level1a"""
    def __init__(self, fname):
        infile = gzip.open(fname, 'rb')
        self.hdulist = fits.open(infile)

    @property
    def image_header(self):
        imgdata = self.hdulist[0]
        return imgdata.header

    @property
    def image(self):
        return self.hdulist[0].data

    def parse_capture(self, string):
        import datetime as dt
        cleaned = string[:-3]+'0'
        time = dt.datetime.strptime(cleaned, '%Y/%j %b %d %H:%M:%S.%f')
        return time

    def plot_img_data(self):
        time = parse_capture(self.image_header['CAPTURE'])
        fig, ax = plt.subplots()#figsize=(8, 6))
        ax.imshow(self.image)
        ax.set_title("{xuv}, {time}".format(time=time.isoformat(),
                                            xuv=self.header['XUV']))
        return ax



