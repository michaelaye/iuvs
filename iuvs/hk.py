import numpy as np
from astropy.io import fits

import SpiceyPy as spice

from .spice import load_kernels


def format_times(sec, subsec):
    return "{:010d}.{:05d}".format(int(sec), int(subsec))


def calc_utc_from_sclk(coarse, fine):
    load_kernels()  # load SPICE kernels
    timestamp_ = coarse
    ss = fine
    mavenid, _ = spice.bodn2c('MAVEN')
    timestamp = timestamp_ + ss / 65536  # timestamp_ and ss are already float
    sec = np.uint64(timestamp)
    subsec = np.uint64((timestamp - sec) * 65536)
    sclkch = np.vectorize(format_times)(sec, subsec)
    sclkdp = np.vectorize(spice.scencd)(mavenid, sclkch)
    et = np.vectorize(spice.sct2e)(mavenid, sclkdp)
    return np.vectorize(spice.et2utc)(et, 'ISOC', 50, 100)


class HKReader(object):
    def __init__(self, fname):
        self.hdulist = fits.open(fname)
        for hdu in self.hdulist[1:]:
            name = hdu.header['EXTNAME']
            setattr(self, name + '_header', hdu.header)
            setattr(self, name, hdu.data)

        temp_cols = [value for value in self.AnalogConv_header.values()
                     if 'temp' in str(value).lower()]
        self.temp_cols = temp_cols

    def calc_utc_from_table(self, table):
        utc = calc_utc_from_sclk(table['SC_CLK_COARSE'], table['SC_CLK_FINE'])
