import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SpiceyPy as spice
from astropy.io import fits


def format_times(sec, subsec):
    return "{:010d}.{:05d}".format(int(sec), int(subsec))


def calc_utc_from_sclk(coarse, fine):
    """Requires to have SPICE kernels loaded (use load_kernels()).

    I'm not doing the loading of SPICE kernels in here so that I can
    run this function in a loop without using up kernel space.
    The caller should load and unload kernels as required.
    """
    # load_kernels()  # load SPICE kernels
    timestamp_ = coarse
    ss = fine
    mavenid, _ = spice.bodn2c('MAVEN')
    timestamp = timestamp_ + ss / 65536  # timestamp_ and ss are already float
    sec = np.uint64(timestamp)
    subsec = np.uint64((timestamp - sec) * 65536)
    sclkch = np.vectorize(format_times)(sec, subsec)
    sclkdp = np.vectorize(spice.scencd)(mavenid, sclkch)
    et = np.vectorize(spice.sct2e)(mavenid, sclkdp)
    utc = np.vectorize(spice.et2utc)(et, 'ISOC', 50, 100)
    return pd.to_datetime(utc)


class HKReader(object):

    def __init__(self, fname):
        self.hdulist = fits.open(fname)
        for hdu in self.hdulist[1:]:
            name = hdu.header['EXTNAME']
            setattr(self, name + '_header', hdu.header)
            setattr(self, name, hdu.data)

        temp_cols = [value for value in getattr(self, 'AnalogConv_header').values()
                     if 'temp' in str(value).lower()]
        self.temp_cols = temp_cols
        self.get_temp_table()
        self.fpath = Path(fname)

    def get_temp_table(self):
        table = getattr(self, 'AnalogConv')  # this table is set during init.
        utc = calc_utc_from_sclk(table['SC_CLK_COARSE'],
                                 table['SC_CLK_FINE'])

        d = {}
        for col in self.temp_cols:
            data = table[col]
            sys_byteorder = ('>', '<')[sys.byteorder == 'little']
            if data.dtype.byteorder not in ('=', sys_byteorder):
                d[col] = data.byteswap().newbyteorder(sys_byteorder)
            else:
                d[col] = data
        self.temp_df = pd.DataFrame(d, index=utc)

    def save_temps_to_hdf(self, fname):
        self.temp_df.to_hdf(fname, 'df')
