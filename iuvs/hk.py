import argparse
import sys

import numpy as np
import pandas as pd
import spiceypy as spice
from astropy.io import fits
from pathlib import Path

from . import io
from .spice import load_kernels

dbpath = Path('/home/klay6683/to_keep/HK_DB_stage.h5')

load_kernels()


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


def process_hk_fname(fname):
    hkfile = HKReader(str(fname))
    hkfile.temp_df.to_hdf(str(dbpath), 'df', mode='a', format='table',
                          append=True)


def check_database_status():
    currentdiff = dbpath.parent / 'current_datediff.hdf'
    print("Give me 10 seconds to read the data...")
    index = pd.read_hdf(str(dbpath), columns='index')
    dates_used = pd.DatetimeIndex(np.unique(index.index.date))
    hkfname_df = io.get_filename_df('hk', env='stage')
    diff = hkfname_df.index.difference(dates_used)
    if len(diff) > 0:
        print("Found these new dates of HK files that are not yet in the data-base:\n{}"
              .format(pd.Series(diff)))
    else:
        print("The HK database file {} is up to date.".format(str(dbpath)))
        currentdiff.unlink()
    diff.to_series().to_hdf(str(currentdiff), 'df')


def update_database():
    diff = pd.DatetimeIndex(pd.read_hdf(str(dbpath.parent / 'current_datediff.hdf')))
    if len(diff) > 0:
        hkfname_df = io.get_filename_df('hk', env='stage')
        for index, row in hkfname_df.loc[diff].iterrows():
            print("Adding {}".format(index.date()))
            process_hk_fname(str(row.p))
    else:
        print("Nothing to update.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['status', 'update'],
                        help="Choose one of the available sub-commands.")

    args = parser.parse_args()
    if args.cmd == 'status':
        check_database_status()
    elif args.cmd == 'update':
        update_database()
