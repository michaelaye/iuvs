import argparse
import datetime as dt
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spiceypy as spice
from astropy.io import fits
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

from . import io
from .spice import load_kernels

dataroot = Path.home() / 'to_keep'
dbpath = dataroot / 'HK_DB_stage.h5'
orbitfile = dataroot / 'orbit_numbers.hdf'

load_kernels()

sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_context('talk')
sns.set_palette(sns.color_palette('husl', 7))

instr_temps = ['FUV_CHIP_TEMP_C',
               'FUV_DET_TEMP_C',
               'FUV_INT_TEMP_C',
               'MUV_CHIP_TEMP_C',
               'MUV_DET_TEMP_C',
               'MUV_INT_TEMP_C']

iuvs_temps = ['IUVS_1_TEMP_C',
              'IUVS_2_TEMP_C',
              'IUVS_3_TEMP_C',
              'IUVS_4_TEMP_C']

htr_mot_temps = ['ZONE_1_HTR_TEMP_C', 'ZONE_2_HTR_TEMP_C',
                 'HV_POWR_TEMP_C', 'GRAT_MOT_TEMP_C', 'SCAN_MOT_TEMP_C']

op_htr_temps = ['OP_1_HTR_TEMP_C',
                'OP_2_HTR_TEMP_C',
                'POWER_BD1_TEMP_C',
                'POWER_BD2_TEMP_C']

temp_groups = [instr_temps, iuvs_temps, htr_mot_temps, op_htr_temps]


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
        print("HK database is up to date.")


def update_orbit_file(force=False):
    orbits = pd.read_hdf(str(orbitfile), 'df')
    today = dt.date.today()
    sevendays = dt.timedelta(days=7)
    # check if orbit file already has today's date
    if (today - orbits.index[-1].date()) < sevendays and not force:
        print("Orbit file is younger than a week. Won't update.\n"
              "(Force with '--force', as it takes 1-2 mins.)")
        return

    print("Updating orbit file.")
    print("Reading all L1A filenames... [1-2 mins]")
    l1afnames = io.get_filename_df('l1a', env='stage')

    def get_orbit_number(x):
        if x.startswith('orbit'):
            return float(x[5:])
        else:
            return np.nan

    l1afnames['orbit'] = l1afnames['cycle_orbit'].map(get_orbit_number)
    orbits = l1afnames.orbit.dropna().astype(int)
    orbits.to_hdf(str(orbitfile), 'df')
    print("Updated orbit file {}.".format(orbitfile))


def generate_plot(data, timeres, loffset, title):
    resampled = data.resample(timeres, base=0).dropna(how='all')

    orbits = pd.read_hdf(str(orbitfile), 'df')
    orbits = orbits.resample(timeres, loffset=loffset).dropna(how='all')

    resampled['orbit'] = orbits

    for savename, keys in zip(['detector_temps', 'iuvs_temps', 'heater_motor_temps',
                               'op_htr_temps'],
                              temp_groups):
        fig, ax = plt.subplots()
        try:
            resampled['orbit'].plot(legend=True, secondary_y=True, ax=ax)
            resampled[keys].plot(ax=ax, style='*', legend=True, markersize=12)
        except TypeError:
            continue
        ax.set_ylabel('Temperature [$C^\circ$]')
        ax.right_ax.set_ylabel('Orbit number')
        ax.set_title('{}, {} means'.format(title, timeres))
        ax.right_ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.set_ylim(resampled[keys].min().min()-1.5,
                    resampled[keys].max().max()+1.5)
        savepath = '{}_{}.png'.format(savename, title.replace(' ', '_'))
        fig.savefig(savepath, dpi=200)
        print('Created', savepath)
        plt.close(fig)


def plot_date(timestring, timeres='600s'):
    print("Reading data...")
    with pd.HDFStore(str(dbpath)) as store:
        c = store.select_column('df', 'index')
        where = c[pd.DatetimeIndex(c).date == pd.Timestamp(timestring).date()].index
        data = store.select('df', where=where)
    print("Generating plots...")
    generate_plot(data, timeres, 0, timestring)


def plot_times(t1, t2, timeres):
    print("Reading data...")
    data = pd.read_hdf(str(dbpath), 'df',
                       where="index>=t1 and index<t2")
    print("Generating plots...")
    generate_plot(data, timeres, 0, '{}-{}'.format(t1, t2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['status', 'update', 'plot'],
                        help="Choose one of the available sub-commands.")
    parser.add_argument('--date', type=str,
                        help="Provide the date string in YYYYMMDD format.")
    parser.add_argument('--force', action='store_true', default=False,
                        help='Enforce the activity if rejected by program.')
    parser.add_argument('--t1', type=str,
                        help="Start time for plot [UTC format].")
    parser.add_argument('--t2', type=str,
                        help="End time for plot [UTC format].")
    parser.add_argument('--timeres', type=str,
                        help='Time resolution for resampling in seconds.')
    args = parser.parse_args()
    if args.cmd == 'status':
        check_database_status()
    elif args.cmd == 'update':
        update_database()
        update_orbit_file(args.force)
    elif args.cmd == 'plot':
        if args.date:
            if not args.timeres:
                args.timeres = '600s'
            plot_date(args.date, args.timeres)
        elif len(args.t1) > 0 and len(args.t2) > 0:
            if not args.timeres:
                args.timeres = '1800s'
            plot_times(args.t1, args.t2, args.timeres)

if __name__ == '__main__':
    sys.exit(main())
