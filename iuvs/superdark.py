import os

import matplotlib.pyplot as plt
import numpy as np
from iuvs import scaling
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path

from . import io

root = Path('/home/klay6683/superdark')


def read_superdark():
    from scipy.io import idl
    d = idl.readsav(str(root / 'xmas_aurora_superdark_v1.sav'),
                    python_dict=True)
    superdark = d['xmas_aurora_superdark']
    return superdark


def get_sensitivity_curve(l1b):
    return l1b.detector_dark_subtracted[8, 2, :] / l1b.img[8, 2, :]


def fit_superdark_to_localdark(dark):
    superdark = read_superdark()
    x = superdark.ravel()
    y = dark.ravel()
    p = np.poly1d(np.polyfit(x, y, 1))
    return p(superdark)


def recalibrate_primary_muv(l1b):
    superdark = fit_superdark_to_localdark(l1b.detector_dark[1])
    primary_muv = l1b.detector_raw - superdark
    primary_muv_cal = primary_muv / get_sensitivity_curve(l1b)
    return primary_muv_cal


def get_compressed_spectrogram(img):
    return img[:, 2:5, :].mean(axis=1)


def do_my_subtract(l1b):
    mysubbed = []
    for i in range(21):
        darkfitter = scaling.DarkFitter(l1b, i, 1, dn_s=False)
        scaler = darkfitter.scalers[-1]
        fitted_dark = scaler.apply_fit(darkfitter.fulldark)
        mysubbed.append(darkfitter.fullraw - fitted_dark)
    mysubbed = np.array(mysubbed)
    mysubbed = mysubbed / get_sensitivity_curve(l1b)
    return mysubbed


def process_one_file(fname):
    print(os.path.basename(fname))
    l1b = io.L1BReader(fname, stage=False)
    current = get_compressed_spectrogram(l1b.img)

    recal = recalibrate_primary_muv(l1b)
    recal_comp = get_compressed_spectrogram(recal)

    mysubbed = do_my_subtract(l1b)
    mycompressed = get_compressed_spectrogram(mysubbed)
    return current, recal_comp, mycompressed


def get_periapse_fnames(orbit):
    while True:
        s = 'periapse-orbit{}-muv'.format(str(orbit).zfill(5))
        fnames = sorted(list(io.l1b_filenames(s, env='production')))
        if len(fnames) > 0:
            break
        orbit += 1

    print("Found periapse data at orbit:", orbit)
    return fnames[:-1], orbit


class Quicklooker(object):

    def __init__(self, orbit, fnames=None, offset=50):
        """Manage superdark comparisons.

        Parameters
        ----------
        orbit : int
            Orbit number for periapse data to process
        fnames : list of strings
            list of full path filenames to work on (Could be automated)
        offset : int
            number of spectral bins to show, counted from the upper end down.
        """
        if fnames is None:
            fnames, orbit = get_periapse_fnames(orbit)
        self.orbit = orbit
        self.str_orbit = str(orbit).zfill(5)
        self.fnames = fnames
        self.offset = offset
        self.wavelengths = io.L1BReader(fnames[0]).wavelengths

    def update_fnames(self, orbit):
        fnames, orbit = get_periapse_fnames(orbit)
        self.orbit = orbit
        self.fnames = fnames

    def process_data(self):
        current = []
        recal = []
        mysub = []

        for fname in self.fnames:
            for data, container in zip(process_one_file(fname), [current, recal, mysub]):
                container.append(data)

        self.current = np.array(current)
        self.recal = np.array(recal)
        self.mysub = np.array(mysub)

    def plot_profiles(self, offset=None):
        if offset is None:
            offset = self.offset
        fig, ax = plt.subplots(figsize=(8, 6))
        func = ax.semilogy
        func(self.wavelengths[0], self.current.mean(axis=(0, 1))[-offset:],
             label='current')
        func(self.wavelengths[0], self.recal.mean(axis=(0, 1))[-offset:],
             label='superdark_scaled')
        func(self.wavelengths[0], np.nanmean(self.mysub, axis=(0, 1))[-offset:],
             label='localdark_scaled')
        ax.legend(loc='best')
        ax.set_title(self.datatitle)
        ax.set_xlabel('Wavelengths [nm]')
        fig.savefig(str(root / 'profiles_orbit{}.pdf'.format(self.str_orbit)), dpi=120)

    @property
    def datatitle(self):
        return 'periapse-orbit{}-muv'.format(str(self.orbit).zfill(5))

    def make_plots(self):
        for t, data in zip('superdark_scaled(Sonal) current localdark_scaled(Michael)'.split(),
                           [self.recal, self.current, self.mysub]):
            fig = plt.figure(figsize=(8, 6))
            grid = ImageGrid(fig, 111,
                             share_all=True,
                             label_mode='1',
                             aspect=True,
                             nrows_ncols=(4, 3),
                             )

            for g, data in zip(grid, data):
                g.imshow(data[:, -self.offset:], cmap=plt.cm.Reds, interpolation='none')

            fig.text(0.1, 0.1, self.datatitle, fontsize=24)
            fig.text(0.1, 0.85, t, fontsize=24)
            fig.savefig('{}_{}.pdf'.format(self.datatitle, t), dpi=120)
            plt.close(fig)

    def do_all(self):
        self.process_data()
        self.make_plots()
        self.plot_profiles(0)
