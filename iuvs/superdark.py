import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from iuvs import scaling
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path

from . import io

root = Path('/home/klay6683/superdark')

sns.set_style('white')

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
    return primary_muv


def use_both(l1b, mydarks):
    superdark = fit_superdark_to_localdark(l1b.detector_dark[1])
    repeated = np.repeat(superdark[np.newaxis,:], repeats=l1b.n_integrations, axis=0)
    mediandark = np.median([mydarks, repeated], axis=0)
    primary = l1b.detector_raw - mediandark
    return primary


def get_compressed_spectrogram(img):
    return img[:, 2:5, :].mean(axis=1)


def do_my_subtract(l1b):
    mysubbed = []
    for i in range(21):
        darkfitter = scaling.DarkFitter(l1b, i, 1, dn_s=False)
        scaler = darkfitter.scalers[-1]
        fitted_dark = scaler.apply_fit(darkfitter.fulldark)
        mysubbed.append(fitted_dark)
    mysubbed = np.array(mysubbed)
    return mysubbed


def process_one_file(fname):
    print(os.path.basename(fname))
    l1b = io.L1BReader(fname, stage=False)
    current = get_compressed_spectrogram(l1b.detector_dark_subtracted)

    recal = recalibrate_primary_muv(l1b)
    recal_comp = get_compressed_spectrogram(recal)

    mydarks = do_my_subtract(l1b)
    mysubbed = l1b.detector_raw - mydarks
    mycompressed = get_compressed_spectrogram(mysubbed)

    usedboth = use_both(l1b, mydarks)
    usedbothcompressed = get_compressed_spectrogram(usedboth)
    return current, recal_comp, mycompressed, usedbothcompressed


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
        both = []
        for fname in self.fnames:
            for data, container in zip(process_one_file(fname), [current, recal, mysub, both]):
                container.append(data)

        self.current = np.array(current)
        self.recal = np.array(recal)
        self.mysub = np.array(mysub)
        self.both = np.array(both)


    def plot_profiles(self, offset=None):
        if offset is None:
            offset = self.offset
        fig, ax = plt.subplots(figsize=(8, 6))
        func = ax.plot
        for data, l in zip([self.current, self.recal, self.mysub, self.both],
                           ['current', 'superdark_scaled', 'localdark_scaled',
                            'mean of both darks']):
            func(self.wavelengths[0], data.mean(axis=(0, 1))[-offset:],
                 label=l)
        ax.legend(loc='best')
        ax.set_title(self.datatitle)
        ax.set_xlabel('Wavelengths [nm]')
        fig.savefig(str(root / 'profiles_orbit{}.png'.format(self.str_orbit)), dpi=200)


    def plot_profile_ratios(self, offset=None):
        if offset is None:
            offset = self.offset
        fig, ax = plt.subplots(figsize=(8, 6))
        func = ax.plot
        basis = self.current.mean(axis=(0,1))
        for data, l, c in zip([self.recal, self.mysub, self.both],
                              ['superdark_scaled', 'localdark_scaled',
                               'mean of both darks'],
                              sns.color_palette()[1:]):
            ratio = data.mean(axis=(0,1)) / basis
            func(self.wavelengths[0], ratio[-offset:],
                 label=l, color=c)
        ax.legend(loc='best')
        ax.set_title(self.datatitle)
        ax.set_xlabel('Wavelengths [nm]')
        fig.savefig(str(root / 'profile_ratios_orbit{}.png'.format(self.str_orbit)), dpi=200)

    @property
    def datatitle(self):
        return 'periapse-orbit{}-muv'.format(str(self.orbit).zfill(5))

    def make_plots(self):
        for t, data in zip(['superdark_scaled(Sonal)', 'current',
                            'localdark_scaled(Michael)', 'mean_darks'],
                           [self.recal, self.current, self.mysub, self.both]):
            fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True,
                                     figsize=(8, 6))
            # grid = ImageGrid(fig, 111,
            #                  share_all=True,
            #                  label_mode='1',
            #                  aspect=False,
            #                  nrows_ncols=(4, 3),
            #                  )

            for ax, data in zip(axes.ravel(), data):
                ax.imshow(data[:, -self.offset:], cmap='viridis', interpolation='none',
                          aspect='auto')

            plt.subplots_adjust(wspace=0, hspace=0)
            fig.text(0.1, 0.05, self.datatitle, fontsize=20)
            # fig.text(0.1, 0.85, t, fontsize=24)
            fig.suptitle(t, fontsize=20)
            fig.savefig(str(root / '{}_{}.png'.format(self.datatitle, t)), dpi=200)
            plt.close(fig)

    def do_all(self):
        self.process_data()
        self.make_plots()
        self.plot_profiles()
        self.plot_profile_ratios()
