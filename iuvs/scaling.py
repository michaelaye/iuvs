from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

from . import io


def get_corner(data, corner, size):
    if corner == 'ul':
        return data[:size, :size]
    elif corner == 'ur':
        return data[:size, -size:]
    elif corner == 'll':
        return data[-size:, :size]
    elif corner == 'lr':
        return data[-size:, -size:]


def get_ul(data, size=10):
    return get_corner(data, 'ul', size)


def get_ur(data, size=10):
    return get_corner(data, 'ur', size)


def get_ll(data, size=10):
    return get_corner(data, 'll', size)


def get_lr(data, size=10):
    return get_corner(data, 'lr', size)


class DarkScaler(object):

    """Managing the general attributes for scaling darks around.

    This is the base class that is inherited by the other scalers.
    """

    name = "to be overwritten by daughter class"

    def __init__(self, data_in, data_out, alternative=False):
        self.data_in = data_in
        self.data_out = data_out
        self.alternative = alternative

    def do_fit(self):
        self.p, self.pcov = curve_fit(self.model,
                                      self.data_in.ravel(),
                                      self.data_out.ravel())
        if self.alternative:
            self.p = np.array([self.expected()])
        self.perr = np.sqrt(np.diag(self.pcov))

    def model(self):
        "Overwrite in daughter class!"
        pass

    @property
    def scaled(self):
        return self.model(self.data_in, self.p)

    @property
    def residual(self):
        return self.data_out - self.scaled

    @property
    def fractional(self):
        return self.residual / self.data_out

    def apply_fit(self, in_):
        """Apply the currently active fit parameters to provided data.

        After the fit parameters have been determined for the data from
        initialization, one can apply the determined fit parameter with the
        model of this object to externally provided data, for example the
        total of an image, if fit was done on subframe.
        """
        return self.model(in_, self.p)

    @property
    def p_formatted(self):
        return list(reversed(["{:.3f}".format(i) for i in self.p]))

    @property
    def p_dict(self):
        d = dict()
        d[self.name] = float(self.p)
        return d

    @property
    def residual_mean_dict(self):
        d = dict()
        d[self.name] = float(self.residual.mean())
        return d

    @property
    def residual_std_dict(self):
        d = dict()
        d[self.name] = float(self.residual.std())
        return d


class AddScaler(DarkScaler):

    """Additive Scaling model."""
    name = 'AddScaler'
    rank = -1

    def model(self, x, a):
        return a + x

    def expected(self):
        return self.data_out.mean() - self.data_in.mean()


class MultScaler(DarkScaler):

    """Pure Multiplicative scaling model"""

    name = 'MultScaler'
    rank = 0

    def model(self, x, a):
        return a * x

    def expected(self):
        return self.data_out.mean() / self.data_in.mean()


class PolyScaler(DarkScaler):

    """Manage polynomial fits. Default rank is 2."""

    def __init__(self, data_in, data_out, rank=2):
        super(PolyScaler, self).__init__(data_in, data_out)
        self.rank = rank
        self.name = 'Poly' + str(self.rank)

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value

    @property
    def poly(self):
        return np.poly1d(self.p)

    def model(self, x=None, p=None):
        if x is None:
            x = self.data_in
        if p is None:
            p = self.p
        poly = np.poly1d(p)
        return poly(x)

    def do_fit(self):
        self.p = np.polyfit(self.data_in.ravel(),
                            self.data_out.ravel(),
                            self.rank)

    @property
    def perr(self):
        print("Not defined yet for PolyFitter.")
        return

    @property
    def p_dict(self):
        d = dict()
        for i, item in enumerate(self.p[::-1]):
            d['poly{}_{}'.format(self.rank, i)] = item
        return d


class PolyScaler1(PolyScaler):

    def __init__(self, *args):
        super(PolyScaler1, self).__init__(*args, rank=1)


class PolyScaler2(PolyScaler):

    def __init__(self, *args):
        super(PolyScaler2, self).__init__(*args, rank=2)


class PolyScaler3(PolyScaler):

    def __init__(self, *args):
        super(PolyScaler3, self).__init__(*args, rank=3)


class PolyScalerManager(object):

    def __init__(self, data_in, data_out, rankstart, rankend):
        self.rankstart = rankstart
        self.rankend = rankend
        self.data_in = data_in
        self.data_out = data_out
        fractionals = []
        coeffs = []
        polynoms = []
        scalers = []
        for rank in range(rankstart, rankend + 1):
            scaler = PolyScaler(data_in, data_out, rank)
            scaler.do_fit()
            scalers.append(scaler)
            polynoms.append(scaler.poly)
            coeffs.append(scaler.p)
            fractionals.append(scaler.fractional.mean())
        self.fractionals = fractionals
        self.coeffs = coeffs
        self.polynoms = polynoms
        self.scalers = scalers

    def plot_fractionals(self):
        plt.plot(range(self.rankstart, self.rankend + 1),
                 self.fractionals)
        plt.xlabel('Polynomial rank')
        plt.xlim(0, self.rankend + 2)
        plt.title('Mean values of fractional residual'
                  ' over polynomial rank')
        plt.ylabel('Mean value of fracional residual')


class DarkFitter(object):
    Scalers = [AddScaler, MultScaler, PolyScaler1,
               PolyScaler2]

    def __init__(self, fname_or_l1b, raw_integration, dark_integration,
                 spa_slice=None, spe_slice=None):
        if hasattr(fname_or_l1b, 'fname'):  # only true if it is type(L1BReader)
            self.l1b = fname_or_l1b
            l1b = fname_or_l1b
            self.fname = self.l1b.fname
        else:
            self.fname = fname_or_l1b
            l1b = io.L1BReader(fname_or_l1b)
            self.l1b = l1b
        self.raw_integration = raw_integration
        self.dark_integration = dark_integration

        fullraw = l1b.get_integration('raw_dn_s', raw_integration)
        fulldark = l1b.get_integration('dark_dn_s', dark_integration)
        self.fullraw = fullraw
        self.fulldark = fulldark

        # get the calm scaling window
        if spa_slice is None:
            self.define_scaling_window()
        else:
            self.spa_slice = spa_slice
            self.spe_slice = spe_slice
            self.raw_subframe = self.fullraw[self.spa_slice, self.spe_slice]
            self.dark_subframe = self.fulldark[self.spa_slice, self.spe_slice]

        # calculate current residual
        currentsub = l1b.get_integration('dds_dn_s', raw_integration)
        self.currentresidual = currentsub[self.spa_slice, self.spe_slice]
        # this container will keep all scaler objects.
        self.scalers = []
        self.p_dicts = {}
        self.residual_mean = {}
        self.residual_std = {}
        for Scaler in self.Scalers:
            if Scaler == MultScaler:
                scaler = Scaler(self.dark_subframe,
                                self.raw_subframe,
                                alternative=True)
            else:
                scaler = Scaler(self.dark_subframe,
                                self.raw_subframe)

            scaler.do_fit()
            self.scalers.append(scaler)
            self.p_dicts.update(scaler.p_dict)
            self.residual_mean.update(scaler.residual_mean_dict)
            self.residual_std.update(scaler.residual_std_dict)

    def define_scaling_window(self):
        self.spa_slice, self.spe_slice = self.l1b.find_scaling_window(self.fullraw)
        self.raw_subframe = self.fullraw[self.spa_slice, self.spe_slice]
        self.dark_subframe = self.fulldark[self.spa_slice, self.spe_slice]

    def get_title_data(self, data):
        subdata = data[self.spa_slice, self.spe_slice]
        return subdata.mean(), subdata.std()

    def plot_profiles(self, save_token=None):
        fig, axes = plt.subplots(nrows=len(self.scalers) + 3, sharex=True)

        # raw profile
        self.l1b.plot_raw_profile(self.raw_integration, ax=axes[0], log=False)

        # dark profile
        self.l1b.plot_dark_profile(self.dark_integration, ax=axes[1], log=False)

        # background subtracted
        self.l1b.plot_dds_profile(self.raw_integration,
                                  ax=axes[2], log=False)
        dds = self.l1b.dds_dn_s[self.raw_integration]
        title = axes[2].get_title() + ', Mean:{:.1f}, STD:{:.3f}'\
                                      .format(*self.get_title_data(dds))
        axes[2].set_title('')
        axes[2].text(.5, .9, title, horizontalalignment='center',
                     transform=axes[2].transAxes, fontsize=10)

        # scaled fits
        spatial = self.l1b.spatial_size // 2
        for scaler, ax in zip(self.scalers, axes[3:]):
            fitted_dark = scaler.apply_fit(self.fulldark)
            sub = self.fullraw - fitted_dark
            mean, std = self.get_title_data(sub)
            ax.plot(self.l1b.wavelengths[self.raw_integration], sub[spatial])
            ax.set_ylabel('DN/s')

            ax.set_xlim((self.l1b.wavelengths[self.raw_integration][0],
                         self.l1b.wavelengths[self.raw_integration][-1]))
            parameters = scaler.p_formatted
            title = ("{}, {}. Mean:{:.1f}, STD:{:.3f}".format(scaler.name,
                                                              parameters,
                                                              mean, std))
            ax.text(.5, .9, title, horizontalalignment='center',
                    transform=ax.transAxes, fontsize=10)

        # min_, max_ = get_min_max(l1b, integration, spa_slice, spe_slice)
        if int(self.l1b.int_time) == 1:
            min_, max_ = (-5, 15)
        else:
            min_, max_ = (-1, 3)

        for ax in axes:
            ax.set_xlabel('')
            ax.locator_params(axis='y', nbins=6)
            ax.set_ylim(min_, max_)

        axes[-1].set_xlabel("Wavelength [nm]")
        fig.suptitle("{}\nProfiles at middle spatial, Window: [{}:{}, {}:{}]\n"
                     .format(self.l1b.plottitle,
                             self.spa_slice.start,
                             self.spa_slice.stop,
                             self.spe_slice.start,
                             self.spe_slice.stop),
                     fontsize=11)
        fig.subplots_adjust(top=0.90, bottom=0.07)
        if save_token is not None:
            fname = io.HOME / 'plots' / (self.l1b.plotfname + '_2_' + save_token + '.png')
            fig.savefig(str(fname), dpi=150)


def poly_fitting(l1b, integration, spatialslice, spectralslice):
    fullraw, fulldark = l1b.get_light_and_dark(integration)
    light = fullraw[spatialslice, spectralslice]
    dark = fulldark[spatialslice, spectralslice]
    scaler = PolyScaler(dark, light)
    scaler.do_fit()
    return scaler.apply_fit(fulldark)


def get_min_max(l1b, integration, spa_slice, spe_slice):
    fitted_dark = poly_fitting(l1b, integration, spa_slice, spe_slice)
    light, dark = l1b.get_light_and_dark(integration)
    sub = light - fitted_dark
    min_, max_ = np.percentile(sub, (2, 98))
    return min_, max_


def do_all(l1b, integration, fullraw=None, fulldark=None, log=False):
    if fullraw is None or fulldark is None:
        fullraw, fulldark = l1b.get_light_and_dark(integration)
    spa_slice, spe_slice = l1b.find_scaling_window(fullraw)
    light_subframe = fullraw[spa_slice, spe_slice]
    dark_subframe = fulldark[spa_slice, spe_slice]

    scalers = [AddScaler, MultScaler, PolyScaler1,
               PolyScaler2]

    fig, axes = plt.subplots(nrows=len(scalers) + 3, sharex=True)

    l1b.plot_raw_profile(integration, ax=axes[0], log=log)
    axes[0].set_ylim(*np.percentile(fullraw, (2, 96)))

    l1b.plot_dark_profile(integration, ax=axes[1], log=log)

    # min_, max_ = get_min_max(l1b, integration, spa_slice, spe_slice)
    if int(l1b.int_time) == 1:
        min_, max_ = (-5, 15)
    else:
        min_, max_ = (-1, 3)

    l1b.plot_some_profile('detector_dark_subtracted', integration,
                          ax=axes[2], scale=True, log=log)
    dds = l1b.get_integration('dds_dn_s', integration)
    subdds = dds[spa_slice, spe_slice]
    title = axes[2].get_title() + ', Mean:{:.1f}, STD:{:.3f}'\
                                  .format(subdds.mean(),
                                          subdds.std())

    axes[2].set_title('')
    axes[2].text(.5, .9, title, horizontalalignment='center',
                 transform=axes[2].transAxes, fontsize=10)

    spatial = l1b.spatial_size // 2
    for Scaler, ax in zip(scalers, axes[3:]):
        if Scaler == MultScaler:
            scaler = Scaler(dark_subframe, light_subframe, alternative=True)
        else:
            scaler = Scaler(dark_subframe, light_subframe)
        scaler.do_fit()
        fitted_dark = scaler.apply_fit(fulldark)
        sub = fullraw - fitted_dark
        mean = sub[spa_slice, spe_slice].mean()
        std = sub[spa_slice, spe_slice].std()
        if log:
            ax.semilogy(l1b.wavelengths[integration], sub[spatial])
            ax.set_ylabel('log(DN/s)')
        else:
            ax.plot(l1b.wavelengths[integration], sub[spatial])
            ax.set_ylabel('DN/s')
        ax.set_xlim((l1b.wavelengths[integration][0],
                     l1b.wavelengths[integration][-1]))
        parameters = scaler.p_formatted
        if isinstance(scaler, AddScaler) or isinstance(scaler, MultScaler):
            parameters = "{} +- {:.3f}".format(scaler.p_formatted[0], scaler.perr[0])
        title = ("{}, {}. Mean:{:.1f}, STD:{:.3f}".format(scaler.name,
                                                          parameters,
                                                          mean, std))
        ax.text(.5, .9, title, horizontalalignment='center',
                transform=ax.transAxes, fontsize=12)

    for ax in axes[:-1]:
        ax.set_xlabel('')
    for ax in axes:
        if not log:
            ax.locator_params(axis='y', nbins=6)
        ax.set_ylim(min_, max_)

    axes[-1].set_xlabel("Wavelength [nm]")
    fig.suptitle("{}\nProfiles at middle spatial, Window: [{}:{}, {}:{}]\n"
                 .format(l1b.plottitle,
                         spa_slice.start,
                         spa_slice.stop,
                         spe_slice.start,
                         spe_slice.stop),
                 fontsize=11)
    fig.subplots_adjust(top=0.90, bottom=0.07)
    fig.savefig('/home/klay6683/plots/' + l1b.plotfname + '_2.png', dpi=150)


class KindHeader(fits.Header):

    """FITS header with the 'kind' card."""

    def __init__(self, kind='original dark'):
        super(KindHeader, self).__init__()
        self.set('kind', kind, comment='The kind of image')


class PrimHeader(KindHeader):

    """FITS primary header with a name card."""

    def __init__(self):
        super(PrimHeader, self).__init__()
        self.set('name', 'dark1')


class FittedHeader(KindHeader):

    """FITS header with a kind and a rank card."""

    def __init__(self, rank):
        super(FittedHeader, self).__init__('fitted dark')
        comment = "The degree of polynom used for the scaling."
        self.set('rank', rank, comment=comment)
        self.add_comment("The rank is '-1' for 'Additive' fitting, '0' is "
                         "for 'Multiplicative' fitting without additive "
                         "offset. For all ranks larger than 0 it is "
                         "equivalent to the degree of the polynomial fit.")


class DarkWriter(object):

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

            polyscaler: type of Polyscaler

        """
        if type(scaler) == PolyScaler:
            rank = scaler.rank
        elif type(scaler) == MultScaler:
            rank = 0
        elif type(scaler) == AddScaler:
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
