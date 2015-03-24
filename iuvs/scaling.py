from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from . import io


def multimodel(x, a):
    return a*x


def addmodel(x, a):
    return a+x


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


class DarkScaler:
    """Managing the general attributes for scaling darks around.

    This is the base class that is inherited by the other scalers.
    """
    def __init__(self, data_in, data_out):
        self.data_in = data_in
        self.data_out = data_out

    def do_fit(self):
        self.p, self.pcov = curve_fit(self.model,
                                      self.data_in.ravel(),
                                      self.data_out.ravel())
        self.perr = np.sqrt(np.diag(self.pcov))

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
        return self.model(in_, self.p)

    @property
    def p_formatted(self):
        return list(reversed(["{:.2f}".format(i) for i in self.p]))


class AddScaler(DarkScaler):
    """Additive Scaling model."""
    name = 'AddScaler'
    rank = -1

    def model(self, x, a):
        return a + x


class MultScaler(DarkScaler):
    """Pure Multiplicative scaling model"""
    name = 'MultScaler'
    rank = 0

    def model(self, x, a):
        return a * x


class PolyScaler(DarkScaler):
    """Manage polynomial fits. Default rank is 2."""
    def __init__(self, data_in, data_out, rank=2):
        super().__init__(data_in, data_out)
        self.rank = rank
        self.name = 'Poly'+str(self.rank)

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


class PolyScaler1(PolyScaler):
    def __init__(self, *args):
        super().__init__(*args, rank=1)


class PolyScaler2(PolyScaler):
    def __init__(self, *args):
        super().__init__(*args, rank=2)


class PolyScaler3(PolyScaler):
    def __init__(self, *args):
        super().__init__(*args, rank=3)


class PolyScalerManager:

    def __init__(self, data_in, data_out, rankstart, rankend):
        self.rankstart = rankstart
        self.rankend = rankend
        self.data_in = data_in
        self.data_out = data_out
        fractionals = []
        coeffs = []
        polynoms = []
        scalers = []
        for rank in range(rankstart, rankend+1):
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
        plt.plot(range(self.rankstart, self.rankend+1),
                 self.fractionals)
        plt.xlabel('Polynomial rank')
        plt.xlim(0, self.rankend+2)
        plt.title('Mean values of fractional residual'
                  ' over polynomial rank')
        plt.ylabel('Mean value of fracional residual')


class DarkFitter:
    def __init__(self, fname):
        self.fname = fname
        l1b = io.L1BReader(fname)
        self.l1b = l1b
        darkout = l1b.detector_dark[-1]
        darkin = l1b.detector_dark[-2]
        scaler = PolyScaler(darkin, darkout)
        scaler.do_fit()
        self.scaler = scaler


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
    min_, max_ = np.percentile(sub, (2, 92))
    return min_, max_


def do_all(l1b, integration, spa_slice, spe_slice):

    fullraw, fulldark = l1b.get_light_and_dark(integration)
    light_subframe = fullraw[spa_slice, spe_slice]
    dark_subframe = fulldark[spa_slice, spe_slice]

    scalers = [AddScaler, MultScaler, PolyScaler1,
               PolyScaler2, PolyScaler3]

    fig, axes = plt.subplots(nrows=len(scalers)+3, sharex=True)

    l1b.plot_raw_profile(integration, ax=axes[0])
    axes[0].set_ylim(*np.percentile(fullraw, (2, 96)))

    l1b.plot_dark_profile(integration, ax=axes[1])

    min_, max_ = get_min_max(l1b, integration, spa_slice, spe_slice)

    l1b.plot_some_profile('detector_background_subtracted', integration,
                          ax=axes[2], scale=True)
    axes[2].set_ylim(min_, max_)

    spatial = fullraw.shape[0]//2

    for Scaler, ax in zip(scalers, axes[3:]):
        scaler = Scaler(dark_subframe, light_subframe)
        scaler.do_fit()
        fitted_dark = scaler.apply_fit(fulldark)
        sub = fullraw - fitted_dark
        ax.plot(l1b.wavelengths[integration], sub[spatial])
        ax.set_ylim(min_, max_)
        ax.set_xlim((l1b.wavelengths[integration][0],
                     l1b.wavelengths[integration][-1]))

        title = ("{}, {}".format(scaler.name,
                                 scaler.p_formatted))
        ax.text(.5, .9, title, horizontalalignment='center',
                transform=ax.transAxes, fontsize=12)

    for ax in axes[:-1]:
        ax.set_xlabel('')
    for ax in axes:
        ax.locator_params(axis='y', nbins=6)
    axes[-1].set_xlabel("Wavelength [nm]")
    fig.suptitle("{}\nSlice: [{}:{}, {}:{}]\n"
                 .format(l1b.plottitle,
                         spa_slice.start,
                         spa_slice.stop,
                         spe_slice.start,
                         spe_slice.stop),
                 fontsize=11)
    fig.subplots_adjust(top=0.90, bottom=0.07)
    fig.savefig('plots/'+l1b.plotfname+'_2.png', dpi=150)


class DarkWriter:

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
