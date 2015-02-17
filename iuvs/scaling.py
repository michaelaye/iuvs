from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import io


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


class AddScaler(DarkScaler):

    def model(self, x, a):
        return a + x


class MultScaler(DarkScaler):

    def model(self, x, a):
        return a * x


class PolyScaler(DarkScaler):

    """Manage polynomial fits. Default rank is 2."""

    def __init__(self, data_in, data_out, rank=2):
        super().__init__(data_in, data_out)
        self.rank = rank

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
        L1BReader
        self.l1b = l1b
        darkout = l1b.detector_dark[-1]
        darkin = l1b.detector_dark[-2]
        scaler = PolyScaler(darkin, darkout)
        scaler.do_fit()
        self.scaler = scaler
