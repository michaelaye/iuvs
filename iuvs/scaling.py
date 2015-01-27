from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def multimodel(x, a):
    return a*x


def addmodel(x, a):
    return a+x


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
        for rank in range(rankstart, rankend+1):
            scaler = PolyScaler(data_in, data_out, rank)
            scaler.do_fit()
            polynoms.append(scaler.poly)
            coeffs.append(scaler.p)
            fractionals.append(scaler.fractional.mean())
        self.fractionals = fractionals
        self.coeffs = coeffs
        self.polynoms = polynoms

    def plot_fractionals(self):
        plt.plot(range(self.rankstart, self.rankend+1),
                 self.fractionals)
        plt.xlabel('Polynomial rank')
        plt.xlim(0, self.rankend+2)
        plt.title('Mean values of fractional residual'
                  ' over polynomial rank')
        plt.ylabel('Mean value of fracional residual')
