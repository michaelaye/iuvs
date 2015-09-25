import os
import sys

import numpy as np
import pandas as pd

from . import io
from .exceptions import ApoapseNonUniqueSpectralPixel


def check_endian(data):
    sys_byteorder = ('>', '<')[sys.byteorder == 'little']
    if data.dtype.byteorder not in ('=', sys_byteorder):
        return data.byteswap().newbyteorder(sys_byteorder)
    else:
        return data


def process_integration_i(i, b, spectral_pix=15):
    selector = b.p_alts[i].apply(lambda x: all(~(x > 0)), axis=1)
    # take column 4 as it's the pixel center ground coords.
    lats = b.p_lats[i][selector][4]
    lons = b.p_lons[i][selector][4]
    values = b.spec[i][selector][spectral_pix]
    return lats, lons, values


def return_and_convert(lats, lons, data):
    return lats, lons, data


class Apoapse(object):

    def __init__(self, fname, wavelength):
        self.fname = fname
        self.wavelength = wavelength
        self.l1b = io.L1BReader(fname)

        # altitude over limb
        self.p_alts = self.get_and_check_PixelGeom('PIXEL_CORNER_MRH_ALT')
        # for now only use the center pixel coords for lats and lons
        self.p_lats = self.get_and_check_PixelGeom('PIXEL_CORNER_LAT')[:, :, 4]
        self.p_lons = self.get_and_check_PixelGeom('PIXEL_CORNER_LON')[:, :, 4]
        # this would fail if the data array is not 3-dim
        self.spec = pd.Panel(check_endian(self.l1b.img))

        spec_pix = self.get_spectral_pixel()

        # get boolean selector dataframe for pixels on disk
        # the all(...) term refers to the fact that I want all 5 coords
        # (corners & center) to be on-disk to be chosen.
        selector = self.p_alts.apply(lambda x: all(~(x > 0)), axis=2)

        # values.ravel to go back to 1D numpy arrays
        self.lats = self.p_lats[selector].values.ravel()
        self.lons = self.p_lons[selector].values.ravel()
        self.data = self.spec[:, :, spec_pix][selector].values.ravel()

    def get_spectral_pixel(self):
        """return spectral pixels closest to chosen wavelength.

        To use only 1 number for all pixel is a cop-out as I could not find
        quickly a way to slice a 3D array with a list of indices for matching wavelengths
        in case they ever differ (I think so far all pixels have the same wavelengths,
        but that might differ in the future).
        TODO: Make this work for a list of differing indices for given wavelengths.
        """
        idx = (np.abs(self.l1b.wavelengths-self.wavelength)).argmin(axis=1)
        if len(set(idx)) > 1:
            raise ApoapseNonUniqueSpectralPixel(idx)
        return idx[0]

    def get_and_check_PixelGeom(self, colname):
        return pd.Panel(check_endian(self.l1b.PixelGeometry[colname]))


def process_fnames(fnames, wavelength):
    lats = []
    lons = []
    data = []
    for fname in fnames:
        print(os.path.basename(fname))
        apo = Apoapse(fname, wavelength)
        idx = ~np.isnan(apo.lats)
        lats.append(apo.lats[idx])
        lons.append(apo.lons[idx])
        data.append(apo.data[idx])

    df = pd.DataFrame({'lats': np.concatenate(lats),
                       'lons': np.concatenate(lons),
                       'data': np.concatenate(data)})

    return df


def process_day(daystring, wavelength):
    """process day of apoapse data

    Parameters
    ----------
    daystring: <str>
        Usual format of YYYYmmdd (%Y%m%d)

    Returns
    -------
    pd.DataFrame
        Also saving the dataframe in ~/to_keep/apoapse
    """
    globstr = "apoapse*-muv_{}".format(daystring)
    fnames = io.l1b_filenames(globstr)
    df = process_fnames(fnames, wavelength)
    df.to_hdf()
