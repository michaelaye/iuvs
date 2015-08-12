import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from iuvs import scaling

xfail = pytest.mark.xfail

xslice = slice(2, 4)
yslice = slice(3, 5)


def get_random_arrays():
    spec = 2 * np.random.rand(4, 5)
    dark = np.random.rand(4, 5)
    return spec, dark


def get_ordered_arrays():
    spec = 2 * np.arange(20).reshape(4, 5)
    dark = np.arange(20).reshape(4, 5)
    return spec, dark


def test_addscaler_ordered():
    spec, dark = get_ordered_arrays()
    scaler = scaling.AddScaler(dark[xslice, yslice], spec[xslice, yslice])
    scaler.do_fit()
    target = spec[xslice, yslice].mean()
    start = dark[xslice, yslice].mean()
    e = 1e-12
    assert (target - start) - scaler.p[0] < e


def test_multscaler_ordered():
    spec, dark = get_ordered_arrays()
    scaler = scaling.MultScaler(dark[xslice, yslice], spec[xslice, yslice])
    scaler.do_fit()
    target = spec[xslice, yslice].mean()
    start = dark[xslice, yslice].mean()
    e = 1e-12
    assert (target / start) - scaler.p[0] < e


def test_addscaler_random():
    spec, dark = get_random_arrays()
    scaler = scaling.AddScaler(dark[xslice, yslice], spec[xslice, yslice])
    scaler.do_fit()
    target = spec[xslice, yslice].mean()
    start = dark[xslice, yslice].mean()
    e = 1e-8
    expected = target - start
    assert (expected - scaler.p[0]) < e


@xfail(reason="bug 13")
def test_multscaler_random():
    spec, dark = get_random_arrays()
    scaler = scaling.MultScaler(dark[xslice, yslice], spec[xslice, yslice])
    scaler.do_fit()
    target = spec[xslice, yslice].mean()
    start = dark[xslice, yslice].mean()
    e = 1e-10
    expected = target / start
    assert (expected - scaler.p[0]) < e


def test_polyscaler_p_dict():
    "check if the p_dict is created properly."
    spec, dark = get_ordered_arrays()
    scaler = scaling.PolyScaler1(dark[xslice, yslice], spec[xslice, yslice])
    scaler.do_fit()
    assert int(scaler.p_dict['poly1_1']) == 2
