from __future__ import division

import os

import hypothesis.strategies as st
import pytest
from hypothesis import given
from hypothesis.extra.datetime import datetimes

from iuvs import io

skiptravis = pytest.mark.skipif('TRAVIS' in os.environ and \
                                os.environ['TRAVIS'] == 'true',
                                reason='does not work on travis')


@skiptravis
@given(st.sampled_from(['l0', 'l1a', 'l1b']),
       st.sampled_from(['stage', 'production'])
       )
def test_get_data_path(level, env):
    path = str(io.get_data_path(level, env))
    if os.path.exists(path):
        assert True
    elif os.access(os.path.dirname(path), os.R_OK):
        assert True
    else:
        assert False
    # assert osp.exists(str(path)) == True


@given(datetimes(timezones=[], min_year=1900),
       st.text(min_size=3, max_size=3),
       )
def test_iuvs_utc_to_dtime(date, extra):
    newmics = date.microsecond // 10 * 10
    expected = date.replace(microsecond=newmics)
    format = '%Y/%j %b %d %H:%M:%S.%f'
    teststr = date.strftime(format)[:-1] + extra
    assert expected == io.iuvs_utc_to_dtime(teststr)


def test_get_hk_filenames(monkeypatch):
    pass


def test_Filename():
    p = '/Users/klay6683/data/iuvs/level1b/'\
        'mvn_iuv_l1b_cruisecal2-mode080-muv_20140521T120029_v01_r01.fits.gz'
    fn = io.Filename(p)
    assert fn.mission == 'mvn'
    assert fn.instrument == 'iuv'
    assert fn.basename == 'mvn_iuv_l1b_cruisecal2-mode080-'\
                          'muv_20140521T120029_v01_r01.fits.gz'
    assert fn.root == '/Users/klay6683/data/iuvs/level1b'

def test_HKFilename():
    p = '/maven_iuvs/stage/products/housekeeping/level1a/'\
        'mvn_iuv_analog_l0_20140405_v003.fits.gz'
    hkfn = io.HKFilename(p)
    assert hkfn.datestring == '20140405'
    assert hkfn.kind == 'analog'
    assert hkfn.version == 'v003'
    assert hkfn.level == 'l0'


def test_ScienceFilename():
    p = '/Users/klay6683/data/iuvs/level1b/'\
        'mvn_iuv_l1b_cruisecal2-mode080-muv_20140521T120029_v01_r01.fits.gz'
    fn = io.ScienceFilename(p)
    assert fn.channel == 'muv'
    assert fn.cycle_orbit == 'mode080'
    assert fn.mode == 'N/A'
    assert fn.phase == 'cruisecal2'
    assert fn.obs_id == 'mvn_iuv_l1b_cruisecal2-mode080-muv_20140521T120029'
    assert fn.version == 'v01'
    assert fn.revision == 'r01'

class TestFitsFile:

    "Collector for test for the FitsFile class."

    def test_init(self):
        "init claims it needs an absolute path as input. test this here"
        pass
