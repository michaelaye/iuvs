from __future__ import division

import os

import hypothesis.strategies as st
import pytest
from hypothesis import given
from hypothesis.extra.datetime import datetimes

from iuvs import io

skiptravis = pytest.mark.skipif(os.environ['TRAVIS'] == 'true',
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


class TestFitsFile:

    "Collector for test for the FitsFile class."

    def test_init(self):
        "init claims it needs an absolute path as input. test this here"
        pass
