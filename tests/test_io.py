import datetime as dt
import os
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given

from iuvs import io


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


def test_iuvs_utc_to_dtime():
    s = '2015/085 Mar 26 17:45:19.96275UTC'
    expected = dt.datetime.strptime("2015-03-26 17:45:19.962750",
                                    '%Y-%m-%d %H:%M:%S.%f')
    assert expected == io.iuvs_utc_to_dtime(s)


def test_get_hk_filenames(monkeypatch):
    pass


class TestFitsFile:

    "Collector for test for the FitsFile class."

    def test_init(self):
        "init claims it needs an absolute path as input. test this here"
        pass
