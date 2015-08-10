import datetime as dt
import os.path as osp
import socket
from pathlib import Path

from iuvs import io
from iuvs.io import iuvs_utc_to_dtime

host = socket.gethostname()


def test_get_data_path():

    if host.startswith('maven-iuvs-itf'):
        for key, level in zip(['l0', 'l1a', 'l1b'],
                              ['level0', 'level1a', 'level1b']):
            for env in ['stage', 'production']:
                root = '/maven_iuvs/' + env + '/products'
                expected = osp.join(root, level)
            assert io.get_data_path(key, env) == Path(expected)


def test_iuvs_utc_to_dtime():
    s = '2015/085 Mar 26 17:45:19.96275UTC'
    expected = dt.datetime.strptime("2015-03-26 17:45:19.962750",
                                    '%Y-%m-%d %H:%M:%S.%f')
    assert expected == iuvs_utc_to_dtime(s)


def test_get_hk_filenames(monkeypatch):
    pass


class TestFitsFile:

    "Collector for test for the FitsFile class."

    def test_init(self):
        "init claims it needs an absolute path as input. test this here"
        pass
