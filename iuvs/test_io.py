import socket
from . import io
import os.path as osp

host = socket.gethostname()

def test_get_data_path():

    if host.startswith('maven-iuvs-itf'):
        for level in ['level0', 'level1a', 'level1b']:
            for env in ['stage','production']:
                root = '/maven_iuvs/stage/products'
                expected = osp.join(root, env, level)
            assert io.get_data_path(level, env) == expected

def test_iuvs_utc_to_dtime():
    from .io import iuvs_utc_to_dtime
    import datetime as dt
    s = '2015/085 Mar 26 17:45:19.96275UTC'
    expected = dt.datetime.strptime("2015-03-26 17:45:19.962750",
                                    '%Y-%m-%d %H:%M:%S.%f')
    assert expected == iuvs_utc_to_dtime(s)


class TestFitsFile:

    "Collector for test for the FitsFile class."

    def test_init():
        "init claims it needs an absolute path as input. test this here"
        pass
