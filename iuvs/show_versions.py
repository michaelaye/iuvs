#!/usr/bin/env python
from . import io
import pandas as pd
import argparse


def get_version_string_from_fname(fname):
    iuvsfname = io.Filename(fname)
    return iuvsfname.version + iuvsfname.revision


def show_existing_versions(level, stage):
    fnames = io.get_filenames(level, pattern=None, stage=stage)
    df = pd.DataFrame(fnames, columns=['fname'])
    df['version_string'] = df.fname.map(get_version_string_from_fname)
    print("Version    Counts")
    print(df.version_string.value_counts)


def show_existing_l1a_stage_versions():
    show_existing_versions('l1a', True)


def show_existing_l1a_production_versions():
    show_existing_versions('l1a', False)


def show_existing_l1b_stage_versions():
    show_existing_versions('l1b', True)


def show_existing_l1b_production_versions():
    show_existing_versions('l1b', False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("level", type=str, choices=['l1a', 'l1b'],
                        help='The level you want to check.')
    parser.add_argument('stage', type=str, choices=['stage', 'production'],
                        help='Which stage you want to look at.')
    parser.parse_args()
