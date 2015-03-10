import pandas as pd
from . import io

stagel1asummary = str(io.stagelevel1apath / 'IUVS_L1A_index_summary.txt')
stagel1bsummary = str(io.stagelevel1bpath / 'IUVS_L1B_index_summary.txt')
prodl1asummary = str(io.productionlevel1apath / 'IUVS_L1A_index_summary.txt')
prodl1bsummary = str(io.productionlevel1bpath / 'IUVS_L1B_index_summary.txt')


def get_summary_df(fname):
    return pd.read_table(fname, skiprows=26, skipinitialspace=True,
                         sep=' ', index_col=0)


def l1a_summary(stage=True):
    if stage:
        return get_summary_df(stagel1asummary)
    else:
        return get_summary_df(prodl1asummary)


def l1b_summary(stage=True):
    if stage:
        return get_summary_df(stagel1bsummary)
    else:
        return get_summary_df(prodl1bsummary)


def get_full_frames(df):
    return df[(df.NX == 1024) & (df.NY == 1024)]
