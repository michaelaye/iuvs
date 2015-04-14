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


def produce_summary_df(df):
    cols = ('filename obs_id xuv int_time naxis naxis1 naxis2 naxis3 spe_ofs'
            ' spa_ofs spe_size spa_size mcp_gain mcp_volt shutter_num shutter_off'
            ' shutter_on det_temp case_temp mir_deg n_fill sw_ver bin_tbl'.upper().split())
    return df[cols]


def produce_summary_txt(df, outfname):
    df = produce_summary_df(df)
    oldwidth = pd.get_option("display.max_colwidth")
    pd.set_option("display.max_colwidth", 10000)
    with open(outfname, 'w') as f:
        f.write(df.to_string())
    pd.set_option("display.max_colwidth", oldwidth)


def main():
    df = pd.read_hdf(str(io.HOME / 'l1a_summary.h5', 'df'))



if __name__ == '__main__':
    main()
