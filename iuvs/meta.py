import pandas as pd
from . import io


def get_summary_df(level, env):
    folder = io.get_data_path(level, env)
    name = 'IUVS_{}_index_summary.txt'.format(level.upper())
    fname = folder / name
    return pd.read_table(fname, skiprows=26, skipinitialspace=True,
                         sep=' ', index_col=0)


def l1a_summary(env='production'):
    return get_summary_df('l1a', env)


def l1b_summary(env='production'):
    return get_summary_df('l1b')


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


def clean_up_dark_scan(df):
    # drop columns that don't seem to change
    to_drop = []
    for col in df:
        n_unique = len(df[col].unique())
        if n_unique == 1:
            to_drop.append(col)
    df = df.drop(to_drop, axis=1)

    # time stuff
    df = df.drop('et timestamp capture'.upper().split(), axis=1)
    df.UTC = df.UTC.map(io.iuvs_utc_to_dtime)
    df = df.rename(columns=lambda x: 'TIME_OF_INT' if x == 'UTC' else x)

    # remove XUV as it's same as CHANNEL
    df = df.drop(['XUV'], axis=1)

    # remove PRODUCT_CREATION_DATE as it's same as PROCESS
    df = df.drop(['PRODUCT_CREATION_DATE'], axis=1)

    # create binning tuple
    df['BINNING_SET'] = tuple(zip(df['SPE_OFS'], df['SPE_SIZE'], df['SPA_OFS'], df['SPA_SIZE']))

    df = df.drop('spe_size spe_ofs spa_size spa_ofs'.upper().split(), axis=1)

    df['INT_TIME'] = (df.INT_TIME*1000).astype('int')
    df.set_index("TIME_OF_INT", inplace=True)
    df.sort_index(inplace=True)
    s = df.PRODUCT_ID.str.split('_')
    s = s.map(lambda x: x[3]).str.split('-')
    df['activity'] = s.map(lambda x: x[0])
    return df
