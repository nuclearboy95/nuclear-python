import pandas as pd
from tabulate import tabulate


__all__ = ['pretty', 'split_by_column']


def pretty(df, index=True, headers=True):
    """

    :param pd.DataFrame df:
    :param bool index:
    :param bool headers:
    :return:
    """
    headers = 'keys' if headers else ()
    return tabulate(df, tablefmt='psql', headers=headers, showindex=index)


def split_by_column(df, col, drop=True) -> dict:
    """

    :param pd.DataFrame df:
    :param str col:
    :param bool drop:
    :return:
    """
    dfs = {key: df for key, df in df.groupby(col)}
    if drop:
        for key in dfs:
            del dfs[key][col]
    return dfs
