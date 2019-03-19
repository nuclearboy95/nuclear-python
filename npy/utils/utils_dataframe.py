import pandas as pd
from tabulate import tabulate


def pretty(df, index=True, headers=True):
    headers = 'keys' if headers else ()
    return tabulate(df, tablefmt='psql', headers=headers, showindex=index)
