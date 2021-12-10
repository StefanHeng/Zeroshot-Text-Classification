import json
import os
from functools import reduce
from datetime import datetime

import numpy as np
import torch
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns


from data_path import *


rcParams['figure.constrained_layout.use'] = True
sns.set_style('darkgrid')


def get(dic, ks):
    """
    :param dic: Potentially multi-level dictionary
    :param ks: Potentially `.`-separated keys
    """
    ks = ks.split('.')
    return reduce(lambda acc, elm: acc[elm], ks, dic)


def set_(dic, ks, val):
    ks = ks.split('.')
    node = reduce(lambda acc, elm: acc[elm], ks[:-1], dic)
    node[ks[-1]] = val


def keys(dic, prefix=''):
    """
    :return: Generator for all potentially-nested keys
    """
    def _full(k_):
        return k_ if prefix == '' else f'{prefix}.{k_}'
    for k, v in dic.items():
        if isinstance(v, dict):
            for k__ in keys(v, prefix=_full(k)):
                yield k__
        else:
            yield _full(k)


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(os.path.join(PATH_BASE, DIR_PROJ, 'config.json'), 'r') as f:
            config.config = json.load(f)
    return get(config.config, attr)


def now(as_str=True):
    d = datetime.now()
    return d.strftime('%Y-%m-%d %H:%M:%S') if as_str else d


def get_torch_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_points(arr, **kwargs):
    """
    :param arr: Array of 2d points to plot
    :param kwargs: Arguments are forwarded to `matplotlib.axes.Axes.plot`
    """
    arr = np.asarray(arr)
    kwargs_ = dict(
        marker='.', lw=0.5, ms=1,
        c='orange',
    )
    plt.plot(arr[:, 0], arr[:, 1], **(kwargs_ | kwargs))


if __name__ == '__main__':
    from icecream import ic

    ic(config('fine-tune'))
