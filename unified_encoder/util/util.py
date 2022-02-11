import sys
import json
import math
from typing import Union
from functools import reduce
from datetime import datetime

import numpy as np
import torch
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns


from .data_path import *


rcParams['figure.constrained_layout.use'] = True
sns.set_style('darkgrid')


def get_python_version():
    vi = sys.version_info
    return dict(
        major=vi[0],
        minor=vi[1]
    )


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
        with open(os.path.join(PATH_BASE, DIR_PROJ, 'util', 'config.json'), 'r') as f:
            config.config = json.load(f)
    return get(config.config, attr)


def now(as_str=True):
    d = datetime.now()
    return d.strftime('%Y-%m-%d %H:%M:%S') if as_str else d


def fmt_num(n: Union[float, int]):
    """
    Convert number to human-readable format, in e.g. Thousands, Millions
    """
    if not hasattr(fmt_num, 'posts'):
        fmt_num.posts = ['', 'K', 'M', 'B', 'T']
    n = float(n)
    idx_ = max(0, min(len(fmt_num.posts) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    return '{:.0f}{}'.format(n / 10 ** (3 * idx_), fmt_num.posts[idx_])


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
    kwargs = {**kwargs_, **kwargs}  # Support versions below 3.9
    # vers = get_python_version()
    # assert vers['major'] == 3
    # if vers['minor'] < 9:
    #     kwargs = {**kwargs_, **kwargs}
    # else:
    #     kwargs = kwargs_ | kwargs
    plt.plot(arr[:, 0], arr[:, 1], **kwargs)


if __name__ == '__main__':
    from icecream import ic

    ic(config('fine-tune'))

    ic(fmt_num(124439808))

