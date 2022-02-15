import re
import sys
import json
import math
import logging
from typing import Union
from datetime import datetime
from functools import reduce
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import colorama
import sty

from .data_path import *


rcParams['figure.constrained_layout.use'] = True
sns.set_style('darkgrid')
LN_KWARGS = dict(marker='o', ms=0.3, lw=0.25)  # matplotlib line plot default args


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
        with open(os.path.join(PATH_BASE, DIR_PROJ, PKG_NM, 'util', 'config.json'), 'r') as f:
            config.config = json.load(f)
    return get(config.config, attr)


def now(as_str=True, sep=':'):
    d = datetime.now()
    return d.strftime(f'%Y-%m-%d %H{sep}%M{sep}%S') if as_str else d  # Considering file output path


def log(s, c: str = 'log', c_time='green', as_str=False):
    """
    Prints `s` to console with color `c`
    """
    if not hasattr(log, 'reset'):
        log.reset = colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
    if not hasattr(log, 'd'):
        log.d = dict(
            log='',
            warn=colorama.Fore.YELLOW,
            error=colorama.Fore.RED,
            err=colorama.Fore.RED,
            success=colorama.Fore.GREEN,
            suc=colorama.Fore.GREEN,
            info=colorama.Fore.BLUE,
            i=colorama.Fore.BLUE,
            w=colorama.Fore.RED,

            y=colorama.Fore.YELLOW,
            yellow=colorama.Fore.YELLOW,
            red=colorama.Fore.RED,
            r=colorama.Fore.RED,
            green=colorama.Fore.GREEN,
            g=colorama.Fore.GREEN,
            blue=colorama.Fore.BLUE,
            b=colorama.Fore.BLUE,

            m=colorama.Fore.MAGENTA
        )
    if c in log.d:
        c = log.d[c]
    if as_str:
        return f'{c}{s}{log.reset}'
    else:
        print(f'{c}{log(now(), c=c_time, as_str=True)}| {s}{log.reset}')


def logs(s, c):
    return log(s, c=c, as_str=True)


def logi(s):
    """
    Syntactic sugar for logging `info` as string
    """
    return logs(s, c='i')


def hex2rgb(hx: str) -> Union[tuple[int], tuple[float]]:
    # Modified from https://stackoverflow.com/a/62083599/10732321
    if not hasattr(hex2rgb, 'regex'):
        hex2rgb.regex = re.compile(r'#[a-fA-F0-9]{3}(?:[a-fA-F0-9]{3})?$')
    m = hex2rgb.regex.match(hx)
    assert m is not None
    if len(hx) <= 4:
        return tuple(int(hx[i]*2, 16) for i in range(1, 4))
    else:
        return tuple(int(hx[i:i+2], 16) for i in range(1, 7, 2))


class MyTheme:
    """
    Theme based on `sty` and `Atom OneDark`
    """
    COLORS = OrderedDict([
        ('yellow', 'E5C07B'),
        ('green', '00BA8E'),
        ('blue', '61AFEF'),
        ('cyan', '2AA198'),
        ('red', 'E06C75'),
        ('purple', 'C678DD')
    ])
    yellow, green, blue, cyan, red, purple = (
        hex2rgb(f'#{h}') for h in ['E5C07B', '00BA8E', '61AFEF', '2AA198', 'E06C75', 'C678DD']
    )

    @staticmethod
    def set_color_type(t: str):
        """
        Sets the class attribute accordingly

        :param t: One of ['rgb`, `sty`]
            If `rgb`: 3-tuple of rgb values
            If `sty`: String for terminal styling prefix
        """
        for color, hex_ in MyTheme.COLORS.items():
            val = hex2rgb(f'#{hex_}')  # For `rgb`
            if t == 'sty':
                setattr(sty.fg, color, sty.Style(sty.RgbFg(*val)))
                val = getattr(sty.fg, color)
            setattr(MyTheme, color, val)


class MyFormatter(logging.Formatter):
    """
    Modified from https://stackoverflow.com/a/56944256/10732321

    Default styling: Time in green, metadata indicates severity, plain log message
    """
    RESET = sty.rs.fg + sty.rs.bg + sty.rs.ef

    MyTheme.set_color_type('sty')
    yellow, green, blue, cyan, red, purple = (
        MyTheme.yellow, MyTheme.green, MyTheme.blue, MyTheme.cyan, MyTheme.red, MyTheme.purple
    )

    KW_TIME = '%(asctime)s'
    KW_MSG = '%(message)s'
    KW_LINENO = '%(lineno)d'
    KW_FNM = '%(filename)s'
    KW_FUNCNM = '%(funcName)s'
    KW_NAME = '%(name)s'

    DEBUG = INFO = BASE = RESET
    WARN, ERR, CRIT = yellow, red, purple
    CRIT += sty.Style(sty.ef.bold)

    LVL_MAP = {  # level => (abbreviation, style)
        logging.DEBUG: ('DBG', DEBUG),
        logging.INFO: ('INFO', INFO),
        logging.WARNING: ('WARN', WARN),
        logging.ERROR: ('ERR', ERR),
        logging.CRITICAL: ('CRIT', CRIT)
    }

    def __init__(self, fmt_time=green):
        super().__init__()

        sty_kw, reset = MyFormatter.blue, MyFormatter.RESET
        fmt_time = f'{fmt_time}{MyFormatter.KW_TIME}{sty_kw}| {reset}'

        def fmt_meta(meta_abv, meta_style):
            return f'{MyFormatter.purple}[{MyFormatter.KW_NAME}]' \
                   f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FUNCNM}' \
                   f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FNM}' \
                   f'{MyFormatter.blue}:{MyFormatter.purple}{MyFormatter.KW_LINENO}' \
                   f'{MyFormatter.blue}, {meta_style}{meta_abv}{reset}'

        self.formats = {
            level: fmt_time + fmt_meta(*args) + f'{sty_kw} - {reset}{MyFormatter.KW_MSG}' + reset
            for level, args in MyFormatter.LVL_MAP.items()
        }
        self.formatter = {
            lv: logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S') for lv, fmt in self.formats.items()
        }

    def format(self, entry):
        return self.formatter[entry.levelno].format(entry)


def fmt_num(n: Union[float, int]):
    """
    Convert number to human-readable format, in e.g. Thousands, Millions
    """
    if not hasattr(fmt_num, 'posts'):
        fmt_num.posts = ['', 'K', 'M', 'B', 'T']
    n = float(n)
    idx_ = max(0, min(len(fmt_num.posts) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    return '{:.0f}{}'.format(n / 10 ** (3 * idx_), fmt_num.posts[idx_])


def model_param_size(m: torch.nn.Module, as_str=True) -> Union[int, str]:
    num = sum(p.numel() for p in m.parameters())
    return fmt_num(num) if as_str else num


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

