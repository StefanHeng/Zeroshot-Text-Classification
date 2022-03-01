import os
import re
import sys
import json
import math
import logging
import datetime
import itertools
from typing import Union, Tuple, List, Dict, Iterable, TypeVar
from functools import reduce
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import datasets
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import colorama
import sty

from zeroshot_encoder.util.data_path import PATH_BASE, DIR_DSET, DIR_PROJ, DIR_MDL, PKG_NM


pd.set_option('expand_frame_repr', False)
pd.set_option('display.precision', 2)
pd.set_option('max_colwidth', 40)

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = (16, 9)
sns.set_style('darkgrid')

LN_KWARGS = dict(marker='o', ms=0.3, lw=0.25)  # matplotlib line plot default args


def get_python_version():
    vi = sys.version_info
    return dict(
        major=vi[0],
        minor=vi[1]
    )


T = TypeVar('T')
K = TypeVar('K')


def join_its(its: Iterable[Iterable[T]]) -> Iterable[T]:
    out = itertools.chain()
    for it in its:
        out = itertools.chain(out, it)
    return out


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
    config.config['UTCD']['dataset_id2name'] = {  # Convert str keys to int
        int(k): v for k, v in config.config['UTCD']['dataset_id2name'].items()
    }
    return get(config.config, attr)


def get_hostname() -> str:
    return os.uname().nodename


def now(as_str=True, sep=':'):
    d = datetime.datetime.now()
    return d.strftime(f'%Y-%m-%d %H{sep}%M{sep}%S') if as_str else d  # Considering file output path


def fmt_dt(secs: Union[int, float, datetime.timedelta]):
    if isinstance(secs, datetime.timedelta):
        secs = secs.seconds + (secs.microseconds/1e6)
    if secs >= 86400:
        d = secs // 86400  # // floor division
        return f'{round(d)}d{fmt_dt(secs-d*86400)}'
    elif secs >= 3600:
        h = secs // 3600
        return f'{round(h)}h{fmt_dt(secs-h*3600)}'
    elif secs >= 60:
        m = secs // 60
        return f'{round(m)}m{fmt_dt(secs-m*60)}'
    else:
        return f'{round(secs)}s'


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


def log_s(s, c):
    return log(s, c=c, as_str=True)


def logi(s):
    """
    Syntactic sugar for logging `info` as string
    """
    return log_s(s, c='i')


def log_dict(d: Dict = None, with_color=True, **kwargs) -> str:
    """
    Syntactic sugar for logging dict with coloring for console output
    """
    if d is None:
        d = kwargs
    pairs = (f'{k}: {logi(v) if with_color else v}' for k, v in d.items())
    pref = log_s('{', c='m') if with_color else '{'
    post = log_s('}', c='m') if with_color else '}'
    return pref + ', '.join(pairs) + post


def hex2rgb(hx: str) -> Union[Tuple[int], Tuple[float]]:
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

    def __init__(self, with_color=True, color_time=green):
        super().__init__()
        self.with_color = with_color

        sty_kw, reset = MyFormatter.blue, MyFormatter.RESET
        color_time = f'{color_time}{MyFormatter.KW_TIME}{sty_kw}| {reset}'

        def args2fmt(args_):
            if self.with_color:
                return color_time + self.fmt_meta(*args_) + f'{sty_kw} - {reset}{MyFormatter.KW_MSG}' + reset
            else:
                return f'{MyFormatter.KW_TIME}| {self.fmt_meta(*args_)} - {MyFormatter.KW_MSG}'

        self.formats = {level: args2fmt(args) for level, args in MyFormatter.LVL_MAP.items()}
        self.formatter = {
            lv: logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S') for lv, fmt in self.formats.items()
        }

    def fmt_meta(self, meta_abv, meta_style=None):
        if self.with_color:
            return f'{MyFormatter.purple}[{MyFormatter.KW_NAME}]' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FUNCNM}' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FNM}' \
               f'{MyFormatter.blue}:{MyFormatter.purple}{MyFormatter.KW_LINENO}' \
               f'{MyFormatter.blue}, {meta_style}{meta_abv}{MyFormatter.RESET}'
        else:
            return f'[{MyFormatter.KW_NAME}] {MyFormatter.KW_FUNCNM}::{MyFormatter.KW_FNM}' \
                   f':{MyFormatter.KW_LINENO}, {meta_abv}'

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
    kwargs = {**kwargs_, **kwargs}  # python3.6 compatibility
    plt.plot(arr[:, 0], arr[:, 1], **kwargs)


def get_output_base():
    # Remote machine `clarity2`, save the models somewhere else to save `/home` disk space
    if 'clarity' in get_hostname():
        return os.path.join('/data')
    else:
        return PATH_BASE


def process_utcd_dataset(join=False):
    ext = config('UTCD.dataset_ext')
    path_dsets = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
    path_out = os.path.join(get_output_base(), DIR_PROJ, DIR_DSET)

    def path2dsets(dnm: str, d_dset: Dict) -> Union[datasets.DatasetDict, Dict[str, pd.DataFrame]]:
        path = d_dset['path']
        path = os.path.join(path_dsets, f'{path}.{ext}')
        with open(path) as f:
            dsets_: Dict = json.load(f)
        d_lbs = config(f'UTCD.datasets.{dnm}.labels')

        def json2dset(split: str, dset: List) -> Union[datasets.Dataset, pd.DataFrame]:
            assert all(sample[0] != '' for sample in dset)
            dset = [dict(text=txt, label=lb) for (txt, lb) in dset]  # Heuristic on how the `json` are stored
            df_ = pd.DataFrame(dset)
            if join:  # Leave processing labels til later
                return df_
            else:
                # Sort the string labels, enforce deterministic order
                lbs = sorted(df_.label.unique())
                assert lbs == d_lbs[split]  # Sanity check
                lbs = datasets.ClassLabel(names=lbs)
                features_ = datasets.Features(text=datasets.Value(dtype='string'), label=lbs)
                # Map to integer labels so that compatible to current training infrastructure in `gpt2.py`
                df_.label.replace(to_replace=lbs.names, value=range(lbs.num_classes), inplace=True)
                return datasets.Dataset.from_pandas(df_, features=features_)
        return datasets.DatasetDict({split: json2dset(split, dset) for split, dset in dsets_.items()})
    d_dsets = {dnm: path2dsets(dnm, d) for dnm, d in config('UTCD.datasets').items()}
    if join:
        dnm2id = config('UTCD.dataset_name2id')

        def pre_concat(dnm: str, df_: pd.DataFrame) -> pd.DataFrame:
            df_['dataset_id'] = [dnm2id[dnm]] * len(df_)  # Add dataset source information to each row
            return df_
        # Global label across all datasets, all splits
        # Needed for inversely mapping to local label regardless of joined split, e.g. train/test,
        #   in case some label only in certain split
        lbs_lb = sorted(set(join_its(df.label.unique() for dsets in d_dsets.values() for df in dsets.values())))
        lbs_lb = datasets.ClassLabel(names=lbs_lb)

        def dfs2dset(dfs: Iterable[pd.DataFrame]) -> datasets.Dataset:
            df = pd.concat(dfs)
            # The string labels **may overlap** across the datasets
            # Keep internal feature label ordering same as dataset id
            lbs_dset = sorted(dnm2id, key=dnm2id.get)
            df.label.replace(to_replace=lbs_lb.names, value=range(lbs_lb.num_classes), inplace=True)
            features = datasets.Features(
                text=datasets.Value(dtype='string'), label=lbs_lb, dataset_id=datasets.ClassLabel(names=lbs_dset)
            )
            return datasets.Dataset.from_pandas(df, features=features)
        tr = dfs2dset(pre_concat(dnm, dsets['train']) for dnm, dsets in d_dsets.items())
        vl = dfs2dset(pre_concat(dnm, dsets['test']) for dnm, dsets in d_dsets.items())
        dsets = datasets.DatasetDict(train=tr, test=vl)
        dsets.save_to_disk(os.path.join(path_out, 'processed', 'UTCD'))
    else:
        for dnm, dsets in d_dsets.items():
            dsets.save_to_disk(os.path.join(path_out, 'processed', dnm))


def map_ag_news():
    dnm = 'ag_news'
    d_dset = config(f'UTCD.datasets.{dnm}')
    ext = config('UTCD.dataset_ext')
    path_dset = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
    path = d_dset['path']
    path = os.path.join(path_dset, f'{path}.{ext}')
    with open(path) as f:
        dsets: Dict = json.load(f)
    d_lb2desc = config(f'baselines.gpt2-nvidia.label-descriptors.{dnm}')
    for split, dset in dsets.items():
        dsets[split] = [[txt, d_lb2desc[lb]] for txt, lb in dset]
    with open(os.path.join(path_dset, f'{dnm}.json'), 'w') as f:
        json.dump(dsets, f, indent=4)


if __name__ == '__main__':
    from icecream import ic

    # ic(config('fine-tune'))

    # ic(fmt_num(124439808))

    # process_utcd_dataset()
    process_utcd_dataset(join=True)

    def sanity_check():
        dset = datasets.load_from_disk(
            os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed', 'UTCD')
        )['test']
        lbs = dset.features['label']
        ic(lbs)
        ic(dset[60], lbs.int2str(118))
    sanity_check()

    # map_ag_news()
