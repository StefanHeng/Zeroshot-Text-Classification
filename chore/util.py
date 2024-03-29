import os
import math
from os.path import join as os_join
from copy import deepcopy
from typing import Tuple, Dict, Iterable, Callable, Any, Union
from collections import OrderedDict

import pandas as pd

from stefutil import *
from zeroshot_classifier.util import *


__all__ = ['get_chore_base', 'ChoreConfig', 'cconfig', 'get_dnm2csv_path_fn', 'prettier_setup', 'dataset_acc']


def get_chore_base() -> str:
    return u.proj_path


class ChoreConfig:
    def __init__(
            self, shuffle: bool = 'new', lower_5ep_sanity_check: bool = False, explicit_version: int = 2,
            train_trial: str = 'default', with_arxiv: bool = False, gpt2_embed_sim: bool = True,
            new_bert_eot: bool = True, after_best_val: bool = False
    ):
        """
        :param shuffle: Fixed shuffling to standard ML training
        :param lower_5ep_sanity_check: Sanity check on 5 epochs made worse performance
        :param explicit_version: implementation of explicit
        :param train_trial: Kinds of experiments ran on training
        :param with_arxiv: Whether to include obsolete arxiv dataset

        """
        ca.check_mismatch('Shuffle Strategy', shuffle, ['ori', 'new'])
        ca.check_mismatch('Explicit Training Version', explicit_version, [1, 2])
        # on all data, on intent-only data, on data of equal aspect #sample size
        ca.check_mismatch('Train Trial', train_trial, ['default', 'intent-only', 'asp-norm'])

        if after_best_val:  # rather recent changes
            assert not lower_5ep_sanity_check
            assert explicit_version == 2
            assert not with_arxiv
            assert gpt2_embed_sim
            assert new_bert_eot
            assert train_trial == 'asp-norm'  # TODO

            setup2path = {
                # (model name, sampling strategy, training strategy, eval dataset domain, #epochs)
                ('binary-bert', 'rand', 'vanilla', 'in', '8ep'):
                    ['2022-10-11_00-57-59_Binary-BERT-vanilla-rand-aspect-norm', '22-10-13_in-domain'],
                ('binary-bert', 'rand', 'vanilla', 'out', '8ep'):
                    ['2022-10-11_00-57-59_Binary-BERT-vanilla-rand-aspect-norm', '22-10-13_out-of-domain'],
                ('binary-bert', 'rand', 'implicit', 'in', '8ep'):
                    ['2022-10-12_01-13-02_Binary-BERT-implicit-rand-aspect-norm', '22-10-13_in-domain'],
                ('binary-bert', 'rand', 'implicit', 'out', '8ep'):
                    ['2022-10-12_01-13-02_Binary-BERT-implicit-rand-aspect-norm', '22-10-13_out-of-domain'],
                ('binary-bert', 'rand', 'implicit-on-text-encode-aspect', 'in', '8ep'): [
                    '2022-10-12_01-16-59_Binary-BERT-implicit-on-text-encode-aspect-rand-aspect-norm',
                    '22-10-13_in-domain'
                ],
                ('binary-bert', 'rand', 'implicit-on-text-encode-aspect', 'out', '8ep'): [
                    '2022-10-12_01-16-59_Binary-BERT-implicit-on-text-encode-aspect-rand-aspect-norm',
                    '22-10-13_out-of-domain'
                ],
                ('binary-bert', 'rand', 'implicit-on-text-encode-sep', 'in', '8ep'): [
                    '2022-10-12_01-21-08_Binary-BERT-implicit-on-text-encode-sep-rand-aspect-norm',
                    '22-10-14_in-domain'
                ],
                ('binary-bert', 'rand', 'implicit-on-text-encode-sep', 'out', '8ep'): [
                    '2022-10-12_01-21-08_Binary-BERT-implicit-on-text-encode-sep-rand-aspect-norm',
                    '22-10-14_out-of-domain'
                ],
                ('binary-bert', 'rand', 'explicit', 'in', '8ep'):
                    ['2022-10-13_11-56-36_Binary-BERT-explicit-rand-aspect-norm', '22-10-13_in-domain'],
                ('binary-bert', 'rand', 'explicit', 'out', '8ep'):
                    ['2022-10-13_11-56-36_Binary-BERT-explicit-rand-aspect-norm', '22-10-14_out-of-domain'],

                ('bi-encoder', 'rand', 'vanilla', 'in', '8ep'): [
                    # le 2e-5, ood performance better
                    # '2022-10-09_09-10-01_Bi-Encoder-vanilla-rand-aspect-norm', '22-10-14_in-domain'
                    '2022-10-11_02-41-38_Bi-Encoder-vanilla-rand-aspect-norm', '22-10-15_in-domain'  # lr 1e-5
                ],
                ('bi-encoder', 'rand', 'vanilla', 'out', '8ep'): [
                    # '2022-10-09_09-10-01_Bi-Encoder-vanilla-rand-aspect-norm', '22-10-14_out-of-domain'
                    '2022-10-11_02-41-38_Bi-Encoder-vanilla-rand-aspect-norm', '22-10-15_out-of-domain'
                ],
                ('bi-encoder', 'rand', 'implicit', 'in', '8ep'): [
                    '2022-10-15_05-20-08_Bi-Encoder-implicit-rand-aspect-norm', '22-10-15_in-domain'
                ],
                ('bi-encoder', 'rand', 'implicit', 'out', '8ep'): [
                    '2022-10-15_05-20-08_Bi-Encoder-implicit-rand-aspect-norm', '22-10-16_out-of-domain'
                ],
                ('bi-encoder', 'rand', 'implicit-on-text-encode-aspect', 'in', '8ep'): [
                    '2022-10-15_05-22-45_Bi-Encoder-implicit-on-text-encode-aspect-rand-aspect-norm',
                    '22-10-16_in-domain'
                ],
                ('bi-encoder', 'rand', 'implicit-on-text-encode-aspect', 'out', '8ep'): [
                    '2022-10-15_05-22-45_Bi-Encoder-implicit-on-text-encode-aspect-rand-aspect-norm',
                    '22-10-16_out-of-domain'
                ],
                ('bi-encoder', 'rand', 'implicit-on-text-encode-sep', 'in', '8ep'): [
                    '2022-10-15_01-57-03_Bi-Encoder-implicit-on-text-encode-sep-rand-aspect-norm',
                    '22-10-16_in-domain'
                ],
                ('bi-encoder', 'rand', 'implicit-on-text-encode-sep', 'out', '8ep'): [
                    '2022-10-15_01-57-03_Bi-Encoder-implicit-on-text-encode-sep-rand-aspect-norm',
                    '22-10-16_out-of-domain'
                ],
                ('bi-encoder', 'rand', 'explicit', 'in', '8ep'): [
                    '2022-10-14_07-20-07_Bi-Encoder-explicit-rand-aspect-norm', '22-10-15_in-domain'
                ],
                ('bi-encoder', 'rand', 'explicit', 'out', '8ep'): [
                    '2022-10-14_07-20-07_Bi-Encoder-explicit-rand-aspect-norm', '22-10-15_out-of-domain'
                ],

                ('gpt2-nvidia', 'NA', 'vanilla', 'in', '8ep'): [
                    # 0.01 warmup
                    # '2022-10-11_23-44-51_NVIDIA-GPT2-gpt2-medium-vanilla-aspect-norm', '22-10-14_in-domain',
                    # 0.1 warmup, better ood performance
                    '2022-10-14_07-45-31_NVIDIA-GPT2-gpt2-medium-vanilla-aspect-norm', '22-10-16_in-domain'
                ],
                ('gpt2-nvidia', 'NA', 'vanilla', 'out', '8ep'): [
                    # '2022-10-11_23-44-51_NVIDIA-GPT2-gpt2-medium-vanilla-aspect-norm', '22-10-14_out-of-domain',
                    '2022-10-14_07-45-31_NVIDIA-GPT2-gpt2-medium-vanilla-aspect-norm', '22-10-16_out-of-domain'
                ],
            }
        else:

            if explicit_version == 1:
                rand_explicit_in = ['binary-bert', 'rand, explicit v2', 'in-domain, 05.25.22']
                rand_explicit_out = ['binary-bert', 'rand, explicit v2', 'out-of-domain, 05.25.22']
            else:  # v1
                rand_explicit_in = ['binary-bert', 'rand, explicit', 'in-domain, 05.13.22']
                rand_explicit_out = ['binary-bert', 'rand, explicit', 'out-of-domain, 05.13.22']

            rand_vanilla_in = [
                'binary-bert', 'rand, vanilla',
                'in-domain, 05.03.22' if lower_5ep_sanity_check else 'in-domain, 03.24.22'
            ]
            rand_vanilla_out = [
                'binary-bert', 'rand, vanilla',
                'out-of-domain, 05.03.22' if lower_5ep_sanity_check else 'out-of-domain, 04.06.22'
            ]
            rand_implicit_sep_in = ['binary-bert', 'rand, implicit-sep', 'in-domain, 04.21.22']
            rand_implicit_sep_out = ['binary-bert', 'rand, implicit-sep', 'out-of-domain, 04.21.22']
            be_rand_vanilla_in = ['bi-encoder', 'rand, vanilla', 'in-domain, 03.26.22']
            be_rand_vanilla_out = ['bi-encoder', 'rand, vanilla', 'out-of-domain, 04.06.22']
            be_rand_implicit_sep_in, be_rand_implicit_sep_out = None, None
            be_rand_explicit_in, be_rand_explicit_out = None, None
            gp_vanilla_in = ['gpt2-nvidia', 'vanilla', 'in-domain, 2022-04-06_23-13-55']
            gp_vanilla_out = ['gpt2-nvidia', 'vanilla', 'out-of-domain, 2022-04-06_23-43-19']
            gp_implicit_sep_in, gp_implicit_sep_out = None, None
            gp_explicit_in, gp_explicit_out = None, None
            seq_vanilla_in, seq_vanilla_out = None, None
            if train_trial == 'intent-only':
                rand_vanilla_in = ['binary-bert', 'rand, vanilla, intent-only', 'in-domain, 06.02.22']
                rand_vanilla_out = ['binary-bert', 'rand, vanilla, intent-only', 'out-of-domain, 06.02.22']
                rand_implicit_sep_in = ['binary-bert', 'rand, implicit-sep, intent-only', 'in-domain, 06.02.22']
                rand_implicit_sep_out = ['binary-bert', 'rand, implicit-sep, intent-only', 'out-of-domain, 06.02.22']
            if train_trial == 'asp-norm':
                seq_vanilla_in = ['bert-seq-cls', 'vanilla', 'in-domain, 06.13.22']
                seq_vanilla_out = ['bert-seq-cls', 'vanilla', 'out-of-domain, 06.14.22']

                assert explicit_version == 2
                rand_vanilla_in = ['binary-bert', 'rand, vanilla, asp-norm', 'in-domain, 06.04.22']
                rand_vanilla_out = ['binary-bert', 'rand, vanilla, asp-norm', 'out-of-domain, 06.04.22']
                rand_implicit_sep_in = ['binary-bert', 'rand, implicit-sep, asp-norm', 'in-domain, 06.04.22']
                rand_implicit_sep_out = ['binary-bert', 'rand, implicit-sep, asp-norm', 'out-of-domain, 06.04.22']
                rand_explicit_in = ['binary-bert', 'rand, explicit, asp-norm', 'in-domain, 06.06.22']
                rand_explicit_out = ['binary-bert', 'rand, explicit, asp-norm', 'out-of-domain, 06.06.22']
                if new_bert_eot:
                    rand_vanilla_in[-1] = 'in-domain, 06.15.22'
                    rand_vanilla_out[-1] = 'out-of-domain, 06.15.22'
                    rand_implicit_sep_in[-1] = 'in-domain, 06.15.22'
                    rand_implicit_sep_out[-1] = 'out-of-domain, 06.15.22'
                    rand_explicit_in[-1] = 'in-domain, 06.15.22'
                    rand_explicit_out[-1] = 'out-of-domain, 06.15.22'

                be_rand_vanilla_in = ['bi-encoder', 'rand, vanilla, asp-norm', 'in-domain, 06.09.22']
                be_rand_vanilla_out = ['bi-encoder', 'rand, vanilla, asp-norm', 'out-of-domain, 06.09.22']
                be_rand_implicit_sep_in = ['bi-encoder', 'rand, implicit-sep, asp-norm', 'in-domain, 06.10.22']
                be_rand_implicit_sep_out = ['bi-encoder', 'rand, implicit-sep, asp-norm', 'out-of-domain, 06.10.22']
                be_rand_explicit_in = ['bi-encoder', 'rand, explicit, asp-norm', 'in-domain, 06.10.22']
                be_rand_explicit_out = ['bi-encoder', 'rand, explicit, asp-norm', 'out-of-domain, 06.10.22']

                gp_vanilla_in = ['gpt2-nvidia', 'vanilla, asp-norm', 'in-domain, 06.10.22']
                gp_vanilla_out = ['gpt2-nvidia', 'vanilla, asp-norm', 'out-of-domain, 06.10.22']
                gp_implicit_sep_in = ['gpt2-nvidia', 'implicit-sep, asp-norm', 'in-domain, 06.13.22']
                gp_implicit_sep_out = ['gpt2-nvidia', 'implicit-sep, asp-norm', 'out-of-domain, 06.13.22']
                gp_explicit_in = ['gpt2-nvidia', 'explicit, asp-norm', 'in-domain, 06.14.22']
                gp_explicit_out = ['gpt2-nvidia', 'explicit, asp-norm', 'out-of-domain, 06.14.22']

                if gpt2_embed_sim:
                    gp_vanilla_in = ['gpt2-nvidia', 'vanilla, asp-norm, emb-sim', 'in-domain, 06.21.22']
                    gp_vanilla_out = ['gpt2-nvidia', 'vanilla, asp-norm, emb-sim', 'out-of-domain, 06.21.22']
                    gp_implicit_sep_in = ['gpt2-nvidia', 'implicit, asp-norm, emb-sim', 'in-domain, 06.21.22']
                    gp_implicit_sep_out = ['gpt2-nvidia', 'implicit, asp-norm, emb-sim', 'out-of-domain, 06.21.22']
                    gp_explicit_in = ['gpt2-nvidia', 'explicit, asp-norm, emb-sim', 'in-domain, 06.21.22']
                    gp_explicit_out = ['gpt2-nvidia', 'explicit, asp-norm, emb-sim', 'out-of-domain, 06.21.22']
            setup2path = OrderedDict({
                ('binary-bert', 'rand', 'vanilla', 'in', '3ep'): rand_vanilla_in,
                ('binary-bert', 'rand', 'vanilla', 'in', '5ep'):
                    ['binary-bert', f'rand, vanilla, 5ep, {shuffle}-shuffle', 'in-domain, 04.26.22'],
                ('binary-bert', 'rand', 'vanilla', 'out', '3ep'): rand_vanilla_out,
                ('binary-bert', 'rand', 'vanilla', 'out', '5ep'):
                    ['binary-bert',  f'rand, vanilla, 5ep, {shuffle}-shuffle', 'out-of-domain, 04.26.22'],
                ('binary-bert', 'rand', 'implicit', 'in', '3ep'):
                    ['binary-bert', 'rand, implicit', 'in-domain, 04.09.22'],
                ('binary-bert', 'rand', 'implicit', 'in', '5ep'):
                    ['binary-bert', 'rand, implicit, 5ep', 'in-domain, 04.25.22'],
                ('binary-bert', 'rand', 'implicit', 'out', '3ep'):
                    ['binary-bert', 'rand, implicit', 'out-of-domain, 04.11.22'],
                ('binary-bert', 'rand', 'implicit', 'out', '5ep'):
                    ['binary-bert', 'rand, implicit, 5ep', 'out-of-domain, 04.25.22'],
                ('binary-bert', 'rand', 'implicit-on-text-encode-aspect', 'in', '3ep'):
                    ['binary-bert', 'rand, implicit-text', 'in-domain, 04.21.22'],
                ('binary-bert', 'rand', 'implicit-on-text-encode-aspect', 'in', '5ep'):
                    ['binary-bert', 'rand, implicit-text, 5ep', 'in-domain, 04.25.22'],
                ('binary-bert', 'rand', 'implicit-on-text-encode-aspect', 'out', '3ep'):
                    ['binary-bert', 'rand, implicit-text', 'out-of-domain, 04.21.22'],
                ('binary-bert', 'rand', 'implicit-on-text-encode-aspect', 'out', '5ep'):
                    ['binary-bert', 'rand, implicit-text, 5ep', 'out-of-domain, 04.25.22'],
                ('binary-bert', 'rand', 'implicit-on-text-encode-sep', 'in', '3ep'):
                    rand_implicit_sep_in,
                ('binary-bert', 'rand', 'implicit-on-text-encode-sep', 'in', '5ep'):
                    ['binary-bert', 'rand, implicit-sep, 5ep', 'in-domain, 04.25.22'],
                ('binary-bert', 'rand', 'implicit-on-text-encode-sep', 'out', '3ep'):
                    rand_implicit_sep_out,
                ('binary-bert', 'rand', 'implicit-on-text-encode-sep', 'out', '5ep'):
                    ['binary-bert', 'rand, implicit-sep, 5ep', 'out-of-domain, 04.25.22'],
                ('binary-bert', 'rand', 'explicit', 'in', '3ep'): rand_explicit_in,
                ('binary-bert', 'rand', 'explicit', 'out', '3ep'): rand_explicit_out,
                ('binary-bert', 'vect', 'vanilla', 'in', '3ep'):
                    ['binary-bert', 'vect, vanilla', 'in-domain, 03.05.22'],

                ('bert-nli', 'rand', 'vanilla', 'in', '3ep'): ['bert-nli', 'rand, vanilla', 'in-domain, 03.24.22'],
                ('bert-nli', 'rand', 'vanilla', 'out', '3ep'): ['bert-nli', 'rand, vanilla', 'out-of-domain, 04.06.22'],
                ('bert-nli', 'rand', 'implicit', 'in', '3ep'): ['bert-nli', 'rand, implicit', 'in-domain, 04.09.22'],
                ('bert-nli', 'rand', 'implicit', 'out', '3ep'):
                    ['bert-nli', 'rand, implicit', 'out-of-domain, 04.11.22'],
                ('bert-nli', 'vect', 'vanilla', 'in', '3ep'): ['bert-nli', 'vect, vanilla', 'in-domain, 03.05.22'],
                ('bert-nli', 'vect', 'vanilla', 'out', '3ep'): ['bert-nli', 'vect, vanilla', 'out-of-domain, 03.09.22'],

                ('bi-encoder', 'rand', 'vanilla', 'in', '3ep'): be_rand_vanilla_in,
                ('bi-encoder', 'rand', 'vanilla', 'out', '3ep'): be_rand_vanilla_out,
                ('bi-encoder', 'rand', 'implicit', 'in', '3ep'):
                    ['bi-encoder', 'rand, implicit', 'in-domain, 04.09.22'],
                ('bi-encoder', 'rand', 'implicit', 'out', '3ep'): [
                    'bi-encoder', 'rand, implicit', 'out-of-domain, 04.11.22'],
                ('bi-encoder', 'rand', 'implicit-on-text-encode-sep', 'in', '3ep'): be_rand_implicit_sep_in,
                ('bi-encoder', 'rand', 'implicit-on-text-encode-sep', 'out', '3ep'): be_rand_implicit_sep_out,
                ('bi-encoder', 'rand', 'explicit', 'in', '3ep'): be_rand_explicit_in,
                ('bi-encoder', 'rand', 'explicit', 'out', '3ep'): be_rand_explicit_out,
                ('bi-encoder', 'vect', 'vanilla', 'in', '3ep'): ['bi-encoder', 'vect, vanilla', 'in-domain, 03.07.22'],

                ('dual-bi-encoder', 'none', 'vanilla', 'in', '3ep'): [
                    'dual-bi-encoder', 'none, vanilla', 'in-domain, 03.23.22'],
                ('dual-bi-encoder', 'none', 'vanilla', 'out', '3ep'): [
                    'dual-bi-encoder', 'none, vanilla', 'out-of-domain, 03.23.22'],

                ('gpt2-nvidia', 'NA', 'vanilla', 'in', '3ep'): gp_vanilla_in,
                ('gpt2-nvidia', 'NA', 'vanilla', 'out', '3ep'): gp_vanilla_out,
                ('gpt2-nvidia', 'NA', 'implicit-on-text-encode-sep', 'in', '3ep'): gp_implicit_sep_in,
                ('gpt2-nvidia', 'NA', 'implicit-on-text-encode-sep', 'out', '3ep'): gp_implicit_sep_out,
                ('gpt2-nvidia', 'NA', 'explicit', 'in', '3ep'): gp_explicit_in,
                ('gpt2-nvidia', 'NA', 'explicit', 'out', '3ep'): gp_explicit_out,

                ('bert-seq-cls', 'NA', 'vanilla', 'in', '3ep'): seq_vanilla_in,
                ('bert-seq-cls', 'NA', 'vanilla', 'out', '3ep'):  seq_vanilla_out
            })
        d_dset_names = {
            'in': OrderedDict([
                ('sentiment', ['go_emotion', 'sentiment_tweets_2020', 'emotion']),
                ('intent', ['sgd', 'clinc_150', 'slurp']),
                ('topic', ['ag_news', 'dbpedia', 'yahoo'])
            ]),
            'out': OrderedDict([
                ('sentiment', ['amazon_polarity', 'finance_sentiment', 'yelp']),
                ('intent', ['banking77', 'snips', 'nlu_evaluation']),
                ('topic', ['multi_eurlex', 'patent', 'consumer_finance'])
            ])
        }
        d_dset_names_all = deepcopy(d_dset_names)
        # intended for writing out table, will write all the data
        if with_arxiv:
            d_dset_names_all['out']['topic'].insert(0, 'arxiv')  # the only difference
        self.config_dict = {
            'domain2aspect2dataset-names': d_dset_names,
            'domain2aspect2dataset-names-all': d_dset_names_all,
            'domain2dataset-names': {  # syntactic sugar
                'in': sum(d_dset_names['in'].values(), start=[]),
                'out': sum(d_dset_names['out'].values(), start=[])
            },
            'domain2dataset-names-all': {
                'in': sum(d_dset_names_all['in'].values(), start=[]),
                'out': sum(d_dset_names_all['out'].values(), start=[])
            },
            'train-setup2dset-eval-path': setup2path,
            'pretty': {
                'model-name': {
                    'binary-bert': 'Binary BERT',
                    'bert-nli': 'BERT-NLI',
                    'bi-encoder': 'Bi-Encoder',
                    'dual-bi-encoder': 'Dual Bi-Encoder',
                    'gpt2-nvidia': 'GPT2-NVIDIA'
                },
                'sampling-strategy': dict(
                    rand='Random Negative Sampling',
                    vect='Word2Vec Average Extremes',
                    none='Positive Labels Only',
                    NA='NA'
                ),
                'training-strategy': {
                    'vanilla': 'Vanilla',
                    'implicit': 'Implicit Labels',
                    'implicit-on-text-encode-aspect': 'Implicit Text with Aspect token',
                    'implicit-on-text-encode-sep': 'Implicit Text with Sep token',
                    'explicit': 'Explicit Loss',
                }
            }
        }

    def __call__(self, key: str):
        return get(self.config_dict, key)

    def __repr__(self):
        return self.config_dict.__repr__()


cconfig = ChoreConfig()


def get_dnm2csv_path_fn(
        model_name: str, sampling_strategy: str = 'rand', training_strategy: str = 'vanilla',
        domain: str = 'in', train_description: str = '3ep', chore_config: ChoreConfig = cconfig
) -> Callable:
    ca(
        model_name=model_name, dataset_domain=domain,
        sampling_strategy=sampling_strategy, training_strategy=training_strategy
    )
    paths = [u.eval_path]
    key = model_name, sampling_strategy, training_strategy, domain, train_description
    _paths = chore_config('train-setup2dset-eval-path')[key]
    if isinstance(_paths, list):
        paths += _paths
    else:
        paths.append(_paths)
    path = os_join(*paths)
    assert os.path.exists(path)  # sanity check
    return lambda d_nm: os_join(path, f'{d_nm}.csv')


def prettier_setup(
        model_name: str, sampling_strategy: str = None, training_strategy: str = None, pprint=True,
        newline: bool = False, chore_config: ChoreConfig = cconfig
) -> Union[Tuple[str, str], str]:
    ca(model_name=model_name)
    if sampling_strategy:
        ca(sampling_strategy=sampling_strategy)
    if training_strategy:
        ca(training_strategy=training_strategy)
    md_nm = chore_config('pretty.model-name')[model_name]
    samp_strat = chore_config('pretty.sampling-strategy')[sampling_strategy] if sampling_strategy else None
    train_strat = chore_config('pretty.training-strategy')[training_strategy] if training_strategy else None
    if pprint:
        ret = md_nm
        if newline:
            pref_1st, pref_later = '\n  w/ ', '\n  & '
        else:
            pref_1st, pref_later = ' w/ ', ' & '
        gone_1st = False
        if samp_strat and samp_strat != 'NA':
            pref = pref_later if gone_1st else pref_1st
            ret = f'{ret}{pref}{samp_strat}'
            gone_1st = True
        if train_strat:
            pref = pref_later if gone_1st else pref_1st
            ret = f'{ret}{pref}{train_strat} training'
            # gone_1st = True
        return ret
    else:
        ret = md_nm, samp_strat
        if training_strategy:
            ret = ret + (training_strategy,)
        return ret


def dataset_acc(
        dataset_names: Union[Iterable[str], str], dnm2csv_path: Callable = None,
        suppress_not_found: Union[bool, Any] = False, return_type: str = 'dict'
) -> Union[Dict[str, Union[float, Any]], float]:
    def get_single(dnm) -> Union[float, Any]:
        path = dnm2csv_path(dnm)
        if not os.path.exists(path) and suppress_not_found:
            return None
        df = pd.read_csv(path)
        df.rename(columns={'Unnamed: 0': 'class'}, inplace=True)  # Per csv files
        df = df.iloc[-3:, :].reset_index(drop=True)
        df.support = df.support.astype(int)
        acc_row = df.iloc[0, :]  # The accuracy row
        acc = acc_row['f1-score']
        assert all(math.isclose(a, acc, abs_tol=1e-8) for a in (acc_row.precision, acc_row.recall))
        return acc
    if isinstance(dataset_names, str):
        return get_single(dataset_names)
    else:
        ret_types = ['dict', 'list']
        ca.check_mismatch('Return Type', return_type, ret_types)
        accs = [get_single(d) for d in dataset_names]
        return dict(zip(dataset_names, accs)) if return_type == 'dict' else accs


def setup2accs(dir_name: str = None, domain_dir_nm: str = None) -> Dict[str, float]:
    path = os_join(u.eval_path, dir_name, domain_dir_nm)
    dnms = [dnm for dnm, d in sconfig('UTCD.datasets').items() if d['domain'] == 'out']  # TODO
    return dataset_acc(dataset_names=dnms, dnm2csv_path=lambda dnm: os_join(path, f'{dnm}.csv'))


if __name__ == '__main__':
    mic.output_width = 512

    # mic(cconfig)

    def check_setups():
        # setups = [
        #     '2022-11-06_14-17-06_Binary-BERT_vanilla_rand_asp-norm_{a=3e-5}',
        #     '2022-11-06_14-22-17_Binary-BERT_vanilla_rand_asp-norm_{a=4e-5}',
        #     '2022-11-06_15-34-21_Bi-Encoder_vanilla_rand_asp-norm_{a=3e-5}',
        #     '2022-11-06_15-37-20_Bi-Encoder_vanilla_rand_asp-norm_{a=4e-5}'
        # ]
        # setups = [
        #     dict(dir_name='2022-11-08_20-20-53_Binary-BERT_vanilla_rand_asp-norm_{a=1e-5_m=a}', domain_dir_nm='22-11-13_out-of-domain'),
        #     dict(dir_name='2022-11-11_21-29-22_Binary-BERT_vanilla_rand_asp-norm_{a=2e-5_m=a}', domain_dir_nm='22-11-13_out-of-domain'),
        #     dict(dir_name='2022-11-11_21-34-47_Binary-BERT_vanilla_rand_asp-norm_{a=3e-5_m=a}', domain_dir_nm='22-11-14_out-of-domain'),
        #     dict(dir_name='2022-11-11_21-36-29_Binary-BERT_vanilla_rand_asp-norm_{a=4e-5_m=a}', domain_dir_nm='22-11-14_out-of-domain')
        # ]
        # setups = [
        #     dict(dir_name='2022-11-14_15-43-37_Bi-Encoder_vanilla_rand_asp-norm_{a=1e-5_m=a}', domain_dir_nm='22-11-16_out-of-domain'),
        #     dict(dir_name='2022-11-15_11-00-49_Bi-Encoder_vanilla_rand_asp-norm_{a=2e-5_m=a}', domain_dir_nm='22-11-16_out-of-domain'),
        #     dict(dir_name='2022-11-15_12-23-27_Bi-Encoder_vanilla_rand_asp-norm_{a=3e-5_m=a}', domain_dir_nm='22-11-16_out-of-domain'),
        #     dict(dir_name='2022-11-15_12-25-42_Bi-Encoder_vanilla_rand_asp-norm_{a=4e-5_m=a}', domain_dir_nm='22-11-16_out-of-domain')
        # ]
        # setups = [
        #     # dict(dir_name='2022-11-16_16-40-22_Binary-BERT_vanilla_rand_asp-norm_{a=2e-5}', domain_dir_nm='22-11-17_out-of-domain'),
        #     dict(dir_name='2022-11-17_14-51-46_Binary-BERT_implicit_rand_asp-norm_{a=1e-5}', domain_dir_nm='22-11-18_out-of-domain'),
        #     dict(dir_name='2022-11-17_14-58-43_Binary-BERT_implicit_rand_asp-norm_{a=2e-5}', domain_dir_nm='22-11-18_out-of-domain'),
        #     dict(dir_name='2022-11-17_15-02-28_Binary-BERT_implicit_rand_asp-norm_{a=3e-5}', domain_dir_nm='22-11-18_out-of-domain'),
        #     dict(dir_name='2022-11-17_15-07-09_Binary-BERT_implicit_rand_asp-norm_{a=4e-5}', domain_dir_nm='22-11-18_out-of-domain')
        # ]
        # setups = [
        #     dict(dir_name='2022-11-18_18-39-05_Binary-BERT_{md=i-o-t-e-a, sp=r, na=T}_{a=2e-5}', domain_dir_nm='22-11-20_out-of-domain'),
        #     dict(dir_name='2022-11-18_18-43-37_Binary-BERT_{md=i-o-t-e-a, sp=r, na=T}_{a=3e-5}', domain_dir_nm='22-11-20_out-of-domain'),
        #     dict(dir_name='2022-11-18_18-46-31_Binary-BERT_{md=i-o-t-e-a, sp=r, na=T}_{a=4e-5}', domain_dir_nm='22-11-20_out-of-domain'),
        #     dict(dir_name='2022-11-18_18-49-47_Binary-BERT_{md=imp-sep, sp=r, na=T}_{a=2e-5}', domain_dir_nm='22-11-20_out-of-domain'),
        #     dict(dir_name='2022-11-18_18-51-58_Binary-BERT_{md=imp-sep, sp=r, na=T}_{a=3e-5}', domain_dir_nm='22-11-20_out-of-domain'),
        # ]
        # setups = [
        #     dict(dir_name='2022-11-20_21-06-56_Binary-BERT_{md=imp-sep, sp=r, na=T}_{a=4e-5}', domain_dir_nm='22-11-21_out-of-domain'),
        #     dict(dir_name='2022-11-20_21-14-11_Bi-Encoder_{md=imp, sp=r, na=T}_{a=2e-5}', domain_dir_nm='22-11-21_out-of-domain'),
        #     dict(dir_name='2022-11-20_21-15-41_Bi-Encoder_{md=imp, sp=r, na=T}_{a=3e-5}', domain_dir_nm='22-11-21_out-of-domain'),
        #     dict(dir_name='2022-11-20_21-17-50_Bi-Encoder_{md=imp, sp=r, na=T}_{a=4e-5}', domain_dir_nm='22-11-21_out-of-domain'),
        #     dict(dir_name='2022-11-20_21-20-23_Bi-Encoder_{md=imp-asp, sp=r, na=T}_{a=2e-5}', domain_dir_nm='22-11-21_out-of-domain'),
        # ]
        # setups = [
        #     dict(dir_name='2022-11-22_01-26-10_Binary-BERT_{md=exp, sp=r, na=T}_{a=1e-5}', domain_dir_nm='22-11-23_out-of-domain'),
        #     dict(dir_name='2022-11-22_01-27-20_Binary-BERT_{md=exp, sp=r, na=T}_{a=2e-5}', domain_dir_nm='22-11-23_out-of-domain'),
        #     dict(dir_name='2022-11-22_01-28-56_Binary-BERT_{md=exp, sp=r, na=T}_{a=3e-5}', domain_dir_nm='22-11-23_out-of-domain'),
        #     dict(dir_name='2022-11-22_01-30-18_Binary-BERT_{md=exp, sp=r, na=T}_{a=4e-5}', domain_dir_nm='22-11-23_out-of-domain'),
        #     dict(dir_name='2022-11-21_19-04-32_Bi-Encoder_{md=imp-asp, sp=r, na=T}_{a=3e-5}', domain_dir_nm='22-11-23_out-of-domain'),
        # ]
        # setups = [
        #     dict(dir_name='2022-11-24_00-04-11_Bi-Encoder_{md=imp-asp, sp=r, na=T}_{a=4e-5}', domain_dir_nm='22-11-24_out-of-domain'),
        #     dict(dir_name='2022-11-24_00-06-55_Bi-Encoder_{md=imp-sep, sp=r, na=T}_{a=2e-5}', domain_dir_nm='22-11-24_out-of-domain'),
        #     dict(dir_name='2022-11-24_00-08-00_Bi-Encoder_{md=imp-sep, sp=r, na=T}_{a=3e-5}', domain_dir_nm='22-11-24_out-of-domain'),
        #     dict(dir_name='2022-11-24_00-09-33_Bi-Encoder_{md=imp-sep, sp=r, na=T}_{a=4e-5}', domain_dir_nm='22-11-24_out-of-domain')
        # ]
        # setups = [
        #     dict(dir_name='2022-11-25_22-01-07_Bi-Encoder_{md=exp, sp=r, na=T}_{a=1e-5}', domain_dir_nm='22-11-26_out-of-domain'),
        #     dict(dir_name='2022-11-25_22-02-38_Bi-Encoder_{md=exp, sp=r, na=T}_{a=2e-5}', domain_dir_nm='22-11-26_out-of-domain'),
        #     dict(dir_name='2022-11-25_22-04-46_Bi-Encoder_{md=exp, sp=r, na=T}_{a=3e-5}', domain_dir_nm='22-11-26_out-of-domain'),
        #     dict(dir_name='2022-11-25_22-07-17_Bi-Encoder_{md=exp, sp=r, na=T}_{a=4e-5}', domain_dir_nm='22-11-27_out-of-domain')
        # ]
        # setups = [
        #     dict(dir_name='2022-11-29_12-12-56_NVIDIA-GPT2_{md=van, na=T}_{a=1e-05}', domain_dir_nm='22-12-01_out-of-domain'),
        #     dict(dir_name='2022-11-29_19-15-44_NVIDIA-GPT2_{md=van, na=T}_{a=2e-05}', domain_dir_nm='22-12-01_out-of-domain'),
        #     dict(dir_name='2022-11-29_19-37-13_NVIDIA-GPT2_{md=van, na=T}_{a=3e-05}', domain_dir_nm='22-12-01_out-of-domain'),
        #     dict(dir_name='2022-11-29_19-43-32_NVIDIA-GPT2_{md=van, na=T}_{a=4e-05}', domain_dir_nm='22-12-01_out-of-domain')
        # ]
        # setups = [
        #     dict(dir_name='2022-12-03_10-43-47_NVIDIA-GPT2_{md=imp, na=T}_{a=1e-05}', domain_dir_nm='22-12-05_out-of-domain'),
        #     dict(dir_name='2022-12-03_14-47-52_NVIDIA-GPT2_{md=imp, na=T}_{a=2e-05}', domain_dir_nm='22-12-05_out-of-domain'),
        #     dict(dir_name='2022-12-03_15-03-14_NVIDIA-GPT2_{md=imp, na=T}_{a=3e-05}', domain_dir_nm='22-12-05_out-of-domain'),
        #     dict(dir_name='2022-12-02_21-33-18_NVIDIA-GPT2_{md=imp, na=T}_{a=4e-05}', domain_dir_nm='22-12-05_out-of-domain')
        # ]
        setups = [
            dict(dir_name='2022-12-05_16-13-20_NVIDIA-GPT2_{md=exp, na=T}_{a=1e-05}', domain_dir_nm='22-12-07_out-of-domain'),
            dict(dir_name='2022-12-05_16-25-57_NVIDIA-GPT2_{md=exp, na=T}_{a=2e-05}', domain_dir_nm='22-12-07_out-of-domain'),
            dict(dir_name='2022-12-05_16-52-24_NVIDIA-GPT2_{md=exp, na=T}_{a=3e-05}', domain_dir_nm='22-12-07_out-of-domain'),
            dict(dir_name='2022-12-05_17-12-33_NVIDIA-GPT2_{md=exp, na=T}_{a=4e-05}', domain_dir_nm='22-12-07_out-of-domain')
        ]

        rows = []
        for s in setups:
            # accs = setup2accs(dir_name=s, domain_dir_nm='22-11-07_out-of-domain')
            accs = setup2accs(**s)
            accs['avg'] = sum(accs.values()) / len(accs)
            accs = {k: round(v * 100, 2) for k, v in accs.items()}
            rows.append(dict(dnm=s, **accs))
        df = pd.DataFrame(rows)
        mic(df)
    check_setups()
