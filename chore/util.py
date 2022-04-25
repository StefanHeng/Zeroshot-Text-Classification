from copy import deepcopy
from typing import Any

from zeroshot_encoder.util import *


def get_chore_base() -> str:
    return os.path.join(BASE_PATH, PROJ_DIR, 'chore')


class ChoreConfig:
    def __init__(self):
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
            'train-setup2dset-eval-path': OrderedDict([
                (('binary-bert', 'rand', 'vanilla', 'in'), ['binary-bert', 'rand, vanilla', 'in-domain, 03.24.22']),
                (('binary-bert', 'rand', 'vanilla', 'out'), [
                    'binary-bert', 'rand, vanilla', 'out-of-domain, 04.06.22']),
                (('binary-bert', 'rand', 'implicit', 'in'), ['binary-bert', 'rand, implicit', 'in-domain, 04.09.22']),
                (('binary-bert', 'rand', 'implicit', 'out'), [
                    'binary-bert', 'rand, implicit', 'out-of-domain, 04.11.22']),
                (('binary-bert', 'rand', 'implicit-on-text-encode-aspect', 'in'), [
                    'binary-bert', 'rand, implicit-text-aspect', 'in-domain, 04.21.22']),
                (('binary-bert', 'rand', 'implicit-on-text-encode-aspect', 'out'), [
                    'binary-bert', 'rand, implicit-text-aspect', 'out-of-domain, 04.21.22']),
                (('binary-bert', 'rand', 'implicit-on-text-encode-sep', 'in'), [
                    'binary-bert', 'rand, implicit-text-sep', 'in-domain, 04.21.22']),
                (('binary-bert', 'rand', 'implicit-on-text-encode-sep', 'out'), [
                    'binary-bert', 'rand, implicit-text-sep', 'out-of-domain, 04.21.22']),
                (('binary-bert', 'vect', 'vanilla', 'in'), ['binary-bert', 'vect, vanilla', 'in-domain, 03.05.22']),

                (('bert-nli', 'rand', 'vanilla', 'in'), ['bert-nli', 'rand, vanilla', 'in-domain, 03.24.22']),
                (('bert-nli', 'rand', 'vanilla', 'out'), ['bert-nli', 'rand, vanilla', 'out-of-domain, 04.06.22']),
                (('bert-nli', 'rand', 'implicit', 'in'), ['bert-nli', 'rand, implicit', 'in-domain, 04.09.22']),
                (('bert-nli', 'rand', 'implicit', 'out'), ['bert-nli', 'rand, implicit', 'out-of-domain, 04.11.22']),
                (('bert-nli', 'vect', 'vanilla', 'in'), ['bert-nli', 'vect, vanilla', 'in-domain, 03.05.22']),
                (('bert-nli', 'vect', 'vanilla', 'out'), ['bert-nli', 'vect, vanilla', 'out-of-domain, 03.09.22']),

                (('bi-encoder', 'rand', 'vanilla', 'in'), ['bi-encoder', 'rand, vanilla', 'in-domain, 03.26.22']),
                (('bi-encoder', 'rand', 'vanilla', 'out'), ['bi-encoder', 'rand, vanilla', 'out-of-domain, 04.06.22']),
                (('bi-encoder', 'rand', 'implicit', 'in'), ['bi-encoder', 'rand, implicit', 'in-domain, 04.09.22']),
                (('bi-encoder', 'rand', 'implicit', 'out'), ['bi-encoder', 'rand, implicit', 'out-of-domain, 04.11.22']),
                (('bi-encoder', 'vect', 'vanilla', 'in'), ['bi-encoder', 'vect, vanilla', 'in-domain, 03.07.22']),

                (('dual-bi-encoder', 'none', 'vanilla', 'in'), [
                    'dual-bi-encoder', 'none, vanilla', 'in-domain, 03.23.22']),
                (('dual-bi-encoder', 'none', 'vanilla', 'out'), [
                    'dual-bi-encoder', 'none, vanilla', 'out-of-domain, 03.23.22']),

                (('gpt2-nvidia', 'NA', 'vanilla', 'in'), [
                    'gpt2-nvidia', 'NA, vanilla', 'in-domain, 2022-04-06_23-13-55']),
                (('gpt2-nvidia', 'NA', 'vanilla', 'out'), [
                    'gpt2-nvidia', 'NA, vanilla', 'out-of-domain, 2022-04-06_23-43-19'])
            ]),
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
        domain: str = 'in'
) -> Callable:
    ca(model_name=model_name, domain=domain, sampling_strategy=sampling_strategy, training_strategy=training_strategy)
    paths = [BASE_PATH, PROJ_DIR, 'evaluations']
    paths += cconfig('train-setup2dset-eval-path')[(model_name, sampling_strategy, training_strategy, domain)]
    return lambda d_nm: os.path.join(*paths, f'{d_nm}.csv')


def prettier_setup(
        model_name: str, sampling_strategy: str = None, training_strategy: str = None, pprint=True
) -> Union[Tuple[str, str], str]:
    ca(model_name=model_name)
    if sampling_strategy:
        ca(sampling_strategy=sampling_strategy)
    if training_strategy:
        ca(training_strategy=training_strategy)
    md_nm = cconfig('pretty.model-name')[model_name]
    samp_strat = cconfig('pretty.sampling-strategy')[sampling_strategy] if sampling_strategy else None
    train_strat = cconfig('pretty.training-strategy')[training_strategy] if training_strategy else None
    if pprint:
        ret = md_nm
        pref_1st, pref_later = '\n  w/ ', '\n  & '
        gone_1st = False
        if samp_strat and samp_strat != 'NA':
            pref = pref_later if gone_1st else pref_1st
            ret = f'{ret}{pref}{samp_strat}'
            gone_1st = True
        if train_strat:
            pref = pref_later if gone_1st else pref_1st
            ret = f'{ret}{pref}{train_strat} training'
            gone_1st = True
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
        # ic(dnm, dnm2csv_path(dnm))
        path = dnm2csv_path(dnm)
        if not os.path.exists(path) and suppress_not_found:
            return None
        df = pd.read_csv(path)
        df.rename(columns={'Unnamed: 0': 'class'}, inplace=True)  # Per csv files
        df = df.iloc[-3:, :].reset_index(drop=True)
        df.support = df.support.astype(int)
        acc_row = df.iloc[0, :]  # The accuracy row
        acc = acc_row['f1-score']
        assert acc_row.precision == acc_row.recall == acc
        return acc
    if isinstance(dataset_names, str):
        return get_single(dataset_names)
    else:
        ret_types = ['dict', 'list']
        ca.check_mismatch('Return Type', return_type, ret_types)
        accs = [get_single(d) for d in dataset_names]
        return dict(zip(dataset_names, accs)) if return_type == 'dict' else accs


if __name__ == '__main__':
    from icecream import ic

    ic.lineWrapWidth = 150

    ic(cconfig)
