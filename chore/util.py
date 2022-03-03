from typing import Callable

from zeroshot_encoder.util import *


path_base = os.path.join(PATH_BASE, DIR_PROJ, 'chore')

_CONFIG = {
    'model-name': {'binary-bert': 'Binary BERT', 'bert-nli': 'BERT-NLI'},
    'sampling-strategy': {'rand': 'Random Negative Sampling'}
}

D_DNMS = OrderedDict([
    ('emotion', ['go_emotion', 'sentiment_tweets_2020', 'emotion']),
    ('indent', ['sgd', 'clinc_150', 'slurp']),
    ('topic', ['ag_news', 'dbpedia', 'yahoo'])
])
DNMS = sum(D_DNMS.values(), start=[])


def get_dnm2csv_path_fn(model_name: str, strategy: str) -> Callable:
    if strategy == 'rand':  # Radnom negative sampling
        if model_name == 'binary-bert':
            return (
                lambda d_nm:
                os.path.join(path_base, 'csvs', 'binary-bert-random-negative-sampling', f'binary_bert_rand_{d_nm}.csv')
            )
        elif model_name == 'bert-nli':
            return (
                lambda d_nm:
                os.path.join(path_base, 'csvs', 'bert-nli-random-negative-sampling', f'nli_bert_{d_nm}.csv')
            )
        else:
            raise ValueError('unexpected model name')
    else:
        return lambda d_nm: os.path.join(path_base, 'csvs', 'default', f'binary_bert_{d_nm}.csv')


def md_nm_n_strat2str_out(model_name: str, strategy: str, pprint=False) -> Tuple[str, str]:
    md_nm, strat = _CONFIG['model-name'][model_name], _CONFIG['sampling-strategy'][strategy]
    return f'{md_nm} with {strat}' if pprint else (md_nm, strat)


def dataset_acc_summary(dataset_names: Iterable[str], dnm2csv_path: Callable = None) -> List[Dict]:
    def get_single(d_nm):
        df = pd.read_csv(dnm2csv_path(d_nm))
        df.rename(columns={'Unnamed: 0': 'class'}, inplace=True)  # Per csv files
        df = df.iloc[-3:, :].reset_index(drop=True)
        df.support = df.support.astype(int)
        row_acc = df.iloc[0, :]
        return OrderedDict([
            ('dnm', d_nm), ('aspect', config(f'UTCD.datasets.{d_nm}.aspect')),
            ('precision', row_acc.precision), ('recall', row_acc.recall), ('f1-score', row_acc['f1-score'])
        ])
    return [get_single(d) for d in dataset_names]

