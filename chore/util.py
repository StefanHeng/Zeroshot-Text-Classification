from typing import Callable

from zeroshot_encoder.util import *


PATH_BASE_CHORE = os.path.join(PATH_BASE, DIR_PROJ, 'chore')

_CONFIG = {
    'model-name': {
        'binary-bert': 'Binary BERT',
        'bert-nli': 'BERT-NLI',
        'bi-encoder': 'Bi-encoder'
    },
    'sampling-strategy': dict(
        rand='Random Negative Sampling', vect='Word2Vec Average Extremes'
    )
}

D_DNMS = OrderedDict([
    ('emotion', ['go_emotion', 'sentiment_tweets_2020', 'emotion']),
    ('intent', ['sgd', 'clinc_150', 'slurp']),
    ('topic', ['ag_news', 'dbpedia', 'yahoo'])
])
DNMS = sum(D_DNMS.values(), start=[])


def get_dnm2csv_path_fn(model_name: str, strategy: str) -> Callable:
    paths = [PATH_BASE, DIR_PROJ, 'evaluations']
    if model_name == 'binary-bert':
        paths.append('binary_bert')
    elif model_name == 'bert-nli':
        paths.append('nli_bert')
    elif model_name == 'bi-encoder':
        paths.append('bi-encoder')
    else:
        raise ValueError('unexpected model name')
    assert strategy in ['rand', 'vect']  # # Radnom negative sampling; word2vec average label selection
    paths.append(strategy)
    return lambda d_nm: os.path.join(*paths, 'results', f'{d_nm}.csv')


def md_nm_n_strat2str_out(model_name: str, strategy: str, pprint=False) -> Tuple[str, str]:
    md_nm, strat = _CONFIG['model-name'][model_name], _CONFIG['sampling-strategy'][strategy]
    return f'{md_nm} with {strat}' if pprint else (md_nm, strat)


def dataset_acc_summary(dataset_names: Iterable[str], dnm2csv_path: Callable = None) -> List[Dict]:
    def get_single(d_nm):
        df = pd.read_csv(dnm2csv_path(d_nm))
        df.rename(columns={'Unnamed: 0': 'class'}, inplace=True)  # Per csv files
        df = df.iloc[-3:, :].reset_index(drop=True)
        df.support = df.support.astype(int)
        row_acc = df.iloc[0, :]  # The accuracy row
        return OrderedDict([
            ('dnm', d_nm), ('aspect', config(f'UTCD.datasets.{d_nm}.aspect')),
            ('precision', row_acc.precision), ('recall', row_acc.recall), ('f1-score', row_acc['f1-score'])
        ])
    return [get_single(d) for d in dataset_names]

