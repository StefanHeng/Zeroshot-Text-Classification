from typing import Callable

from zeroshot_encoder.util import *


PATH_BASE_CHORE = os.path.join(PATH_BASE, DIR_PROJ, 'chore')

_CONFIG = {
    'model-name': {
        'binary-bert': 'Binary BERT',
        'bert-nli': 'BERT-NLI',
        'bi-encoder': 'Bi-Encoder',
        'dual-bi-encoder': 'Dual Bi-Encoder',
        'gpt2-nvidia': 'GPT2-NVIDIA'
    },
    'sampling-strategy': dict(
        rand='Random Negative Sampling', vect='Word2Vec Average Extremes',
        none='Positive Labels Only', NA='-',  # not applicable
    )
}

D_DNMS = {
    'in-domain': OrderedDict([
        ('sentiment', ['go_emotion', 'sentiment_tweets_2020', 'emotion']),
        ('intent', ['sgd', 'clinc_150', 'slurp']),
        ('topic', ['ag_news', 'dbpedia', 'yahoo'])
    ]),
    'out-of-domain': OrderedDict([
        ('sentiment', ['amazon_polarity', 'finance_sentiment', 'yelp']),
        ('intent', ['banking77', 'snips', 'nlu_evaluation']),  # TODO: 'arxiv' removed, add another?
        ('topic', ['arxiv', 'patent', 'consumer_finance'])
    ])
}
DNMS_IN = sum(D_DNMS['in-domain'].values(), start=[])
DNMS_OUT = sum(D_DNMS['out-of-domain'].values(), start=[])


def get_dnm2csv_path_fn(model_name: str, strategy: str, in_domain=True) -> Callable:
    paths = [PATH_BASE, DIR_PROJ, 'evaluations']
    assert model_name in ['binary-bert', 'bert-nli', 'bi-encoder', 'dual-bi-encoder', 'gpt2-nvidia']
    if model_name == 'bert-nli':
        paths.append('nli_bert')
    elif model_name == 'binary-bert':
        paths.append('binary_bert')
    else:
        paths.append(model_name)
    assert strategy in ['rand', 'vect', 'none', 'NA']  # # Radnom negative sampling; word2vec average label selection

    if strategy == 'NA':  # GPT2
        assert model_name == 'gpt2-nvidia'
        paths.extend(['in-domain', '2022-03-11 23-50-25'] if in_domain else ['out-of-domain', '2022-03-12 00-25-13'])
    else:  # BERT models
        paths.extend([strategy, 'results'])
        paths.append('in-domain' if in_domain else 'out-of-domain')
    return lambda d_nm: os.path.join(*paths, f'{d_nm}.csv')


def md_nm_n_strat2str_out(model_name: str, strategy: str, pprint=False) -> Union[Tuple[str, str], str]:
    md_nm, strat = _CONFIG['model-name'][model_name], _CONFIG['sampling-strategy'][strategy]
    if pprint:
        return f'{md_nm} w/ {strat}' if strat != '-' else md_nm
    else:
        return md_nm, strat


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
