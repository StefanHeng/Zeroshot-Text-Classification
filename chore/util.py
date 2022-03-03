from typing import Callable

from zeroshot_encoder.util import *


path_base = os.path.join(PATH_BASE, DIR_PROJ, 'chore')


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


_CONFIG = {
    'model-name': {'binary-bert': 'Binary Bert', 'bert-nli': 'Bert-NLI'},
    'sampling-strategy': {'rand': 'Random Negative Sampling'}
}


def md_nm_n_strat2str_out(model_name: str, strategy: str) -> Tuple[str, str]:
    return _CONFIG['model-name'][model_name], _CONFIG['sampling-strategy'][strategy]

