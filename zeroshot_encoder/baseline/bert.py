"""
Implementation of UPenn BERT with MNLI approach

[Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach]
(https://arxiv.org/abs/1909.00161)

Assumes model already pretrained on MNLI
implementing their fine-tuning approach, which serves as our continued pretraining
"""

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments

from zeroshot_encoder.util import *
from zeroshot_encoder.preprocess import *


class ZsBertTokenizer(BertTokenizerFast):
    def __call__(self, samples: Dict[str, List[Union[str, int]]], **kwargs):
        """
        :param sample: Batched samples with keys `text`, `label`, `dataset_id`
            Intended to use with `Dataset.map`
        """
        ic(self.model_max_length)
        ic(list(samples), len(samples['label']))
        exit(1)
        pass


def get_all_setup(
        model_name, dataset_name: str = 'benchmark_joined',
        n_sample=None, random_seed=None, do_eval=True, custom_logging=True
) -> Tuple[BertForSequenceClassification, BertTokenizerFast, datasets.Dataset, datasets.Dataset, Trainer]:
    assert dataset_name == 'benchmark_joined'
    path = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, 'bert', 'mnli-pretrained')
    # ic(os.listdir(path))
    model_ = BertForSequenceClassification.from_pretrained(path)
    # ic(model_.config)
    # ic(model_.config.max_position_embeddings)
    # ic(model_.config.)

    # from transformers import BertConfig
    model_name_ = 'bert-base-uncased'
    # ic(BertConfig.from_pretrained(model_name_))
    tokenizer_ = ZsBertTokenizer.from_pretrained(  # For we load from disk, field missing
        path, model_max_length=BertTokenizerFast.max_model_input_sizes[model_name_]
    )
    # ic(vars(model_), tokenizer_)
    tr, vl = get_dset(
        'benchmark_joined', map_func=tokenizer_, n_sample=n_sample, remove_columns=['label', 'text'], random_seed=random_seed,
        fast='debug' not in model_name
    )
    return model_, tokenizer_, tr, vl,


if __name__ == '__main__':
    from icecream import ic

    seed = config('random-seed')
    n = 1024

    nm = 'debug'
    # nm = 'model'

    get_all_setup(
        model_name=nm, dataset_name='benchmark_joined',
        do_eval=False, custom_logging=True, n_sample=n, random_seed=seed
    )

