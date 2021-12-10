import os

from data_path import *


STSb = 'stsb_multi_mt'
config = {
    'fine-tune': dict(
        eg_sbert=dict(  # Per *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, section 4.2
            dataset_name=STSb,
            embedding_model_name='bert-base-uncased',
            max_seq_length=256,
            batch_size=16,
            n_epochs=4,
            warmup_frac=0.1,
            pooling_model_kwargs=dict(pooling_mode_mean_tokens=True)
        )
    ),
    'datasets': {
        STSb: dict(
            n_sample=dict(
                train=5749,
                dev=1500,
                test=1379
            ),
            label_range=dict(
                min=0,
                max=5
            )
        )
    },
    'random-seed': 77
}


if __name__ == '__main__':
    import json
    from icecream import ic

    fl_nm = 'config.json'
    ic(config)
    with open(os.path.join(PATH_BASE, DIR_PROJ, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
