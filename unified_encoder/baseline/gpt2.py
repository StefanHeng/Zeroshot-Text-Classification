from typing import Callable

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, GPT2Model
from transformers import Trainer, TrainingArguments, SchedulerType
from datasets import load_dataset


def get_dset(dnm='ag_news', map_func: Callable = None, n_sample=1000, random_seed: int = None):
    dset = load_dataset(dnm)
    ic(dset, len(dset))
    ic(dset['train'][0])
    if n_sample:
        dset.select(range(n_sample))
    if map_func is not None:
        dset = dset.map(map_func, batched=True)
    tr, val = dset['train'], dset['test']
    if random_seed:
        tr, val = tr.shuffle(seed=random_seed), val.shuffle(seed=random_seed)
    return tr, val


def get_model_n_tokenizer(name='gpt2'):
    """
    :param name: Model name, one of [`debug`, `gpt2`, `gpt2-medium`]
    """
    MODEL_NMS = dict(small='gpt2', large='gpt2-medium')

    conf = AutoConfig.from_pretrained('gpt2')
    if name == 'debug':  # Try a smaller model for training sanity check
        n_token = 64
        # ic(config)
        conf.update(dict(n_ctx=n_token, n_positions=n_token, n_layer=4))
        ic(conf)
        # exit(1)
        model_ = GPT2Model(config=conf)
        ic(model_)
    else:
        model_nm = MODEL_NMS['small']  # TODO: reduce max seq len to 512 as in paper
        model_ = AutoModel.from_pretrained(model_nm)
        n_token = conf.n_positions
        ic(model_)

    tokenizer_ = AutoTokenizer.from_pretrained('gpt2', use_fast=True, model_max_length=n_token)
    SPEC_TOKS = ['<|question|>', '<|text|>', '<|answer|>']
    tokenizer_.add_special_tokens(dict(pad_token='[PAD]', additional_special_tokens=SPEC_TOKS))
    model_.resize_token_embeddings(len(tokenizer_))
    return model_, tokenizer_


def get_training_args(name='gpt2') -> TrainingArguments:
    D_TRAIN_ARGS = {
        'gpt2': dict(
            learning_rate=3e-5,
            batch_size=32,
            weight_decay=1e-2
        ),
        'gpt2-medium': dict(
            learning_rate=4e-5,
            batch_size=128,
            weight_decay=1e-2
        )
    }
    lr, bsz, decay = (D_TRAIN_ARGS[name][k] for k in ['learning_rate', 'batch_size', 'weight_decay'])

    return TrainingArguments(
        output_dir=os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, 'gpt2'),
        do_train=True, do_eval=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        # TODO: Adam's beta1, beta2, epsilon, what values were used???
        max_grad_norm=1,
        num_train_epochs=1,
        lr_scheduler_type=SchedulerType.COSINE,
        warmup_ratio=1e-2,
        logging_strategy='steps',
        logging_steps=1,
        fp16=torch.cuda.is_available(),  # TODO: dynamic loss scaling??
        optim='adamw_torch'
    )


if __name__ == '__main__':
    from icecream import ic

    from unified_encoder.util import *

    seed = config('random-seed')
    transformers.set_seed(seed)

    model, tokenizer = get_model_n_tokenizer('debug')
    exit(1)
    train_args = get_training_args()

    def tokenize_func(sample):
        return tokenizer(sample['text'], padding='max_length', truncation=True)
    dset_tr, dset_val = get_dset(map_func=tokenize_func, n_sample=128, random_seed=seed)
    # dset_tok = dset_tok.remove_columns('label')  # For autoregressive learning

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dset_tr,
        eval_dataset=dset_val
    )
    trainer.train()
