from typing import List, Callable

import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, GPT2Model, GPT2TokenizerFast
# Head for training
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments, SchedulerType
from transformers import DataCollatorForLanguageModeling, default_data_collator
from datasets import load_dataset, Dataset

from unified_encoder.util import *


def get_dset(
        dnm='ag_news',
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None
) -> tuple[Dataset, ...]:
    dset = load_dataset(dnm)
    ic(dset, len(dset))
    ic(dset['train'][0])
    tr, vl = dset['train'], dset['test']
    if n_sample is not None:
        tr = tr.select(range(n_sample))
        vl = vl.select(range(n_sample))
    if map_func is not None:
        tr = tr.map(map_func, batched=True, remove_columns=remove_columns)
        vl = vl.map(map_func, batched=True, remove_columns=remove_columns)
    if random_seed:
        tr, vl = tr.shuffle(seed=random_seed), vl.shuffle(seed=random_seed)
        ic(tr, vl)
    return tr, vl


def get_model_n_tokenizer(name='gpt2') -> tuple[GPT2Model, GPT2TokenizerFast]:
    """
    :param name: Model name, one of [`debug`, `gpt2`, `gpt2-medium`]
    """
    MODEL_NMS = dict(small='gpt2', large='gpt2-medium')

    conf = AutoConfig.from_pretrained('gpt2')
    if name == 'debug':  # Try a smaller model for training sanity check
        n_token = 64
        conf.update(dict(n_ctx=n_token, n_positions=n_token, n_layer=4))
        # ic(conf)
        # exit(1)
        model_ = GPT2LMHeadModel(config=conf)
        # ic(model_)
    else:
        model_nm = MODEL_NMS['small']  # TODO: reduce max seq len to 512 as in paper
        model_ = AutoModel.from_pretrained(model_nm)
        n_token = conf.n_positions
        # ic(model_)

    tokenizer_ = AutoTokenizer.from_pretrained('gpt2', use_fast=True, model_max_length=n_token)
    SPEC_TOKS = ['<|question|>', '<|text|>', '<|answer|>']
    tokenizer_.add_special_tokens(dict(pad_token='[PAD]', additional_special_tokens=SPEC_TOKS))
    model_.resize_token_embeddings(len(tokenizer_))
    return model_, tokenizer_


def get_train_setup(name='gpt2') -> tuple[TrainingArguments, DataCollatorForLanguageModeling]:
    D_TRAIN_ARGS = {
        'debug': dict(
            learning_rate=3e-5,
            batch_size=4,
            weight_decay=1e-2
        ),
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
        # Adam's beta1, beta2, epsilon taken from
        # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        max_grad_norm=1,
        num_train_epochs=1,
        lr_scheduler_type=SchedulerType.COSINE,
        warmup_ratio=1e-2,
        logging_strategy='steps',
        logging_steps=1,
        fp16=torch.cuda.is_available(),  # TODO: dynamic loss scaling??
        optim='adamw_torch'
    ), DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


if __name__ == '__main__':
    from icecream import ic

    from unified_encoder.util import *

    # from transformers import AutoModelForCausalLM
    # ic(AutoModelForCausalLM.from_pretrained('gpt2'))
    # exit(1)

    seed = config('random-seed')
    transformers.set_seed(seed)

    nm = 'debug'
    model, tokenizer = get_model_n_tokenizer(nm)
    train_args, data_collator = get_train_setup(nm)

    def tokenize_func(sample):
        ret = tokenizer(sample['text'], padding='max_length', truncation=True)
        ret['labels'] = ret['input_ids'].copy()
        # ic(ret)
        # exit(1)
        return ret
    dset_tr, dset_vl = get_dset(map_func=tokenize_func, remove_columns=['label', 'text'], n_sample=16, random_seed=seed)
    # ic(dset_tr, type(dset_vl))

    # dl = DataLoader(dset_tr)
    # for i in dl:
    #     ic(i)
    #     exit(1)
    # ic(data_collator.torch_call(list(dset_tr[:2])))

    trainer = Trainer(
        model=model,
        args=train_args,
        # data_collator=data_collator,
        data_collator=default_data_collator,
        train_dataset=dset_tr,
        eval_dataset=dset_vl
    )
    trainer.train()
