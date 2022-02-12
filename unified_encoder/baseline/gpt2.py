from typing import List, Callable

import numpy as np
import transformers
# LMHead for training
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments, SchedulerType, TrainerCallback
from transformers.training_args import OptimizerNames
from transformers import DataCollatorForLanguageModeling, default_data_collator
from datasets import load_dataset, Dataset
from datasets import load_metric

from unified_encoder.util import *


def get_dset(
        dnm='ag_news',
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None
) -> tuple[Dataset, ...]:
    dset = load_dataset(dnm)
    # ic(dset['train'][0])
    tr, vl = dset['train'], dset['test']
    if n_sample is not None:
        tr = tr.select(range(n_sample))
        vl = vl.select(range(n_sample))
    if map_func is not None:
        tr = tr.map(map_func, batched=True, remove_columns=remove_columns)
        vl = vl.map(map_func, batched=True, remove_columns=remove_columns)
    if random_seed:
        tr, vl = tr.shuffle(seed=random_seed), vl.shuffle(seed=random_seed)
    return tr, vl


def get_model_n_tokenizer(name='gpt2') -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    """
    :param name: Model name, one of [`debug`, `gpt2`, `gpt2-medium`]
    """
    MODEL_NMS = dict(small='gpt2', large='gpt2-medium')

    conf = AutoConfig.from_pretrained('gpt2')
    if name == 'debug':  # Try a smaller model for training sanity check
        n_token = 16
        conf.update(dict(n_ctx=n_token, n_positions=n_token, n_layer=12))
        # ic(conf)
        # exit(1)
        model_ = GPT2LMHeadModel(config=conf)
        # ic(model_)
    else:
        model_nm = MODEL_NMS['small']  # TODO: reduce max seq len to 512 as in paper
        model_ = AutoModelWithLMHead.from_pretrained(model_nm)
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
            learning_rate=5e-2,
            batch_size=4,
            weight_decay=1e-2,
            num_train_epochs=4,
            lr_scheduler_type=SchedulerType.CONSTANT,
        ),
        'gpt2': dict(
            learning_rate=3e-5,
            batch_size=32,
            weight_decay=1e-2,
            num_train_epochs=1,
            lr_scheduler_type=SchedulerType.COSINE,
        ),
        'gpt2-medium': dict(
            learning_rate=4e-5,
            batch_size=128,
            weight_decay=1e-2,
            num_train_epochs=1,
            lr_scheduler_type=SchedulerType.COSINE,
        )
    }
    lr, bsz, decay, n_ep, sch = (D_TRAIN_ARGS[name][k] for k in [
        'learning_rate', 'batch_size', 'weight_decay', 'num_train_epochs', 'lr_scheduler_type'
    ])
    ic(n_ep)

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
        num_train_epochs=n_ep,
        lr_scheduler_type=sch,
        warmup_ratio=1e-2,
        logging_strategy='steps',
        logging_steps=1,
        fp16=torch.cuda.is_available(),  # TODO: dynamic loss scaling??
        optim=OptimizerNames.ADAMW_TORCH
    ), DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        logs = kwargs['logs']
        # ic(kwargs)
        ic(args, state, control)
        if state.is_local_process_zero:
            print('in my logs', logs)

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        print('in my evaluate')


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override `Trainer.compute_loss` for logging accuracy

        Modified from https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/4?u=stefanh
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # ========================== Added ==========================
        if 'labels' in inputs:
            preds = outputs.logits.detach()
            matches: torch.Tensor = (preds.argmax(axis=-1) == inputs['labels'])
            self.log(dict(training_accuracy=round((matches.sum() / matches.numel()).item(), 4)))
            # ic(dict(training_accuracy=round((matches.sum() / matches.numel()).item(), 4)))
        # ========================== End of added ==========================

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    from icecream import ic

    from unified_encoder.util import *

    seed = config('random-seed')
    transformers.set_seed(seed)

    nm = 'debug'
    model, tokenizer = get_model_n_tokenizer(nm)
    train_args, data_collator = get_train_setup(nm)
    ic(train_args)

    def tokenize_func(sample):
        ret = tokenizer(sample['text'], padding='max_length', truncation=True)
        # ret['labels'] = ret['input_ids'].copy()
        return ret
    dset_tr, dset_vl = get_dset(map_func=tokenize_func, remove_columns=['label', 'text'], n_sample=8, random_seed=seed)

    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        labels, predictions = labels.flatten(), predictions.flatten()  # Original 2D tensor gives error
        # ic(logits, labels, predictions, labels.shape, predictions.shape)
        return metric.compute(predictions=predictions, references=labels)

    cb = PrinterCallback()

    # trainer = Trainer(
    trainer = CustomTrainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        # data_collator=default_data_collator,
        train_dataset=dset_tr,
        eval_dataset=dset_vl,
        compute_metrics=compute_metrics,
        # callbacks=[cb]
    )
    trainer.train()
    ic(trainer.evaluate())
