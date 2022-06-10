"""
Pretraining for 2-stage explicit training

Given text, predict aspect with linear classification head
    Binary BERT & Bi-Encoder all pretrained via BERT; TODO: consider +MLM?
    GPT2-NVIDIA pretrained with GPT2

Pretrained weights loaded for finetuning
"""

from os.path import join as os_join

import numpy as np
from transformers import TrainingArguments, SchedulerType
from transformers.training_args import OptimizerNames
from datasets import load_metric

from stefutil import *
from zeroshot_classifier.util import *
import zeroshot_classifier.util.utcd as utcd_util
import zeroshot_classifier.models.binary_bert


__all__ = ['EXPLICIT_BERT_MODEL_NAME', 'EXPLICIT_GPT2_MODEL_NAME', 'get_train_args', 'compute_metrics']


_bert_md_nm = zeroshot_classifier.models.binary_bert.MODEL_NAME,
_gpt2_md_nm = zeroshot_classifier.models.gpt2.MODEL_NAME
EXPLICIT_BERT_MODEL_NAME = f'Explicit Pretrain Aspect {_bert_md_nm}'
EXPLICIT_GPT2_MODEL_NAME = f'Explicit Pretrain Aspect {_gpt2_md_nm}'


def get_train_args(model_name: str, **kwargs) -> TrainingArguments:
    ca.check_mismatch('Model Name', model_name, [_bert_md_nm, _gpt2_md_nm])
    debug = False
    if debug:
        args = dict(
            batch_size=16,
            learning_rate=1e-4,
            weight_decay=0,
            lr_scheduler_type=SchedulerType.CONSTANT,
            num_train_epochs=4
        )
    else:
        # Keep the same as in Binary BERT vanilla training
        args = dict(
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            weight_decay=1e-2,
            num_train_epochs=3,
            lr_scheduler_type=SchedulerType.COSINE,
        )
    if 'batch_size' in args:
        bsz = args.pop('batch_size')
        args['per_device_train_batch_size'] = bsz
        args['per_device_eval_batch_size'] = bsz
    md_nm = model_name.replace(' ', '-')
    dir_nm = f'{now(for_path=True)}_{md_nm}'
    args.update(dict(
        output_dir=os_join(utcd_util.get_output_base(), u.proj_dir, u.model_dir, dir_nm),
        do_train=True, do_eval=True,
        evaluation_strategy='epoch',
        eval_accumulation_steps=128,  # Saves GPU memory
        warmup_ratio=1e-1,
        adam_epsilon=1e-6,
        logging_strategy='steps',
        logging_steps=1,
        save_strategy='epoch',
        optim=OptimizerNames.ADAMW_TORCH,
        report_to='none'  # I have my own tensorboard logging
    ))
    args.update(kwargs)
    return TrainingArguments(**args)


def compute_metrics(eval_pred):
    if not hasattr(compute_metrics, 'acc'):
        compute_metrics.acc = load_metric('accuracy')
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return dict(acc=compute_metrics.acc.compute(predictions=preds, references=labels)['accuracy'])
