from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, SchedulerType


if __name__ == '__main__':
    from icecream import ic

    from unified_encoder.util import *

    seed = config('random-seed')
    transformers.set_seed(seed)

    dnm = 'ag_news'
    dset = load_dataset(dnm)
    ic(dset, len(dset))
    ic(dset['train'][0])

    d_training_args = {
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
    MODEL_NMS = dict(small='gpt2', large='gpt2-medium')
    model_nm = MODEL_NMS['small']
    lr, bsz, decay = (d_training_args[model_nm][k] for k in ['learning_rate', 'batch_size', 'weight_decay'])

    training_args = TrainingArguments(
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
    )
    model = AutoModel.from_pretrained(model_nm)
    tokenizer = AutoTokenizer.from_pretrained(model_nm)  # TODO: reduce max seq len to 512
    SPEC_TOKS = ['<|question|>', '<|text|>', '<|answer|>']
    tokenizer.add_special_tokens(dict(pad_token='[PAD]', additional_special_tokens=SPEC_TOKS))
    model.resize_token_embeddings(len(tokenizer))

    def tokenize_func(sample):
        return tokenizer(sample['text'], padding='max_length', truncation=True)

    dset_tok = dset.map(tokenize_func, batched=True)
    dset_tok = dset_tok.remove_columns('label')  # For autoregressive learning
    ic(dset_tok)

    n = 20
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=dset_tok['train'],
        # eval_dataset=dset_tok['test']
        train_dataset=dset_tok["train"].shuffle(seed=seed).select(range(n)),
        eval_dataset=dset_tok["test"].shuffle(seed=seed).select(range(n))
    )
    trainer.train()
