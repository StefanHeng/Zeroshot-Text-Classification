from datasets import load_dataset
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, SchedulerType


class Gpt2Dataset(Dataset):
    """
    Dataset for paper `Zero-shot Text Classification With Generative Language Models`
    """
    MODEL_NM = 'gpt2'  # For the sake of tokenization

    def __init__(self, dnm: str):
        """
        :param dnm: HuggingFace text classification dataset name
        """
        self.tokenizer = AutoTokenizer.from_pretrained(Gpt2Dataset.MODEL_NM)
        self.dset = load_dataset(dnm)


class Gpt2Model:
    MODEL_NMS = dict(small='gpt2', large='gpt2-medium')

    def __init__(self, model_key: str = 'small'):
        self.model_nm = Gpt2Model.MODEL_NMS[model_key]
        self.model = AutoModel.from_pretrained(self.model_nm)


def setup():
    SPEC_TOKS = ['<|question|>', '<|text|>', '<|answer|>']
    dnm = 'ag_news'

    model = Gpt2Model()
    dset = Gpt2Dataset(dnm)
    dset.tokenizer.add_special_tokens(dict(additional_special_tokens=Gpt2Dataset.SPEC_TOKS))
    model.model.resize_token_embeddings(len(dset.tokenizer))


if __name__ == '__main__':
    from icecream import ic

    from util import *

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
    model_nm = 'gpt2'
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
    ic(tokenizer.max_model_input_sizes, tokenizer.model_input_names)
    SPEC_TOKS = ['<|question|>', '<|text|>', '<|answer|>']
    tokenizer.add_special_tokens(dict(pad_token='[PAD]', additional_special_tokens=SPEC_TOKS))
    model.resize_token_embeddings(len(tokenizer))

    def tokenize_func(sample):
        # out = tokenizer(sample['text'], padding='max_length', truncation=True)
        # ic(type(out), vars(out))
        return tokenizer(sample['text'], padding='max_length', truncation=True)
    # samp = dset['train'][0]
    # tokenize_func(samp)
    # exit(1)

    dset_tok = dset.map(tokenize_func, batched=True)
    dset_tok = dset_tok.remove_columns('label')  # For autoregressive learning
    ic(dset_tok)

    n = 20
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=dset['train'],
        # eval_dataset=dset['test']
        train_dataset=dset_tok["train"].shuffle(seed=seed).select(range(n)),
        eval_dataset=dset_tok["test"].shuffle(seed=seed).select(range(n))
    )
    trainer.train()
