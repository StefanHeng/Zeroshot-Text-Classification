from collections import defaultdict
from typing import List, Callable

import pandas as pd
import transformers
# LMHead for training
from transformers import BatchEncoding
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments, SchedulerType, TrainerCallback
from transformers import DataCollatorForLanguageModeling
from transformers.training_args import OptimizerNames
from datasets import load_dataset, Dataset
from datasets import load_metric

from unified_encoder.util import *


def get_dset(
        dnm='ag_news',
        map_func: Callable = None, remove_columns: Union[str, List[str]] = None,
        n_sample: int = None, random_seed: int = None
) -> Tuple[Dataset, ...]:
    dset = load_dataset(dnm)
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


class ZsTokenizer(GPT2TokenizerFast):
    """
    A wrapper around GPT2 tokenizer for 0-shot classification tokenizing
    """
    SPEC_TOKS = OrderedDict([
        ('pref_ques', '<|question|>'),  # Word embeddings
        ('pref_text', '<|text|>'),
        ('pref_answ', '<|answer|>'),
        ('type_ques', '[QUES]'),  # Type embeddings
        ('type_text', '[TEXT]'),
        ('type_answ', '[ANSW]')
    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pad token cannot be `self.eos_token`
        # cos otherwise `DataCollatorForLanguageModeling` would override normal eos tokens
        self.add_special_tokens(dict(pad_token='[PAD]', additional_special_tokens=list(ZsTokenizer.SPEC_TOKS.values())))
        # for i in range(50254, 50266):  # Sanity check
        #     ic(i, self.decode(i))
        # ic(self.pad_token)
        # exit(1)

        self.templates = config('baselines.gpt2-0shot.templates')
        self.cache: Dict[str, Dict] = dict()  # Mapping from dataset name to labels

        self.ques_token, self.text_token, self.answ_token = (
            ZsTokenizer.SPEC_TOKS[k] for k in ('pref_ques', 'pref_text', 'pref_answ')
        )  # Special tokens
        self.ques_type_token, self.text_type_token, self.answ_type_token = (
            ZsTokenizer.SPEC_TOKS[k] for k in ('type_ques', 'type_text', 'type_answ')
        )  # Type tokens
        # ic(self.is_fast)
        # type_ids = [self.encode(i)[0] for i in (iq, it, ia)]  # Should be encoded into single id
        # eos = self.eos_token
        #

    def call_paren(self, s: str, **kwargs) -> List[int]:
        return super().__call__(s, **kwargs)['input_ids']

    def __call__(self, samples: Dict[str, Union[List, str, int]], **kwargs):
        """
        :param samples: Data sample(s) with keys [`dataset_name`, `label`, `text`]
            Each value an element or a list of elements
        """
        # ic(samples)
        max_length = kwargs.get('max_length', None)
        is_batched = isinstance(samples['label'], (tuple, list))
        if max_length is None:
            max_length = self.model_max_length
        n_token = self.max_model_input_sizes['gpt2']  # Indented number of token positions as in vanilla GPT2 model

        ln = len(samples['label'])
        idxs_tpl = np.random.randint(len(self.templates), size=ln)

        def call_single(i, dnm, text, label):
            # dnm, text, label = (sample[k] for k in ('dataset_name', 'text', 'label'))
            assert dnm == 'ag_news'  # Potentially support dynamic dataset
            if dnm not in self.cache:
                feats = load_dataset(dnm, split='train').features['label']  # Pick a split arbitrarily
                feat2feat_full = {
                    'World': 'World News',
                    'Sports': 'Sports',
                    'Business': 'Business',
                    'Sci/Tech': 'Science & Technology'
                }
                n_cls = feats.num_classes
                lb2feat_str: List[str] = [feat2feat_full[feats.names[i]] for i in range(n_cls)]  # Labels = range
                self.cache[dnm] = dict(n_cls=n_cls, lb2feat_str=lb2feat_str)
            else:
                n_cls, lb2feat_str = (self.cache[dnm][k] for k in ('n_cls', 'lb2feat_str'))

            idx_lbs = np.arange(n_cls)
            np.random.shuffle(idx_lbs)
            strs_lb = ' , '.join(f'" {lb2feat_str[idx]} "' for idx in idx_lbs)
            question = self.templates[idxs_tpl[i]].format(strs_lb)
            answer = lb2feat_str[label]

            def enc_spec(tok: str) -> int:
                id_ = self.encode(tok)
                assert len(id_) == 1
                return id_[0]  # Intended for special tokens
            ids_ques = self.call_paren(question, **kwargs)
            ids_text = self.call_paren(text, **kwargs)
            ids_answ = self.call_paren(answer, **kwargs)
            # Number of contex tokens, up until answer token, inclusive
            n_ques, n_text, n_answ = (1 + len(ids_ques) + 1), (1 + len(ids_text) + 1), (1 + len(ids_answ) + 1)
            n_cont = n_ques + n_text + 1
            ids = [
                enc_spec(self.ques_token), *ids_ques, enc_spec(self.eos_token),
                enc_spec(self.text_token), *ids_text, enc_spec(self.eos_token),
                enc_spec(self.answ_token), *ids_answ, enc_spec(self.eos_token)
            ]
            tids = [enc_spec(self.ques_type_token)] * n_ques + \
                   [enc_spec(self.text_type_token)] * n_text + \
                   [enc_spec(self.answ_type_token)] * n_answ
            msks = [1] * len(ids)  # Encode ids are attended for CLM
            # Context position ids, followed by output position ids
            # adding `n_token` offset for the modified positional embeddings, see `ZsGPT2Model`
            pids = list(range(n_cont)) + [i + n_token for i in range(len(ids)-n_cont)]
            assert all(len(lst_ids) == len(ids) for lst_ids in (ids, tids, msks, pids))  # Sanity check

            def pad(ints: List[int], name) -> List[int]:
                if name == 'attention_mask':
                    int_pad = 0  # Ignore in attention
                else:
                    int_pad = self.encode(self.pad_token)  # input_ids set to `pad_token` will be ignored
                # Pad to max_length, truncate if necessary
                return ints[:max_length] if len(ints) > max_length else (ints + [int_pad] * (max_length - len(ints)))
            # return dict(input_ids=ids, attention_mask=msks, token_type_ids=tids, position_ids=pids)
            out = {k: pad(ints, k) for k, ints in ((
                ('input_ids', ids), ('attention_mask', msks), ('token_type_ids', tids), ('position_ids', pids)
            ))}
            print(text)  # Sanity check
            print(label)
            for k, v in out.items():
                if 'position' in k or 'mask' in k:
                    toks = [f'{str(i):>15}' for i in v]
                else:
                    toks = [f'{self.decode(i):>15}' for i in v]
                print(f'{k:>20}:', ' '.join(toks))
            print()
            exit(1)
            return out

        # def pad(ints: List[List[int]], int_pad) -> List[List[int]]:
        #     # Pad to max_length, truncate if necessary
        #     lst = [l[:max_length] if len(l) > max_length else (l + [int_pad] * (max_length-len(l))) for l in ints]
        #     return lst

        if is_batched:
            # ds = [call_single(i, d) for i, d in enumerate(samples)]
            # ic([samples[k] for k in ['dataset_name', 'text', 'label']])
            ds = [call_single(i, dnm, txt, lb) for i, (dnm, txt, lb) in enumerate(zip(
                *[samples[k] for k in ['dataset_name', 'text', 'label']]
            ))]
            return BatchEncoding({k: [d[k] for d in ds] for k in ds[0]})  # Stack all the ids
        else:
            return BatchEncoding(call_single(0, *[samples[k] for k in ['dataset_name', 'text', 'label']]))
        # idx_lbs = np.tile(np.arange(n_cls), (ln, 1))  # Shuffle the labels as in 0-shot pre-training
        # [np.random.shuffle(row) for row in idx_lbs]


def tokenize_func(tokenizer_, dnm='ag_news', max_length=None):
    # TODO: support pre-training task, variable option sizes?

    def _tokenize_func(sample: Dict[str, List]):
        """
        :param sample: A batch of data samples
        """
        # ic(sample)
        #
        # def join_parts(parts: List[Dict[str, List]]):
        #     # Calling tokenizer with `is_split_into_words` doesn't produce same result
        #     # No special token is added by tokenizer
        #     gen = (tokenizer_(elm) for elm in parts)  # Get lists for now
        #     ids_, msks_ = zip(*((d['input_ids'], d['attention_mask']) for d in gen))
        #     # return sum(ids_, []), sum(msks_, [])  # Keyword `start` omitted for python3.6 colab compatibility
        #     return dict(input_ids=sum(ids_, []), attention_mask=sum(msks_, []))
        #
        # def single_sample2str(i, cont: str, lb: int):
        #     strs_lb = ' , '.join(f'" {lb2feat_full[idx]} "' for idx in idx_lbs[i])
        #     question = templates[idxs_tpl[i]].format(strs_lb)
        #     # return join_parts([  # Ensures no space around special tokens
        #     #     sq, question, eos,
        #     #     st, cont, eos,
        #     #     sa, lb2feat_full[lb], eos
        #     # ])
        #     ques = join_parts([sq, question, eos])
        #     text = join_parts([st, cont, eos])
        #     answ = join_parts([sa, lb2feat_full[lb], eos])
        #     for d, tid in zip((ques, text, answ), type_ids):
        #         d['token_type_ids'] = [tid] * len(d['input_ids'])
        #         # ic(tid, d['token_type_ids'])
        #
        #     # def concat_dict(ds: List[Dict]):
        #     #     d_out = defaultdict(list)
        #     #     for k in ds[0]:
        #     #         for dic in ds:  # Extend in that order
        #     #             d_out[k].extend(dic[k])
        #     #     # return d_out
        #     d_out = {k: sum((dic[k] for dic in (ques, text, answ)), []) for k in ques}
        #     idx_type_ans = d_out['input_ids'].index(tokenizer_.encode(sa)[0])
        #     assert idx_type_ans != -1
        #
        #     # Sanity check
        #     # d_out['input_ids'][0] = 50262
        #     # ic([i for i in type_ids])
        #     assert all(all(i != tid for tid in type_ids) for i in d_out['input_ids'] + d_out['attention_mask'])
        #     ic(d_out)
        #     exit(1)
        # ids_n_msks = [
        #     single_sample2str(i, cont, lb) for i, (cont, lb) in enumerate(zip(sample['text'], sample['label']))
        # ]
        # ids, msks = zip(*((i, m) for i, m in ids_n_msks))
        # # for txt, label, row in zip(sample['text'], sample['label'], ids):
        # #     print(txt)
        # #     print(label)
        # #     print(' '.join(tokenizer_.decode(i) for i in row))
        # #     print()
        # # exit(1)
        #
        # return BatchEncoding(
        #     # `pad_token_id` will be ignored per `DataCollatorForLanguageModeling`; 0 for masked positions in attention
        #     dict(input_ids=pad(ids, int_pad=tokenizer_.pad_token_id), attention_mask=pad(msks, int_pad=0))
        # )
        sample['dataset_name'] = [dnm] * len(sample['label'])  # Work with single dataset for now
        return tokenizer_(sample)
    return _tokenize_func


def get_model_n_tokenizer(name='gpt2') -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast, DataCollatorForLanguageModeling]:
    """
    :param name: Model name, one of [`debug`, `gpt2`, `gpt2-medium`]
    """
    MODEL_NMS = dict(small='gpt2', large='gpt2-medium')

    conf = AutoConfig.from_pretrained('gpt2')
    if 'debug' in name:  # Try a smaller model for training sanity check
        if 'large' in name:
            n_token = 128
        else:
            n_token = 4
        conf.update(dict(n_ctx=n_token, n_positions=n_token))
        # ic(conf)
        model_ = GPT2LMHeadModel(config=conf)
    else:
        model_nm = MODEL_NMS['small']  # TODO: reduce max seq len to 512 as in paper
        model_ = AutoModelWithLMHead.from_pretrained(model_nm)
        n_token = conf.n_positions

    tokenizer_ = ZsTokenizer.from_pretrained('gpt2', use_fast=True, model_max_length=n_token)
    model_.resize_token_embeddings(len(tokenizer_))

    return model_, tokenizer_, DataCollatorForLanguageModeling(tokenizer=tokenizer_, mlm=False)


def get_train_setup(name='gpt2') -> TrainingArguments:
    D_TRAIN_ARGS = {
        'debug': dict(
            learning_rate=1e-4,
            batch_size=4,
            weight_decay=1e-2,
            num_train_epochs=4,
            lr_scheduler_type=SchedulerType.CONSTANT,
        ),
        'debug-large': dict(
            learning_rate=1e-4,
            batch_size=32,
            weight_decay=1e-2,
            num_train_epochs=16,
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

    return TrainingArguments(
        output_dir=os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, 'gpt2'),
        do_train=True, do_eval=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        # Adam's beta1, beta2, epsilon taken from the GPT2 config in
        # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
        learning_rate=lr,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        max_grad_norm=1,
        num_train_epochs=n_ep,
        lr_scheduler_type=sch,
        warmup_ratio=1e-2,
        log_level='warning',
        logging_strategy='steps',
        logging_steps=1,
        fp16=torch.cuda.is_available(),  # TODO: dynamic loss scaling??
        optim=OptimizerNames.ADAMW_TORCH,
        disable_tqdm=True,
        report_to='none'
    )


def compute_metrics(eval_pred):
    if not hasattr(compute_metrics, 'metric'):
        compute_metrics.metric = load_metric('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels, predictions = labels.flatten(), predictions.flatten()  # Original 2D tensor gives error
    return compute_metrics.metric.compute(predictions=predictions, references=labels)


def get_all_setup(
        name, n_sample=None, random_seed=None
) -> Tuple[
    GPT2LMHeadModel, GPT2TokenizerFast, DataCollatorForLanguageModeling, TrainingArguments,
    Dataset, Dataset, Trainer
]:
    model_, tokenizer_, data_collator_ = get_model_n_tokenizer(name)
    train_args_ = get_train_setup(name)
    dset_tr_, dset_vl_ = get_dset(
        map_func=tokenize_func(tokenizer_), remove_columns=['label', 'text'], n_sample=n_sample, random_seed=random_seed
    )
    trainer_ = CustomTrainer(
        model=model_,
        args=train_args_,
        data_collator=data_collator_,
        train_dataset=dset_tr_,
        eval_dataset=dset_vl_,
        compute_metrics=compute_metrics
    )
    trainer_.add_callback(MyLoggingCallback(trainer_, interactive=False))
    return model_, tokenizer_, data_collator_, train_args_, dset_tr_, dset_vl_, trainer_


class TrainPlot:
    """
    An interactive matplotlib graph to log metrics during training
    """
    def __init__(
            self,
            title='Transformer Training', train_args: TrainingArguments = None,
            interactive=True, save_plot=True
    ):
        self.title = title
        self.axes = None
        self.lines = []
        self.first = True

        self.interactive = interactive
        self.save_plot = save_plot
        self.colors = sns.color_palette(palette='husl', n_colors=7)
        self.c_tr, self.c_vl = self.colors[0], self.colors[3]

        self.train_args = train_args
        lr, bsz, n_ep = train_args.learning_rate, train_args.per_device_train_batch_size, train_args.num_train_epochs
        self.title_plot = rf'{title}, $\alpha = {lr}$, batch size=${bsz}$, #epochs=${n_ep}$'
        self.title_save = f'{title}, a={lr}, bsz={bsz}, n_ep={n_ep}, {now(sep="-")}'

    def make_plot(self):
        fig, self.axes = plt.subplots(2, 1, figsize=(16, 9))
        fig.suptitle(self.title_plot)
        self.axes[0].set_xlabel('Step')
        self.axes[0].set_ylabel('Loss')
        self.axes[1].set_xlabel('Step')
        self.axes[1].set_ylabel('Accuracy (%)')
        if self.interactive:
            plt.ion()

    def update(self, stats: List[Dict]):
        """
        Updates the plot with a new data point

        :param stats: List of training step stats
        """
        df = pd.DataFrame(stats)
        step, tr_acc, tr_loss, vl_acc, vl_loss = (
            df[k] for k in ('step', 'train_acc', 'train_loss', 'eval_acc', 'eval_loss')
        )
        ax1, ax2 = self.axes
        # Re-plot, since x and y lim may change
        while ax1.lines:
            ax1.lines[-1].remove()
        while ax2.lines:
            ax2.lines[-1].remove()
        ax1.plot(step, tr_loss, label='Training Loss', c=self.c_tr, **LN_KWARGS)
        ax1.plot(step, vl_loss, label='Validation Loss', c=self.c_vl, **LN_KWARGS)
        ax2.plot(step, tr_acc, label='Training Accuracy', c=self.c_tr, **LN_KWARGS)
        ax2.plot(step, vl_acc, label='Validation Accuracy', c=self.c_vl, **LN_KWARGS)
        ax1.legend()
        ax2.legend()
        plt.draw()  # Needed for `ion`

    def plot_single(self, stats):
        """
        Make single static plot
        """
        self.make_plot()
        self.update(stats)
        self.finish()

    def finish(self):
        plt.ioff()  # Keep the plot window
        if self.save_plot:
            self.save()
        plt.show()

    def save(self):
        plt.savefig(os.path.join(self.train_args.output_dir, f'{self.title_save}.png'), dpi=300)


class MyLoggingCallback(TrainerCallback):
    """
    Requires
        - Tuple of (custom compute_loss log, internal training log, internal validation log) for each step
            - Intended for coupled training and evaluation
        - Accuracy as a metric is passed to `Trainer` and training metric computed in `compute_loss` and logged
    """
    def __init__(self, trainer: Trainer, name='GPT-2 Training', mode='train', interactive=True, save_plot=True):
        """
        :param trainer: The parent Trainer
        :param name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        from icecream import ic
        # ic(self.logger.handlers)
        had_handler = False
        for hd in self.logger.handlers:
            if hasattr(hd, 'name_for_my_logging') and getattr(hd, 'name_for_my_logging') == name:
                # ic(hd, vars(hd))
                had_handler = True
        if not had_handler:
            handler = logging.StreamHandler(stream=sys.stdout)  # For my own coloring
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(MyFormatter())
            # For ipython compatibility, potentially update it instead of adding new handler
            setattr(handler, 'name_for_my_logging', name)
            self.logger.addHandler(handler)
        # self.logger.propagate = False

        self.out_dict: Dict = None
        self.parent_trainer = trainer
        self.called_val_init = False
        self.log_hist: List[Dict] = []

        self.mode = mode
        self.train_begin, self.train_end = None, None
        self.t_strt, self.t_end = None, None

        self.interactive = interactive
        self.plot = TrainPlot(title=name, train_args=trainer.args, save_plot=save_plot)

    def set_mode(self, mode: str):
        """
        :param mode: One of ['train', 'eval']
        """
        self.mode = mode

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        args = self.parent_trainer.args
        lr, bsz, n_ep = args.learning_rate, args.per_device_train_batch_size, args.num_train_epochs
        d = OrderedDict([('learning rate', lr), ('batch size', bsz), ('#epochs', n_ep)])
        self.logger.info(f'Training started with {log_dict(d)}')
                         # f'learning rate: {logi(lr)}, batch size: {logi(bsz)}, #epochs: {logi(n_ep)}... ')
        self.t_strt = datetime.datetime.now()

        self.mode = 'train'
        self.train_begin = True
        if self.interactive:
            self.plot.make_plot()

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        if self.train_begin:
            self.train_begin = False
            self.train_end = True

            self.t_end = datetime.datetime.now()
            self.logger.info(f'Training completed in {logi(fmt_dt(self.t_end - self.t_strt))} ')

            if self.interactive:
                self.plot.finish()
            else:  # If didn't show plot before
                self.plot.plot_single(self.log_hist)
        self.mode = 'eval'

    def on_log(self, args: TrainingArguments, state, control, logs: Dict = None, **kwargs):
        def out_dict2str(d: Dict):
            keys_ = ['step', 'epoch', 'train_loss', 'eval_loss', 'train_acc', 'eval_acc']
            fmt = [':>4', ':6.2f', ':7.4f', ':7.4f', ':6.2f', ':6.2f']
            s_fmts = [f'{{{k}{fmt_}}}' for k, fmt_ in zip(keys_, fmt)]  # Enforce ordering

            d = {k: (('loss' in k and round(v, 4)) or ('acc' in k and round(v*100, 4)) or v) for k, v in d.items()}
            s_outs = [(k, fmt_.format(**{k: d[k]})) for fmt_, k in zip(s_fmts, keys_) if k in d]
            return ', '.join(f'{k}={logi(s)}' for (k, s) in s_outs)

        def log_update(d_out):
            self.logger.info(out_dict2str(d_out))
            self.log_hist.append(d_out)
            if self.interactive:
                self.plot.update(self.log_hist)

        if state.is_local_process_zero:
            if self.mode == 'train':
                step = state.global_step
                if 'src' in logs and logs['src'] == 'compute_loss':  # Custom added metric computation
                    if step == 0:  # Before model runs, initial call
                        if not self.called_val_init:  # Prevents circular logging call, see Trainer.evaluate()
                            self.called_val_init = True
                            tr_acc, tr_loss, n_ep = (logs.get(k, None) for k in ('acc', 'loss', 'epoch'))

                            # Other `on_log` calls invoked inside `evaluate` ignored
                            out: Dict = self.parent_trainer.evaluate()
                            n_ep_, vl_acc, vl_loss = (out.get(k, None) for k in ('epoch', 'eval_accuracy', 'eval_loss'))
                            assert all(elm is not None for elm in (n_ep, vl_acc, vl_loss))
                            assert n_ep == n_ep_
                            # Training step in range (1, total steps); Epoch troublesome to calculate TODO
                            # Prep for Trainer internal evaluation call
                            self.out_dict = dict(step=step, epoch=0, train_acc=tr_acc, train_loss=tr_loss)
                            # python3.6 compatibility
                            out = {**self.out_dict, **dict(eval_acc=vl_acc, eval_loss=vl_loss)}
                            log_update(out)
                    else:  # Need to look for the accuracy calculated for the training batch
                        acc, loss = logs.get('acc', None), logs.get('loss', None)
                        assert acc is not None and loss is not None
                        if self.out_dict is None:  # Heuristic: 1st call to `compute_loss` corresponds to training
                            # Now is the 1st call, after logging for last batch completes
                            self.out_dict = dict(step=step, train_acc=acc, train_loss=loss)
                elif 'loss' in logs:  # Internal training log
                    # Edge case step = 1: Before training start, i.e. step=1, stats for training already logged,
                    # But log anyway, for after gradient update, evaluation loss changes
                    tr_loss, lr, n_ep = (logs.get(k, None) for k in ('loss', 'learning_rate', 'epoch'))
                    assert all(elm is not None for elm in (tr_loss, lr, n_ep))
                    tr_loss_compute = self.out_dict.get('train_loss', None)
                    # Without overriding `_maybe_log_save_evaluate`, can only get the training loss with 4 decimal place
                    assert round(tr_loss_compute, 4) == tr_loss
                    # See Trainer.train(); compute_loss executes before step increments
                    assert self.out_dict['step'] == step-1  # Override step & loss
                    self.out_dict.update(dict(step=step, train_loss=tr_loss, lr=lr, epoch=n_ep))
                elif 'eval_loss' in logs:
                    if step != 0:
                        vl_loss, vl_acc, n_ep_ = (logs.get(k, None) for k in ('eval_loss', 'eval_accuracy', 'epoch'))
                        assert all(elm is not None for elm in (vl_loss, vl_acc, n_ep_))
                        assert step == self.out_dict['step']
                        assert n_ep_ == self.out_dict['epoch']
                        # python3.6 compatibility
                        out = dict(**self.out_dict, **dict(eval_loss=vl_loss, eval_acc=vl_acc))
                        log_update(out)
                        self.out_dict = None
                elif any('runtime' in k for k in logs.keys()):
                    self.logger.info(log_dict(logs) if isinstance(logs, dict) else logs)
                else:
                    print('unhandled case')
                    exit(1)
            else:
                if 'src' not in logs:  # Skip custom compute_loss logging
                    self.logger.info(log_dict(logs) if isinstance(logs, dict) else logs)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert 'args' in kwargs and 'callbacks' in kwargs
        # Expect a custom logging callback passed in to **replace** the internal callback
        callbacks = self.callback_handler.callbacks
        self.callback_handler.callbacks = [
            c for c in callbacks if str(c.__class__) != "<class 'transformers.trainer_callback.PrinterCallback'>"
        ]

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override `Trainer.compute_loss` for logging accuracy
            - Note that both training and validation calls `compute_loss`
                => Further logic needs to determine accuracy for which dataset

        Modified from https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/4?u=stefanh
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # ========================== Begin of added ==========================
        if 'labels' in inputs:
            preds = outputs.logits.detach()
            matches: torch.Tensor = (preds.argmax(axis=-1) == inputs['labels'])
            d_log = dict(src='compute_loss', acc=round((matches.sum() / matches.numel()).item(), 4))
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

        # ========================== Begin of added ==========================
        if 'labels' in inputs:
            d_log['loss'] = loss.detach().item()
            self.log(d_log)
        # ========================== End of added ==========================

        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    from icecream import ic

    from unified_encoder.util import *

    seed = config('random-seed')
    transformers.set_seed(seed)

    # nm, n = 'debug', 8
    nm, n = 'debug-large', 128
    model, tokenizer, data_collator, train_args, dset_tr, dset_vl, trainer = get_all_setup(
        nm, n_sample=n, random_seed=seed
    )
    trainer.train()
    trainer.save_model(os.path.join(trainer.args.output_dir, now(sep='-')))
    trainer.evaluate()
