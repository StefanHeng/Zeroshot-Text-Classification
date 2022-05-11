import os
import math
from os.path import join
from os.path import join as os_join
from pathlib import Path
from typing import List, Type, Dict, Optional, Union
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    BertConfig, BertModel, BertPreTrainedModel, BertTokenizer,
    get_scheduler
)
from transformers.utils import logging
from sklearn.metrics import classification_report
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from zeroshot_encoder.util.load_data import (
    get_data, binary_explicit_format, in_domain_data_path, out_of_domain_data_path
)
from stefutil import *
from zeroshot_encoder.util import *


logging.set_verbosity_info()
logger = logging.get_logger(__name__)
set_seed(42)


class BertZeroShotExplicit(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.binary_cls = nn.Linear(self.config.hidden_size, 2)
        self.aspect_cls = nn.Linear(self.config.hidden_size, 3)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        binary_logits = self.binary_cls(pooled_output)
        aspect_logits = self.aspect_cls(pooled_output)
        
        loss = None
        
        logits = {'cls': binary_logits, 'aspect': aspect_logits}
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, logits, outputs.hidden_states, outputs.attentions


class ExplicitCrossEncoder:
    def __init__(self, name="bert-base-uncased", device: Union[str, torch.device] = 'cuda', max_length=None) -> None:
        self.config = BertConfig.from_pretrained(name)
        self.model = BertZeroShotExplicit(self.config)
        self.tokenizer = BertTokenizer.from_pretrained(name)
        self.device = device
        self.max_length = max_length

        self.writer = None
        self.model_meta = dict(model='BinaryBERT', mode='explicit')
    
    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []
        aspects = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)
            aspects.append(example.aspect)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        aspects = torch.tensor(aspects, dtype=torch.long).to(self.device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized, labels, aspects

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized

    def fit(
            self,
            train_dataloader: DataLoader,
            epochs: int = 1,
            scheduler: str = 'linear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            output_path: str = None,
            max_grad_norm: float = 1,
            show_progress_bar: bool = True
    ):
        os.makedirs(output_path, exist_ok=True)
        mdl, md = self.model_meta['model'], self.model_meta['mode']
        log_fnm = f'{now(for_path=True)}, {mdl}, md={md}, #ep={epochs}'
        self.writer = SummaryWriter(os_join(output_path, f'tb - {log_fnm}.log'))

        train_dataloader.collate_fn = self.smart_batching_collate
        self.model.to(self.device)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        num_training_steps = int(len(train_dataloader) * epochs)

        lr_scheduler = get_scheduler(
            name=scheduler, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )

        def _get_lr() -> float:
            return lr_scheduler.get_last_lr()[0]

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            tr_loss = 0
            self.model.zero_grad()
            self.model.train()

            # for features, labels, aspects in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
            with tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar) as it:
                for features, lbs, aspects in it:
                    model_predictions = self.model(**features, return_dict=True)

                    pooled_output = model_predictions[1]
                    loss_fct = CrossEntropyLoss()

                    task_loss_value = loss_fct(pooled_output['aspect'].view(-1, 3), aspects.view(-1))
                    binary_loss_value = loss_fct(pooled_output['cls'].view(-1, 2), lbs.view(-1))

                    cls_loss, asp_loss = binary_loss_value.detach().item(), task_loss_value.detach().item()
                    it.set_postfix(cls_loss=cls_loss, asp_loss=asp_loss)
                    step = training_steps
                    self.writer.add_scalar('Train/learning rate', _get_lr(), step)
                    self.writer.add_scalar('Train/Binary Classification Loss', cls_loss, step)
                    self.writer.add_scalar('Train/Aspect Classification Loss', asp_loss, step)

                    loss = task_loss_value + binary_loss_value
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    training_steps += 1
                    tr_loss += loss.item()
            
            average_loss = tr_loss/training_steps
            print(f'Epoch: {epoch+1}\nAverage loss: {average_loss:f}\n Current Learning Rate: {lr_scheduler.get_last_lr()}')
    
        self.save(output_path)

    def predict(self, sentences: List[List[str]], batch_size: int = 32):
        
        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, shuffle=False)

        show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")
        
        pred_scores = []
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = model_predictions[1]['cls']

                if len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        return pred_scores

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def parse_args():
    modes = [
        'vanilla',
        'implicit',
        'implicit-on-text-encode-aspect',  # encode each of the 3 aspects as 3 special tokens, followed by text
        'implicit-on-text-encode-sep',  # encode aspects normally, but add special token between aspect and text
        'explicit'
    ]

    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    parser_train = subparser.add_parser('train')
    parser_test = subparser.add_parser('test')

    # set train arguments
    parser_train.add_argument('--output', type=str, required=True)
    parser_train.add_argument('--sampling', type=str, choices=['rand', 'vect'], required=True)
    parser_train.add_argument('--mode', type=str, choices=modes, default='vanilla')
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=3)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    # set test arguments
    parser_test.add_argument('--model_path', type=str, required=True)
    parser_test.add_argument('--domain', type=str, choices=['in', 'out'], required=True)
    parser_test.add_argument('--mode', type=str, choices=modes, default='vanilla')
    
    return parser.parse_args()


if __name__ == "__main__":
    from icecream import ic

    args = parse_args()

    if args.command == 'train':
        dvc = 'cuda' if torch.cuda.is_available() else 'cpu'

        n_sample = 1024  # TODO: debugging
        data = get_data(in_domain_data_path, n_sample=n_sample)
        # get keys from data dict
        datasets = list(data.keys())
        train = binary_explicit_format(data)

        train_batch_size = args.batch_size
        lr = args.learning_rate
        num_epochs = args.epochs
        model_save_path = join(args.output, args.sampling)

        dl = DataLoader(train, shuffle=True, batch_size=train_batch_size)

        model = ExplicitCrossEncoder('bert-base-uncased', device=dvc)

        warmup_steps_ = math.ceil(len(dl) * num_epochs * 0.1)  # 10% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps_))

        model.fit(
            train_dataloader=dl,
            epochs=num_epochs,
            warmup_steps=warmup_steps_,
            optimizer_params={'lr': lr},
            output_path=model_save_path)
    
    if args.command == 'test':
        mode = args.mode
        pred_path = join(args.model_path, 'preds/{}/'.format(args.domain))
        result_path = join(args.model_path, 'results/{}/'.format(args.domain))
        Path(pred_path).mkdir(parents=True, exist_ok=True)
        Path(result_path).mkdir(parents=True, exist_ok=True)
        if args.domain == 'in':
            data = get_data(in_domain_data_path)
        else:  # out
            data = get_data(out_of_domain_data_path)
        # get keys from data dict
        datasets = list(data.keys())

        model = ExplicitCrossEncoder(args.model_path)

        label_map = ["false", "true"]

        # loop through all datasets
        for dataset in datasets:
            examples = data[dataset]["test"]
            label_options = data[dataset]['labels']
            aspect = data[dataset]['aspect']
            preds = []
            gold = []
            correct = 0

            if mode == 'vanilla':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[txt, lb] for lb in lbs]
            elif mode == 'implicit':
                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[txt, f'{lb} {aspect}'] for lb in lbs]
            elif mode == 'implicit-on-text-encode-aspect':
                aspect_token = sconfig('training.implicit-on-text.encode-aspect.aspect2aspect-token')[aspect]

                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[f'{aspect_token} {txt}', lb] for lb in lbs]
            elif mode == 'implicit-on-text-encode-sep':
                sep_token = sconfig('training.implicit-on-text.encode-sep.aspect-sep-token')

                def txt_n_lbs2query(txt: str, lbs: List[str]) -> List[List[str]]:
                    return [[f'{aspect} {sep_token} {txt}', lb] for lb in lbs]
            else:
                raise NotImplementedError(f'{logi(mode)} not supported yet')

            # loop through each test example
            print(f'Evaluating dataset: {logi(dataset)}')
            for index, (txt_, gold_labels) in enumerate(tqdm(examples.items())):
                query = txt_n_lbs2query(txt_, label_options)
                results = model.predict(query)

                # compute which pred is higher
                pred = label_options[results[:, 1].argmax()]
                preds.append(pred)
               
                if pred in gold_labels:
                    correct += 1
                    gold.append(pred)
                else:
                    gold.append(gold_labels[0])
            
            print(f'{logi(dataset)} Dataset Accuracy: {logi(correct/len(examples))}')
            report = classification_report(gold, preds, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv('{}/{}.csv'.format(result_path, dataset))
