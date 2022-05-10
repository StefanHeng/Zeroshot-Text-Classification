"""
Get the text samples that models perform the worst on, and look for insights

See `zeroshot_encoder.baseline.binary_bert` in test mode
"""

import json
from os.path import join as os_join
from typing import List, Dict, Union, Any
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertviz import head_view

from stefutil import *
from zeroshot_encoder.util import *


def get_bad_samples(d_loss: Dict[str, np.array], k: int = 32, save: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    :param d_loss: The loss of each text sample in each dataset by a model, in iteration order
    :param k: top #samples to keep
    :return: A list of text samples with the respective loss that the model performs the worst on, sorted by performance
    :param save: Save the results to a directory path
    """
    d_out, split = dict(), 'test'
    for dnm, loss in d_loss.items():
        idxs_top = np.argpartition(loss, -k)[-k:]
        s_idxs_top = set(idxs_top)
        out = []
        for i, (txt, lbs) in enumerate(utcd.get_dataset(dnm, split).items()):
            if i in s_idxs_top:
                out.append(dict(text=txt, labels=lbs, loss=float(loss[i])))
        d_out[dnm] = sorted(out, key=lambda x: -x['loss'])
    if save:
        fnm = os_join(save, f'{now(for_path=True)}, bad_samples.json')
        with open(fnm, 'w') as fl:
            json.dump(d_out, fl, indent=4)
    return d_out


class AttentionVisualizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.dataset_cache: Dict[str, Dict[str, List[str]]] = dict()

        self.model_cache = defaultdict(lambda: defaultdict(dict))  # dataset name => text => label => visualization args

        self.logger = get_logger('Binary BERT Attention Visualizer')

    def visualize(self, dataset_name: str, text: str, label: str = None):
        """
        Visualize the attention weights of a text, label pair
            Intended for binary bert

        Previously computed attention weights are cached

        Should be called in a notebook only per `bertviz`
        """
        split = 'test'
        if dataset_name not in self.dataset_cache:
            self.dataset_cache[dataset_name] = utcd.get_dataset(dataset_name, split)
        label_options = sconfig(f'UTCD.datasets.{dataset_name}.splits.{split}.labels')
        self.logger.info(f'Visualizing dataset {logi(dataset_name)} with label options {logi(label_options)}... ')
        if label is None:  # just assume not run on this text before
            label, args = self._get_pair(dataset_name, text, label_options)
        elif label not in self.model_cache[dataset_name][text]:
            args = self._get_pair(dataset_name, text, label)
        else:
            args = self.model_cache[dataset_name][text][label]
        self.logger.info(f'Visualizing on {log_dict(text=text, label=label)} ... ')
        head_view(**args)

    def _get_pair(self, dataset_name: str, text: str, label: Union[str, List[str]]):
        batched = isinstance(label, list)
        if batched:
            text_in, label_in = [text] * len(label), label
        else:  # single label
            text_in, label_in = [text], [label]
        tok_args = dict(padding=True, truncation='longest_first', return_tensors='pt')
        # ic(text_in, label_in)
        inputs = self.tokenizer(text_in, label_in, **tok_args)
        input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        # ic(type(outputs))
        # ic(outputs.keys())
        attn = outputs.attentions
        # ic(logits.shape, logits)
        #         # ic(type(attention), len(attention))
        #         # for t in attention:
        #     ic(t.shape)
        if batched:
            for i, (lb, iids, tids) in enumerate(zip(label, input_ids, token_type_ids)):
                toks = self.tokenizer.convert_ids_to_tokens(iids)
                b_strt = tids.tolist().index(1)
                a = tuple(a[None, i] for a in attn)
                self.model_cache[dataset_name][text][lb] = dict(attention=a, tokens=toks, sentence_b_start=b_strt)
            scores = outputs.logits[:, 1]
            lb = label[scores.argmax()]  # pick the label with the highest score
            return lb, self.model_cache[dataset_name][text][lb]
        else:
            b_strt = token_type_ids[0].tolist().index(1)
            toks = self.tokenizer.convert_ids_to_tokens(input_ids[0])  # remove batch dimension
            # ic(toks)
            arg = dict(attention=attn, tokens=toks, sentence_b_start=b_strt)
            self.model_cache[dataset_name][text][label] = arg  # index into the 1-element list
            return arg


if __name__ == '__main__':
    import pickle

    from icecream import ic
    ic.lineWrapWidth = 5112

    model_dir_nm = os_join('binary-bert-rand-vanilla-old-shuffle-05.03.22', 'rand')
    mdl_path = os_join(u.proj_path, u.model_dir, model_dir_nm)

    def get_bad_eg():
        path_eval = os_join(mdl_path, 'eval', 'in-domain, 05.09.22')
        with open(os_join(path_eval, 'eval_loss.pkl'), 'rb') as f:
            d = pickle.load(f)
        save_path = os_join(u.proj_path, 'eval', 'binary-bert', 'rand, vanilla', 'in-domain, 05.09.22')
        get_bad_samples(d, save=save_path)
    get_bad_eg()

    def visualize():
        av = AttentionVisualizer(mdl_path)
        dnm = 'emotion'
        txt = 'i feel like the writer wants me to think so and proclaiming he no longer liked pulsars is a petty and ' \
              'hilarious bit of character '
        # lbl = 'anger'
        lbl = None
        av.visualize(dataset_name=dnm, text=txt, label=lbl)
    # visualize()
