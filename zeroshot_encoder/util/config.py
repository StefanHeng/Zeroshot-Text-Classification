# import itertools
# import os
# import re
# import json
# from typing import Tuple, List, Dict, Callable
#
# import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# from zeroshot_encoder.util import PATH_BASE, DIR_PROJ, DIR_DSET, PKG_NM
# from zeroshot_encoder.util import get_logger
from zeroshot_encoder.util import *


STSb = 'stsb_multi_mt'  # Per Hugging Face
config = {
    'fine-tune': dict(
        eg_sbert=dict(  # Per *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, section 4.2
            dataset_name=STSb,
            embedding_model_name='bert-base-uncased',
            max_seq_length=256,
            batch_size=16,
            n_epochs=4,
            n_eval=100,  # Total number of evaluations during training
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
    'baselines': {
        'gpt2-nvidia': {
            'templates': [
                'To which category does the following document belong? : {}',
                'To which category does the following text belong? : {}',
                'To which category does the text belong? : {}',
                'To which category does the article belong? : {}',
                'How would you describe the following document? : as {}',
                'How would you describe the text? : as {}',
                'How would you describe the following text? : as {}',
                'Which best describes the text? : {}',
                'Which best describes the document? : {}',
                'Which best describes the following document? : {}',
                'Which best describes the following text? : {}',
                'The following document is _ ? : {}',
                'The following text is _ ? : {}',
                'The text is _ ? : {}',
                'The document is _ ? : {}',
                'How is the text best described? : {}',
                'How is the document best described? : {}',
                'How is the following text best described? : {}',
                'How is the following document best described? : {}',
                'Which of these choices best describes the text? : {}',
                'Which of these options best describes the text? : {}',
                'Which of these choices best describes the document? : {}',
                'Which of these options best describes the document? : {}',
                'Which of these categories best describes the following document? : {}',
                'Which of these choices best describes the following document? : {}',
                'Which of these options best describes the following text? : {}'
            ],
            'label-descriptors': dict(  # string label to natural language descriptor, as in paper
                ag_news={
                    'World': 'World News',
                    'Sports': 'Sports',
                    'Business': 'Business',
                    'Sci/Tech': 'Science & Technology'
                }
            )
        },
        'bert-mnli': dict(
            templates=dict(
                sentiment='This text expresses a {} sentiment',
                intent='This text expresses the intent of {}',
                topic='This text belongs to the topic of {}'
            )
        ),
    },
    'UTCD': dict(
        datasets=dict(
            # in-domain evaluation has the same labels as training
            # emotion=dict(
            #     path='UTCD/in-domain/emotion', aspect='sentiment', eval_labels_same=True, out_of_domain=False),
            # go_emotion=dict(
            #     path='UTCD/in-domain/go_emotion', aspect='sentiment', eval_labels_same=True, out_of_domain=False),
            # sentiment_tweets_2020=dict(
            #     path='UTCD/in-domain/sentiment_tweets_2020', aspect='sentiment',
            #     eval_labels_same=True, out_of_domain=False
            # ),
            # clinc_150=dict(
            #     path='UTCD/in-domain/clinc_150', aspect='intent', eval_labels_same=True, out_of_domain=False),
            # # `eval_labels_same` := has some unique test labels
            # # TODO: fix `labels`
            # # sgd=dict(path='UTCD/in-domain/sgd', aspect='intent', eval_labels_same=False, out_of_domain=False),
            # slurp=dict(path='UTCD/in-domain/slurp', aspect='intent', eval_labels_same=False, out_of_domain=False),
            ag_news=dict(path='UTCD/in-domain/ag_news', aspect='topic', eval_labels_same=True, out_of_domain=False),
            dbpedia=dict(path='UTCD/in-domain/dbpedia', aspect='topic', eval_labels_same=True, out_of_domain=False),
            yahoo=dict(path='UTCD/in-domain/yahoo', aspect='topic', eval_labels_same=True, out_of_domain=False),
            # Out-of-domain datasets: test split intended to evaluation
            # TODO: until new multi-label format supported
            # amazon_polarity=dict(
            #     path='UTCD-ood/amazon_polarity', aspect='sentiment', eval_labels_same=True, out_of_domain=True
            # ),
            # finance_sentiment=dict(
            #     path='UTCD-ood/finance_sentiment', aspect='sentiment', eval_labels_same=True, out_of_domain=True
            # ),
            # yelp=dict(path='UTCD-ood/yelp', aspect='sentiment', eval_labels_same=True, out_of_domain=True),
            # # Removed for too many options blow up GPT2's 1024 token length; TODO: remove, keep now cos plotting
            # arxiv=dict(path='UTCD-ood/arxiv', aspect='topic', eval_labels_same=True, out_of_domain=True),
            # multi_eurlex=dict(path='UTCD-ood/multi_eurlex', aspect='topic', eval_labels_same=True, out_of_domain=True),
            # patent=dict(path='UTCD-ood/patent', aspect='topic', eval_labels_same=True, out_of_domain=True),
            # consumer_finance=dict(
            #     path='UTCD-ood/consumer_finance', aspect='topic', eval_labels_same=True, out_of_domain=True
            # ),
            # banking77=dict(path='UTCD-ood/banking77', aspect='intent', eval_labels_same=True, out_of_domain=True),
            # snips=dict(path='UTCD-ood/snips', aspect='intent', eval_labels_same=True, out_of_domain=True),
            # nlu_evaluation=dict(
            #     path='UTCD-ood/nlu_evaluation', aspect='intent', eval_labels_same=True, out_of_domain=True
            # )
        ),
        dataset_ext='json'  # all in json
    ),
    'random-seed': 77
}

path_dset = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
ext = config['UTCD']['dataset_ext']

# from icecream import ic  # TODO: debugging


def _re_call() -> Callable[[str], int]:
    if not hasattr(_re_call, 'token_pattern'):
        # taken from sklearn.CountVectorizer, which was `r"(?u)\b\w\w+\b"`
        _re_call.token_pattern = re.compile(r'(?u)\b\w+\b')
    return lambda x: len(_re_call.token_pattern.findall(x))


def _hf_call(model_name) -> Callable[[str], int]:
    if not hasattr(_hf_call, 'd'):
        _hf_call.d = {}
    d = _hf_call.d
    if model_name not in d:
        d[model_name] = AutoTokenizer.from_pretrained(model_name)
    return lambda x: len(d[model_name](x)['input_ids'])


def get_tokenizer_len(s: str, mode: str = 're') -> int:
    assert mode in ['re', 'bert', 'gpt2']
    if not hasattr(get_tokenizer_len, 'd_f'):
        get_tokenizer_len.d_f = dict(
            re=_re_call(),
            bert=_hf_call('bert-base-cased'),
            gpt2=_hf_call('gpt2')
        )
    return get_tokenizer_len.d_f[mode](s)


tokenize_modes = ['re', 'bert', 'gpt2']


from icecream import ic  # TODO: debugging


def path2dataset_info(d: Dict) -> Tuple[Dict, Dict]:
    """
    :return: 2-tuple of (dataset information for `config`, number of tokens for plot)
    """
    path = os.path.join(path_dset, f'{d["path"]}.{ext}')
    with open(path) as fl:
        dsets: Dict = json.load(fl)
    # ic(path, dsets.keys())

    def split2info(split, dset: Dict[str, List[str]]) -> Dict:
        # Based on heuristics on how the `json` are stored
        # ic(type(dset), len(dset.keys()))
        # txts, lbs = list(dset.keys()), sum([lbs for lbs in dset.values()], start=[])
        ic('start call', now())
        # lbs = sum([lbs for lbs in dset.values()], start=[])
        # ic('computed lbs', now())
        # lst_n_txts, lst_n_lbs = [], []
        # n_text, n_pair = len(txts), sum([len(lbs) for lbs in dset.values()])
        # creating a list of all the strings consume mems
        n_text_, n_pair_ = len(dset.keys()), sum([len(lbs) for lbs in dset.values()])
        ic('computed text & pair counts', now())
        # n_text_, n_pair_ = len(dset.keys()), len(lbs)
        lbs_uniq = set().union(*dset.values())
        n_multi_label = sum([len(lbs_) > 1 for lbs_ in dset.values()])
        ic('computed n_multi_label', now())
        # txt_n_toks = {mode: [get_tokenizer_len(t, mode) for t in txts] for mode in modes}
        # lb_n_toks = {mode: [get_tokenizer_len(t, mode) for t in lbs] for mode in modes}
        txt_n_toks, lb_n_toks = dict(), dict()
        for mode in tokenize_modes:
            txt_n_toks_, lb_n_toks_ = [], []
            n, desc_t, desc_l = 15, f'{split}-{mode}-text', f'{split}-{mode}-label'

            # if mode == 're':
            #     for lb in lbs_uniq:
            #         ic(lb, get_tokenizer_len(lb, mode))
            lb2tokenize_len = {lb: get_tokenizer_len(lb, mode) for lb in lbs_uniq}

            for t in tqdm(dset.keys(), total=len(dset), desc=f'{desc_t:>{n}}'):
                txt_n_toks_.append(get_tokenizer_len(t, mode))
            for t in tqdm(dset.values(), desc=f'{desc_l:>{n}}'):
                lb_n_toks_.extend([lb2tokenize_len[lb] for lb in t])
            # txt_n_toks[mode] = [get_tokenizer_len(t, mode) for t in txts]
            # lb_n_toks[mode] = [get_tokenizer_len(t, mode) for t in lbs]
            txt_n_toks[mode], lb_n_toks[mode] = txt_n_toks_, lb_n_toks_
        # exit(1)
        return dict(
            labels=sorted(lbs_uniq),
            n_label=len(lbs_uniq),
            n_text=n_text_,
            n_pair=n_pair_,
            multi_label=n_text_ < n_pair_,
            n_multi_label=n_multi_label,
            txt_n_toks=txt_n_toks,
            lb_n_toks=lb_n_toks
        )
    labels, aspect = dsets.pop('labels'), dsets.pop('aspect')
    assert aspect == d['aspect']
    d_out = {split: split2info(split, dset) for split, dset in dsets.items()}  # Labels for each split
    assert all(split in ['train', 'test'] for split in d_out.keys())
    # sum over all splits of the dataset for token length computation
    txt_n_toks_all = [d_out.pop('txt_n_toks') for d_out in d_out.values()]
    lb_n_toks_all = [d_out.pop('lb_n_toks') for d_out in d_out.values()]
    txt_n_toks_all = {mode: sum([toks[mode] for toks in txt_n_toks_all], start=[]) for mode in tokenize_modes}
    lb_n_toks_all = {mode: sum([toks[mode] for toks in lb_n_toks_all], start=[]) for mode in tokenize_modes}
    n_text_all = sum([d_out['n_text'] for d_out in d_out.values()])
    n_pair_all = sum([d_out['n_pair'] for d_out in d_out.values()])
    avg_toks = {f'{mode}-txt_avg_tokens': sum(txt_n_toks_all[mode]) / n_text_all for mode in tokenize_modes} | \
               {f'{mode}-lb_avg_tokens': sum(lb_n_toks_all[mode]) / n_pair_all for mode in tokenize_modes}
    # ic(d['path'])
    # ic(sorted(labels), sorted(set().union(*[set(d['labels']) for d in d_out.values()])), sorted(d_out['train']['labels']))
    # ic(len(d_out['train']['labels']), len(d_out['test']['labels']))
    # ic(len(set().union(*[set(d['labels']) for d in d_out.values()])))
    # assert sorted(labels) == sorted(d_out['train']['labels'])
    assert set(labels) == set().union(*[set(d['labels']) for d in d_out.values()])
    if d['eval_labels_same']:
        assert d_out['train']['labels'] == d_out['test']['labels']
    # return d_out
    return d_out | avg_toks, dict(text=txt_n_toks_all, label=lb_n_toks_all)


def extract_utcd_meta() -> Dict:
    d_dsets: Dict = config['UTCD']['datasets']
    logger = get_logger('Process UTCD')
    d_n_toks = dict()
    for dnm, d_dset in d_dsets.items():
        logger.info(f'Processing {logi(dnm)}... ')
        d_meta, d_n_toks[dnm] = path2dataset_info(d_dset)
        d_dset.update(dict(splits=d_meta))

    dnms = sorted(d_dsets)  # All datasets, in- and out-of-domain, share the same dataset <=> id mapping
    config['UTCD']['dataset_name2id'] = {dnm: i for i, dnm in enumerate(dnms)}
    config['UTCD']['dataset_id2name'] = {i: dnm for i, dnm in enumerate(dnms)}

    return d_n_toks


def plot_utcd_n_toks(d_n_toks: Dict):
    d_df = dict()
    text_types = ['text', 'label']
    for text_type, mode in itertools.product(text_types, tokenize_modes):
        dnm2n_tok = {dnm: d_n_toks[dnm][text_type][mode] for dnm in d_n_toks.keys()}
        toks_unrolled = sum([[(n_tok, dnm) for n_tok in n_toks] for dnm, n_toks in dnm2n_tok.items()], start=[])
        d_df[(text_type, mode)] = pd.DataFrame(toks_unrolled, columns=['n_token', 'dataset_name'])
        # ic(text_type, mode, d_df[(text_type, mode)])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i_row, text_type in enumerate(text_types):
        for i_col, mode in enumerate(tokenize_modes):
            # ic(i_row, i_col, text_type, mode)
            ax = axes[i_row, i_col]
            df = d_df[(text_type, mode)]
            n_bin = df.n_token.max() - df.n_token.min() + 1
            legend = i_row == 0 and i_col == 0
            sns.histplot(data=df, x='n_token', hue='dataset_name', kde=True, bins=n_bin, legend=legend, ax=ax)
            ax.set_title(f'{text_type} with {mode} tokenization')
    # for ax in axes:
    #     ax.set_xlim([0, 100])  # empirical
    title, xlabel = 'Histogram of #tokens per sequence', '#token'
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.suptitle('Tokenization length distribution across datasets')
    # if save:
    #     plt.savefig(os.path.join(PATH_BASE, DIR_PROJ, 'plot', f'{title}, {now(for_path=True)}.png'), dpi=300)
    # else:
    plt.show()


plot_utcd_n_toks(extract_utcd_meta())


if __name__ == '__main__':
    from icecream import ic

    fl_nm = 'config.json'
    ic(config)
    exit(1)
    with open(os.path.join(PATH_BASE, DIR_PROJ, PKG_NM, 'util', 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
