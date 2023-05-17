"""
For uploading UTCD to HuggingFace, each split needs to be stored in a separate file.
"""

import json
import glob
import os.path
from os.path import join as os_join

from zeroshot_classifier.models import BinaryBertCrossEncoder, ZsGPT2Tokenizer, ZsGPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from stefutil import *
from zeroshot_classifier.util import *
from chore.util import *


def save_each_dataset_split(dir_nm):
    """
    :param dir_nm: directory name containing datasets to split into separate files by dataset split
    """
    path = os_join(u.proj_path, u.dset_dir, dir_nm)
    assert os.path.exists(path)  # sanity check
    path_out = os_join(u.proj_path, u.dset_dir, f'{dir_nm}_split')
    assert not os.path.exists(path_out)
    os.mkdir(path_out)

    paths = list(glob.iglob(os_join(path, '*.json')))
    it = tqdm(paths, desc=f'Saving each split on {dir_nm}')
    for path in it:
        dnm = stem(path)
        it.set_postfix(dnm=pl.i(dnm))
        with open(path) as f:
            d = json.load(f)
        assert set(d.keys()).issubset({'train', 'eval', 'test', 'labels', 'aspect'})

        path_out_dset = os_join(path_out, dnm)
        os.mkdir(path_out_dset)
        splits = ['train', 'eval', 'test'] if 'eval' in d else ['train', 'test']
        for split in splits:
            with open(os_join(path_out_dset, f'{split}.json'), 'w') as f:
                json.dump(d[split], f)


if __name__ == '__main__':
    import pandas as pd

    def save_dset_split():
        # save_each_dataset_split('in-domain')
        # save_each_dataset_split('out-of-domain')
        # save_each_dataset_split('aspect-normalized-in-domain')
        save_each_dataset_split('aspect-normalized-out-of-domain')
    # save_dset_split()

    def check_previous_model_performance():
        dom2dnms = cconfig('domain2dataset-names-all')
        dnms_in, dnms_out = dom2dnms['in'], dom2dnms['out']
        mic(dnms_in, dnms_out)

        path = os_join(u.eval_path, 'model-upload')

        d = dict()
        for md_nm in ['binary-bert', 'bi-encoder']:
            for strat in ['vanilla', 'implicit-sep', 'explicit']:
                dir_nm = f'{md_nm}-{strat}'
                path_dnm = os_join(path, dir_nm)
                paths = os.listdir(path_dnm)
                paths_in = [p for p in paths if p.startswith('in-domain')]
                paths_out = [p for p in paths if p.startswith('out-of-domain')]
                assert len(paths_in) == len(paths_out) == 1

                dom = 'out'
                path_in = os_join(path_dnm, paths_in[0])
                path_out = os_join(path_dnm, paths_out[0])
                # mic(path_in, path_out)

                fnms = os.listdir(path_in) if dom == 'in' else os.listdir(path_out)
                accs = []
                for dnm in (dnms_in if dom == 'in' else dnms_out):
                    dnm_ = f'{dnm}.csv'
                    assert dnm_ in fnms
                    df = pd.read_csv(os_join((path_in if dom == 'in' else path_out), dnm_))
                    df = df.iloc[-3:, :].reset_index(drop=True)  # only keep the aggregated rows
                    accs.append(df.loc[0, 'f1-score'])  # f1-score from accuracy row
                d[dir_nm] = sum(accs) / len(accs)
                # raise NotImplementedError
        mic(d)
    # check_previous_model_performance()

    org_nm = 'claritylab'

    def upload_model():
        # model_type = 'binary-bert'
        # model_type = 'bi-encoder'
        model_type = 'gpt2'

        model_base_path = os_join(u.base_path, 'models-upload')

        if model_type == 'gpt2':
            # md_nm = '2022-06-19_13-08-17_NVIDIA-GPT2-gpt2-medium-vanilla-aspect-norm'
            # md_nm = '2022-06-19_13-09-36_NVIDIA-GPT2-gpt2-medium-implicit-aspect-norm'
            md_nm = '2022-06-13_19-09-32_NVIDIA-GPT2-explicit-aspect-norm'
            model_path = os_join(model_base_path, md_nm, R'trained')
        else:
            if model_type == 'binary-bert':
                # md_nm = '06.03.22_binary-bert-asp-norm-vanilla'
                # md_nm = '06.03.22_binary-bert-asp-norm-implicit-sep'
                md_nm = '06.04.22_binary-bert-explicit-finetune'
            else:
                assert model_type == 'bi-encoder'
                # md_nm = 'bi-encoder, vanilla, asp-norm, 06.08.22'
                # md_nm = 'bi-encoder, implicit-sep, asp-norm, 06.09.22'
                md_nm = 'bi-encoder, explicit, asp-norm, 06.09.22'
            model_path = os_join(model_base_path, md_nm)
        mic(model_path)
        assert os.path.exists(model_path)  # sanity check
        # mic(os.listdir(model_path))

        if 'vanilla' in md_nm:
            strat = 'vanilla'
        elif 'implicit' in md_nm:
            strat = 'implicit'
        else:
            assert 'explicit' in md_nm
            strat = 'explicit'
        md_nm_hf = f'zero-shot-{strat}-{model_type}'

        if model_type == 'binary-bert':
            model = BinaryBertCrossEncoder(model_name=model_path)
            model, tokenizer = model.model, model.tokenizer
            mic(type(model), tokenizer)

            # `sentence-transformers` uses huggingface-hub 0.4.0, uploading to hub w/ org code different
            mic(model.push_to_hub(repo_path_or_name=md_nm_hf, organization=org_nm))
            # tensorflow framework not supported for `CrossEncoder` in `sentence-transformers` don't support TF
            mic(tokenizer.push_to_hub(repo_path_or_name=md_nm_hf, organization=org_nm))
        elif model_type == 'bi-encoder':
            model = SentenceTransformer(model_path)
            mic(model)
            # mic(model._modules)
            # mic(model._first_module())
            # raise NotImplementedError
            mic(model.save_to_hub(repo_name=md_nm_hf, organization=org_nm))
        else:
            # tensorflow framework not supported for there's no `TFZsGPT2LMHeadModel`
            assert model_type == 'gpt2'

            model = ZsGPT2LMHeadModel.from_pretrained(model_path)
            # have to pass `form` for correct tokenization
            tokenizer = ZsGPT2Tokenizer.from_pretrained(model_path, form=strat)
            mic(type(model), tokenizer)

            # repo_id = f'{org_nm}/{md_nm_hf}'

            model.push_to_hub(repo_path_or_name=md_nm_hf, organization=org_nm)
            tokenizer.push_to_hub(repo_path_or_name=md_nm_hf, organization=org_nm)
            # mic(model.push_to_hub(repo_id))
            # mic(tokenizer.push_to_hub(repo_id))
    # upload_model()

    def check_uploaded_binary_bert():
        from transformers import AutoTokenizer

        model_kind = 'binary-bert'
        # strat = 'vanilla'
        strat = 'implicit'
        # strat = 'explicit'
        md_nm = f'zero-shot-{strat}-{model_kind}'
        md_nm = f'{org_nm}/{md_nm}'
        mic(md_nm)
        model = BinaryBertCrossEncoder(model_name=md_nm)
        tokenizer = AutoTokenizer.from_pretrained(md_nm)
        mic(type(model), type(tokenizer))

        # dnm = 'emotion'
        dnm = 'snips'
        d_dset = sconfig(f'UTCD.datasets.{dnm}')
        aspect = d_dset['aspect']
        # mic(aspect)
        input_tpl = TrainStrategy2PairMap(train_strategy=strat)(aspect)
        # text = 'i feel like reds and purples are just so rich and kind of perfect'  # in-domain `emotion`
        text = "I'd like to have this track onto my Classical Relaxations playlist."  # out-of-domain `SNIPS`
        labels = get(d_dset, 'splits.test.labels')
        # mic(labels)

        query = input_tpl(text, labels)
        mic(query)

        logits = model.predict(query, apply_softmax=True)
        mic(logits)
    # check_uploaded_binary_bert()

    def check_uploaded_bi_encoder():
        from sentence_transformers import util as sbert_util
        text = "I'd like to have this track onto my Classical Relaxations playlist."
        dnm = 'snips'
        d_dset = sconfig(f'UTCD.datasets.{dnm}')
        aspect = d_dset['aspect']

        # strat = 'vanilla'
        # strat = 'implicit'
        strat = 'explicit'
        model = SentenceTransformer(model_name_or_path=f'claritylab/zero-shot-{strat}-bi-encoder')
        labels = get(d_dset, 'splits.test.labels')

        mode2map = TrainStrategy2PairMap(train_strategy=strat)
        text = [mode2map.map_text(text, aspect=aspect)]
        labels = [mode2map.map_label(lb, aspect=aspect) for lb in labels]

        text_embed = model.encode(text)
        label_embeds = model.encode(labels)
        scores = [sbert_util.cos_sim(text_embed, lb_embed).item() for lb_embed in label_embeds]
        mic(labels, scores)
    check_uploaded_bi_encoder()

    def check_upload_gpt2():
        md_nm = 'gpt2'
        strat = 'vanilla'
        # strat = 'implicit'
        # strat = 'explicit'
        md_nm = f'{org_nm}/zero-shot-{md_nm}-{strat}'

        model = ZsGPT2LMHeadModel.from_pretrained(md_nm)
        tokenizer = ZsGPT2Tokenizer.from_pretrained(md_nm, form=strat)


