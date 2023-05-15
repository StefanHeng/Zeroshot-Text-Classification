"""
For uploading UTCD to HuggingFace, each split needs to be stored in a separate file.
"""

import json
import glob
import os.path
from os.path import join as os_join

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

    def upload_models():
        from zeroshot_classifier.models.architecture import BinaryBertCrossEncoder

        from transformers import BertForSequenceClassification, TFBertForSequenceClassification
        from sentence_transformers import SentenceTransformer

        # model_type = 'binary-bert'
        model_type = 'bi-encoder'

        if model_type == 'binary-bert':
            # md_nm = '06.03.22_binary-bert-asp-norm-vanilla'
            # md_nm = '06.03.22_binary-bert-asp-norm-implicit-sep'
            md_nm = '06.04.22_binary-bert-explicit-finetune'
        else:
            assert model_type == 'bi-encoder'
            # md_nm = 'bi-encoder, vanilla, asp-norm, 06.08.22'
            # md_nm = 'bi-encoder, implicit-sep, asp-norm, 06.09.22'
            md_nm = 'bi-encoder, explicit, asp-norm, 06.09.22'
        model_path = os_join(u.base_path, 'models-upload', md_nm)
        mic(model_path)
        assert os.path.exists(model_path)  # sanity check
        # mic(os.listdir(model_path))

        if 'vanilla' in md_nm:
            strat = 'vanilla'
        elif 'implicit-sep' in md_nm:
            strat = 'implicit'
        else:
            assert 'explicit' in md_nm
            strat = 'explicit'
        md_nm_hf = f'zero-shot-{strat}-{model_type}'

        if model_type == 'binary-bert':
            model = BinaryBertCrossEncoder(model_name=model_path)
            model, tokenizer = model.model, model.tokenizer
            assert isinstance(model, BertForSequenceClassification)  # sanity check
            model_tf = TFBertForSequenceClassification.from_pretrained(model_path, from_pt=True)
            mic(type(model), type(model_tf), tokenizer)

            # `sentence-transformers` uses huggingface-hub 0.4.0, uploading to hub w/ org code different
            model.push_to_hub(repo_path_or_name=md_nm_hf, organization=org_nm)
            model_tf.push_to_hub(repo_path_or_name=md_nm_hf, organization=org_nm)
            tokenizer.push_to_hub(repo_path_or_name=md_nm_hf, organization=org_nm)
        else:
            model = SentenceTransformer(model_path)
            mic(model)
            # mic(model._modules)
            # mic(model._first_module())
            # raise NotImplementedError
            mic(model.save_to_hub(repo_name=md_nm_hf, organization=org_nm))
    upload_models()
