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


def save_each_split(dir_nm):
    """
    :param dir_nm: directory name containing datasets to split into separate files by dataset split
    """
    path = os_join(u.proj_path, u.dset_dir, dir_nm)
    assert os.path.exists(path)  # sanity check
    path_out = os_join(u.proj_path, u.dset_dir, f'{dir_nm}_split')
    # mic(path_out)
    assert not os.path.exists(path_out)
    os.mkdir(path_out)

    paths = list(glob.iglob(os_join(path, '*.json')))
    # mic(list(paths))
    it = tqdm(paths, desc=f'Saving each split on {dir_nm}')
    for path in it:
        dnm = stem(path)
        it.set_postfix(dnm=pl.i(dnm))
        with open(path) as f:
            d = json.load(f)
        # mic(d.keys())
        assert set(d.keys()).issubset({'train', 'eval', 'test', 'labels', 'aspect'})

        path_out_dset = os_join(path_out, dnm)
        os.mkdir(path_out_dset)
        splits = ['train', 'eval', 'test'] if 'eval' in d else ['train', 'test']
        for split in splits:
            with open(os_join(path_out_dset, f'{split}.json'), 'w') as f:
                json.dump(d[split], f)
        # raise NotImplementedError


if __name__ == '__main__':
    # save_each_split('in-domain')
    # save_each_split('out-of-domain')
    # save_each_split('aspect-normalized-in-domain')
    save_each_split('aspect-normalized-out-of-domain')
