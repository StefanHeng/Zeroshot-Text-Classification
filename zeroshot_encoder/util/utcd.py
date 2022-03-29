# from .util import *
from zeroshot_encoder.util.util import *


def get_output_base():
    # For remote machines, save heavy-duty data somewhere else to save `/home` disk space
    hnm = get_hostname()
    if 'clarity' in hnm:  # Clarity lab
        return '/data'
    elif 'arc-ts' in hnm:  # Great Lakes; `profmars0` picked arbitrarily among [`profmars0`, `profmars1`]
        # Per https://arc.umich.edu/greatlakes/user-guide/
        return os.path.join('/scratch', 'profmars_root', 'profmars0', 'stefanhg')
    else:
        return PATH_BASE


def process_utcd_dataset(in_domain=False, join=False, group_labels=False):
    """
    :param in_domain: If True, process all the in-domain datasets; otherwise, process all the out-of-domain datasets
    :param join: If true, all datasets are joined to a single dataset
    :param group_labels: If true, the datasets are converted to a multi-label format

    .. note::
        1. The original dataset format is list of (text, label) pairs
        2. `group_labels` supported only when datasets are not jointed, intended for evaluation

    Save processed datasets to disk
    """
    logger = get_logger('Process UTCD')

    nm_dsets = 'UTCD-ood' if in_domain else 'UTCD'
    ext = config('UTCD.dataset_ext')
    path_dsets = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
    path_out = os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed')
    logger.info(f'Processing UTCD datasets with '
                f'{log_dict(dict(in_domain=in_domain, join=join, group_labels=group_labels))}... ')

    def path2dsets(dnm: str, d_dset: Dict) -> Union[datasets.DatasetDict, Dict[str, pd.DataFrame]]:
        logger.info(f'Processing dataset {logi(dnm)}... ')
        path = d_dset['path']
        path = os.path.join(path_dsets, f'{path}.{ext}')
        with open(path) as f:
            dsets_: Dict = json.load(f)

        def json2dset(split: str, dset: List) -> Union[datasets.Dataset, pd.DataFrame]:
            assert all(sample[0] != '' for sample in dset)
            if group_labels:
                # Otherwise, process just normally
                dset = sorted(dset)  # Sort first by text then by label, for `groupby`
                # Group the label for each unique text
                lbs_: List[str] = config(f'UTCD.datasets.{dnm}.splits.{split}.labels')
                # index is label per `lbs_` ordering, same with `datasets.ClassLabel`
                lb2id = {lb: i for i, lb in enumerate(lbs_)}
                dset = [  # Map to integer labels
                    dict(text=k, labels=[lb2id[lb] for txt, lb in v])
                    for k, v in itertools.groupby(dset, key=lambda pr: pr[0])
                ]
                lbs = datasets.Sequence(  # if not multi-label, `Sequence` of single element
                    datasets.ClassLabel(names=lbs_),
                    length=-1 if config(f'UTCD.datasets.{dnm}.splits.{split}.multi_label') else 1
                )
                return datasets.Dataset.from_pandas(
                    pd.DataFrame(dset),
                    features=datasets.Features(text=datasets.Value(dtype='string'), labels=lbs)
                )
            else:
                dset = [dict(text=txt, label=lb) for (txt, lb) in dset]  # Heuristic on how the `json` are stored
                df_ = pd.DataFrame(dset)
                if join:  # Leave processing labels til later
                    return df_
                else:
                    # Sort the string labels, enforce deterministic order
                    lbs = sorted(df_.label.unique())
                    assert lbs == config(f'UTCD.datasets.{dnm}.splits.{split}.labels')  # Sanity check
                    lbs = datasets.ClassLabel(names=lbs)
                    features_ = datasets.Features(text=datasets.Value(dtype='string'), label=lbs)
                    # Map to integer labels so that compatible to current training infrastructure in `gpt2.py`
                    df_.label.replace(to_replace=lbs.names, value=range(lbs.num_classes), inplace=True)
                    return datasets.Dataset.from_pandas(df_, features=features_)
        return datasets.DatasetDict({split: json2dset(split, dset) for split, dset in dsets_.items()})
    d_dsets = {
        dnm: path2dsets(dnm, d) for dnm, d in config('UTCD.datasets').items() if d['out_of_domain'] == (not in_domain)
    }
    if join:
        dnm2id = config('UTCD.dataset_name2id')

        def pre_concat(dnm: str, df_: pd.DataFrame) -> pd.DataFrame:
            df_['dataset_id'] = [dnm2id[dnm]] * len(df_)  # Add dataset source information to each row
            return df_
        # Global label across all datasets, all splits
        # Needed for inversely mapping to local label regardless of joined split, e.g. train/test,
        #   in case some label only in certain split
        lbs_lb = sorted(set(join_its(df.label.unique() for dsets in d_dsets.values() for df in dsets.values())))
        lbs_lb = datasets.ClassLabel(names=lbs_lb)

        def dfs2dset(dfs: Iterable[pd.DataFrame]) -> datasets.Dataset:
            df = pd.concat(dfs)
            # The string labels **may overlap** across the datasets
            # Keep internal feature label ordering same as dataset id
            lbs_dset = sorted(dnm2id, key=dnm2id.get)
            df.label.replace(to_replace=lbs_lb.names, value=range(lbs_lb.num_classes), inplace=True)
            features = datasets.Features(
                text=datasets.Value(dtype='string'), label=lbs_lb, dataset_id=datasets.ClassLabel(names=lbs_dset)
            )
            return datasets.Dataset.from_pandas(df, features=features)
        tr = dfs2dset(pre_concat(dnm, dsets['train']) for dnm, dsets in d_dsets.items())
        vl = dfs2dset(pre_concat(dnm, dsets['test']) for dnm, dsets in d_dsets.items())
        dsets = datasets.DatasetDict(train=tr, test=vl)
        dsets.save_to_disk(os.path.join(path_out, nm_dsets))
    else:
        for dnm, dsets in d_dsets.items():
            dsets.save_to_disk(os.path.join(path_out, f'{dnm}-label-grouped' if group_labels else dnm))
    logger.info(f'Dataset(s) saved to {logi(path_out)}')


def map_ag_news():
    dnm = 'ag_news'
    d_dset = config(f'UTCD.datasets.{dnm}')
    ext = config('UTCD.dataset_ext')
    path_dset = os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET)
    path = d_dset['path']
    path = os.path.join(path_dset, f'{path}.{ext}')
    with open(path) as f:
        dsets: Dict = json.load(f)
    d_lb2desc = config(f'baselines.gpt2-nvidia.label-descriptors.{dnm}')
    for split, dset in dsets.items():
        dsets[split] = [[txt, d_lb2desc[lb]] for txt, lb in dset]
    with open(os.path.join(path_dset, f'{dnm}.json'), 'w') as f:
        json.dump(dsets, f, indent=4)


def get_utcd_info() -> pd.DataFrame:
    """
    Metadata about each dataset in UTCD
    """
    k_avg_tok = [f'{mode}-{text_type}_avg_tokens' for text_type in ['txt', 'lb'] for mode in ['re', 'bert', 'gpt2']]
    infos = [
        dict(dataset_name=dnm, aspect=d_dset['aspect'], out_of_domain=d_dset['out_of_domain'])
        | {f'{split}-{k}': v for split, d_info in d_dset['splits'].items() for k, v in d_info.items()}
        | {k: d_dset[k] for k in k_avg_tok}
        for dnm, d_dset in config('UTCD.datasets').items()
    ]
    return pd.DataFrame(infos)


if __name__ == '__main__':
    from icecream import ic

    def sanity_check(dsets_nm):
        path = os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed', dsets_nm)
        ic(path)
        dset = datasets.load_from_disk(path)
        te, vl = dset['train'], dset['test']
        ic(len(te), len(vl))
        lbs = vl.features['label']
        ic(lbs)
        ic(vl[60], lbs.int2str(118))

    def get_utcd():
        process_utcd_dataset(join=True)
        sanity_check('UTCD')
    # get_utcd()

    def get_utcd_ood():
        process_utcd_dataset(in_domain=True, join=True)
        sanity_check('UTCD-ood')
    # get_utcd_ood()

    # process_utcd_dataset(in_domain=True, join=False, group_labels=False)
    # process_utcd_dataset(in_domain=False, join=False, group_labels=False)
    # process_utcd_dataset(in_domain=True, join=False, group_labels=True)
    # process_utcd_dataset(in_domain=False, join=False, group_labels=True)

    def sanity_check_ln_eurlex():
        path = os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed', 'multi_eurlex')
        ic(path)
        dset = datasets.load_from_disk(path)
        ic(dset, len(dset))
    # sanity_check_ln_eurlex()
    # ic(lst2uniq_ids([5, 6, 7, 6, 5, 1]))

    def output_utcd_info():
        df = get_utcd_info()
        ic(df)
        df.to_csv(os.path.join(PATH_BASE, DIR_PROJ, DIR_DSET, 'utcd-info.csv'), float_format='%.3f')
    output_utcd_info()
