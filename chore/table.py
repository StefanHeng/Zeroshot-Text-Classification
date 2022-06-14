import os
import csv
from typing import List, Tuple, Dict, Union
from collections import OrderedDict

from stefutil import *
from chore.util import *


def summaries2table_row(summaries: List[Dict], exp='latex', acc_only: bool = True) -> Union[str, List[str]]:
    def get_single(d: Dict) -> str:
        if acc_only:
            return f'{d["f1-score"]*100:4.1f}'
        else:
            return '/'.join(f'{d[k]*100:4.1f}' for k in ('precision', 'recall', 'f1-score'))
    if exp == 'latex':
        return ' & '.join(f'${get_single(d)}$' for d in summaries)
    elif exp == 'csv':
        return [get_single(d) for d in summaries]
    else:
        raise ValueError('Unexpected type')


def write_csv_train_strat_in_col(
        train_strategies: Tuple[str, ...] = ('vanilla', 'implicit'), chore_config: ChoreConfig = cconfig
):
    """
    On only the Bert models & GPT2, random sampling
    """
    dom2dnms = chore_config('domain2dataset-names-all')
    dnms_in, dnms_out = dom2dnms['in'], dom2dnms['out']
    train_strategies = list(train_strategies)
    header_in = sum([[f'in/{dnm}'] * len(train_strategies) for dnm in dnms_in], start=[])
    header_out = sum([[f'out/{dnm}'] * len(train_strategies) for dnm in dnms_out], start=[])
    rows: List[List] = [  # header
        ['domain/dataset name'] + header_in + header_out,
        ['training strategy\\\nmodel'] + train_strategies * (len(dnms_in) + len(dnms_out))
    ]
    setups = [
        ('binary-bert', 'rand'),
        # ('bert-nli', 'rand'),
        ('bi-encoder', 'rand'),
        ('gpt2-nvidia', 'NA'),
    ]
    for model_name, samp_strat in setups:
        col_name = prettier_setup(model_name, samp_strat, chore_config=chore_config)
        row = [col_name]
        for domain, dnms in zip(['in', 'out'], [dnms_in, dnms_out]):
            for dnm in dnms:
                for train_strat in train_strategies:
                    if model_name == 'gpt2-nvidia' and train_strat != 'implicit-on-text-encode-sep':  # TODO: not ready yet
                        acc = None
                    else:
                        dnm2csv_path = get_dnm2csv_path_fn(
                            model_name, samp_strat, train_strat, domain=domain, chore_config=chore_config
                        )
                        acc = dataset_acc(dnm, dnm2csv_path=dnm2csv_path, suppress_not_found=True)
                        acc = f'{acc * 100:4.1f}' if bool(acc) else None
                    row.append(acc)
        rows.append(row)

    fnm = f'Model classification accuracy by dataset & training strategy, {now(for_path=True)}.csv'
    fnm = os.path.join(get_chore_base(), 'table', fnm)
    with open(fnm, 'w') as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)


def write_csv_train_strat_in_row(
        train_strategies: Tuple[str, ...] = ('vanilla', 'implicit'), chore_config: ChoreConfig = cconfig,
        domain: str = 'in'
):
    dom2dnms = chore_config('domain2dataset-names-all')
    dnms = dom2dnms[domain]
    train_strategies = list(train_strategies)
    rows: List[List] = [  # header
        ['architecture', 'training strategy\\dataset name'] + dnms + ['avg']
    ]
    setups = [
        ('binary-bert', 'rand'),
        ('bi-encoder', 'rand'),
        ('gpt2-nvidia', 'NA'),
    ]
    for model_name, samp_strat in setups:
        for train_strat in train_strategies:
            col_name = prettier_setup(model_name, samp_strat, chore_config=chore_config)
            row = [col_name, train_strat]
            accs = []
            for dnm in dnms:
                if model_name == 'gpt2-nvidia' and train_strat != 'implicit-on-text-encode-sep':  # TODO: not ready yet
                    acc = None
                else:
                    dnm2csv_path = get_dnm2csv_path_fn(
                        model_name, samp_strat, train_strat, domain=domain, chore_config=chore_config
                    )
                    acc = dataset_acc(dnm, dnm2csv_path=dnm2csv_path, suppress_not_found=True)
                    accs.append(acc)
                    acc = f'{acc * 100:4.1f}' if bool(acc) else None
                row.append(acc)
            if accs:
                row.append(f'{sum(accs) / len(accs) * 100:4.1f}')
            else:
                row.append(None)
            rows.append(row)
    domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
    fnm = f'{now(for_path=True)}_{domain_str.capitalize()} classification accuracy by dataset & training strategy.csv'
    fnm = os.path.join(get_chore_base(), 'table', fnm)
    with open(fnm, 'w') as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)


if __name__ == '__main__':
    # def quick_table(dataset_names, model_name, strategy):
    #     fn = get_dnm2csv_path_fn(model_name, strategy)
    #     summaries = dataset_acc(dataset_names, dnm2csv_path=fn)
    #     dnm_, asp, prec, rec, f1 = summaries[0].keys()
    #
    #     n_metric = 10
    #
    #     def print_single(with_color=True):
    #         if with_color:
    #             perfs = (f'      {k*100:>4.1f}' for k in (prec, rec, f1))
    #             print(f'{dnm_:>24}{asp:>10}' + ''.join(logi(p) for p in perfs))
    #         else:
    #             print(f'{dnm_:>24}{asp:>10}{prec:>{n_metric}}{rec:>{n_metric}}{f1:>{n_metric}}')
    #
    #     print_single(with_color=False)
    #     for summary in summaries:
    #         dnm_, asp, prec, rec, f1 = summary.values()
    #         print_single(with_color=True)
    # # quick_table(config('UTCD.datasets'))
    #
    # def get_latex_table_row():
    #     for model_name, strategy in [('binary-bert', 'rand'), ('bert-nli', 'rand'), ('gpt2-nvidia', 'NA')]:
    #         summaries = dataset_acc(IN_DATASET_NAMES, dnm2csv_path=get_dnm2csv_path_fn(model_name, strategy))
    #         print(summaries2table_row(summaries, exp='csv'))
    # # get_latex_table_row()
    #
    # def get_csv(domain: str = 'in'):
    #     assert domain in ['in', 'out']
    #     dnms = IN_DATASET_NAMES if domain == 'in' else OUT_DATASET_NAMES
    #     with open(os.path.join(
    #             PATH_BASE_CHORE, 'table', f'in-domain classification accuracy, {now(for_path=True)}.csv'
    #     ), 'w') as f:
    #         writer = csv.writer(f)
    #
    #         writer.writerow([''] * 2 + sum(([aspect] * 3 for aspect in ('emotion', 'intent', 'topic')), start=[]))
    #         writer.writerow(['model', 'sampling strategy'] + dnms)
    #
    #         setups_in = [
    #             ('binary-bert', 'rand'), ('binary-bert', 'vect'),
    #             ('bert-nli', 'rand'), ('bert-nli', 'vect'),
    #             ('bi-encoder', 'rand'), ('bi-encoder', 'vect'),
    #             ('gpt2-nvidia', 'NA')
    #         ]
    #         setups_out = [
    #             ('binary-bert', 'rand'),
    #             ('bert-nli', 'rand'), ('bert-nli', 'vect'),
    #             ('bi-encoder', 'rand'), ('bi-encoder', 'vect'),
    #             ('gpt2-nvidia', 'NA')
    #         ]
    #         for model_name, strategy in (setups_in if domain == 'in' else setups_out):
    #             fn = get_dnm2csv_path_fn(model_name, strategy, domain=domain == 'in')
    #             summaries = dataset_acc(dnms, dnm2csv_path=fn)
    #             model_name, strategy = prettier_model_name_n_sample_strategy(model_name, strategy)
    #             writer.writerow([model_name, strategy] + summaries2table_row(summaries, exp='csv'))
    # # get_csv(domain='out')

    def write_csv_model_setup_by_dataset(training_strategy: str = 'vanilla'):
        if training_strategy != 'all':
            ca(training_strategy=training_strategy)
        train_setup2csv_path: OrderedDict = cconfig('train-setup2dset-eval-path')
        # # Use the dict as a set, enforce row ordering in output
        if training_strategy == 'all':  # all unique setups apart from `domain`
            setups = OrderedDict([(conf[:3], None) for conf in train_setup2csv_path.keys()])
        else:
            setups = OrderedDict([
                (conf[:2], None) for conf in train_setup2csv_path.keys() if conf[2] == training_strategy
            ])
        dom2dnms = cconfig('domain2dataset-names-all')  # include deprecated `arxiv`
        dnms_in, dnms_out = dom2dnms['in'], dom2dnms['out']

        n_in, n_out = len(dnms_in), len(dnms_out)
        # rows of headers
        rows: List[List] = [
            ['', 'domain'] + ['in'] * n_in + ['out'] * n_out,
            ['model_name', 'sampling_strategy'] + dnms_in + dnms_out
        ]

        def get_row(model_name, sampling_strategy, train_strat):
            row = [model_name, sampling_strategy]
            key = [model_name, sampling_strategy, train_strat]
            key_in, key_out = tuple([*key, 'in']), tuple([*key, 'out'])
            if key_in in train_setup2csv_path:  # for ignoring `arxiv`
                d_out = dataset_acc(dnms_in, dnm2csv_path=get_dnm2csv_path_fn(*key_in))
                row += [f'{d_out[dnm]*100:4.1f}' if bool(d_out[dnm]) else None for dnm in dnms_in]
            else:
                row += [''] * n_in
            if key_out in train_setup2csv_path:
                d_out = dataset_acc(dnms_out, dnm2csv_path=get_dnm2csv_path_fn(*key_out), suppress_not_found=True)
                row += [f'{d_out[dnm]*100:4.1f}' if bool(d_out[dnm]) else None for dnm in dnms_out]
            else:
                row += [''] * n_out
            return row

        if training_strategy == 'all':
            for setup in setups:
                rows.append(get_row(*setup))
        else:
            for setup in setups:
                rows.append(get_row(*setup, training_strategy))

        fnm = f'{training_strategy.capitalize()} classification accuracy, {now(for_path=True)}.csv'
        fnm = os.path.join(get_chore_base(), 'table', fnm)
        with open(fnm, 'w') as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)
    # write_csv()
    # write_csv(training_strategy='implicit')
    # write_csv_model_setup_by_dataset(training_strategy='all')


    def write():
        ttrial = 'asp-norm'
        chore_config = ChoreConfig(train_trial=ttrial)
        tr_strats = (
            'vanilla',
            # 'implicit',
            # 'implicit-on-text-encode-aspect',
            'implicit-on-text-encode-sep',
            'explicit'
        )
        # dom = 'in'
        dom = 'out'
        write_csv_train_strat_in_row(train_strategies=tr_strats, chore_config=chore_config, domain=dom)
    # write()

    def get_seq_cls_numbers(domain: str = 'in'):
        md_nm = 'bert-seq-cls'
        dnms = cconfig(f'domain2dataset-names-all.{domain}')
        samp_strat, train_strat = 'NA', 'vanilla'
        dnm2csv_path = get_dnm2csv_path_fn(md_nm, samp_strat, train_strat, domain=domain, chore_config=cconfig)
        for dnm in dnms:
            acc = dataset_acc(dnm, dnm2csv_path=dnm2csv_path)
            mic(dnm, acc)
            exit(1)
    get_seq_cls_numbers()

