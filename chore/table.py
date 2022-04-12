import csv

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


if __name__ == '__main__':
    from icecream import ic

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

    def write_csv(training_strategy: str = 'vanilla'):
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
        fnm = f'{training_strategy.capitalize()} classification accuracy, {now(for_path=True)}.csv'
        path_out = os.path.join(get_chore_base(), 'table', fnm)

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

        with open(path_out, 'w') as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)
    # write_csv()
    # write_csv(training_strategy='implicit')
    write_csv(training_strategy='all')
