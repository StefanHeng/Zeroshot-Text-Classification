import csv
from typing import Callable

from zeroshot_encoder.util import *


def summaries2table_row(summaries: List[Dict], exp='latex', acc_only: bool = True) -> Union[str, List[str]]:
    def out_single(d: Dict) -> str:
        if acc_only:
            return f'{d["f1-score"]*100:4.1f}'
        else:
            return '/'.join(f'{d[k]*100:4.1f}' for k in ('precision', 'recall', 'f1-score'))
    if exp == 'latex':
        return ' & '.join(f'${out_single(d)}$' for d in summaries)
    elif exp == 'csv':
        return [out_single(d) for d in summaries]
    else:
        raise ValueError('Unexpected type')


if __name__ == '__main__':
    from chore import *

    def quick_table(dataset_names, model_name, strategy):
        fn = get_dnm2csv_path_fn(model_name, strategy)
        summaries = dataset_acc_summary(dataset_names, dnm2csv_path=fn)
        dnm_, asp, prec, rec, f1 = summaries[0].keys()

        n_metric = 10

        def print_single(with_color=True):
            if with_color:
                perfs = (f'      {k*100:>4.1f}' for k in (prec, rec, f1))
                print(f'{dnm_:>24}{asp:>10}' + ''.join(logi(p) for p in perfs))
            else:
                print(f'{dnm_:>24}{asp:>10}{prec:>{n_metric}}{rec:>{n_metric}}{f1:>{n_metric}}')

        print_single(with_color=False)
        for summary in summaries:
            dnm_, asp, prec, rec, f1 = summary.values()
            print_single(with_color=True)
    # quick_table(config('UTCD.datasets'))

    def get_latex_table_row():
        for model_name, strategy in [('binary-bert', 'rand'), ('bert-nli', 'rand'), ('gpt2-nvidia', 'NA')]:
            summaries = dataset_acc_summary(DNMS_IN, dnm2csv_path=get_dnm2csv_path_fn(model_name, strategy))
            print(summaries2table_row(summaries, exp='csv'))
    get_latex_table_row()

    def get_csv(dataset_names: Iterable[str]):
        with open(os.path.join(
                PATH_BASE_CHORE, 'table', f'in-domain classification accuracy, {now(sep="-")}.csv'
        ), 'w') as f:
            writer = csv.writer(f)

            writer.writerow([''] * 2 + sum(([aspect] * 3 for aspect in ('emotion', 'intent', 'topic')), start=[]))
            writer.writerow(['model', 'sampling strategy'] + DNMS_IN)

            setups = [
                ('binary-bert', 'rand'), ('binary-bert', 'vect'),
                ('bert-nli', 'rand'), ('bert-nli', 'vect'),
                ('bi-encoder', 'rand'), ('bi-encoder', 'vect'),
                ('gpt2-nvidia', 'NA')
            ]
            for model_name, strategy in setups:
                fn = get_dnm2csv_path_fn(model_name, strategy)
                summaries = dataset_acc_summary(dataset_names, dnm2csv_path=fn)
                model_name, strategy = md_nm_n_strat2str_out(model_name, strategy)
                writer.writerow([model_name, strategy] + summaries2table_row(summaries, exp='csv'))
    # get_csv(DNMS)
