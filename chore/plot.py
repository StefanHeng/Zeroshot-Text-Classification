import csv
from typing import Callable

from zeroshot_encoder.util import *


def dataset_acc_summary(dataset_names: Iterable[str], dnm2csv_path: Callable = None) -> List[Dict]:
    def get_single(d_nm):
        df = pd.read_csv(dnm2csv_path(d_nm))
        df.rename(columns={'Unnamed: 0': 'class'}, inplace=True)  # Per csv files
        df = df.iloc[-3:, :].reset_index(drop=True)
        df.support = df.support.astype(int)
        row_acc = df.iloc[0, :]
        return OrderedDict([
            ('dnm', d_nm), ('aspect', config(f'UTCD.datasets.{d_nm}.aspect')),
            ('precision', row_acc.precision), ('recall', row_acc.recall), ('f1-score', row_acc['f1-score'])
        ])
    return [get_single(d) for d in dataset_names]


def summaries2table_row(summaries: List[Dict], exp='latex') -> Union[str, List[str]]:
    def out_single(d: Dict) -> str:
        return '/'.join(f'{d[k]*100:4.1f}' for k in ('precision', 'recall', 'f1-score'))
    if exp == 'latex':
        return ' & '.join(f'${out_single(d)}$' for d in summaries)
    elif exp == 'csv':
        return [out_single(d) for d in summaries]
    else:
        raise ValueError('Unexpected type')


def get_dnm2csv_path_fn(model_name: str, strategy: str) -> Callable:
    if strategy == 'rand':  # Radnom negative sampling
        if model_name == 'binary-bert':
            return (
                lambda d_nm:
                os.path.join(path_base, 'csvs', 'binary-bert-random-negative-sampling', f'binary_bert_rand_{d_nm}.csv')
            )
        elif model_name == 'bert-nli':
            return (
                lambda d_nm:
                os.path.join(path_base, 'csvs', 'bert-nli-random-negative-sampling', f'nli_bert_{d_nm}.csv')
            )
        else:
            raise ValueError('unexpected model name')
    else:
        return lambda d_nm: os.path.join(path_base, 'csvs', 'default', f'binary_bert_{d_nm}.csv')


_CONFIG = {
    'model-name': {'binary-bert': 'Binary Bert', 'bert-nli': 'Bert-NLI'},
    'sampling-strategy': {'rand': 'Random Negative Sampling'}
}


def md_nm_n_strat2str_out(model_name: str, strategy: str) -> Tuple[str, str]:
    return _CONFIG['model-name'][model_name], _CONFIG['sampling-strategy'][strategy]


def plot_class_heatmap(
        dataset_name: str, save=False, dir_save: str = '', dnm2csv_path: Callable = None,
        normalize=False, cmap: str = 'mako', global_only: bool = True, approach: str = None
):
    df = pd.read_csv(dnm2csv_path(dataset_name))
    df.rename(columns={'Unnamed: 0': 'class'}, inplace=True)  # Per csv files
    df.support = df.support.astype(int)
    df, df_global = df.iloc[:-3, :], df.iloc[-3:, :].reset_index(drop=True)  # Heuristics, of global performance
    # csv seems wrong, get it from last column, last row
    total_support = df_global.at[0, 'support'] = df_global.iloc[-1, -1]
    if normalize:
        anchor_acc, anchor_support = dict(vmin=0, vmax=1), dict(vmin=0, vmax=total_support)
    else:
        anchor_acc, anchor_support = dict(), dict()

    width_ratios = [3, 3 / 10, 1, 3 / 10]
    if global_only:
        n_row = len(df_global)
        fig, axes = plt.subplots(1, 4, figsize=(6, n_row / 3 + 1), gridspec_kw=dict(width_ratios=width_ratios))
        sns.heatmap(
            pd.DataFrame(df_global[['precision', 'recall', 'f1-score']]), annot=True,
            ax=axes[0], cbar_ax=axes[1], yticklabels=df_global['class'], cmap=cmap, **anchor_acc
        )
        sns.heatmap(
            pd.DataFrame(df_global['support']), annot=True, fmt='g',
            ax=axes[2], cbar_ax=axes[3], yticklabels=False, cmap=cmap, **anchor_support
        )
        axes[0].xaxis.set_ticks_position('top') and axes[0].tick_params(top=False)
        axes[2].xaxis.set_ticks_position('top') and axes[2].tick_params(top=False)
    else:
        n_row = len(df) + len(df_global)
        fig, axes = plt.subplots(2, 4, figsize=(6, n_row/3 + 1), gridspec_kw=dict(
            width_ratios=width_ratios, height_ratios=[len(df), len(df_global)]
        ))
        sns.heatmap(
            pd.DataFrame(df[['precision', 'recall', 'f1-score']]), annot=True,
            ax=axes[0, 0], cbar_ax=axes[0, 1], yticklabels=df['class'], cmap=cmap, **anchor_acc
        )
        sns.heatmap(
            pd.DataFrame(df['support']), annot=True, fmt='g',
            ax=axes[0, 2], cbar_ax=axes[0, 3], yticklabels=False, cmap=cmap, **anchor_support
        )
        axes[0, 0].xaxis.set_ticks_position('top') and axes[0, 0].tick_params(top=False)
        axes[0, 2].xaxis.set_ticks_position('top') and axes[0, 2].tick_params(top=False)

        sns.heatmap(
            pd.DataFrame(df_global[['precision', 'recall', 'f1-score']]), annot=True,
            ax=axes[1, 0], cbar_ax=axes[1, 1], yticklabels=df_global['class'], cmap=cmap, **anchor_acc
        )
        sns.heatmap(
            pd.DataFrame(df_global['support']), annot=True, fmt='g',
            ax=axes[1, 2], cbar_ax=axes[1, 3], yticklabels=False, cmap=cmap, **anchor_support
        )
        axes[1, 0].xaxis.set_ticks_position('top') and axes[1, 0].tick_params(top=False)
        axes[1, 2].xaxis.set_ticks_position('top') and axes[1, 2].tick_params(top=False)
    title = f'Heatmap of in-domain evaluation classification performance \n' \
            f' - Binary Bert on {dataset_name} with {approach}'
    plt.suptitle(title)
    if save:
        title = title.replace('\n', '')
        plt.savefig(os.path.join(dir_save, f'{title}.png'), dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    from icecream import ic

    # dnm = 'slurp'
    # dnm = 'yahoo'
    path_base = os.path.join(PATH_BASE, DIR_PROJ, 'chore')

    def pick_cmap():
        cmaps = [
            'mako',
            'CMRmap',
            'RdYlBu',
            'Spectral',
            'bone',
            'gnuplot',
            'gnuplot2',
            'icefire',
            'rainbow',
            'rocket',
            'terrain',
            'twilight',
            'twilight_shifted'
        ]
        for cm in cmaps:
            plot_class_heatmap('slurp', cmap=cm, save=True, dir_save=os.path.join(path_base, 'plot'))

    # plot_class_heatmap(dnm, save=False, dir_save=os.path.join(path_base, 'plot'))

    def save_plots(model_name, strategy):
        fn = get_dnm2csv_path_fn(model_name, strategy)
        md_nm, strat = md_nm_n_strat2str_out(model_name, strategy)
        dir_save = os.path.join(path_base, 'plot', f'{now(sep="-")}, {md_nm} with {strat}')
        os.makedirs(dir_save, exist_ok=True)
        for dnm_ in config('UTCD.datasets'):
            plot_class_heatmap(dnm_, save=True, dir_save=dir_save, dnm2csv_path=fn, approach=strategy)
    # save_plots(split='neg-samp', approach='Random Negative Sampling')

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

    dnms = [
        'go_emotion', 'sentiment_tweets_2020', 'emotion',
        'sgd', 'clinc_150', 'slurp',
        'ag_news', 'dbpedia', 'yahoo'
    ]

    def get_latex_table_row(model_name, strategy):
        summaries = dataset_acc_summary(dnms, dnm2csv_path=get_dnm2csv_path_fn(model_name, strategy))
        print(summaries2table_row(summaries))
    # get_latex_table_row()

    def get_csv(dataset_names):
        with open(os.path.join(path_base, 'table', 'in-domain evaluation.csv'), 'w') as f:
            writer = csv.writer(f)

            writer.writerow([''] * 2 + sum(([aspect] * 3 for aspect in ('emotion', 'intent', 'topic')), start=[]))
            writer.writerow(['model', 'sampling strategy'] + dnms)

            for model_name, strategy in [('binary-bert', 'rand'), ('bert-nli', 'rand')]:
                fn = get_dnm2csv_path_fn(model_name, strategy)
                summaries = dataset_acc_summary(dataset_names, dnm2csv_path=fn)
                model_name, strategy = md_nm_n_strat2str_out(model_name, strategy)
                writer.writerow([model_name, strategy] + summaries2table_row(summaries, exp='csv'))
    get_csv(dnms)
