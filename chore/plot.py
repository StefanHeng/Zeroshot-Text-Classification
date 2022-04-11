import os
from typing import Callable

from zeroshot_encoder.util import *


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

    from chore import *

    # dnm = 'slurp'
    # dnm = 'yahoo'

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
            plot_class_heatmap('slurp', cmap=cm, save=True, dir_save=os.path.join(PATH_BASE_CHORE, 'plot'))

    # plot_class_heatmap(dnm, save=False, dir_save=os.path.join(path_base, 'plot'))

    def save_plots(model_name, strategy):
        fn = get_dnm2csv_path_fn(model_name, strategy)
        md_nm, strat = md_nm_n_strat2str_out(model_name, strategy)
        dir_save = os.path.join(PATH_BASE_CHORE, 'plot', f'{now(for_path=True)}, {md_nm} with {strat}')
        os.makedirs(dir_save, exist_ok=True)
        for dnm_ in config('UTCD.datasets'):
            plot_class_heatmap(dnm_, save=True, dir_save=dir_save, dnm2csv_path=fn, approach=strategy)
    # save_plots(split='neg-samp', approach='Random Negative Sampling')

    def plot_approaches_performance(
            setups: List[Tuple[str, str]], domain: str = 'in', train_strategy: str = 'vanilla', save=False
    ):
        domains = ['in', 'out']
        assert domain in domains, f'Unexpected domain: expected one of {domains}, got {domain}'
        train_strats = ['vanilla', 'implicit', 'explicit']
        assert train_strategy in train_strats, \
            f'Unexpected train strategy: expected one of {logi(train_strats)}, got {logi(train_strategy)}'

        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
        d_dnms = D_DNMS[domain_str]
        fig, axes = plt.subplots(1, len(d_dnms), figsize=(16, 6))
        # Ordered, uniq list names, for consistent color code between runs
        models = list(OrderedDict((md, None) for md, strat in setups))
        n_color = len(models)+1
        cs = sns.color_palette(palette='husl', n_colors=n_color)

        for ax, (aspect, dnms) in zip(axes, d_dnms.items()):
            for md_nm, sample_strat in setups:
                path = get_dnm2csv_path_fn(
                    model_name=md_nm, sample_strategy=sample_strat, train_strategy=train_strategy, domain=domain)
                # As percentage
                scores = [s['f1-score'] * 100 for s in dataset_acc_summary(dataset_names=dnms, dnm2csv_path=path)]
                dnm_ints = list(range(len(dnms)))
                ls = ':' if sample_strat == 'vect' else '-'
                i_color = models.index(md_nm)
                label = md_nm_n_strat2str_out(md_nm, sample_strat, pprint=True)
                ax.plot(dnm_ints, scores, c=cs[i_color], ls=ls, lw=1, marker='.', ms=8, label=label)
            dnm_ints = list(range(len(dnms)))
            ax.set_xticks(dnm_ints, labels=[dnms[i] for i in dnm_ints])
            ax.set_title(f'{aspect} split')
        scores = np.concatenate([l.get_ydata() for ax in axes for l in ax.lines])
        edges = [np.concatenate([l.get_xdata() for l in ax.lines]) for ax in axes]
        ma, mi = scores.max(), scores.min()
        ma, mi = min(round(ma, -1)+10, 100), max(round(mi, -1)-10, -5)
        for ax, edges_ in zip(axes, edges):
            ax.set_ylim([mi, ma])
            ma_, mi_ = float(edges_.max()), float(edges_.min())
            assert ma_.is_integer() and mi_.is_integer()
            ax.set_xlim([mi_-0.25, ma_+0.25])
        title = f'{train_strategy.capitalize()} Training Clasisifcation Accuracy - {domain_str} evaluation'
        plt.suptitle(title)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        fig.supylabel('Classification Accuracy (%)')
        fig.supxlabel('Dataset')
        if save:  # TODO: distance between dsets
            plt.savefig(os.path.join(PATH_BASE_CHORE, 'plot', f'{now(for_path=True)}, {title}.png'), dpi=300)
        else:
            plt.show()
    # plot_approaches_performance(save=True)
    # plot_approaches_performance(setups_in, in_domain=True, save=False)

    def plot_in_domain():
        setups = [
            ('binary-bert', 'rand'),
            ('bert-nli', 'rand'),
            ('bi-encoder', 'rand'),
            # ('dual-bi-encoder', 'none'),
            ('gpt2-nvidia', 'NA')
        ]
        # plot_approaches_performance(setups_in, in_domain=True, save=False)
        plot_approaches_performance(setups, domain='in', save=True)
    # plot_in_domain()

    def plot_out_of_domain():
        setups = [
            ('binary-bert', 'rand'),
            ('bert-nli', 'rand'),
            # ('bert-nli', 'vect'),
            ('bi-encoder', 'rand'),
            # ('dual-bi-encoder', 'none'),
            ('gpt2-nvidia', 'NA'),
        ]
        # plot_approaches_performance(setups_out, in_domain=False, save=False)
        plot_approaches_performance(setups, domain='out', save=True)
    # plot_out_of_domain()

    def plot_in_implicit():
        setups = [
            ('binary-bert', 'rand'),
            ('bert-nli', 'rand'),
            ('bi-encoder', 'rand'),
            # ('gpt2-nvidia', 'NA')
        ]
        plot_approaches_performance(setups, domain='in', train_strategy='implicit', save=True)
    plot_in_implicit()
