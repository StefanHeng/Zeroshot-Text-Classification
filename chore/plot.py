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
        dir_save = os.path.join(PATH_BASE_CHORE, 'plot', f'{now(sep="-")}, {md_nm} with {strat}')
        os.makedirs(dir_save, exist_ok=True)
        for dnm_ in config('UTCD.datasets'):
            plot_class_heatmap(dnm_, save=True, dir_save=dir_save, dnm2csv_path=fn, approach=strategy)
    # save_plots(split='neg-samp', approach='Random Negative Sampling')

    def plot_approaches_performance(save=False):
        setups = [('binary-bert', 'rand'), ('binary-bert', 'vect'), ('bert-nli', 'rand'), ('bert-nli', 'vect')]
        fig, axes = plt.subplots(1, (len(D_DNMS)), figsize=(16, 6))
        n_color = len(setups)
        cs = sns.color_palette(palette='husl', n_colors=n_color)

        for ax, (aspect, dnms) in zip(axes, D_DNMS.items()):
            for i_color, (md_nm, strat) in enumerate(setups):
                # if md_nm == 'bert-nli':  # Uses half the color; TODO: intended for 2 classes
                #     i_color += 2*n_color//3
                scores = [
                    s['f1-score']
                    for s in dataset_acc_summary(dataset_names=dnms, dnm2csv_path=get_dnm2csv_path_fn(md_nm, strat))
                ]
                dnm_ints = list(range(len(dnms)))
                ax.plot(
                    dnm_ints, scores, c=cs[i_color], lw=1, ms=16, marker='.',
                    label=md_nm_n_strat2str_out(md_nm, strat, pprint=True)
                )
                ax.set_xticks(dnm_ints, labels=[dnms[i] for i in dnm_ints])
            ax.set_title(f'{aspect} split')
        scores = np.concatenate([l.get_ydata() for ax in axes for l in ax.lines])
        ma, mi = scores.max(), scores.min()
        ma, mi = round(ma, 1), round(mi, 1)-0.1
        for ax in axes:
            ax.set_ylim([mi, ma])
        title = 'Clasisifcation Accuracy - In-domain evaluation'
        plt.suptitle(title)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        if save:
            plt.savefig(os.path.join(PATH_BASE_CHORE, 'plot', f'{now(sep="-")}, {title}.png'), dpi=300)
        else:
            plt.show()
    plot_approaches_performance(save=False)