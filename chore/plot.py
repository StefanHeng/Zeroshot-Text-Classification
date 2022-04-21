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

    from chore.util import *

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
            plot_class_heatmap('slurp', cmap=cm, save=True, dir_save=os.path.join(get_chore_base(), 'plot'))

    # plot_class_heatmap(dnm, save=False, dir_save=os.path.join(path_base, 'plot'))

    def save_plots(model_name, strategy):
        fn = get_dnm2csv_path_fn(model_name, strategy)
        md_nm, strat = prettier_model_name_n_sample_strategy(model_name, strategy)
        dir_save = os.path.join(get_chore_base(), 'plot', f'{now(for_path=True)}, {md_nm} with {strat}')
        os.makedirs(dir_save, exist_ok=True)
        for dnm_ in config('UTCD.datasets'):
            plot_class_heatmap(dnm_, save=True, dir_save=dir_save, dnm2csv_path=fn, approach=strategy)
    # save_plots(split='neg-samp', approach='Random Negative Sampling')

    def plot_setups_acc(
            setups: List[Dict[str, str]], domain: str = 'in', train_strategy: str = 'vanilla', save=False
    ):
        ca(domain=domain)

        domain_str = 'in-domain' if domain == 'in' else 'out-of-domain'
        aspect2dnms = cconfig('domain2aspect2dataset-names')[domain]
        fig, axes = plt.subplots(1, len(aspect2dnms), figsize=(16, 6))
        # color-code by model name
        models = list(OrderedDict((d['model_name'], None) for d in setups))
        cs = sns.color_palette(palette='husl', n_colors=len(models))

        for ax, (aspect, dnms) in zip(axes, aspect2dnms.items()):
            for s in setups:
                md_nm, sample_strat = s['model_name'], s['sampling_strategy']
                train_strat = s.get('training_strategy', train_strategy)
                ca(model_name=md_nm, sampling_strategy=sample_strat, training_strategy=train_strategy)
                path = get_dnm2csv_path_fn(
                    model_name=md_nm, sampling_strategy=sample_strat, training_strategy=train_strat, domain=domain)
                # As percentage
                scores = [a*100 for a in dataset_acc(dataset_names=dnms, dnm2csv_path=path, return_type='list')]
                dnm_ints = list(range(len(dnms)))
                ls = ':' if sample_strat == 'vect' else '-'
                i_color = models.index(md_nm)
                label = prettier_model_name_n_sample_strategy(md_nm, sample_strat, pprint=True)
                ax.plot(dnm_ints, scores, c=cs[i_color], ls=ls, lw=1, marker='.', ms=8, label=label)
            dnm_ints = list(range(len(dnms)))
            ax.set_xticks(dnm_ints, labels=[dnms[i] for i in dnm_ints])
            ax.set_title(f'{aspect} split')
        scores = np.concatenate([l.get_ydata() for ax in axes for l in ax.lines])
        edges = [np.concatenate([l.get_xdata() for l in ax.lines]) for ax in axes]
        # ic(scores)
        ma, mi = scores.max(), scores.min()
        ma, mi = min(round(ma, -1)+10, 100), max(round(mi, -1)-10, -5)
        for ax, edges_ in zip(axes, edges):
            ax.set_ylim([mi, ma])
            ma_, mi_ = float(edges_.max()), float(edges_.min())
            assert ma_.is_integer() and mi_.is_integer()
            ax.set_xlim([mi_-0.25, ma_+0.25])
        title = f'Training Clasisifcation Accuracy - {domain_str} evaluation'
        plt.suptitle(title)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        fig.supylabel('Classification Accuracy (%)')
        fig.supxlabel('Dataset')
        if save:
            plt.savefig(os.path.join(get_chore_base(), 'plot', f'{now(for_path=True)}, {title}.png'), dpi=300)
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
        plot_setups_acc(setups, domain='in', save=True)
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
        plot_setups_acc(setups, domain='out', save=True)
    # plot_out_of_domain()

    def plot_in_implicit():
        setups = [
            ('binary-bert', 'rand'),
            ('bert-nli', 'rand'),
            ('bi-encoder', 'rand'),
            # ('gpt2-nvidia', 'NA')
        ]
        plot_setups_acc(setups, domain='out', train_strategy='implicit', save=True)
        # plot_setups_acc(setups, domain='out', train_strategy='implicit', save=False)
    # plot_in_implicit()

    def plot_berts_implicit():
        setups = [
            ('binary-bert', 'rand', 'vanilla'),
            ('binary-bert', 'rand', 'implicit'),
            ('binary-bert', 'rand', 'implicit-text-aspect'),
            # ('binary-bert', 'implicit-text-sep'),
        ]
        setups = [dict(zip(['model', 'sampling_strategy', 'training_strategy'], s)) for s in setups]
        plot_setups_acc(setups, domain='in', save=False)
    plot_berts_implicit()
