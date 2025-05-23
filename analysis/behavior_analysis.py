import itertools
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import yaml
from matplotlib.ticker import MaxNLocator

import nwb_utils.utils_behavior as bhv_utils
from nwb_utils import utils_io
from nwb_utils.utils_misc import get_continuous_time_periods
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame
from nwb_wrappers import nwb_reader_functions as nwb_read

warnings.filterwarnings("ignore")


def plot_single_session(combine_bhv_data, color_palette, saving_path):
    sessions_list = np.unique(combine_bhv_data['session_id'].values[:])
    n_sessions = len(sessions_list)
    expert_sessions_table = []
    print(f"N sessions : {n_sessions}")
    for session_id in sessions_list:
        session_table, switches, block_size = bhv_utils.get_standard_single_session_table(combine_bhv_data, session=session_id)
        if session_table['behavior'].values[0] == 'free_licking':
            print(f"No plot for {session_table['behavior'].values[0]} sessions")
            continue

        # Set plot parameters.
        raster_marker = 2
        marker_width = 2
        figsize = (15, 8)

        d = session_table.loc[session_table.early_lick == 0][int(block_size / 2)::block_size]
        marker = itertools.cycle(['o', 's'])
        markers = [next(marker) for i in d["opto_stim"].unique()]

        if session_table['behavior'].values[0] in ['context', 'whisker_context']:
            figure, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                                   gridspec_kw={'height_ratios': [2, 3]},
                                                   sharex=True)
            # Plot the lines :
            # # Plot the global perf
            # d['correct_a'] = d.hr_a
            # d['correct_n'] = 1 - d.hr_n
            # d['correct_w'] = [1 - d.hr_w.values[i] if d.context.values[i] == 0 else d.hr_w.values[i] for i in
            #                   range(len(d))]
            # ax0.plot(d.trial, d.correct, color='r', linewidth=2, linestyle='--')
            # ax0.axhline(y=0.6, xmin=0, xmax=1, color='g', linewidth=2, linestyle='--')
            # sns.lineplot(data=d, x='trial', y='correct_n', color='gray', ax=ax0,
            #              markers=markers)
            # if 'correct_w' in list(d.columns) and (not np.isnan(d.correct_w.values[:]).all()):
            #     sns.lineplot(data=d, x='trial', y='correct_w', color=color_palette[2], ax=ax0, markers=markers)
            # if 'correct_a' in list(d.columns) and (not np.isnan(d.correct_a.values[:]).all()):
            #     sns.lineplot(data=d, x='trial', y='correct_a', color=color_palette[0], ax=ax0, markers=markers)
            # ax0.set_ylim([-0.05, 1.05])
            # ax0.set_ylabel('Correct choice')

            # Plot the contrast perf
            hr_w_contrast = [(np.abs(d.hr_w.values[i] - d.hr_w.values[i - 1]) +
                              np.abs(d.hr_w.values[i] - d.hr_w.values[i + 1])) / 2 for
                             i in np.arange(1, d.hr_w.size - 1)]
            hr_w_contrast.insert(0, np.nan)
            hr_w_contrast.insert(len(hr_w_contrast), np.nan)
            d['contrast_w'] = hr_w_contrast

            hr_a_contrast = [(np.abs(d.hr_a.values[i] - d.hr_a.values[i - 1]) +
                              np.abs(d.hr_a.values[i] - d.hr_a.values[i + 1])) / 2 for
                             i in np.arange(1, d.hr_a.size - 1)]
            hr_a_contrast.insert(0, np.nan)
            hr_a_contrast.insert(len(hr_a_contrast), np.nan)
            d['contrast_a'] = hr_a_contrast

            hr_n_contrast = [(np.abs(d.hr_n.values[i] - d.hr_n.values[i - 1]) +
                              np.abs(d.hr_n.values[i] - d.hr_n.values[i + 1])) / 2 for
                             i in np.arange(1, d.hr_n.size - 1)]
            hr_n_contrast.insert(0, np.nan)
            hr_n_contrast.insert(len(hr_n_contrast), np.nan)
            d['contrast_n'] = hr_n_contrast

            sns.lineplot(data=d, x='trial', y='contrast_n',
                         color=color_palette[4], ax=ax1, markers=markers)
            if 'contrast_w' in list(d.columns) and (not np.isnan(d.contrast_w.values[:]).all()):
                sns.lineplot(data=d, x='trial', y='contrast_w',
                             color=color_palette[2], ax=ax1, markers=markers)
            if 'contrast_a' in list(d.columns) and (not np.isnan(d.contrast_a.values[:]).all()):
                sns.lineplot(data=d, x='trial', y='contrast_a',
                             color=color_palette[0], ax=ax1, markers=markers)
            ax1.set_ylim([-0.05, 1.05])
            ax1.set_ylabel('Contrast Lick Probability')
            ax1.axhline(y=0.375, xmin=0, xmax=1, color='g', linewidth=2, linestyle='--')
            if d.contrast_w.count() > 2:
                bootstrap_res = scipy.stats.bootstrap(data=(d.contrast_w,), statistic=np.nanmean, n_resamples=10000)
                y_err = np.zeros((2, 1))
                y_err[0, 0] = np.nanmean(d.contrast_w) - bootstrap_res.confidence_interval.low
                y_err[1, 0] = bootstrap_res.confidence_interval.high - np.nanmean(d.contrast_w)
                ci_low = bootstrap_res.confidence_interval.low
                ci_high = bootstrap_res.confidence_interval.high
            else:
                y_err = np.zeros((2, 1))
                y_err[0, 0] = 0
                y_err[1, 0] = 0
                ci_low = np.nanmean(d.contrast_w)
                ci_high = np.nanmean(d.contrast_w)
            ax1.errorbar(max(d.trial) + 10, np.nanmean(d.contrast_w),
                         yerr=y_err,
                         xerr=None, fmt='o', color=color_palette[2], ecolor=color_palette[2], elinewidth=2)
            rwd_hr_w = d.loc[d.context == 1].hr_w
            non_rwd_hr_w = d.loc[d.context == 0].hr_w
            d_prime = (np.nanmean(rwd_hr_w) - np.nanmean(non_rwd_hr_w)) / np.sqrt(0.5 * (np.var(rwd_hr_w) + np.var(non_rwd_hr_w)))
            d_prime_lsens = scipy.stats.norm.ppf(min(np.nanmean(rwd_hr_w), 0.999)) - scipy.stats.norm.ppf(max(np.nanmean(non_rwd_hr_w), 0.001))
            perf_dict = {'mouse_id': [session_id[0:5]],
                         'session_id': [session_id],
                         'w_contrast_thresh': [0.375],
                         'w_contrast_mean': [np.nanmean(d.contrast_w)],
                         'w_contrast_ci_low': [ci_low],
                         'w_contrast_ci_high': [ci_high],
                         'w_context_expert': [ci_low > 0.375],
                         'd_prime': [d_prime],
                         'lsens_d_prime': [d_prime_lsens]}
            expert_sessions_table.append(pd.DataFrame.from_dict(perf_dict))
            if ci_low > 0.375:
                ax1.plot(max(d.trial) + 10, 0.9, marker='*', color=color_palette[2])
        else:
            figure, ax2 = plt.subplots(1, 1, figsize=figsize)

        # Plot the lines
        sns.lineplot(data=d, x='trial', y='hr_n', color='k', ax=ax2,
                     markers=markers)

        if 'hr_w' in list(d.columns) and (not np.isnan(d.hr_w.values[:]).all()):
            sns.lineplot(data=d, x='trial', y='hr_w', color=color_palette[2], ax=ax2, markers=markers)
        if 'hr_a' in list(d.columns) and (not np.isnan(d.hr_a.values[:]).all()):
            sns.lineplot(data=d, x='trial', y='hr_a', color=color_palette[0], ax=ax2, markers=markers)

        if session_table['behavior'].values[0] in ['context', 'whisker_context']:
            rewarded_bloc_bool = list(d.context.values[:])
            bloc_limites = np.arange(start=0, stop=len(session_table.index), step=block_size)
            bloc_area_color = ['green' if i == 1 else 'firebrick' for i in rewarded_bloc_bool]
            if bloc_limites[-1] < len(session_table.index):
                bloc_area = [(bloc_limites[i], bloc_limites[i + 1]) for i in range(len(bloc_limites) - 1)]
                bloc_area.append((bloc_limites[-1], len(session_table.index)))
                if len(bloc_area) > len(bloc_area_color):
                    bloc_area = bloc_area[0: len(bloc_area_color)]
                for index, coords in enumerate(bloc_area):
                    color = bloc_area_color[index]
                    ax2.axvspan(coords[0], coords[1], alpha=0.25, facecolor=color, zorder=1)

        # Plot the trials :
        ax2.scatter(x=session_table.loc[session_table.lick_flag == 0]['trial'],
                    y=session_table.loc[session_table.lick_flag == 0]['outcome_n'] - 0.1,
                    color=color_palette[4], marker=raster_marker, linewidths=marker_width)
        ax2.scatter(x=session_table.loc[session_table.lick_flag == 1]['trial'],
                    y=session_table.loc[session_table.lick_flag == 1]['outcome_n'] - 1.1,
                    color='k', marker=raster_marker, linewidths=marker_width)

        if 'hr_a' in list(d.columns) and (not np.isnan(d.hr_w.values[:]).all()):
            ax2.scatter(x=session_table.loc[session_table.lick_flag == 0]['trial'],
                        y=session_table.loc[session_table.lick_flag == 0]['outcome_a'] - 0.15,
                        color=color_palette[1], marker=raster_marker, linewidths=marker_width)
            ax2.scatter(x=session_table.loc[session_table.lick_flag == 1]['trial'],
                        y=session_table.loc[session_table.lick_flag == 1]['outcome_a'] - 1.15,
                        color=color_palette[0], marker=raster_marker, linewidths=marker_width)

        if 'hr_w' in list(d.columns) and (not np.isnan(d.hr_w.values[:]).all()):
            ax2.scatter(x=session_table.loc[session_table.lick_flag == 0]['trial'],
                        y=session_table.loc[session_table.lick_flag == 0]['outcome_w'] - 0.2,
                        color=color_palette[3], marker=raster_marker, linewidths=marker_width)
            ax2.scatter(x=session_table.loc[session_table.lick_flag == 1]['trial'],
                        y=session_table.loc[session_table.lick_flag == 1]['outcome_w'] - 1.2,
                        color=color_palette[2], marker=raster_marker, linewidths=marker_width)

        ax2.set_ylim([-0.2, 1.05])
        ax2.set_xlabel('Trial number')
        ax2.set_ylabel('Lick probability')
        figure_title = f"{session_table.mouse_id.values[0]}, {session_id[0:14]}, {session_table.behavior.values[0]} " \
                       f"{session_table.day.values[0]}"
        plt.suptitle(figure_title)
        sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        figure_name = f"{session_table.mouse_id.values[0]}_{session_table.behavior.values[0]}_" \
                      f"{session_table.day.values[0]}"
        session_saving_path = os.path.join(saving_path, f"{session_table.mouse_id.values[0]}",
                                           f'{session_table.session_id.values[0]}_{session_table.behavior.values[0]}_{session_table.day.values[0]}')
        if not os.path.exists(session_saving_path):
            os.makedirs(session_saving_path)
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{session_saving_path}', f'{figure_name}.{save_format}'),
                           format=f"{save_format}")

        plt.close()

    if expert_sessions_table:
        expert_sessions_table = pd.concat(expert_sessions_table)
        session_index = []
        for mouse in expert_sessions_table['mouse_id'].unique():
            session_index.extend(np.arange(0, len(expert_sessions_table.loc[expert_sessions_table.mouse_id == mouse])))
        expert_sessions_table['session_index'] = session_index
        expert_sessions_table.to_excel(os.path.join('//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Pol_Bech/Session_list', 'context_expert_sessions.xlsx'))
        fig = sns.relplot(
            data=expert_sessions_table, x='session_index', y="w_contrast_mean", col="mouse_id", hue='w_context_expert',
            height=1.5, aspect=1, col_wrap=4, legend=True)
        fig.set_ylabels('Whisker contrast')
        fig.set(ylim=(-0.05, 1.05))
        fig.fig.suptitle('Global whisker context performance')
        fig.tight_layout()
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', f'whisker_context_perf.{save_format}'), format=f"{save_format}")


def plot_single_mouse_across_days(combine_bhv_data, color_palette, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot average across days for {n_mice} mice")
    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only Auditory and Whisker days
        mouse_table = mouse_table[mouse_table.behavior.isin(('auditory', 'whisker', 'whisker_psy'))]

        # Select columns for plot
        cols = ['outcome_a', 'outcome_w', 'outcome_n', 'day', 'opto_stim']
        df = mouse_table.loc[mouse_table.early_lick == 0, cols]

        # Compute hit rates. Use transform to propagate hit rate to all entries.
        df['hr_w'] = df.groupby(['day', 'opto_stim'], as_index=False)['outcome_w'].transform(np.nanmean)
        df['hr_a'] = df.groupby(['day', 'opto_stim'], as_index=False)['outcome_a'].transform(np.nanmean)
        df['hr_n'] = df.groupby(['day', 'opto_stim'], as_index=False)['outcome_n'].transform(np.nanmean)

        # Average by day for this mouse
        df_by_day = df.groupby(['day', 'opto_stim'], as_index=False).agg(np.nanmean)

        # Do the plot
        figsize = (4, 6)
        figure, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(data=df_by_day, x='day', y='hr_n', color='k', ax=ax, marker='o')
        sns.lineplot(data=df_by_day, x='day', y='hr_a', color=color_palette[0], ax=ax, marker='o')
        if max(df_by_day['day'].values) >= 0:  # This means there's one whisker training day at least
            sns.lineplot(data=df_by_day, x='day', y='hr_w', color=color_palette[2], ax=ax, marker='o')

        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        ax.set_title(f"{mouse_id}")
        sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{mouse_id}_fast_learning.{save_format}'),
                           format=f"{save_format}")

        plt.close()


def plot_single_mouse_weight_across_days(combine_bhv_data, color_palette, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot normalized mouse weight across days for {n_mice} mice")

    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only Auditory and Whisker days
        mouse_table = mouse_table[mouse_table.behavior.isin(('auditory', 'whisker', 'whisker_psy'))]

        # Select columns for plot
        cols = ['normalized_weight', 'day']
        df = mouse_table.loc[mouse_table.early_lick == 0, cols]
        df.normalized_weight = df.normalized_weight * 100

        # Do the plot
        figsize = (4, 2)
        figure, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(data=df, x='day', y='normalized_weight', color='grey', ax=ax, marker='o')

        ax.set_ylim([70, 100])
        ax.set_yticks([70, 80, 90, 100])
        ax.set_xlabel('Day')
        ax.set_ylabel('Normalized weight [%]')
        ax.set_title(f"{mouse_id}")
        sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{mouse_id}_weight.{save_format}'),
                           format=f"{save_format}",
                           bbox_inches='tight'
                           )

        plt.close()


def categorical_context_lineplot(data, hue, palette, mouse_id, saving_path, figname):
    figsize = (6, 9)
    figure, ax0 = plt.subplots(1, 1, figsize=figsize)

    sns.pointplot(data.loc[data['opto_stim'] == 0], x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], ax=ax0, markers='o')
    sns.pointplot(data.loc[data['opto_stim'] == 0], x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], ax=ax0, markers='o')
    sns.pointplot(data.loc[data['opto_stim'] == 0], x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], ax=ax0, markers='o')

    ax0.set_ylim([-0.1, 1.05])
    ax0.set_xlabel('Day')
    ax0.set_ylabel('Lick probability')
    sns.despine()

    sns.move_legend(
        ax0, "lower center",
        bbox_to_anchor=(0.5, 1), ncol=3, title=f"{mouse_id}", frameon=False,
    )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def categorical_context_boxplot(data, hue, palette, mouse_id, saving_path, figname):
    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    sns.boxplot(data=data, x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], ax=ax0)
    sns.boxplot(data=data, x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], ax=ax1)
    sns.boxplot(data=data, x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], ax=ax2)

    for ax in [ax0, ax1, ax2]:
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def categorical_context_stripplot(data, hue, palette, mouse_id, saving_path, figname):
    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    sns.stripplot(data=data, x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], dodge=True,
                  jitter=0.2, ax=ax0)
    sns.stripplot(data=data, x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], dodge=True,
                  jitter=0.2, ax=ax1)
    sns.stripplot(data=data, x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], dodge=True,
                  jitter=0.2, ax=ax2)
    for ax in [ax0, ax1, ax2]:
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def categorical_context_pointplot(data, hue, palette, mouse_id, saving_path, figname):
    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax0)
    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax1)
    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax2)

    for ax in [ax0, ax1, ax2]:
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def categorical_context_opto(data, hue, palette, mouse_id, saving_path, figname):

    data = data.groupby('day').filter(lambda x: x['opto_stim'].sum()>0)

    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)
    hue_order = sorted(data[hue].unique())

    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_n', hue=hue, hue_order=hue_order, palette=palette['catch_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax0)
    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_a', hue=hue, hue_order=hue_order, palette=palette['aud_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax1)
    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_w', hue=hue, hue_order=hue_order, palette=palette['wh_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax2)

    sns.pointplot(data=data.loc[data['opto_stim'] == 1], x='day', y='hr_n', hue=hue, hue_order=hue_order, palette=palette['catch_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax0, markers='*', linestyles='dashed')
    sns.pointplot(data=data.loc[data['opto_stim'] == 1], x='day', y='hr_a', hue=hue, hue_order=hue_order, palette=palette['aud_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax1, markers='*', linestyles='dashed')
    sns.pointplot(data=data.loc[data['opto_stim'] == 1], x='day', y='hr_w', hue=hue, hue_order=hue_order, palette=palette['wh_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax2, markers='*', linestyles='dashed')

    for ax in [ax0, ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(title=f'{mouse_id}', handles=handles, labels=['Non Rewarded', 'Rewarded', 'Non Rewarded opto', 'Rewarded opto'])
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def categorical_context_opto_avg(data, hue, palette, mouse_id, saving_path, figname):

    data = data.groupby('day').filter(lambda x: x['opto_stim'].sum()>0)
    hue_order = sorted(data[hue].unique())

    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    sns.pointplot(data=data, x='opto_stim', y='hr_n', hue=hue, hue_order=hue_order,
                  palette=palette['catch_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax0)
    sns.pointplot(data=data, x='opto_stim', y='hr_a', hue=hue,  hue_order=hue_order,
                  palette=palette['aud_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax1)
    sns.pointplot(data=data, x='opto_stim', y='hr_w', hue=hue, hue_order=hue_order,
                  palette=palette['wh_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax2)

    for ax in [ax0, ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(title=f'{mouse_id}', handles=handles,
                  labels=['Not-Rewarded', 'Rewarded'])
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Opto stim')
        ax.set_ylabel('Lick probability')
        ax.set_xticklabels(["Stim off", "Stim on"])
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def plot_single_mouse_across_context_days(combine_bhv_data, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot average across context days for {n_mice} mice")
    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only Context days
        mouse_table = mouse_table[mouse_table.behavior.isin(['whisker_context'])]
        mouse_table = mouse_table.reset_index(drop=True)
        if mouse_table.empty:
            print(f"No context day: return")
            return

        # Add column with string for rewarded and non-rewarded context
        mouse_table['context_rwd_str'] = mouse_table['context']
        mouse_table = mouse_table.replace({'context_rwd_str': {1: 'Rewarded', 0: 'Non-Rewarded'}})

        # Select columns for the first plot
        cols = ['outcome_a', 'outcome_w', 'outcome_n', 'day', 'context', 'context_background', 'context_rwd_str', 'opto_stim']
        df = mouse_table.loc[mouse_table.early_lick == 0, cols]

        # Compute hit rates. Use transform to propagate hit rate to all entries.
        df['hr_w'] = df.groupby(['day', 'context', 'context_rwd_str', 'opto_stim'], as_index=False)['outcome_w'] \
            .transform(np.nanmean)
        df['hr_a'] = df.groupby(['day', 'context', 'context_rwd_str', 'opto_stim'], as_index=False)['outcome_a'] \
            .transform(np.nanmean)
        df['hr_n'] = df.groupby(['day', 'context', 'context_rwd_str', 'opto_stim'], as_index=False)['outcome_n'] \
            .transform(np.nanmean)

        # Average by day and context blocks for this mouse
        df_by_day = df.groupby(['day', 'context', 'context_rwd_str', 'context_background', 'opto_stim'], as_index=False).agg(np.nanmean)

        # Look at the mean difference in Lick probability between rewarded and non-rewarded context
        df_by_day_diff = df_by_day.sort_values(by=['day', 'context_rwd_str'], ascending=True)
        df_by_day_diff['hr_w_diff'] = df_by_day_diff.loc[df_by_day_diff['opto_stim'] == 0].groupby('day')['hr_w'].diff()
        df_by_day_diff['hr_a_diff'] = df_by_day_diff.loc[df_by_day_diff['opto_stim'] == 0].groupby('day')['hr_a'].diff()
        df_by_day_diff['hr_n_diff'] = df_by_day_diff.loc[df_by_day_diff['opto_stim'] == 0].groupby('day')['hr_n'].diff()

        df_by_day_diff['hr_w_diff_opto'] = df_by_day_diff.groupby(['day', 'context'])['hr_w'].diff()
        df_by_day_diff['hr_a_diff_opto'] = df_by_day_diff.groupby(['day', 'context'])['hr_a'].diff()
        df_by_day_diff['hr_n_diff_opto'] = df_by_day_diff.groupby(['day', 'context'])['hr_n'].diff()

        # Plot the diff
        figsize = (6, 9)
        figure, ax0 = plt.subplots(1, 1, figsize=figsize)

        sns.pointplot(df_by_day_diff.dropna(subset=['hr_w_diff']), x='day', y='hr_n_diff', color='black', ax=ax0, markers='o')
        sns.pointplot(df_by_day_diff.dropna(subset=['hr_w_diff']), x='day', y='hr_a_diff', color='mediumblue', ax=ax0, markers='o')
        sns.pointplot(df_by_day_diff.dropna(subset=['hr_w_diff']), x='day', y='hr_w_diff', color='green', ax=ax0, markers='o')

        ax0.set_ylim([-0.2, 1.05])

        ax0.set_xlabel('Day')
        ax0.set_ylabel("\u0394 Lick probability")
        ax0.set_title(f"{mouse_id}")
        sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        figname = f"{mouse_id}_context_diff"
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                           format=f"{save_format}")
        plt.close()

        # Get each session to look at each individual
        mouse_session_list = list(np.unique(mouse_table['session_id'].values[:]))
        by_block_data = []
        for mouse_session in mouse_session_list:
            session_table, switches, block_size = bhv_utils.get_standard_single_session_table(mouse_table, session=mouse_session,
                                                                                     verbose=False)
            session_table = session_table.loc[session_table.early_lick == 0][int(block_size / 2)::block_size]
            by_block_data.append(session_table)
        by_block_data = pd.concat(by_block_data, ignore_index=True)
        by_block_data['context_rwd_str'] = by_block_data['context']
        by_block_data = by_block_data.replace({'context_rwd_str': {1: 'Rewarded', 0: 'Non-Rewarded'}})

        # Define the two colort palette
        context_name_palette = {
            'catch_palette': {'brown': 'black', 'pink': 'darkgray'},
            'wh_palette': {'brown': 'green', 'pink': 'limegreen'},
            'aud_palette': {'brown': 'mediumblue', 'pink': 'cornflowerblue'}
        }
        context_reward_palette = {
            'catch_palette': {'Rewarded': 'black', 'Non-Rewarded': 'darkgray'},
            'wh_palette': {'Rewarded': 'green', 'Non-Rewarded': 'firebrick'},
            'aud_palette': {'Rewarded': 'mediumblue', 'Non-Rewarded': 'cornflowerblue'}
        }

        # Do the plots  : one point per day per context
        # Keep the context by background
        # categorical_context_lineplot(data=df_by_day, hue='context_background', palette=context_name_palette,
        #                              mouse_id=mouse_id, saving_path=saving_path, figname=f"{mouse_id}_context_name")

        # Keep the context by rewarded
        categorical_context_lineplot(data=df_by_day, hue='context_rwd_str', palette=context_reward_palette,
                                     mouse_id=mouse_id, saving_path=saving_path, figname=f"{mouse_id}_context_reward")

        # Do the plots : with context block distribution for each day
        # # Boxplots
        # categorical_context_boxplot(data=by_block_data, hue='context_background', palette=context_name_palette,
        #                             mouse_id=mouse_id, saving_path=saving_path,
        #                             figname=f"{mouse_id}_box_context_name_bloc")
        #
        # categorical_context_boxplot(data=by_block_data, hue='context_rwd_str', palette=context_reward_palette,
        #                             mouse_id=mouse_id, saving_path=saving_path,
        #                             figname=f"{mouse_id}_box_context_reward")
        #
        # # Stripplots
        # categorical_context_stripplot(data=by_block_data, hue='context_background', palette=context_name_palette,
        #                               mouse_id=mouse_id, saving_path=saving_path,
        #                               figname=f"{mouse_id}_strip_context_name_bloc")
        #
        # categorical_context_stripplot(data=by_block_data, hue='context_rwd_str', palette=context_reward_palette,
        #                               mouse_id=mouse_id, saving_path=saving_path,
        #                               figname=f"{mouse_id}_strip_context_reward")

        # Point-plots
        # categorical_context_pointplot(data=by_block_data, hue='context_background', palette=context_name_palette,
        #                               mouse_id=mouse_id, saving_path=saving_path,
        #                               figname=f"{mouse_id}_point_context_name_bloc")

        categorical_context_pointplot(data=by_block_data, hue='context_rwd_str', palette=context_reward_palette,
                                      mouse_id=mouse_id, saving_path=saving_path,
                                      figname=f"{mouse_id}_point_context_reward")

        if not df_by_day_diff['hr_w_diff_opto'].dropna().empty:
            categorical_context_opto(data=by_block_data, hue='context_rwd_str', palette=context_reward_palette,
                                      mouse_id=mouse_id, saving_path=saving_path,
                                      figname=f"{mouse_id}_point_context_opto")
            opto_diff_by_day(data=df_by_day_diff, mouse_id=mouse_id, palette=context_reward_palette, saving_path=saving_path,
                             figname=f"{mouse_id}_context_opto_diff")

            opto_palette = {
                'catch_palette': {0: 'black', 1: 'darkgray'},
                'wh_palette': {0: 'green', 1: 'firebrick'},
                'aud_palette': {0: 'mediumblue', 1: 'cornflowerblue'}
            }
            categorical_context_opto_avg(data=by_block_data, hue='context_rwd_str', palette=context_reward_palette,
                                      mouse_id=mouse_id, saving_path=saving_path,
                                      figname=f"{mouse_id}_point_context_opto_avg")
            opto_diff_avg(data=df_by_day_diff, mouse_id=mouse_id, palette=context_reward_palette, saving_path=saving_path,
                             figname=f"{mouse_id}_context_opto_diff_avg")


def opto_diff_by_day(data, mouse_id, palette, saving_path, figname):
    figsize = (6, 9)
    figure, ax0 = plt.subplots(1, 1, figsize=figsize)

    sns.pointplot(data, x='day', y='hr_n_diff_opto', hue='context_rwd_str', palette=palette['catch_palette'], ax=ax0, markers='o')
    sns.pointplot(data, x='day', y='hr_a_diff_opto', hue='context_rwd_str', palette=palette['aud_palette'], ax=ax0, markers='o')
    sns.pointplot(data, x='day', y='hr_w_diff_opto', hue='context_rwd_str', palette=palette['wh_palette'], ax=ax0, markers='o')
    ax0.set_ylim([-1.05, 1.05])
    # else:

    ax0.set_xlabel('Day')
    ax0.set_ylabel("\u0394 Lick probability")
    ax0.set_title(f"{mouse_id} opto")
    sns.despine()

    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def opto_diff_avg(data, mouse_id, palette, saving_path, figname):
    figsize = (6, 9)
    figure, ax0 = plt.subplots(1, 1, figsize=figsize)

    sns.pointplot(data, x='context_rwd_str', y='hr_n_diff_opto', palette=palette['catch_palette'], ax=ax0, markers='o')
    sns.pointplot(data, x='context_rwd_str', y='hr_a_diff_opto', palette=palette['aud_palette'], ax=ax0, markers='o')
    sns.pointplot(data, x='context_rwd_str', y='hr_w_diff_opto', palette=palette['wh_palette'], ax=ax0, markers='o')
    ax0.set_ylim([-0.55, 0.55])
    # else:

    ax0.set_xlabel('Day')
    ax0.set_ylabel("\u0394 Lick probability")
    ax0.set_title(f"{mouse_id} opto")
    sns.despine()

    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def get_single_session_time_to_switch(combine_bhv_data, do_single_session_plot=False):
    sessions_list = np.unique(combine_bhv_data['session_id'].values[:])
    n_sessions = len(sessions_list)
    print(f"N sessions : {n_sessions}")
    to_rewarded_transitions_prob = dict()
    to_non_rewarded_transitions_prob = dict()
    for session_id in sessions_list:
        session_table, switches, block_size = bhv_utils.get_standard_single_session_table(combine_bhv_data, session=session_id)

        # Keep only the session with context
        if session_table['behavior'].values[0] not in 'whisker_context':
            continue

        # Keep only the whisker trials
        whisker_session_table = session_table.loc[session_table.trial_type == 'whisker_trial']
        whisker_session_table = whisker_session_table.reset_index(drop=True)

        # extract licks array
        licks = whisker_session_table.outcome_w.values[:]

        # Extract transitions rwd to non rwd and opposite
        rewarded_transitions = np.where(np.diff(whisker_session_table.context.values[:]) == 1)[0]
        non_rewarded_transitions = np.where(np.diff(whisker_session_table.context.values[:]) == -1)[0]

        # Build rewarded transitions matrix from trial -3 to trial +3
        wh_switches = np.where(np.diff(whisker_session_table.context.values[:]))[0]
        n_trials_around = min(np.diff(wh_switches))
        trials_above = n_trials_around + 1
        trials_below = n_trials_around - 1
        rewarded_transitions_mat = np.zeros((len(rewarded_transitions), 2 * n_trials_around))
        for index, transition in enumerate(list(rewarded_transitions)):
            if transition + trials_above > len(licks):
                rewarded_transitions_mat = rewarded_transitions_mat[0: len(rewarded_transitions) - 1, :]
                continue
            else:
                rewarded_transitions_mat[index, :] = licks[np.arange(transition - trials_below, transition + trials_above)]
        rewarded_transition_prob = np.mean(rewarded_transitions_mat, axis=0)
        to_rewarded_transitions_prob[session_id] = rewarded_transition_prob

        # Build non_rewarded transitions matrix from trial -3 to trial +3
        non_rewarded_transitions_mat = np.zeros((len(non_rewarded_transitions), 2 * n_trials_around))
        for index, transition in enumerate(list(non_rewarded_transitions)):
            if transition + trials_above > len(licks):
                non_rewarded_transitions_mat = non_rewarded_transitions_mat[0: len(non_rewarded_transitions) - 1, :]
                continue
            else:
                non_rewarded_transitions_mat[index, :] = licks[np.arange(transition - trials_below, transition + trials_above)]
        non_rewarded_transition_prob = np.mean(non_rewarded_transitions_mat, axis=0)
        to_non_rewarded_transitions_prob[session_id] = non_rewarded_transition_prob

        # Do single session plot
        if do_single_session_plot:
            figsize = (6, 4)
            figure, ax = plt.subplots(1, 1, figsize=figsize)
            scale = np.arange(-n_trials_around, n_trials_around + 1)
            scale = np.delete(scale, n_trials_around)
            before_switch = scale[np.where(scale < 0)[0]]
            after_switch = scale[np.where(scale > 0)[0]]
            # ax.plot(scale, rewarded_transition_prob, '--go')
            # ax.plot(scale, non_rewarded_transition_prob, '--ro')
            ax.plot(before_switch, rewarded_transition_prob[0: len(before_switch)], '--ro')
            ax.plot(before_switch, non_rewarded_transition_prob[0: len(before_switch)], '--go')
            ax.plot(after_switch, rewarded_transition_prob[len(before_switch):], '--go')
            ax.plot(after_switch, non_rewarded_transition_prob[len(before_switch):], '--ro')
            ax.plot(([-1, 1]), ([rewarded_transition_prob[len(before_switch) - 1], rewarded_transition_prob[len(before_switch)]]),
                    color='grey', zorder=0)
            ax.plot(([-1, 1]), ([non_rewarded_transition_prob[len(before_switch) - 1], non_rewarded_transition_prob[len(before_switch)]]),
                    color='grey', zorder=1)
            ax.set_xlabel('Trial number')
            ax.set_ylabel('Lick probability')
            figure_title = f"{session_table.mouse_id.values[0]}, {session_table.behavior.values[0]} " \
                           f"{session_table.day.values[0]}"
            ax.set_title(figure_title)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylim([-0.1, 1.05])
            plt.xticks(range(-n_trials_around, n_trials_around + 1))
            plt.show()

    return to_rewarded_transitions_prob, to_non_rewarded_transitions_prob


def plot_single_mouse_reaction_time_across_days(combine_bhv_data, color_palette, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot reaction time across days for {n_mice} mice")
    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only Auditory and Whisker days
        mouse_table = mouse_table[mouse_table.behavior.isin(('auditory', 'whisker', 'context'))]

        # Select columns for plot
        cols = ['start_time', 'stop_time', 'lick_time', 'trial_type', 'lick_flag', 'early_lick', 'context',
                'day', 'response_window_start_time']

        # first df with only rewarded context as no trial stop ttl in non-rewarded context: compute reaction time
        df = mouse_table.loc[(mouse_table.early_lick == 0) & (mouse_table.lick_flag == 1) &
                             (mouse_table.context == 1), cols]
        df['computed_reaction_time'] = df['lick_time'] - df['response_window_start_time']
        df = df.replace({'context': {1: 'Rewarded', 0: 'Non-Rewarded'}})

        # second df with only rewarded context as no trial stop ttl in non-rewarded context: compute reaction time
        df_2 = mouse_table.loc[(mouse_table.early_lick == 0) & (mouse_table.lick_flag == 1), cols]
        df_2['computed_reaction_time'] = df_2['lick_time'] - df_2['response_window_start_time']
        df_2 = df_2.replace({'context': {1: 'Rewarded', 0: 'Non-Rewarded'}})

        trial_types = np.sort(list(np.unique(mouse_table.trial_type.values[:])))
        colors = [color_palette[0], color_palette[4], color_palette[2]]
        context_reward_palette = {
            'auditory_trial': {'Rewarded': 'mediumblue', 'Non-Rewarded': 'cornflowerblue'},
            'no_stim_trial': {'Rewarded': 'black', 'Non-Rewarded': 'darkgray'},
            'whisker_trial': {'Rewarded': 'green', 'Non-Rewarded': 'firebrick'},
        }

        # Do the plot with all trials
        figsize = (18, 18)
        figure, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=figsize)
        figname = f"{mouse_id}_reaction_time"

        for index, ax in enumerate([ax0, ax1, ax2]):

            sns.boxenplot(df.loc[df.trial_type == trial_types[index]], x='day', y='computed_reaction_time',
                          color=colors[index], ax=ax)

            ax.set_ylim([-0.1, 1.25])
            ax.set_xlabel('Day')
            ax.set_ylabel(f'{trial_types[index].capitalize()} \n Reaction time (s)')
            if index == 0:
                ax.set_title(f"{mouse_id}")
            sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                           format=f"{save_format}")
        plt.close()

        # Do the plot by context
        figsize = (18, 18)
        figure, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=figsize)
        figname = f"{mouse_id}_reaction_time_context"

        for index, ax in enumerate([ax0, ax1, ax2]):

            sns.boxenplot(df_2.loc[df_2.trial_type == trial_types[index]], x='day', y='computed_reaction_time',
                          hue='context', palette=context_reward_palette.get(trial_types[index]), ax=ax)

            ax.set_ylim([-0.1, 1.25])
            ax.set_xlabel('Day')
            ax.set_ylabel(f'{trial_types[index].capitalize()} \n Reaction time (s)')
            if index == 0:
                ax.set_title(f"{mouse_id}")
            sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                           format=f"{save_format}")
        plt.close()


def plot_single_mouse_psychometrics_across_days(combine_bhv_data, color_palette, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)

    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only whisker psychophysical days
        mouse_table = mouse_table[mouse_table.behavior.isin(['whisker_psy'])]

        # Remap individual whisker stim amplitude to 5 levels including no_stim_trials (level 0)
        for day_idx in mouse_table['day'].unique():
            print(mouse_id, 'day', day_idx)
            mouse_table_day = mouse_table.loc[mouse_table['day'] == day_idx]
            wh_stim_amp_mapper = {k: idx for idx, k in
                                  enumerate(np.unique(mouse_table_day['whisker_stim_amplitude'].values))}

            mouse_table.loc[mouse_table['day'] == day_idx, 'whisker_stim_levels'] = mouse_table.loc[
                mouse_table['day'] == day_idx, 'whisker_stim_amplitude'].map(wh_stim_amp_mapper)

        stim_amplitude_levels = np.unique(mouse_table['whisker_stim_levels'].values)
        print('Stim levels', stim_amplitude_levels)

        # Plot psychometric curve
        g = sns.FacetGrid(mouse_table, col='day', col_wrap=4, height=4, aspect=1, sharey=True, sharex=True)
        figname = f"{mouse_id}_psychometric_curve"
        g.map(sns.pointplot,
              'whisker_stim_levels',
              'lick_flag',
              'opto_stim',
              order=sorted(stim_amplitude_levels),
              estimator='mean',
              errorbar=('ci', 95),
              n_boot=1000,
              seed=42,
              palette={0: 'forestgreen', 1:'royalblue'}
              )

    g.set_axis_labels('Stimulus amplitude [mT]', 'P(lick)')
    g.set(xticks=range(5), xticklabels=[0, 10, 20, 25, 30])
    g.set(ylim=(0, 1.1))

    # Save figures
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        g.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                  format=f"{save_format}",
                  bbox_inches='tight')
    plt.close()
    return


def plot_behavior(nwb_list, output_folder, plots):
    bhv_data = bhv_utils.build_standard_behavior_table(nwb_list)

    # Plot all single session figures
    colors = ['#225ea8', '#00FFFF', '#238443', '#d51a1c', '#cccccc']
    if 'single_session' in plots:
        print('Plotting single sessions')
        plot_single_session(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder)

    if 'across_days' in plots:
        plot_single_mouse_across_days(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder)

    if 'psycho' in plots:
        plot_single_mouse_psychometrics_across_days(combine_bhv_data=bhv_data, color_palette=colors,
                                                    saving_path=output_folder)
    if 'reaction_time' in plots:
        plot_single_mouse_reaction_time_across_days(combine_bhv_data=bhv_data, color_palette=colors,
                                                    saving_path=output_folder)
    if 'across_context_days' in plots:
        print('Plotting across_context_days')
        plot_single_mouse_across_context_days(combine_bhv_data=bhv_data, saving_path=output_folder)

    if 'context_switch' in plots:
        print('Plotting context_switch')
        get_single_session_time_to_switch(combine_bhv_data=bhv_data, do_single_session_plot=True)

    if 'multiple' in plots:
        plot_multiple_mice_training(bhv_data, saving_path=output_folder, reward_group_hue=False)

    if 'opto_grid' in plots:
        plot_single_mouse_opto_grid(bhv_data, saving_path=output_folder)

    if 'opto_grid_multiple' in plots:
        control_mice = ['PB177']
        plot_multiple_mice_opto_grid(bhv_data, control_mice, saving_path=output_folder)
        plot_control_subtracted_opto_grid(bhv_data, control_mice, saving_path=output_folder)

    if 'context_block_perf' in plots:
        plot_context_performance_stats(combine_bhv_data=bhv_data, expert_table=context_session_table,
                                       saving_path=output_folder)

    return


def plot_group_behavior(nwb_list, info_path):

    # Exclude mouse and format info
    mouse_info_df = pd.read_excel(info_path)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[mouse_info_df['exclude'] == 0]

    mouse_list = mouse_info_df['mouse_id'].values[:]
    mouse_info_df['reward_group'] = mouse_info_df['reward_group'].map({'R+': 1,
                                                                       'R-': 0,
                                                                       'R+proba': 2})

    # Get combined behavior data
    nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in mouse_list)]
    bhv_data = bhv_utils.build_standard_behavior_table(nwb_list)
    bhv_data = bhv_data.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

    print('Number of mice: ', bhv_data['mouse_id'].nunique())

    # Plot group figures

    #plot_multiple_mice_training(bhv_data, colors, reward_group_hue=False)
    #plot_multiple_mice_training(bhv_data, colors, reward_group_hue=True)
    plot_multiple_mice_history(bhv_data)

    return

def plot_multiple_mice_history(bhv_data):

    bhv_data = bhv_data[bhv_data.day==0]

    # Call single mouse function
    _, all_mouse_df = plot_single_mouse_history(bhv_data)

    # Figure content
    trial_types_to_plot = ['whisker_trial', 'auditory_trial']

    for group in all_mouse_df.reward_group.unique():
        if group==0:
            figname = 'all_mice_trial_history_nonrewarded'
            order = ['Miss', 'Hit']
            palette = [lighten_color('crimson', 0.8),
                       lighten_color('mediumblue', 0.8),
                       ]
        elif group==1:
            figname = 'all_mice_trial_history_rewarded'
            order = ['Miss', 'Hit']
            palette = [lighten_color('forestgreen', 0.8),
                       lighten_color('mediumblue', 0.8),
                       ]
        elif group==2:
            figname = 'all_mice_trial_history_rewarded_proba'
            order = ['Miss', 'Hit R+', 'Hit R-']
            palette = [lighten_color('forestgreen', 0.8),
                        lighten_color('mediumblue', 0.8),
                        ]

        # Subset of data
        all_mouse_df_sub = all_mouse_df[all_mouse_df.reward_group==group]

        # Plot
        g = sns.catplot(
            data=all_mouse_df_sub,
            x='outcome',
            y='value',
            kind='strip',
            estimator='mean',
            seed=42,
            orient='v',
            order=order,
            col='history',
            col_wrap=3,
            height=4,
            aspect=0.6,
            sharey=True,
            sharex=True,
            hue='trial_type',
            hue_order=trial_types_to_plot,
            palette=palette,
            legend=False,
        )

        # Axes
        g.set_axis_labels('Outcome', r'P(lick) | Hit at trial $n$')
        g.set(ylim=(-0.05, 1.1))

        plt.show()


        # Save
        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            saving_path = r'M:\analysis\Axel_Bisi\results\history'
            g.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                           format=f"{save_format}",
                           bbox_inches='tight'
                           )


        # Plot
        figname = figname + '_mean'
        g = sns.catplot(
            data=all_mouse_df_sub,
            x='outcome',
            y='value',
            kind='bar',
            seed=42,
            orient='v',
            order=order,
            col='history',
            col_wrap=3,
            height=4,
            aspect=0.6,
            sharey=True,
            sharex=True,
            hue='trial_type',
            hue_order=trial_types_to_plot,
            palette=palette,
            legend=False,
        )

        # Axes
        g.set_axis_labels('Outcome', r'P(lick) | Hit at trial $n$')
        g.set(ylim=(-0.05, 1.1))

        plt.show()

        # Save
        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            saving_path = r'M:\analysis\Axel_Bisi\results\history'
            g.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                      format=f"{save_format}",
                      bbox_inches='tight'
                      )

    return

def plot_multiple_mice_training(bhv_data, saving_path, reward_group_hue=False):
    mice_list = np.unique(bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot average across days for {n_mice} mice")

    # Keep only Auditory and Whisker days
    bhv_data = bhv_data[bhv_data.behavior.isin(('auditory', 'whisker'))]
    # Select columns for plot
    # cols = ['mouse_id', 'reward_group', 'outcome_a', 'outcome_w', 'outcome_n', 'day']
    cols = ['mouse_id', 'outcome_a', 'outcome_w', 'outcome_n', 'day']
    df = bhv_data.loc[bhv_data.early_lick == 0, cols]

    # Compute hit rates. Use transform to propagate hit rate to all entries.
    df['hr_w'] = df.groupby(['mouse_id', 'day'], as_index=False)['outcome_w'].transform(np.nanmean)
    df['hr_a'] = df.groupby(['mouse_id', 'day'], as_index=False)['outcome_a'].transform(np.nanmean)
    df['hr_n'] = df.groupby(['mouse_id', 'day'], as_index=False)['outcome_n'].transform(np.nanmean)

    # Average by day per mouse
    df_by_day = df.groupby(['mouse_id', 'day'], as_index=False).agg(np.nanmean)

    # Select days to show
    first_before_wday = 6
    last_after_wday = 2
    df_by_day = df_by_day[(df_by_day['day'] >= -first_before_wday) &
                                  (df_by_day['day'] <= last_after_wday)]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    remove_top_right_frame(ax)
    sns.set_style('whitegrid', {'grid.color': '.6', 'grid.linestyle': ':'})

    if reward_group_hue:
        reward_group_hue = 'reward_group'
        hue_order=[1,0]
    else:
        reward_group_hue = None
        hue_order=None

    # Plot all mice
    sns.lineplot(data=df_by_day,
                 ax=ax,
                 x='day',
                 y='hr_n',
                 style='mouse_id',
                 hue=reward_group_hue,
                 hue_order=hue_order, # 1 first in case of reward_group_hue is False
                 palette=['k', 'dimgrey'],
                 dashes=False,
                 lw=1,
                 alpha=0.5,
                 legend=False
                 )

    sns.lineplot(data=df_by_day,
                 ax=ax,
                 x='day',
                 y='hr_a',
                 style='mouse_id',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['blue', 'lightblue'],
                 dashes=False,
                 lw=1,
                 alpha=0.5,
                 legend=False
                 )

    sns.lineplot(data=df_by_day[df_by_day['day'].isin(range(0,last_after_wday+1))],
                 ax=ax,
                 x='day',
                 y='hr_w',
                 style='mouse_id',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['forestgreen', 'crimson'],
                 dashes=False,
                 lw=1,
                 alpha=0.5,
                 legend=False
                 )


    # Plot mean over mice
    sns.lineplot(data=df_by_day,
                 x='day',
                 y='hr_n',
                 marker='o',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['k', 'dimgrey'],
                 mew=0,
                 estimator='mean',
                 err_style='bars',
                 errorbar='se',
                 lw=2.5,
                 alpha=1.0)

    sns.lineplot(data=df_by_day,
                 x='day',
                 y='hr_a',
                 marker='o',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['lightblue', 'mediumblue'],
                 mew=0,
                 estimator='mean',
                 err_style='bars',
                 errorbar='se',
                 lw=2.5,
                 alpha=1.0)

    sns.lineplot(data=df_by_day,
                 x='day',
                 y='hr_w',
                 marker='o',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['forestgreen', 'crimson'],
                 mew=0,
                 estimator='mean',
                 err_style='bars',
                 errorbar='se',
                 lw=2.5,
                 alpha=1.0)


    # Axes
    ax.set_xlabel('Days to whisker learning', fontsize=20)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('P(lick)', fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # ensures integers i.e. days
    fig.tight_layout()
    plt.title(r'Training performance, $n={}$'.format(n_mice), fontsize=20)

    plt.show()

    # Save
    save_formats = ['pdf', 'png', 'svg']
    saving_path = saving_path
    for save_format in save_formats:
        if reward_group_hue:
            filename = r'training_performance_reward_group.{}'.format(save_format)
        else:
            filename = r'training_performance.{}'.format(save_format)
        fig.savefig(os.path.join(f'{saving_path}', filename),
                    format=f"{save_format}",
                    bbox_inches='tight')

    plt.close()
    return fig

def plot_multiple_mouse_training_stats(bhv_data, colors, reward_group_hue=False):
    mice_list = np.unique(bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot training statistics for {n_mice} mice")

    # Keep only Auditory and Whisker days
    bhv_data = bhv_data[bhv_data.behavior.isin(('auditory', 'whisker'))]
    # Select columns for plot
    cols = ['mouse_id', 'reward_group', 'outcome_a', 'outcome_w', 'outcome_n', 'day']
    df = bhv_data.loc[bhv_data.early_lick == 0, cols]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    remove_top_right_frame(ax)
    sns.set_style('whitegrid', {'grid.color': '.6', 'grid.linestyle': ':'})

    # Save
    save_formats = ['pdf', 'png', 'svg']
    saving_path = r'M:\analysis\Axel_Bisi\results\training'
    for save_format in save_formats:
        if reward_group_hue:
            filename = r'training_performance_reward_group.{}'.format(save_format)
        else:
            filename = r'training_performance.{}'.format(save_format)
        fig.savefig(os.path.join(f'{saving_path}', filename),
                    format=f"{save_format}",
                    bbox_inches='tight')

    plt.close()
    return fig


def plot_single_mouse_history(bhv_data, saving_path=None):

    # Keep only first trials
    N = 150

    # Init. container for all mouse figure
    all_mouse_df = []

    reward_groups_to_plot = [0,1,2] #R-, R+, R+proba
    trial_types_to_plot = ['whisker_trial',
                           'auditory_trial',
                           ]

    # Keep only whisker day
    bhv_data_sub = bhv_data[bhv_data.behavior == 'whisker']


    # For each mouse_id, find frequency of pairs of outcomes
    mice_list = np.unique(bhv_data_sub['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot trial history for {n_mice} mice")

    for mouse_id in mice_list:

        # Copy needed here for execution
        bhv_data_sub_copy = bhv_data_sub.copy()
        # Data subset for mouse_id
        bhv_data_sub_copy = bhv_data_sub_copy[bhv_data_sub_copy.mouse_id == mouse_id]
        bhv_data_sub_copy = bhv_data_sub_copy[bhv_data_sub_copy.trial_number <= N]
        bhv_data_sub_copy = bhv_data_sub_copy[bhv_data_sub_copy.trial_type.isin(trial_types_to_plot)]

        # Check reward group
        reward_group = bhv_data_sub_copy.reward_group.unique()[0]
        if reward_group not in reward_groups_to_plot:
            continue
        elif reward_group == 0:
            reward_group_hue = 'R-'
        elif reward_group == 1:
            reward_group_hue = 'R+'
        elif reward_group == 2:
            reward_group_hue = 'R+proba'

        # Init. dataframe to store pairs
        pairs_df = pd.DataFrame(columns=['trial_type',
                                         'outcome',
                                         'history',
                                         'trial_index'])
        # Iterate over trial types
        for trial_type in trial_types_to_plot:
            # Get lick trials for trial type
            bhv_data_type = bhv_data_sub_copy[bhv_data_sub_copy.trial_type == trial_type]
            bhv_data_type.reset_index(inplace=True)
            lick_trials = bhv_data_type[bhv_data_type.lick_flag == 1]
            # Iterate over range of trial histories
            for trial_back in [1,2,3]:
                # Iterate over lick trials
                lick_trials_ids = lick_trials.index[trial_back:]  # ignore n first licks
                for lick_idx in lick_trials_ids:

                    if reward_group_hue in ['R+', 'R-']:
                        outcome = bhv_data_type.iloc[lick_idx-trial_back, :]['perf']
                        trial_perf_map = {0: 'Miss', 2: 'Hit', 1: 'Miss', 3: 'Hit'}

                    elif reward_group_hue == 'R+proba':
                        outcome = bhv_data_type.iloc[lick_idx-trial_back, :]['perf']
                        reward_available = bhv_data_type.iloc[lick_idx-trial_back, :]['reward_available']

                        # Note: this assumes only whisker has reward proba
                        if outcome == 2 and reward_available == 1:
                            outcome = outcome
                        elif outcome == 2 and reward_available == 0:
                            outcome = 20

                        trial_perf_map = {0: 'Miss', 2: 'Hit R+', 1: 'Miss', 3: 'Hit R+',
                                          20: 'Hit R-'}

                    entry_dict = {'trial_type': trial_type,
                                  'outcome': outcome,
                                  'history': 'n-{}'.format(trial_back),
                                  'trial_index': lick_idx}
                    pairs_df = pd.concat([pairs_df, pd.DataFrame(entry_dict, index=[lick_idx])],
                                         ignore_index=False)

        # Format columns for plotting
        pairs_df['outcome'] = pairs_df['outcome'].map(trial_perf_map)


        # Calculate lick rate for each trial type, per history values
        if reward_group_hue in ['R+', 'R-']:
            # Map to licks
            lick_map = {'Miss': 0, 'Hit': 1}
            pairs_df['lick_flag'] = pairs_df['outcome'].map(lick_map)

            # Get rate per conditions -> this could be improved, see for R+proba
            hit_groupby_df = pairs_df.groupby(['trial_type', 'history'])['lick_flag'].mean().unstack()
            hit_groupby_df['outcome'] = 'Hit'
            hit_groupby_df.reset_index(inplace=True)
            miss_groupby_df = 1 - pairs_df.groupby(['trial_type', 'history'])['lick_flag'].mean().unstack()
            miss_groupby_df['outcome'] = 'Miss'
            miss_groupby_df.reset_index(inplace=True)
            history_rate_df = pd.concat([hit_groupby_df, miss_groupby_df], axis=0)
            history_rate_df = history_rate_df.melt(id_vars=['trial_type', 'outcome'],
                                                   var_name='history')
        elif reward_group_hue == 'R+proba':

            history_rate_df = pairs_df.groupby(['trial_type', 'history', 'outcome',]).size().reset_index(name='lick_count') # get pair counts per group and outcome
            sum_df = history_rate_df.groupby(['trial_type', 'history'])['lick_count'].transform('sum') # get sum per group
            history_rate_df['value'] = history_rate_df['lick_count'].div(sum_df) # get proportions per group

        # Append to all mouse df
        history_rate_df['mouse_id'] = mouse_id
        history_rate_df['reward_group'] = bhv_data_sub_copy['reward_group'].values[0]
        all_mouse_df.append(history_rate_df)

        if saving_path:
            # Plot based on trial type at single mouse_id level
            figname = f"{mouse_id}_trial_history"

            g = sns.catplot(
                data=history_rate_df,
                x='outcome',
                y='value',
                kind='bar',
                orient='v',
                order=['Miss', 'Hit'],
                col='history',
                col_wrap=3,
                height=4,
                aspect=0.6,
                sharey=True,
                sharex=True,
                hue='trial_type',
                hue_order=trial_types_to_plot,
                palette=[lighten_color('forestgreen', 0.8),
                         lighten_color('mediumblue', 0.8),
                        # lighten_color('k', 0.8),
                         ],
                legend=False,
            )

            # Axes
            g.set_axis_labels('Outcome', r'P(lick) | Hit at trial $n$')
            g.set(ylim=(-0.05, 1.1))

            plt.show()

            save_formats = ['pdf', 'png', 'svg']
            for save_format in save_formats:
                g.savefig(os.path.join(f'{saving_path}', f'{mouse_id}_{figname}.{save_format}'),
                               format=f"{save_format}",
                               bbox_inches='tight'
                               )

            plt.close()

        else:
            g=None
            pass

    return g, pd.concat(all_mouse_df)


def plot_single_mouse_opto_grid(data, saving_path):

    for mouse, mouse_data in data.groupby('mouse_id'):
        data_stim = mouse_data.loc[mouse_data.opto_stim == 1].drop_duplicates()
        fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig.suptitle(f'Opto grid performance {mouse}')

        fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig1.suptitle(f'Opto grid trial density {mouse}')
        for name, group in data_stim.groupby(by=['context_background', 'trial_type']):
            group['opto_grid_no_global'] = group.groupby(by=['session_id', 'opto_grid_no']).ngroup()
            if 'whisker_trial' in name:
                outcome = 'outcome_w'
                col = 2
            elif 'auditory_trial' in name:
                outcome = 'outcome_a'
                col = 1
            else:
                outcome = 'outcome_n'
                col = 0

            row = group.context.unique()[0]-1
            grid = group.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).reset_index()
            sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0, vmax=1,
                        ax=ax[row, col])
            ax[row, col].invert_yaxis()
            ax[row, col].invert_xaxis()

            grid = group.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].count().reset_index()
            sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0,
                        ax=ax1[row, col])
            ax1[row, col].invert_yaxis()
            ax1[row, col].invert_xaxis()

        cols = ['No stim', 'Auditory', 'Whisker']
        rows = ['Rewarded', 'No rewarded']
        for a, col in zip(ax[0], cols):
            a.set_title(col)
        for a, row in zip(ax[:, 0], rows):
            a.set_ylabel(row)

        fig.tight_layout()
        fig.show()
        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', f'{mouse}_opto_grid_performance.{save_format}'),
                           format=f"{save_format}")

        for a, col in zip(ax1[0], cols):
            a.set_title(col)
        for a, row in zip(ax1[:, 0], rows):
            a.set_ylabel(row)

        fig1.tight_layout()
        fig1.show()
        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            fig1.savefig(os.path.join(f'{saving_path}', f'{mouse}_opto_grid_trial_density.{save_format}'),
                        format=f"{save_format}")

    return

def plot_multiple_mice_opto_grid(data, control_mice, saving_path):
    fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig.suptitle('Pop opto grid performance')

    fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig1.suptitle('Pop opto grid trial density')
    data_stim = data.loc[(~data.mouse_id.isin(control_mice)) & (data.opto_stim == 1)].drop_duplicates()
    control_data = data.loc[(data.mouse_id.isin(control_mice)) & (data.opto_stim == 1)].drop_duplicates()

    for name, group in data_stim.groupby(by=['context', 'trial_type']):

        group['opto_grid_no_global'] = group.groupby(by=['session_id', 'opto_grid_no']).ngroup()

        if 'whisker_trial' in name:
            outcome = 'outcome_w'
            col = 2
        elif 'auditory_trial' in name:
            outcome = 'outcome_a'
            col = 1
        else:
            outcome = 'outcome_n'
            col = 0

        row = group.context.unique()[0] - 1

        grid = group.groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
        sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0, vmax=1,
                    ax=ax[row, col])
        ax[row, col].invert_yaxis()
        ax[row, col].invert_xaxis()

        grid = group.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].count().reset_index()
        sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0,
                    ax=ax1[row, col])
        ax1[row, col].invert_yaxis()
        ax1[row, col].invert_xaxis()

    cols = ['No stim', 'Auditory', 'Whisker']
    rows = ['Rewarded', 'No rewarded']
    for a, col in zip(ax[0], cols):
        a.set_title(col)
    for a, row in zip(ax[:, 0], rows):
        a.set_ylabel(row)

    fig.tight_layout()
    fig.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', f'Pop_opto_grid_performance.{save_format}'),
                       format=f"{save_format}")

    for a, col in zip(ax1[0], cols):
        a.set_title(col)
    for a, row in zip(ax1[:, 0], rows):
        a.set_ylabel(row)

    fig1.tight_layout()
    fig1.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig1.savefig(os.path.join(f'{saving_path}', f'Pop_opto_grid_trial_density.{save_format}'),
                    format=f"{save_format}")


    fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig.suptitle('Pop opto grid performance')

    fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig1.suptitle('Pop opto grid trial density')

    for name, group in control_data.groupby(by=['context', 'trial_type']):

        group['opto_grid_no_global'] = group.groupby(by=['session_id', 'opto_grid_no']).ngroup()

        if 'whisker_trial' in name:
            outcome = 'outcome_w'
            col = 2
        elif 'auditory_trial' in name:
            outcome = 'outcome_a'
            col = 1
        else:
            outcome = 'outcome_n'
            col = 0

        row = group.context.unique()[0] - 1

        grid = group.groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
        sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0, vmax=1,
                    ax=ax[row, col])
        ax[row, col].invert_yaxis()
        ax[row, col].invert_xaxis()

        grid = group.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].count().reset_index()
        sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0,
                    ax=ax1[row, col])
        ax1[row, col].invert_yaxis()
        ax1[row, col].invert_xaxis()

    cols = ['No stim', 'Auditory', 'Whisker']
    rows = ['Rewarded', 'No rewarded']
    for a, col in zip(ax[0], cols):
        a.set_title(col)
    for a, row in zip(ax[:, 0], rows):
        a.set_ylabel(row)

    fig.tight_layout()
    fig.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', f'Control_Pop_opto_grid_performance.{save_format}'),
                       format=f"{save_format}")

    for a, col in zip(ax1[0], cols):
        a.set_title(col)
    for a, row in zip(ax1[:, 0], rows):
        a.set_ylabel(row)

    fig1.tight_layout()
    fig1.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig1.savefig(os.path.join(f'{saving_path}', f'Control_Pop_opto_grid_trial_density.{save_format}'),
                    format=f"{save_format}")

    return


# def plot_control_subtracted_opto_grid(data, control_mice, saving_path):
#     from matplotlib.colors import LinearSegmentedColormap
#     cyanmagenta = ['#00FFFF', '#FFFFFF', '#FF00FF']
#     hotcold = ['#aefdff', '#60fdfa', '#2adef6', '#2593ff', '#2d47f9', '#3810dc', '#3d019d',
#                '#313131',
#                '#97023d', '#d90d39', '#f8432d', '#ff8e25', '#f7da29', '#fafd5b', '#fffda9']
#     cmap = LinearSegmentedColormap.from_list("Custom", cyanmagenta)
#     cmap.set_bad(color='white')
#     fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
#     fig.suptitle('Pop opto grid performance')
#
#     data_stim = data.loc[data.opto_stim == 1].drop_duplicates()
#
#     for name, group in data_stim.groupby(by=['context', 'trial_type']):
#
#         group['opto_grid_no_global'] = group.groupby(by=['session_id', 'opto_grid_no']).ngroup()
#
#         if 'whisker_trial' in name:
#             outcome = 'outcome_w'
#             col = 2
#         elif 'auditory_trial' in name:
#             outcome = 'outcome_a'
#             col = 1
#         else:
#             outcome = 'outcome_n'
#             col = 0
#
#         row = group.context.unique()[0] - 1
#
#         grid = group.loc[~group.mouse_id.isin(control_mice)].groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
#         control_grid = group.loc[group.mouse_id.isin(control_mice)].groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
#
#         grid = grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome)
#         # control_grid = control_grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome)
#         if 'whisker' in name[1]:
#             if name[0] == 0:
#                 control_grid = 0.35
#             else:
#                 control_grid = 0.8
#
#         elif 'auditory' in name[1]:
#             control_grid = 1
#         else:
#             control_grid = 0.15
#
#         sns.heatmap(grid-control_grid, vmin=-1, vmax=1,
#                     ax=ax[row, col], cmap='seismic')
#         ax[row, col].invert_yaxis()
#         ax[row, col].invert_xaxis()
#
#     cols = ['No stim', 'Auditory', 'Whisker']
#     rows = ['Rewarded', 'No rewarded']
#     for a, col in zip(ax[0], cols):
#         a.set_title(col)
#     for a, row in zip(ax[:, 0], rows):
#         a.set_ylabel(row)
#
#     fig.tight_layout()
#     fig.show()
#     save_formats = ['pdf', 'png', 'svg']
#     for save_format in save_formats:
#         fig.savefig(os.path.join(f'{saving_path}', f'Substracted_Pop_opto_grid_performance.{save_format}'),
#                        format=f"{save_format}")
#
#     fig1, ax1 = plt.subplots(1, 3, figsize=(8, 3), dpi=300)
#     fig1.suptitle('Pop opto grid no context')
#     for name, group in data_stim.groupby(by=['trial_type']):
#
#         group['opto_grid_no_global'] = group.groupby(by=['session_id', 'opto_grid_no']).ngroup()
#
#         if 'whisker_trial' in name:
#             outcome = 'outcome_w'
#             col = 2
#         elif 'auditory_trial' in name:
#             outcome = 'outcome_a'
#             col = 1
#         else:
#             outcome = 'outcome_n'
#             col = 0
#
#         grid = group.loc[~group.mouse_id.isin(control_mice)].groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
#         control_grid = group.loc[group.mouse_id.isin(control_mice)].groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
#
#         grid = grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome)
#         # control_grid = control_grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome)
#         if 'whisker' in name:
#             control_grid = 0.5
#         elif 'auditory' in name:
#             control_grid = 1
#         else:
#             control_grid = 0.15
#
#         sns.heatmap(grid-control_grid, vmin=-1, vmax=1,
#                     ax=ax1[col], cmap='seismic')
#         ax1[col].invert_yaxis()
#         ax1[col].invert_xaxis()
#
#     cols = ['No stim', 'Auditory', 'Whisker']
#     for a, col in zip(ax1, cols):
#         a.set_title(col)
#
#     fig1.tight_layout()
#     fig1.show()
#     save_formats = ['pdf', 'png', 'svg']
#     for save_format in save_formats:
#         fig1.savefig(os.path.join(f'{saving_path}', f'Substracted_Pop_opto_grid_performance_no_context.{save_format}'),
#                        format=f"{save_format}")
#
#     return


def plot_control_subtracted_opto_grid(data, control_mice, saving_path):
    from matplotlib.colors import LinearSegmentedColormap
    cyanmagenta = ['#00FFFF', '#FFFFFF', '#FF00FF']
    hotcold = ['#aefdff', '#60fdfa', '#2adef6', '#2593ff', '#2d47f9', '#3810dc', '#3d019d',
               '#313131',
               '#97023d', '#d90d39', '#f8432d', '#ff8e25', '#f7da29', '#fafd5b', '#fffda9']
    cmap = LinearSegmentedColormap.from_list("Custom", cyanmagenta)
    cmap.set_bad(color='white')
    fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig.suptitle('Pop opto grid performance')

    fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig1.suptitle('Pop opto grid performance')

    data_stim = data.loc[data.opto_stim == 1].drop_duplicates()
    data_nostim = data.loc[data.opto_stim == 0].drop_duplicates()

    for name, group in data_stim.groupby(by=['context', 'trial_type']):

        group['opto_grid_no_global'] = group.groupby(by=['session_id', 'opto_grid_no']).ngroup()

        if 'whisker_trial' in name:
            outcome = 'outcome_w'
            col = 2
        elif 'auditory_trial' in name:
            outcome = 'outcome_a'
            col = 1
        else:
            outcome = 'outcome_n'
            col = 0

        row = group.context.unique()[0] - 1

        grid = group.loc[~group.mouse_id.isin(control_mice)].groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
        # control_grid = group.loc[group.mouse_id.isin(control_mice)].groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()


        # control_grid = control_grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome)

        control_data = data_nostim.groupby(by=['context', 'trial_type']).get_group(name)
        control_grid = control_data[outcome].mean()
        # if 'whisker' in name[1]:
        #     if name[0] == 0:
        #         control_grid = 0.35
        #     else:
        #         control_grid = 0.8
        #
        # elif 'auditory' in name[1]:
        #     control_grid = 1
        # else:
        #     control_grid = 0.15

        grid[f'{outcome}_sub'] = grid[outcome] - control_grid
        grid[f'{outcome}_abs'] = abs(grid[f'{outcome}_sub']) * 100
        # sns.scatterplot(data=grid, x='opto_grid_ml', y='opto_grid_ap', hue=f'{outcome}_sub',
        #                 hue_norm=plt.Normalize(-1, 1), s=350, palette='seismic', ax=ax1[row, col])
        # ax1[row, col].set_xlim([0, 6.5])
        # ax1[row, col].set_xticks(np.arange(0.5,6,1))
        # ax1[row, col].set_ylim([-4, 4])
        # ax1[row, col].set_yticks(np.arange(-3.5, 4, 1))
        # ax1[row, col].set_aspect(1)
        # ax1[row, col].invert_xaxis()
        # ax1[row, col].spines['top'].set_visible(False)
        # ax1[row, col].spines['right'].set_visible(False)
        # ax1[row, col].spines['bottom'].set_visible(False)
        # ax1[row, col].spines['left'].set_visible(False)
        # ax1[row, col].get_legend().remove()

        plot_opto_on_allen(grid, outcome, os.path.join(saving_path, group.context.map({0: 'No_Rewarded', 1:'Rewarded'}).unique()[0]))

        grid = grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome)
        sns.heatmap(grid-control_grid, vmin=-1, vmax=1,
                    ax=ax[row, col], cmap='seismic')
        ax[row, col].invert_yaxis()
        ax[row, col].invert_xaxis()

    cols = ['No stim', 'Auditory', 'Whisker']
    rows = ['Rewarded', 'No rewarded']
    for a, col in zip(ax[0], cols):
        a.set_title(col)
    for a, row in zip(ax[:, 0], rows):
        a.set_ylabel(row)

    for a, col in zip(ax1[0], cols):
        a.set_title(col)
    for a, row in zip(ax1[:, 0], rows):
        a.set_ylabel(row)

    fig.tight_layout()
    fig.show()
    fig1.tight_layout()
    fig1.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', f'Substracted_Pop_opto_grid_performance.{save_format}'),
                       format=f"{save_format}")
        fig1.savefig(os.path.join(f'{saving_path}', f'Substracted_Pop_opto_grid_performance_dots.{save_format}'),
                       format=f"{save_format}")

    fig1, ax1 = plt.subplots(1, 3, figsize=(8, 3), dpi=300)
    fig1.suptitle('Pop opto grid no context')
    for name, group in data_stim.groupby(by=['trial_type']):

        group['opto_grid_no_global'] = group.groupby(by=['session_id', 'opto_grid_no']).ngroup()

        if 'whisker_trial' in name:
            outcome = 'outcome_w'
            col = 2
        elif 'auditory_trial' in name:
            outcome = 'outcome_a'
            col = 1
        else:
            outcome = 'outcome_n'
            col = 0

        grid = group.loc[~group.mouse_id.isin(control_mice)].groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
        # control_grid = group.loc[group.mouse_id.isin(control_mice)].groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
        control_data = data_nostim.groupby(by='trial_type').get_group(name)
        control_grid = control_data[outcome].mean()

        grid[f'{outcome}_sub'] = grid[outcome] - control_grid
        grid[f'{outcome}_abs'] = abs(grid[f'{outcome}_sub']) * 100

        plot_opto_on_allen(grid, outcome, os.path.join(saving_path, 'no_context'))


        # control_grid = control_grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome)
        # if 'whisker' in name:
        #     control_grid = 0.5
        # elif 'auditory' in name:
        #     control_grid = 1
        # else:
        #     control_grid = 0.15

        grid = grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome)
        sns.heatmap(grid-control_grid, vmin=-1, vmax=1,
                    ax=ax1[col], cmap='seismic')
        ax1[col].invert_yaxis()
        ax1[col].invert_xaxis()

    cols = ['No stim', 'Auditory', 'Whisker']
    for a, col in zip(ax1, cols):
        a.set_title(col)

    fig1.tight_layout()
    fig1.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig1.savefig(os.path.join(f'{saving_path}', f'Substracted_Pop_opto_grid_performance_no_context.{save_format}'),
                       format=f"{save_format}")

    return


def plot_opto_on_allen(grid, outcome, result_path):
    from utils.wf_plotting_utils import get_colormap, get_wf_scalebar, get_allen_ccf
    from skimage.transform import rescale
    cmap = get_colormap('gray')
    cmap.set_bad(color='white')
    bregma = (488, 290)
    scale = 4
    scalebar = get_wf_scalebar(scale=scale)
    iso_mask, atlas_mask, allen_bregma = get_allen_ccf(bregma)

    grid['opto_grid_ml_wf'] = bregma[0] - grid['opto_grid_ml'] * scalebar
    grid['opto_grid_ap_wf'] = bregma[1] - grid['opto_grid_ap'] * scalebar

    fig, ax = plt.subplots(1, figsize=(5, 5), dpi=200)
    single_frame = np.rot90(rescale(np.ones([125, 160]), scale, anti_aliasing=False))
    single_frame = np.pad(single_frame, [(0, 650 - single_frame.shape[0]), (0, 510 - single_frame.shape[1])],
                          mode='constant', constant_values=np.nan)
    im = ax.imshow(single_frame, cmap=cmap, vmin=0, vmax=1)
    sns.scatterplot(data=grid, x='opto_grid_ml_wf', y='opto_grid_ap_wf', hue=f'{outcome}_sub',
                    hue_norm=plt.Normalize(-1, 1), s=900, palette='seismic', ax=ax)
    ax.contour(atlas_mask, levels=np.unique(atlas_mask), colors='gray',
               linewidths=1)
    ax.contour(iso_mask, levels=np.unique(np.round(iso_mask)), colors='black',
               linewidths=2, zorder=2)
    ax.scatter(bregma[0], bregma[1], marker='+', c='r', s=300, linewidths=4,
               zorder=3)
    ax.set_xticks(np.unique(grid['opto_grid_ml_wf']), np.arange(5.5, 0, -1))
    ax.set_yticks(np.unique(grid['opto_grid_ap_wf']), np.arange(3.5, -4, -1))
    ax.set_aspect(1)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_axis_off()
    ax.get_legend().remove()
    ax.hlines(5, 5, 5 + scalebar * 3, linewidth=2, colors='k')
    # ax.text(50, 100, "3 mm", size=10)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    fig.savefig(os.path.join(result_path, f'{outcome}.png'))
    # fig.savefig(os.path.join(result_path, f'{outcome}.pdf'))


def plot_context_performance_stats(combine_bhv_data, expert_table, saving_path):
    sessions_list = np.unique(combine_bhv_data['session_id'].values[:])
    n_sessions = len(sessions_list)
    print(f"N sessions : {n_sessions}")
    combined_d = []
    above_threshold_table = []
    for session_id in sessions_list:
        session_table, switches, block_size = bhv_utils.get_standard_single_session_table(combine_bhv_data,
                                                                                          session=session_id)
        if session_table['behavior'].values[0] not in ['context', 'whisker_context']:
            print(f"No plot for {session_table['behavior'].values[0]} sessions")
            continue

        d = session_table.loc[session_table.early_lick == 0][int(block_size / 2)::block_size]
        # Add the proportion of correct choices
        d['correct_a'] = d.hr_a
        d['correct_n'] = 1 - d.hr_n
        d['correct_w'] = [1 - d.hr_w.values[i] if d.context.values[i] == 0 else d.hr_w.values[i] for i in
                          range(len(d))]

        # Add the contrast perf (the average diff between one block and the 2 surrounding blocks)
        hr_w_contrast = [(np.abs(d.hr_w.values[i] - d.hr_w.values[i - 1]) +
                          np.abs(d.hr_w.values[i] - d.hr_w.values[i + 1])) / 2 for
                         i in np.arange(1, d.hr_w.size - 1)]
        hr_w_contrast.insert(0, np.nan)
        hr_w_contrast.insert(len(hr_w_contrast), np.nan)
        d['contrast_w'] = hr_w_contrast

        hr_a_contrast = [(np.abs(d.hr_a.values[i] - d.hr_a.values[i - 1]) +
                          np.abs(d.hr_a.values[i] - d.hr_a.values[i + 1])) / 2 for
                         i in np.arange(1, d.hr_a.size - 1)]
        hr_a_contrast.insert(0, np.nan)
        hr_a_contrast.insert(len(hr_a_contrast), np.nan)
        d['contrast_a'] = hr_a_contrast

        hr_n_contrast = [(np.abs(d.hr_n.values[i] - d.hr_n.values[i - 1]) +
                          np.abs(d.hr_n.values[i] - d.hr_n.values[i + 1])) / 2 for
                         i in np.arange(1, d.hr_n.size - 1)]
        hr_n_contrast.insert(0, np.nan)
        hr_n_contrast.insert(len(hr_n_contrast), np.nan)
        d['contrast_n'] = hr_n_contrast

        combined_d.append(d)

    combined_d = pd.concat(combined_d)
    cols = ['mouse_id', 'session_id',
            'block', 'context', 'context_background',
            'correct', 'correct_a', 'correct_n', 'correct_w',
            'contrast_a', 'contrast_n', 'contrast_w']
    combined_d = combined_d[cols]
    combined_d = combined_d.reset_index(drop=True)
    combined_d['six_contrast_w'] = combined_d.contrast_w > 0.375

    session_index = []
    for mouse in combined_d['mouse_id'].unique():
        for sess_ind, sess in enumerate(combined_d.loc[combined_d.mouse_id == mouse].session_id.unique()):
            above_threshold = dict()
            n_blocks = len(combined_d.loc[(combined_d.mouse_id == mouse) & (combined_d.session_id == sess)])
            session_index.extend([sess_ind for i in range(n_blocks)])
            above_thresh = combined_d.loc[(combined_d.mouse_id == mouse) & (combined_d.session_id == sess)].six_contrast_w.values[:]
            continuous_periods = get_continuous_time_periods(above_thresh)
            len_above_thresh = len(np.where(np.array([np.diff(i) for i in continuous_periods]) >= 4)[0])
            above_threshold['mouse_id'] = [mouse]
            above_threshold['session_id'] = [sess]
            above_threshold['n_blocks'] = [n_blocks]
            above_threshold['n_good_blocks'] = [len(np.where(above_thresh)[0])]
            above_threshold['n_4_successive_good_blocks'] = [len_above_thresh]
            above_threshold_table.append(pd.DataFrame.from_dict(above_threshold))

    combined_d['session_index'] = session_index
    combined_d["context"] = combined_d["context"].map({0: "Non-Rewarded", 1: "Rewarded"})
    above_threshold_table = pd.concat(above_threshold_table)

    # ---------------------------------------------- Do some plots --------------------------------------------------- #
    # Plot distribution of contrast value across all blocks
    fig, (ax0, ax1) = plt.subplots(2,  1, sharex=True, figsize=(5, 10))
    sns.histplot(
        data=combined_d, x="contrast_w", binwidth=1/16, stat='percent', kde=True, ax=ax0)
    sns.histplot(
        data=combined_d, x="contrast_w", binwidth=1/16, stat='count', kde=True, ax=ax1)
    sns.despine(top=True, right=True)
    ax0.set_ylabel('% blocks')
    ax0.set_xlabel('Contrast lick probability')
    ax0.axvline(x=0.375, ymin=0, ymax=1, color='r', linestyle='--')
    ax1.set_ylabel('N blocks')
    ax1.set_xlabel('Contrast lick probability')
    ax1.axvline(x=0.375, ymin=0, ymax=1, color='r', linestyle='--')
    plt.suptitle('Distribution of whisker-contrast values')
    fig.tight_layout()
    save_formats = ['pdf', 'png', 'svg']
    if not os.path.exists(os.path.join(f'{saving_path}', 'block_distribution')):
        os.makedirs(os.path.join(f'{saving_path}', 'block_distribution'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'block_distribution',
                                 f'all_block_distribution.{save_format}'),
                    format=f"{save_format}")
    plt.close()

    # Plot distribution of contrast value across all blocks & distinguish context
    hist_stat = ['count', 'percent']
    ylabels = ['N Block', '% Block']
    for i, stat in enumerate(hist_stat):
        fig = sns.displot(
            data=combined_d, x="contrast_w", col="context", binwidth=1/16, kde=True,
            kind="hist", stat=stat)
        fig.set_ylabels(f'{ylabels[i]}')
        fig.set_xlabels('Contrast lick probability')
        fig.set_titles(col_template="{col_name} context")
        for ax in fig.axes.flat:
            ax.axvline(x=0.375, ymin=0, ymax=1, color='r', linestyle='--')
        plt.suptitle('Distribution of whisker-contrast values')
        fig.tight_layout()
        save_formats = ['pdf', 'png', 'svg']
        if not os.path.exists(os.path.join(f'{saving_path}', 'block_distribution')):
            os.makedirs(os.path.join(f'{saving_path}', 'block_distribution'))
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', 'block_distribution',
                                                       f'all_block_distribution_{stat}_context_split.{save_format}'),
                        format=f"{save_format}")
    plt.close()

    # Plot distribution of contrast value across all block distinguish context / mice / session order
    # Figure 1
    fig = sns.displot(
        data=combined_d, x="contrast_w", col="mouse_id", hue="context",
        kind="hist", binwidth=1/16, height=2, aspect=0.7, col_wrap=4, legend=True)
    for ax in fig.axes.flat:
        ax.axvline(x=0.375, ymin=0, ymax=1, color='r', linestyle='--')
    fig.set_titles(col_template="{col_name}")
    fig.set_xlabels('Contrast')
    fig.set_ylabels('N Blocks')
    plt.suptitle('Distribution of whisker-contrast values')
    fig.tight_layout()
    save_formats = ['pdf', 'png', 'svg']
    if not os.path.exists(os.path.join(f'{saving_path}', 'block_distribution')):
        os.makedirs(os.path.join(f'{saving_path}', 'block_distribution'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'block_distribution',
                                                   f'all_block_distribution_mouse_context_split.{save_format}'),
                    format=f"{save_format}")
    plt.close()
    # Figure 2
    fig = sns.displot(
        data=combined_d, y="contrast_w", col="mouse_id", hue="context",
        kind="ecdf", height=2, aspect=0.7, col_wrap=4, legend=True)
    for ax in fig.axes.flat:
        ax.axhline(y=0.375, xmin=0, xmax=1, color='r', linestyle='--', zorder=-1)
    fig.set_ylabels('Contrast')
    fig.set_titles(col_template="{col_name}")
    plt.suptitle('Cumulative distribution of whisker-contrast values')
    fig.tight_layout()
    save_formats = ['pdf', 'png', 'svg']
    if not os.path.exists(os.path.join(f'{saving_path}', 'block_distribution')):
        os.makedirs(os.path.join(f'{saving_path}', 'block_distribution'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'block_distribution',
                                                   f'all_block_distribution_cumulative_mouse_context_split.{save_format}'),
                    format=f"{save_format}")
    plt.close()
    # Figure 3
    show_legend = True
    if len(combined_d.session_index.unique()) > 10:
        show_legend = False
    fig = sns.displot(
        data=combined_d, y="contrast_w", col="mouse_id", hue='session_index',
        kind="ecdf", height=2, aspect=0.7, palette='coolwarm', col_wrap=4, legend=show_legend)
    for ax in fig.axes.flat:
        ax.axhline(y=0.375, xmin=0, xmax=1, color='r', linestyle='--', zorder=-1)
    fig.set_ylabels('Contrast')
    fig.set_titles(col_template="{col_name}")
    plt.suptitle('Cumulative distribution of whisker-contrast values')
    fig.tight_layout()
    save_formats = ['pdf', 'png', 'svg']
    if not os.path.exists(os.path.join(f'{saving_path}', 'block_distribution')):
        os.makedirs(os.path.join(f'{saving_path}', 'block_distribution'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'block_distribution',
                                                   f'all_block_distribution_cumulative_mouse_session_context_split.{save_format}'),
                    format=f"{save_format}")
    plt.close()

    # Plot distribution of block above threshold
    hist_stat = ['count', 'percent']
    ylabels = ['N Block', '% Block']
    for i, stat in enumerate(hist_stat):
        fig = sns.displot(
            data=combined_d, x="six_contrast_w", col="mouse_id", hue='context',
            kind="hist", stat=stat, common_norm=False, binwidth=0.2, height=2, aspect=0.7, col_wrap=4)
        fig.set_ylabels(f'{ylabels[i]}')
        fig.set_xlabels('Good block')
        fig.set_titles(col_template="{col_name}")
        plt.suptitle('Distribution of above threshold blocks')
        fig.tight_layout()
        save_formats = ['pdf', 'png', 'svg']
        if not os.path.exists(os.path.join(f'{saving_path}', 'good_blocks')):
            os.makedirs(os.path.join(f'{saving_path}', 'good_blocks'))
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', 'good_blocks',
                                                       f'good_block_distribution_{stat}_mouse_context_split.{save_format}'),
                        format=f"{save_format}")
    plt.close()

    # Plot paired-plot of performance
    fig = sns.pairplot(above_threshold_table, hue='mouse_id')
    plt.suptitle('Block statistics all mice')
    fig.tight_layout()
    save_formats = ['pdf', 'png', 'svg']
    if not os.path.exists(os.path.join(f'{saving_path}', 'session_results')):
        os.makedirs(os.path.join(f'{saving_path}', 'session_results'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'session_results',
                                                   f'session_block_stats.{save_format}'),
                    format=f"{save_format}")
    plt.close()
    # Plot paired-plot of performance for each mouse
    for mouse in above_threshold_table.mouse_id.unique():
        fig = sns.pairplot(above_threshold_table.loc[above_threshold_table.mouse_id == mouse],
                           hue='session_id', palette='coolwarm')
        plt.suptitle(f'Block statistics {mouse}')
        fig.tight_layout()
        save_formats = ['pdf', 'png', 'svg']
        if not os.path.exists(os.path.join(f'{saving_path}', 'session_results')):
            os.makedirs(os.path.join(f'{saving_path}', 'session_results'))
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', 'session_results',
                                                       f'{mouse}_session_block_stats.{save_format}'),
                        format=f"{save_format}")
        plt.close("all")
    #  ---------------------------------------------------------------------------------------------------------- #
    # Explore session contrast table
    session_perf_df = pd.read_excel(expert_table)
    session_perf_df["w_context_expert"] = session_perf_df["w_context_expert"].map({False: "Non-Expert", True: "Expert"})

    # Distribution of session contrast
    fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))
    sns.histplot(
        data=session_perf_df, x="w_contrast_mean", kde=True, ax=ax0)
    ax0.set_ylabel('N Sessions')
    ax0.set_xlabel('Contrast lick probability')
    ax0.axvline(x=0.375, ymin=0, ymax=1, color='r', linestyle='--')
    sns.despine(top=True, right=True)
    plt.suptitle('Distribution of whisker-contrast session values')
    fig.tight_layout()
    save_formats = ['pdf', 'png', 'svg']
    if not os.path.exists(os.path.join(f'{saving_path}', 'session_results')):
        os.makedirs(os.path.join(f'{saving_path}', 'session_results'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'session_results',
                                                   f'session_contrast_distribution.{save_format}'),
                    format=f"{save_format}")
    plt.close()

    # Distribution of session contrast by mouse
    fig = sns.displot(
        data=session_perf_df, x="w_contrast_mean", col="mouse_id", kde=True, col_wrap=4,
        kind="hist", height=2, aspect=0.7)
    fig.set_xlabels('Mean contrast')
    fig.set_ylabels('N sessions')
    for ax in fig.axes.flat:
        ax.axvline(x=0.375, ymin=0, ymax=1, color='r', linestyle='--')
        ax.set_xlim(0, 1)
    fig.set_titles(col_template="{col_name}")
    plt.suptitle('Distribution of mean whisker-contrast values')
    fig.tight_layout()
    if not os.path.exists(os.path.join(f'{saving_path}', 'session_results')):
        os.makedirs(os.path.join(f'{saving_path}', 'session_results'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'session_results',
                                                   f'session_contrast_distribution_mouse_split.{save_format}'),
                    format=f"{save_format}")
    plt.close()

    # Link session 'whisker-contrast' value with d-prime
    fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))
    sns.regplot(data=session_perf_df, x='w_contrast_mean', y='d_prime', ax=ax0)
    sns.scatterplot(data=session_perf_df.loc[session_perf_df.w_context_expert == 'Expert'],
                    x='w_contrast_mean', y='d_prime', color='green', ax=ax0)
    sns.scatterplot(data=session_perf_df.loc[session_perf_df.w_context_expert == 'Non-Expert'],
                    x='w_contrast_mean', y='d_prime', color='gray', ax=ax0)
    ax0.set_ylabel("d'")
    ax0.set_xlabel('Mean contrast')
    ax0.axvline(x=0.375, ymin=0, ymax=1, color='r', linestyle='--', zorder=-1)
    sns.despine(top=True, right=True)
    plt.suptitle("Correlation between 'whisker-contrast' and d'")
    fig.tight_layout()
    if not os.path.exists(os.path.join(f'{saving_path}', 'session_results')):
        os.makedirs(os.path.join(f'{saving_path}', 'session_results'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'session_results',
                                                   f'whisker_contrast_vs_d-prime.{save_format}'),
                    format=f"{save_format}")
    plt.close()
    # Temporary addition to compare expert sessions with threshold to the ones manually defined
    # config_file = "C:/Users/rdard/Documents//python_repos/CICADA/cicada/src/cicada/config/group.yaml"
    # with open(config_file, 'r', encoding='utf8') as stream:
    #     config_dict = yaml.safe_load(stream)
    # old_exp_sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
    # old_exp_sessions = [session[0] for session in old_exp_sessions]
    # old_exp = [session for session in old_exp_sessions if session not in session_perf_df.loc[session_perf_df.w_context_expert == True].session_id.values[:].tolist()]
    # sns.scatterplot(data=session_perf_df.loc[session_perf_df.session_id.isin(old_exp)],
    #                 x='w_contrast_mean', y='d_prime', color='red', ax=ax0)
    # new_exp_sessions = [session for session in
    #                 session_perf_df.loc[session_perf_df.w_context_expert == True].session_id.values[:].tolist() if
    #                 session not in old_exp_sessions]
    # sns.scatterplot(data=session_perf_df.loc[session_perf_df.session_id.isin(new_exp_sessions)],
    #                 x='w_contrast_mean', y='d_prime', color='blue', ax=ax0)

    # Show learning with mean contrast across sessions for all mice in one plot
    fig = sns.relplot(
        data=session_perf_df, x="session_index", y='w_contrast_mean', col="mouse_id", hue='w_context_expert',
        col_wrap=4, height=2, aspect=1)
    for ax in fig.axes.flat:
        ax.axhline(y=0.375, xmin=0, xmax=1, color='r', linestyle='--')
        ax.set_ylim(-0.05, 1.05)
    fig.set_xlabels('Session index')
    fig.set_ylabels('Session contrast')
    fig.set_titles(col_template="{col_name}")
    plt.suptitle('Session whisker-contrast across training')
    fig.tight_layout()
    if not os.path.exists(os.path.join(f'{saving_path}', 'learning')):
        os.makedirs(os.path.join(f'{saving_path}', 'learning'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'learning',
                                                   f'whisker_contrast_mouse_split.{save_format}'),
                    format=f"{save_format}")
    plt.close()
    # Show learning with mean contrast across sessions for each mouse
    for mouse in session_perf_df.mouse_id.unique():
        fig, ax0 = plt.subplots(1, 1, figsize=(6, 6))
        sns.scatterplot(data=session_perf_df.loc[session_perf_df.mouse_id == mouse],
                        x="session_index", y='w_contrast_mean', hue='w_context_expert',
                        hue_order=['Non-Expert', 'Expert'], ax=ax0)
        ax0.set_ylabel("Contrast lick probability")
        ax0.set_xlabel('Session index')
        ax0.axhline(y=0.375, xmin=0, xmax=1, color='r', linestyle='--', zorder=-1)
        ax0.set_title(f'{mouse} context training')
        ax0.set_ylim(-0.05, 1.05)
        xerror = np.zeros((2, len(session_perf_df.loc[session_perf_df.mouse_id == mouse])))
        xerror[0, :] = (session_perf_df.loc[session_perf_df.mouse_id == mouse].w_contrast_mean -
                        session_perf_df.loc[session_perf_df.mouse_id == mouse].w_contrast_ci_low)
        xerror[1, :] = (session_perf_df.loc[session_perf_df.mouse_id == mouse].w_contrast_ci_high
                        - session_perf_df.loc[session_perf_df.mouse_id == mouse].w_contrast_mean)
        ax0.errorbar(session_perf_df.loc[session_perf_df.mouse_id == mouse].session_index,
                     session_perf_df.loc[session_perf_df.mouse_id == mouse].w_contrast_mean,
                     yerr=xerror, xerr=None, fmt='none', capsize=5, zorder=-1, color='gray')
        sns.despine(top=True, right=True)
        fig.tight_layout()
        if not os.path.exists(os.path.join(f'{saving_path}', 'learning')):
            os.makedirs(os.path.join(f'{saving_path}', 'learning'))
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', 'learning',
                                                       f'{mouse}_whisker_contrast.{save_format}'),
                        format=f"{save_format}")
        plt.close()


if __name__ == '__main__':

    experimenter = 'Anthony_Renard'
    output_path = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'results')
    excel_path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\mice_info\session_metadata.xlsx'
    nwb_path = utils_io.get_experimenter_nwb_folder('AR')

    # Path to performance table for context sessions (generated by plot singe session)
    context_session_table = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Pol_Bech/Session_list/context_perf_table.xlsx'


    # Set the saving folder
    output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'Pop_results',
                               'Context_behaviour', 'opto_meeting_with_carl')
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # context_session_table = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/Context_behaviour_analysis_20240502/context_expert_sessions.xlsx'

    # Get NWBs files from cicada custom-made group saved in yaml file
    # config_file = r"M:\analysis\Pol_Bech\Sessions_list\context_good_params_sessions_path.yaml"
    config_file = r"M:\z_LSENS\Share\Pol_Bech\Session_list\context_sessions_opto.yaml"


excel_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\mice_info\\session_metadata.xlsx'
db = read_excel_db(excel_path)
experimenters = ['AR', 'GF', 'MI']

mice_rew = db.loc[(db['2P_calcium_imaging']==True)
                  & (db.exclude != 'exclude')
                  & (db.reward_group == 'R+'), 'subject_id'].unique()
mice_rew = list(mice_rew)
mice_non_rew = db.loc[(db['2P_calcium_imaging']==True)
                      & (db.exclude != 'exclude')
                      & (db.reward_group == 'R-'), 'subject_id'].unique()
mice_non_rew = list(mice_non_rew)

behavior_types = ['auditory', 'whisker']
days = [-2, -1, 0, 1, 2]

nwb_path = utils_io.get_experimenter_nwb_folder('AR')
nwb_list = []
# Reduce the number of files, restricting to mice with imaging.
for mouse_id in mice_rew + mice_non_rew:
    nwb_list.extend([nwb for nwb in os.listdir(nwb_path) if mouse_id in nwb])
nwb_list = sorted([os.path.join(nwb_path,nwb) for nwb in nwb_list])

nwb_list = [nwb for nwb in nwb_list if os.path.basename(nwb)[-25:-23] in experimenters]

nwb_metadata = {nwb: nwb_read.get_session_metadata(nwb) for nwb in nwb_list}
nwb_list = [nwb for nwb, metadata in nwb_metadata.items()
                if (os.path.basename(nwb)[-25:-20] in mice_rew)
                & ('twophoton' in metadata['session_type'])
                & (metadata['behavior_type'] in behavior_types)
                & (metadata['day'] in days)
                ]



    # Choose session from dict wit keys
    nwb_files = config_dict['Session path']
    # sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
    # sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
    # nwb_files = [session[1] for session in config_dict['Sessions']]

    # Get NWBs files from context expert table
    # session_perf_df = pd.read_excel(context_session_table)
    # session_to_do = session_perf_df.loc[session_perf_df.w_context_expert == True].session_id.values[:].tolist()
    # all_nwb_names = [name for name in all_nwb_names if name[0:21] in session_to_do]
    # sessions_to_exclude = ['RD039_20240117_125330', 'RD039_20240306_142242']
    # all_nwb_names = [name for name in all_nwb_names if name[0:21] not in sessions_to_exclude]
    # nwb_files = [os.path.join(root_path, name) for name in all_nwb_names]

    # Get NWBs file specifying both subjects and sessions to do
    # subject_ids = ['PB164', 'PB165', 'PB166', 'PB167', 'PB168', 'PB170', 'PB171', 'PB172', 'PB173', 'PB174', 'PB175',
    #                'RD027', 'RD028', 'RD029', 'RD030', 'RD031', 'RD032', 'RD033', 'RD034', 'RD035', 'RD036', 'RD037',
    #                'RD038', 'RD039', 'RD041', 'RD042', 'RD043', 'RD044', 'RD045']
    # subject_ids = ['PB176', "PB177", "PB178", "PB179", "PB180", "PB181"]
    # PB_nwb_names = os.listdir(root_path)
    # all_nwb_names = [file for file in PB_nwb_names if 'PB' in file]
    # session_to_do = ["PB170_20240309_110026",
    #                  "PB170_20240311_091835",
    #                  "PB170_20240312_104013",
    #                  "PB171_20240309_130312",
    #                  "PB171_20240311_105637",
    #                  "PB171_20240312_133450",
    #                  "PB172_20240309_151856",
    #                  "PB172_20240311_131938",
    #                  "PB172_20240312_152926",
    #                  "PB173_20240313_114224",
    #                  "PB173_20240314_093012",
    #                  "PB175_20240313_133937",
    #                  "PB175_20240314_105858"
    #                  ]

    # pop_nwb_files = []
    # nwb_files = []
    # for subject_id in subject_ids:
    #     print(" ")
    #     print(f"Subject ID : {subject_id}")
    #     experimenter = 'Pol_Bech' if 'PB' in subject_id else 'Robin_Dard'
    #     root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWB')
    #     nwb_names = [name for name in all_nwb_names if subject_id in name]
    # #     #     for session in session_to_do:
    # #     #         nwb_files += [os.path.join(root_path, name) for name in nwb_names if session in name]
    #     nwb_files += [os.path.join(root_path, name) for name in nwb_names]
    #
    #     pop_nwb_files+=nwb_files
    #     if nwb_files.__len__() == 0:
    #         continue
    #
    #     results_path = os.path.join(output_path, subject_id)
    #     if not os.path.exists(results_path):
    #         os.makedirs(results_path)

    # Define plots to do
    # plots_to_do = ['single_session', 'across_context_days', 'context_switch']
    # plots_to_do = ['context_block_perf']
    plots_to_do = ["opto_grid", "opto_grid_multiple"]
    #
    print(f"Analyzing {plots_to_do}")
    plot_behavior(nwb_list=nwb_files, output_folder=output_path, plots=plots_to_do)

    # plots_to_do = ['context_block_perf']
    # print(f"Analyzing {plots_to_do}")
    # plot_behavior(nwb_list=nwb_files, output_folder=output_path, plots=plots_to_do)
