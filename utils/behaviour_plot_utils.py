import itertools
import os
import glob
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import yaml
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator
from utils.wf_plotting_utils import get_colormap, get_wf_scalebar, get_allen_ccf
from statsmodels.stats.multitest import multipletests
from skimage.transform import rescale

import nwb_utils.utils_behavior as bhv_utils
from nwb_utils.utils_misc import get_continuous_time_periods
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame

warnings.filterwarnings("ignore")


# def plot_single_session(combine_bhv_data, color_palette, save_path):
#     raster_marker = 2
#     marker_width = 2
#     figsize = (15, 8)
#     marker = itertools.cycle(['o', 's'])
#     markers = [next(marker) for i in d["opto_stim"].unique()]
#
#     return


def save_fig(fig, save_path, name, save_format=['pdf', 'png', 'svg']):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for format in save_format:
        fig.savefig(os.path.join(save_path, name), format=format)


def plot_distributions(df, stat, fig, ax, x, y, hue, binwidth, title, xlabel, ylabel, line, line_loc):
    sns.histplot(
        data=df, x=x, y=y, hue=hue, binwidth=binwidth, stat=stat, kde=True, ax=ax)
    sns.despine(top=True, right=True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.axvline(x=0.375, ymin=0, ymax=1, color='r', linestyle='--')
    ax.set_title(title)
    if line:
        ax.axvline(x=line_loc, ymin=0, ymax=1, color='r', linestyle='--')
    fig.tight_layout()
    plt.close()


def plot_performance(df, df_by_block, fig, color_palette, context=False):

    raster_marker = 2
    marker_width = 2

    if context:
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
    else:
        ax2 = fig.add_subplot(1, 1, 1)

    sns.lineplot(data=df, x='trial', y='hr_n', color='k', ax=ax2,
                 markers='o')

    if 'hr_w' in list(df_by_block.columns) and (not np.isnan(df_by_block.hr_w.values[:]).all()):
        sns.lineplot(data=df_by_block, x='trial', y='hr_w', color=color_palette[2], ax=ax2, markers='o')
    if 'hr_a' in list(df_by_block.columns) and (not np.isnan(df_by_block.hr_a.values[:]).all()):
        sns.lineplot(data=df_by_block, x='trial', y='hr_a', color=color_palette[0], ax=ax2, markers='o')

    # Plot the trials :
    ax2.scatter(x=df.loc[df.lick_flag == 0]['trial'],
                y=df.loc[df.lick_flag == 0]['outcome_n'] - 0.1,
                color=color_palette[4], marker=raster_marker, linewidths=marker_width)
    ax2.scatter(x=df.loc[df.lick_flag == 1]['trial'],
                y=df.loc[df.lick_flag == 1]['outcome_n'] - 1.1,
                color='k', marker=raster_marker, linewidths=marker_width)

    if 'hr_a' in list(df_by_block.columns) and (not np.isnan(df_by_block.hr_w.values[:]).all()):
        ax2.scatter(x=df.loc[df.lick_flag == 0]['trial'],
                    y=df.loc[df.lick_flag == 0]['outcome_a'] - 0.15,
                    color=color_palette[1], marker=raster_marker, linewidths=marker_width)
        ax2.scatter(x=df.loc[df.lick_flag == 1]['trial'],
                    y=df.loc[df.lick_flag == 1]['outcome_a'] - 1.15,
                    color=color_palette[0], marker=raster_marker, linewidths=marker_width)

    if 'hr_w' in list(df_by_block.columns) and (not np.isnan(df_by_block.hr_w.values[:]).all()):
        ax2.scatter(x=df.loc[df.lick_flag == 0]['trial'],
                    y=df.loc[df.lick_flag == 0]['outcome_w'] - 0.2,
                    color=color_palette[3], marker=raster_marker, linewidths=marker_width)
        ax2.scatter(x=df.loc[df.lick_flag == 1]['trial'],
                    y=df.loc[df.lick_flag == 1]['outcome_w'] - 1.2,
                    color=color_palette[2], marker=raster_marker, linewidths=marker_width)

    if context:
        sns.lineplot(data=df_by_block, x='trial', y='contrast_n',
                     color=color_palette[4], ax=ax1, markers='o')
        if 'contrast_w' in list(df_by_block.columns) and (not np.isnan(df_by_block.contrast_w.values[:]).all()):
            sns.lineplot(data=df_by_block, x='trial', y='contrast_w',
                         color=color_palette[2], ax=ax1, markers='o')
        if 'contrast_a' in list(df_by_block.columns) and (not np.isnan(df_by_block.contrast_a.values[:]).all()):
            sns.lineplot(data=df_by_block, x='trial', y='contrast_a',
                         color=color_palette[0], ax=ax1, markers='o')

        bloc_area_color = ['green' if i == 'Rewarded' else 'firebrick' for i in df_by_block.context.values[:]]
        if df_by_block.switches.values[-1] < len(df.index):
            bloc_area = [(df_by_block.switches.values[i], df_by_block.switches.values[i + 1]) for i in range(len(df_by_block.switches) - 1)]
            bloc_area.append((df_by_block.switches.values[-1], len(df.index)))
            if len(bloc_area) > len(bloc_area_color):
                bloc_area = bloc_area[0: len(bloc_area_color)]
            for index, coords in enumerate(bloc_area):
                color = bloc_area_color[index]
                ax2.axvspan(coords[0], coords[1], alpha=0.25, facecolor=color, zorder=1)


def plot_single_mouse_opto_grid(data, result_path):

    for mouse, mouse_data in data.groupby('mouse_id'):
        saving_path = os.path.join(result_path, mouse)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig.suptitle(f'Opto grid performance {mouse}')

        fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig1.suptitle(f'Opto grid control substracted {mouse}')

        fig2, ax2 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig2.suptitle(f'Opto grid trial density {mouse}')

        fig3, ax3 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig3.suptitle(f'Performance {mouse} sigma from shuffle')

        fig4, ax4 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig4.suptitle(f'Performance substracted {mouse} sigma from shuffle_sub')

        fig5, ax5 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig5.suptitle(f'Performance percentile {mouse}')

        fig6, ax6 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig6.suptitle(f'Performance percentile substracted {mouse}')

        fig7, ax7 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig7.suptitle(f'P value fdr corrected {mouse}')

        fig8, ax8 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
        fig8.suptitle(f'P value substracted fdr corrected {mouse}')

        mouse_shuffle_grid_agg = []
        for name, group in mouse_data.groupby(by=['context_background', 'trial_type']):
            trial_type_grid = []
            if 'whisker_trial' in name:
                outcome = 'outcome_w'
                col = 2
            elif 'auditory_trial' in name:
                outcome = 'outcome_a'
                col = 1
            else:
                outcome = 'outcome_n'
                col = 0

            control = group.loc[group.opto_stim == 0].drop_duplicates()
            stim = group.loc[group.opto_stim == 1].drop_duplicates()

            stim['opto_grid_no_global'] = stim.groupby(by=['session_id', 'opto_grid_no']).ngroup()

            row = group.context.unique()[0]-1
            density_grid = stim.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].count().reset_index()

            trial_grid = stim.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).reset_index()
            nostim_grid = control.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).reset_index()
            trial_grid[f"{outcome}_sub"] = trial_grid[outcome] - nostim_grid[outcome].values[0]

            fig, ax[row, col] = plot_opto_on_allen(trial_grid, outcome=outcome, palette='viridis', vmin=0, vmax=1, fig=fig, ax=ax[row, col], result_path=None)
            fig1, ax1[row, col] = plot_opto_on_allen(trial_grid, outcome=f"{outcome}_sub", palette='seismic', vmin=-1, vmax=1, fig=fig1, ax=ax1[row, col], result_path=None)
            fig2, ax2[row, col] = plot_opto_on_allen(density_grid, outcome=outcome, palette='viridis', vmin=0, vmax=density_grid[outcome].max(), fig=fig2, ax=ax2[row, col], result_path=None)

            for i in tqdm(range(10000)):
                shuffle_group = group.copy().reset_index()
                shuffle_group[outcome] = shuffle_group.groupby('session_id')[outcome].apply(lambda x:np.random.permutation(x.values)).reset_index().explode(outcome, ignore_index=True)[outcome]
                control = shuffle_group.loc[shuffle_group.opto_stim == 0]
                stim = shuffle_group.loc[shuffle_group.opto_stim == 1]
                shuffle_grid = stim.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).reset_index()
                nostim_grid = control.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].apply(
                    np.nanmean).reset_index()
                shuffle_grid[f"{outcome}_sub"] = shuffle_grid[outcome] - nostim_grid[outcome].values[0]
                trial_type_grid += [shuffle_grid]

            trial_type_grid = pd.concat(trial_type_grid)
            trial_type_grid_agg = trial_type_grid.groupby(by=['opto_grid_ml', 'opto_grid_ap']).agg(
                shuffle_mean=(outcome, 'mean'), shuffle_std=(outcome, 'std'), shuffle_mean_sub=(f'{outcome}_sub', 'mean'), shuffle_std_sub=(f'{outcome}_sub', 'std'))
            trial_type_grid_agg['shuffle_std'] = trial_type_grid_agg.shuffle_std.mask(trial_type_grid_agg.shuffle_std==0).fillna(0.000001)
            trial_type_grid_agg['shuffle_std_sub'] = trial_type_grid_agg.shuffle_std.mask(trial_type_grid_agg.shuffle_std_sub==0).fillna(0.000001)

            trial_type_grid_agg['shuffle_dist'] = trial_type_grid.reset_index(drop=True).pivot_table(outcome, ['opto_grid_ml', 'opto_grid_ap'], aggfunc=list)
            trial_type_grid_agg['shuffle_dist_sub'] = trial_type_grid.reset_index(drop=True).pivot_table(f"{outcome}_sub", ['opto_grid_ml', 'opto_grid_ap'], aggfunc=list)
            trial_type_grid_agg['data_mean'] = trial_grid.pivot_table(outcome, ['opto_grid_ml', 'opto_grid_ap'])[outcome]
            trial_type_grid_agg['data_mean_sub'] = trial_grid.pivot_table(f'{outcome}_sub', ['opto_grid_ml', 'opto_grid_ap'])[f'{outcome}_sub']
            trial_type_grid_agg['percentile'] = trial_type_grid_agg.apply(lambda x: np.sum(x['data_mean']>=np.asarray(x.shuffle_dist))/len(x.shuffle_dist), axis=1)
            trial_type_grid_agg['percentile_sub'] = trial_type_grid_agg.apply(lambda x: np.sum(x['data_mean_sub']>=np.asarray(x.shuffle_dist_sub))/len(x.shuffle_dist_sub), axis=1)
            trial_type_grid_agg['n_sigma'] = trial_type_grid_agg.apply(lambda x: (x['data_mean']-x['shuffle_mean'])/x['shuffle_std'], axis=1)
            trial_type_grid_agg['n_sigma_sub'] = trial_type_grid_agg.apply(lambda x: (x['data_mean_sub']-x['shuffle_mean_sub'])/x['shuffle_std_sub'], axis=1)
            trial_type_grid_agg['p'] = trial_type_grid_agg.apply(lambda x: 2 * (1 - norm.cdf(abs(x.n_sigma))), axis=1)
            reject, adj_pvals, _, __ = multipletests(trial_type_grid_agg['p'].values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
            trial_type_grid_agg['p_corr'] = adj_pvals
            trial_type_grid_agg['p_sub'] = trial_type_grid_agg.apply(lambda x: 2 * (1 - norm.cdf(abs(x.n_sigma_sub))), axis=1)
            reject, adj_pvals, _, __ = multipletests(trial_type_grid_agg['p_sub'].values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
            trial_type_grid_agg['p_corr_sub'] = adj_pvals
            trial_type_grid_agg['context'] = ['rewarded' if group.context.unique()[0]==1 else 'non-rewarded' for i in range(trial_type_grid_agg.shape[0])]
            trial_type_grid_agg['context_background'] = [name[0] for i in range(trial_type_grid_agg.shape[0])]
            trial_type_grid_agg['trial_type'] = [name[1] for i in range(trial_type_grid_agg.shape[0])]

            trial_type_grid_agg = trial_type_grid_agg.reset_index()
            mouse_shuffle_grid_agg += [trial_type_grid_agg]

            fig3, ax3[row, col] = plot_opto_on_allen(trial_type_grid_agg, outcome='n_sigma', palette='icefire', vmin=-trial_type_grid_agg.n_sigma.abs().max(), vmax=trial_type_grid_agg.n_sigma.abs().max(), fig=fig3, ax=ax3[row, col], result_path=None)
            fig4, ax4[row, col] = plot_opto_on_allen(trial_type_grid_agg, outcome="n_sigma_sub", palette='icefire', vmin=-trial_type_grid_agg.n_sigma_sub.abs().max(), vmax=trial_type_grid_agg.n_sigma_sub.abs().max(), fig=fig4, ax=ax4[row, col], result_path=None)
            percentile_palette = sns.diverging_palette(45, 45, l=0, sep=195, n=200, center="light", as_cmap=True)
            fig5, ax5[row, col] = plot_opto_on_allen(trial_type_grid_agg, outcome='percentile', palette=percentile_palette, vmin=0, vmax=1, fig=fig5, ax=ax5[row, col], result_path=None)
            fig6, ax6[row, col] = plot_opto_on_allen(trial_type_grid_agg, outcome='percentile_sub', palette=percentile_palette, vmin=0, vmax=1, fig=fig6, ax=ax6[row, col], result_path=None)
            fig7, ax7[row, col] = plot_opto_on_allen(trial_type_grid_agg, outcome='p_corr', palette='Greys', vmin=0.0001, vmax=0.05, fig=fig7, ax=ax7[row, col], result_path=None)
            fig8, ax8[row, col] = plot_opto_on_allen(trial_type_grid_agg, outcome='p_corr_sub', palette='Greys', vmin=0.0001, vmax=0.05, fig=fig8, ax=ax8[row, col], result_path=None)

        mouse_shuffle_grid_agg = pd.concat(mouse_shuffle_grid_agg).reset_index()
        mouse_shuffle_grid_agg.to_json(os.path.join(saving_path, 'opto_data.json'))

        cols = ['No stim', 'Auditory', 'Whisker']
        rows = ['Rewarded', 'No rewarded']
        for a, col in zip(ax[0], cols):
            a.set_title(col)
        for a, row in zip(ax[:, 0], rows):
            a.set_ylabel(row)

        fig.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', f'{mouse}_opto_grid_performance.{save_format}'),
                           format=f"{save_format}")

        for a, col in zip(ax1[0], cols):
            a.set_title(col)
        for a, row in zip(ax1[:, 0], rows):
            a.set_ylabel(row)

        fig1.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig1.savefig(os.path.join(f'{saving_path}', f'{mouse}_opto_grid_performance_sub.{save_format}'),
                        format=f"{save_format}")

        for a, col in zip(ax2[0], cols):
            a.set_title(col)
        for a, row in zip(ax2[:, 0], rows):
            a.set_ylabel(row)

        fig2.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig2.savefig(os.path.join(f'{saving_path}', f'{mouse}_opto_grid_trial_density.{save_format}'),
                        format=f"{save_format}")

        for a, col in zip(ax3[0], cols):
            a.set_title(col)
        for a, row in zip(ax3[:, 0], rows):
            a.set_ylabel(row)

        fig3.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig3.savefig(os.path.join(f'{saving_path}', f'{mouse}_performance_n_sigmas.{save_format}'),
                        format=f"{save_format}")

        for a, col in zip(ax4[0], cols):
            a.set_title(col)
        for a, row in zip(ax4[:, 0], rows):
            a.set_ylabel(row)

        fig4.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig4.savefig(os.path.join(f'{saving_path}', f'{mouse}_performance_n_sigmas_sub.{save_format}'),
                        format=f"{save_format}")

        for a, col in zip(ax5[0], cols):
            a.set_title(col)
        for a, row in zip(ax5[:, 0], rows):
            a.set_ylabel(row)

        fig5.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig5.savefig(os.path.join(f'{saving_path}', f'{mouse}_percentile.{save_format}'),
                        format=f"{save_format}")

        for a, col in zip(ax6[0], cols):
            a.set_title(col)
        for a, row in zip(ax6[:, 0], rows):
            a.set_ylabel(row)

        fig6.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig6.savefig(os.path.join(f'{saving_path}', f'{mouse}_percentile_sub.{save_format}'),
                        format=f"{save_format}")

        for a, col in zip(ax5[0], cols):
            a.set_title(col)
        for a, row in zip(ax5[:, 0], rows):
            a.set_ylabel(row)

        fig5.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig5.savefig(os.path.join(f'{saving_path}', f'{mouse}_percentile.{save_format}'),
                        format=f"{save_format}")

        for a, col in zip(ax7[0], cols):
            a.set_title(col)
        for a, row in zip(ax7[:, 0], rows):
            a.set_ylabel(row)

        fig7.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig7.savefig(os.path.join(f'{saving_path}', f'{mouse}_p_corr.{save_format}'),
                        format=f"{save_format}")

        for a, col in zip(ax8[0], cols):
            a.set_title(col)
        for a, row in zip(ax8[:, 0], rows):
            a.set_ylabel(row)

        fig8.tight_layout()
        save_formats = ['png']
        for save_format in save_formats:
            fig8.savefig(os.path.join(f'{saving_path}', f'{mouse}_p_corr_sub.{save_format}'),
                        format=f"{save_format}")
    return


def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def plot_multiple_mice_opto_grid(data, result_path):

    single_mouse_result_files = glob.glob(os.path.join(result_path, "*", "opto_data.json"))
    df = []
    for file in single_mouse_result_files:
        mouse_data = pd.read_json(file)
        mouse_data['mouse_name'] = [file.split("\\")[-2] for i in range(mouse_data.shape[0])]
        df += [mouse_data]
    df = pd.concat(df)
    avg_df = df.groupby(by=['context', 'trial_type', 'opto_grid_ml', 'opto_grid_ap']).agg(
                                                                                 data=('data_mean', list),
                                                                                 data_sub=('data_mean_sub', list),
                                                                                 data_mean=('data_mean', 'mean'),
                                                                                 data_mean_sub=(
                                                                                 'data_mean_sub', 'mean'),
                                                                                 shuffle_dist=('shuffle_dist', 'sum'),
                                                                                 shuffle_dist_sub=(
                                                                                 'shuffle_dist_sub', 'sum'),
                                                                                 percentile_avg=('percentile', 'mean'),
                                                                                 percentile_avg_sub=(
                                                                                 'percentile_sub', 'mean'),
                                                                                 n_sigma_avg=('n_sigma', 'mean'),
                                                                                 n_sigma_avg_sub=(
                                                                                 'n_sigma_sub', 'mean'))
    avg_df['shuffle_mean'] = avg_df.apply(lambda x: np.mean(x.shuffle_dist), axis=1)
    avg_df['shuffle_std'] = avg_df.apply(lambda x: np.std(x.shuffle_dist), axis=1)
    avg_df['shuffle_mean_sub'] = avg_df.apply(lambda x: np.mean(x.shuffle_dist_sub), axis=1)
    avg_df['shuffle_std_sub'] = avg_df.apply(lambda x: np.std(x.shuffle_dist_sub), axis=1)
    avg_df['total_percentile_avg'] = avg_df.apply(lambda x: np.mean(x.percentile_avg), axis=1)
    avg_df['total_percentile_avg_sub'] = avg_df.apply(lambda x: np.mean(x.percentile_avg_sub), axis=1)
    avg_df['total_sigma_avg'] = avg_df.apply(lambda x: np.mean(x.n_sigma_avg), axis=1)
    avg_df['total_sigma_avg_sub'] = avg_df.apply(lambda x: np.mean(x.n_sigma_avg_sub), axis=1)
    avg_df['abs_sigma_avg_sub'] = avg_df.apply(lambda x: np.abs(np.mean(x.n_sigma_avg_sub)), axis=1)
    avg_df['percentile'] = avg_df.apply(lambda x: np.sum(x['data_mean'] >= np.asarray(x.shuffle_dist)) / len(x.shuffle_dist), axis=1)
    avg_df['percentile_sub'] = avg_df.apply(lambda x: np.sum(x['data_mean_sub'] >= np.asarray(x.shuffle_dist_sub)) / len(x.shuffle_dist_sub), axis=1)
    avg_df['n_sigma'] = avg_df.apply(lambda x: (x['data_mean'] - x['shuffle_mean']) / x['shuffle_std'], axis=1)
    avg_df['n_sigma_sub'] = avg_df.apply(lambda x: (x['data_mean_sub'] - x['shuffle_mean_sub']) / x['shuffle_std_sub'], axis=1)
    avg_df['p'] = avg_df.apply(lambda x: 2*min(1-x.percentile, x.percentile), axis=1)
    reject, adj_pvals, _, __ = multipletests(avg_df['p'].values, alpha=0.05, method='fdr_bh',
                                             is_sorted=False, returnsorted=False)
    avg_df['p_corr'] = adj_pvals
    avg_df['p_sub'] = avg_df.apply(lambda x: 2*min(1-x.percentile_sub, x.percentile_sub), axis=1)
    reject, adj_pvals, _, __ = multipletests(avg_df['p_sub'].values, alpha=0.05, method='fdr_bh',
                                             is_sorted=False, returnsorted=False)
    avg_df['p_corr_sub'] = adj_pvals
    avg_df['d'] = avg_df.apply(lambda x: abs(cohen_d(x.shuffle_dist, x.data)), axis=1)
    avg_df['d_sub'] = avg_df.apply(lambda x: abs(cohen_d(x.shuffle_dist_sub, x.data_sub)), axis=1)

    avg_df = avg_df.reset_index()

    fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig.suptitle(f'Opto grid performance')

    fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig1.suptitle(f'Opto grid control substracted')

    fig2, ax2 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig2.suptitle(f'Opto grid trial density')

    fig3, ax3 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig3.suptitle(f'Performance sigma from shuffle')

    fig4, ax4 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig4.suptitle(f'Performance substracted sigma from shuffle_sub')

    fig5, ax5 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig5.suptitle(f'Performance percentile ')

    fig6, ax6 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig6.suptitle(f'Performance percentile substracted')

    fig7, ax7 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig7.suptitle(f'P value fdr corrected')

    fig8, ax8 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig8.suptitle(f'P value substracted fdr corrected')

    fig9, ax9 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig9.suptitle(f'Avg percentile')

    fig10, ax10 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig10.suptitle(f'Avg percentile substracted')

    fig11, ax11 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig11.suptitle(f'Avg sigmas from mean')

    fig12, ax12 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig12.suptitle(f'Avg sigmas from mean substracted')

    fig13, ax13 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig13.suptitle(f'abs(avg_sigma_sub)>1.5')

    for name, group in avg_df.groupby(by=['context', 'trial_type']):
        if 'whisker_trial' in name:
            outcome = 'outcome_w'
            col = 2
        elif 'auditory_trial' in name:
            outcome = 'outcome_a'
            col = 1
        else:
            outcome = 'outcome_n'
            col = 0

        row = 0 if 'rewarded' in name else 1

        data_trial = data.groupby(by=['context', 'trial_type']).get_group((0 if name[0]=='non-rewarded' else 1, name[1]))
        stim = data_trial.loc[data_trial.opto_stim == 1].drop_duplicates()
        density_grid = stim.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].count().reset_index()

        fig, ax[row, col] = plot_opto_on_allen(group, outcome='data_mean', palette='viridis', vmin=0, vmax=1, fig=fig,
                                               ax=ax[row, col], result_path=None)
        fig1, ax1[row, col] = plot_opto_on_allen(group, outcome=f"data_mean_sub", palette='seismic', vmin=-0.5,
                                                 vmax=0.5, fig=fig1, ax=ax1[row, col], result_path=None)
        fig2, ax2[row, col] = plot_opto_on_allen(density_grid, outcome=outcome, palette='viridis', vmin=0,
                                                 vmax=density_grid[outcome].max(), fig=fig2, ax=ax2[row, col],
                                                 result_path=None)

        fig3, ax3[row, col] = plot_opto_on_allen(group, outcome='n_sigma', palette='icefire',
                                                 vmin=-1.5,
                                                 vmax=1.5, fig=fig3,
                                                 ax=ax3[row, col], result_path=None)
        fig4, ax4[row, col] = plot_opto_on_allen(group, outcome="n_sigma_sub", palette='icefire',
                                                 vmin=-1.5,
                                                 vmax=1.5, fig=fig4,
                                                 ax=ax4[row, col], result_path=None)
        percentile_palette = sns.diverging_palette(45, 45, l=0, sep=195, n=200, center="light", as_cmap=True)
        fig5, ax5[row, col] = plot_opto_on_allen(group, outcome='percentile', palette=percentile_palette,
                                                 vmin=0, vmax=1, fig=fig5, ax=ax5[row, col], result_path=None)
        fig6, ax6[row, col] = plot_opto_on_allen(group, outcome='percentile_sub',
                                                 palette=percentile_palette, vmin=0, vmax=1, fig=fig6, ax=ax6[row, col],
                                                 result_path=None)
        fig7, ax7[row, col] = plot_opto_on_allen(group, outcome='p_corr', palette='Greys', vmin=0.0001,
                                                 vmax=0.05, fig=fig7, ax=ax7[row, col], result_path=None)
        fig8, ax8[row, col] = plot_opto_on_allen(group, outcome='p_corr_sub', palette='Greys',
                                                 vmin=0.0001, vmax=0.05, fig=fig8, ax=ax8[row, col], result_path=None)

        fig9, ax9[row, col] = plot_opto_on_allen(group, outcome='total_percentile_avg', palette=percentile_palette,
                                                 vmin=0, vmax=1, fig=fig9, ax=ax9[row, col], result_path=None)
                                                 
        fig10, ax10[row, col] = plot_opto_on_allen(group, outcome='total_percentile_avg_sub',
                                                 palette=percentile_palette, vmin=0, vmax=1, fig=fig10, ax=ax10[row, col],
                                                 result_path=None)
        fig11, ax11[row, col] = plot_opto_on_allen(group, outcome='total_sigma_avg', palette='icefire',
                                                 vmin=-1.5,
                                                 vmax=1.5, fig=fig11,
                                                 ax=ax11[row, col], result_path=None)
        fig12, ax12[row, col] = plot_opto_on_allen(group, outcome="total_sigma_avg_sub", palette='icefire',
                                                 vmin=-1.5,
                                                 vmax=1.5, fig=fig12,
                                                 ax=ax12[row, col], result_path=None)
        fig13, ax13[row, col] = plot_opto_on_allen(group, outcome="total_sigma_avg_sub", palette='Greys',
                                                 vmin=1.5,
                                                 vmax=3, fig=fig13,
                                                 ax=ax13[row, col], result_path=None)

    cols = ['No stim', 'Auditory', 'Whisker']
    rows = ['Rewarded', 'No rewarded']
    for a, col in zip(ax[0], cols):
        a.set_title(col)
    for a, row in zip(ax[:, 0], rows):
        a.set_ylabel(row)

    fig.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{result_path}', f'avg_opto_grid_performance.{save_format}'),
                       format=f"{save_format}")

    for a, col in zip(ax1[0], cols):
        a.set_title(col)
    for a, row in zip(ax1[:, 0], rows):
        a.set_ylabel(row)

    fig1.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig1.savefig(os.path.join(f'{result_path}', f'avg_opto_grid_performance_sub.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax2[0], cols):
        a.set_title(col)
    for a, row in zip(ax2[:, 0], rows):
        a.set_ylabel(row)

    fig2.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig2.savefig(os.path.join(f'{result_path}', f'avg_opto_grid_trial_density.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax3[0], cols):
        a.set_title(col)
    for a, row in zip(ax3[:, 0], rows):
        a.set_ylabel(row)

    fig3.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig3.savefig(os.path.join(f'{result_path}', f'avg_performance_n_sigmas.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax4[0], cols):
        a.set_title(col)
    for a, row in zip(ax4[:, 0], rows):
        a.set_ylabel(row)

    fig4.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig4.savefig(os.path.join(f'{result_path}', f'avg_performance_n_sigmas_sub.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax5[0], cols):
        a.set_title(col)
    for a, row in zip(ax5[:, 0], rows):
        a.set_ylabel(row)

    fig5.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig5.savefig(os.path.join(f'{result_path}', f'avg_percentile.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax6[0], cols):
        a.set_title(col)
    for a, row in zip(ax6[:, 0], rows):
        a.set_ylabel(row)

    fig6.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig6.savefig(os.path.join(f'{result_path}', f'avg_percentile_sub.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax5[0], cols):
        a.set_title(col)
    for a, row in zip(ax5[:, 0], rows):
        a.set_ylabel(row)

    fig5.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig5.savefig(os.path.join(f'{result_path}', f'avg_percentile.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax7[0], cols):
        a.set_title(col)
    for a, row in zip(ax7[:, 0], rows):
        a.set_ylabel(row)

    fig7.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig7.savefig(os.path.join(f'{result_path}', f'avg_p_corr.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax8[0], cols):
        a.set_title(col)
    for a, row in zip(ax8[:, 0], rows):
        a.set_ylabel(row)

    fig8.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig8.savefig(os.path.join(f'{result_path}', f'avg_p_corr_sub.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax9[0], cols):
        a.set_title(col)
    for a, row in zip(ax9[:, 0], rows):
        a.set_ylabel(row)

    fig9.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig9.savefig(os.path.join(f'{result_path}', f'avg_percentile_by_mouse.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax10[0], cols):
        a.set_title(col)
    for a, row in zip(ax10[:, 0], rows):
        a.set_ylabel(row)

    fig10.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig10.savefig(os.path.join(f'{result_path}', f'avg_percentile_by_mouse_sub.{save_format}'),
                     format=f"{save_format}")

    for a, col in zip(ax11[0], cols):
        a.set_title(col)
    for a, row in zip(ax11[:, 0], rows):
        a.set_ylabel(row)

    fig11.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig11.savefig(os.path.join(f'{result_path}', f'avg_sigma_by_mouse.{save_format}'),
                    format=f"{save_format}")

    for a, col in zip(ax12[0], cols):
        a.set_title(col)
    for a, row in zip(ax12[:, 0], rows):
        a.set_ylabel(row)

    fig12.tight_layout()
    save_formats = ['png']
    for save_format in save_formats:
        fig12.savefig(os.path.join(f'{result_path}', f'avg_sigma_by_mouse_sub.{save_format}'),
                    format=f"{save_format}")
    return


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
        control = group.loc[group.opto_stim == 0].drop_duplicates()
        stim = group.loc[group.opto_stim == 1].drop_duplicates()

        stim['opto_grid_no_global'] = stim.groupby(by=['session_id', 'opto_grid_no']).ngroup()

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

        plot_opto_on_allen(grid, f'{outcome}_sub', 'seismic', os.path.join(saving_path, 'no_context'))


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


def plot_opto_on_allen(grid, outcome, palette, result_path, vmin=-1, vmax=1, fig=None, ax=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5), dpi=200)
        new_fig = True
    else:
        new_fig = False
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)

    cmap = get_colormap('gray')
    cmap.set_bad(color='white')
    bregma = (488, 290)
    scale = 4
    scalebar = get_wf_scalebar(scale=scale)
    iso_mask, atlas_mask, allen_bregma = get_allen_ccf(bregma)

    grid['opto_grid_ml_wf'] = bregma[0] - grid['opto_grid_ml'] * scalebar
    grid['opto_grid_ap_wf'] = bregma[1] - grid['opto_grid_ap'] * scalebar

    # fig, ax = plt.subplots(1, figsize=(5, 5), dpi=200)
    single_frame = np.rot90(rescale(np.ones([125, 160]), scale, anti_aliasing=False))
    single_frame = np.pad(single_frame, [(0, 650 - single_frame.shape[0]), (0, 510 - single_frame.shape[1])],
                          mode='constant', constant_values=np.nan)
    im = ax.imshow(single_frame, cmap=cmap, vmin=0, vmax=1)
    g = sns.scatterplot(data=grid, x='opto_grid_ml_wf', y='opto_grid_ap_wf', hue=f'{outcome}',
                    hue_norm=plt.Normalize(vmin, vmax), s=280, palette=palette, ax=ax)
    ax.contour(atlas_mask, levels=np.unique(atlas_mask), colors='gray',
               linewidths=1)
    ax.contour(iso_mask, levels=np.unique(np.round(iso_mask)), colors='black',
               linewidths=2, zorder=2)
    ax.scatter(bregma[0], bregma[1], marker='+', c='r', s=300, linewidths=4,
               zorder=3)
    ax.set_xticks(np.unique(grid['opto_grid_ml_wf']), np.arange(5.5, 0, -1))
    ax.set_yticks(np.unique(grid['opto_grid_ap_wf']), np.arange(3.5, -4, -1))
    ax.set_aspect(1)
    ax.set_axis_off()
    ax.get_legend().remove()
    ax.hlines(5, 5, 5 + scalebar * 3, linewidth=2, colors='k')
    # ax.text(50, 100, "3 mm", size=10)
    if 'p_corr' in outcome:
        norm = colors.LogNorm(vmin, vmax)
    else:
        norm = plt.Normalize(vmin, vmax)

    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm, cax=cax, orientation='horizontal')

    if new_fig and result_path is not None:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        fig.savefig(result_path + ".png")
        fig.savefig(result_path + ".svg")

    if new_fig == False:
        return fig, ax

