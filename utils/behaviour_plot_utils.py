import itertools
import os
import warnings

import matplotlib.pyplot as plt
import yaml
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

import nwb_utils.utils_behavior as bhv_utils
from nwb_utils.utils_misc import get_continuous_time_periods
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame

warnings.filterwarnings("ignore")


def plot_single_session(combine_bhv_data, color_palette, save_path):
    raster_marker = 2
    marker_width = 2
    figsize = (15, 8)
    marker = itertools.cycle(['o', 's'])
    markers = [next(marker) for i in d["opto_stim"].unique()]

    return


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
