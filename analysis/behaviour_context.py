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
import utils.behaviour_plot_utils as plot_utils
from nwb_utils.utils_misc import get_continuous_time_periods
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame

warnings.filterwarnings("ignore")


def plot_single_session(combine_bhv_data, color_palette, saving_path):
    combine_bhv_data = bhv_utils.compute_single_session_metrics(combine_bhv_data)
    combined_by_block = bhv_utils.get_by_block_table(combine_bhv_data)

    above_threshold_table = combined_by_block.groupby('session_id').apply(lambda x: bhv_utils.compute_above_threshold(x, 4))
    above_threshold_table = pd.concat(pd.DataFrame.from_dict(x) for x in above_threshold_table)

    fig, ax = plt.subplots(2,  1, sharex=True, figsize=(5, 10))
    for i, stat in enumerate(['percent', 'count']):
        plot_utils.plot_distributions(df=combined_by_block, stat=stat, fig=fig, ax=ax[i], x='contrast_w', y=None, hue=None, binwidth=1/16, line=True, line_loc=0.375,
                                      title='Distribution of whisker-contrast values',
                                      xlabel='Contrast lick probability',
                                      ylabel='% blocks' if stat == 'percent' else 'N blocks')

    plot_utils.save_fig(fig,
                        save_path=os.path.join(saving_path, 'block_distribution'),
                        name="all_block_distribution",
                        save_format=['pdf', 'png', 'svg'])

    fig, ax = plt.subplots(2,  1, sharex=True, figsize=(5, 10))
    for i, stat in enumerate(['percent', 'count']):
        plot_utils.plot_distributions(df=combined_by_block, stat=stat, fig=fig, ax=ax[i], x='contrast_w', y=None, hue='context', binwidth=1/16, line=True, line_loc=0.375,
                                      title='Distribution of whisker-contrast values',
                                      xlabel='Contrast lick probability',
                                      ylabel='% blocks' if stat == 'percent' else 'N blocks')
    plot_utils.save_fig(fig,
                        save_path=os.path.join(saving_path, 'block_distribution'),
                        name="all_block_distribution_context_overlay",
                        save_format=['pdf', 'png', 'svg'])

    fig, ax = plt.subplots(2,  2, sharex=True, figsize=(5, 10))
    for i, stat in enumerate(['percent', 'count']):
        plot_utils.plot_distributions(df=combined_by_block.loc[combined_by_block['context'] == 'Rewarded'], stat=stat, fig=fig, ax=ax[i, 0], x='contrast_w', y=None, hue=None, binwidth=1/16, line=True, line_loc=0.375,
                                      title='Rewarded',
                                      xlabel='Contrast lick probability',
                                      ylabel='% blocks' if stat == 'percent' else 'N blocks')

        plot_utils.plot_distributions(df=combined_by_block.loc[combined_by_block['context'] == 'Non-Rewarded'], stat=stat, fig=fig, ax=ax[i, 1], x='contrast_w', y=None, hue=None, binwidth=1/16, line=True, line_loc=0.375,
                                      title='Non-Rewarded',
                                      xlabel='Contrast lick probability',
                                      ylabel='% blocks' if stat == 'percent' else 'N blocks')
    plot_utils.save_fig(fig,
                        save_path=os.path.join(saving_path, 'block_distribution'),
                        name="all_block_distribution_context_split",
                        save_format=['pdf', 'png', 'svg'])



    return


def plot_behaviour(nwb_list, save_path):
    combine_bhv_data = bhv_utils.build_standard_behavior_table(nwb_list)

    colors = ['#225ea8', '#00FFFF', '#238443', '#d51a1c', '#cccccc']
    plot_single_session(combine_bhv_data=combine_bhv_data, color_palette=colors, saving_path=save_path)

    return