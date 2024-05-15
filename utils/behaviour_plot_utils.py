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
