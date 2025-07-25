import os
import sys
sys.path.append(os.getcwd())
import yaml
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_1samp, ttest_rel
from tqdm import tqdm
from utils.haas_utils import *
from utils.wf_plotting_utils import plot_single_frame, reduce_im_dimensions, plot_grid_on_allen, generate_reduced_image_df


area_dict= {'(-0.5, 0.5)': 38, '(-0.5, 1.5)': 31, '(-0.5, 2.5)': 24, '(-0.5, 3.5)': 17, '(-0.5, 4.5)': 10, '(-0.5, 5.5)': 3, '(-1.5, 0.5)': 39,
            '(-1.5, 1.5)': 32, '(-1.5, 2.5)': 25, '(-1.5, 3.5)': 18, '(-1.5, 4.5)': 11, '(-1.5, 5.5)': 4, '(-2.5, 0.5)': 40, '(-2.5, 1.5)': 33,
            '(-2.5, 2.5)': 26, '(-2.5, 3.5)': 19, '(-2.5, 4.5)': 12, '(-2.5, 5.5)': 5, '(-3.5, 0.5)': 41, '(-3.5, 1.5)': 34, '(-3.5, 2.5)': 27,
            '(-3.5, 3.5)': 20, '(-3.5, 4.5)': 13, '(-3.5, 5.5)': 6, '(0.5, 0.5)': 37, '(0.5, 1.5)': 30, '(0.5, 2.5)': 23, '(0.5, 3.5)': 16,
            '(0.5, 4.5)': 9, '(0.5, 5.5)': 2, '(1.5, 0.5)': 36, '(1.5, 1.5)': 29, '(1.5, 2.5)': 22, '(1.5, 3.5)': 15, '(1.5, 4.5)': 8,
            '(1.5, 5.5)': 1, '(2.5, 0.5)': 35, '(2.5, 1.5)': 28, '(2.5, 2.5)': 21,  '(2.5, 3.5)': 14, '(2.5, 4.5)': 7, '(2.5, 5.5)': 0}

def preprocess_corr_results(file):

    df = pd.read_parquet(file.replace("\\", "/"), use_threads=False)

    if 'opto' in file:
        
        df['opto_stim_coord'] = df.apply(lambda x: f"({x.opto_grid_ap}, {x.opto_grid_ml})",axis=1)
        avg_df = df[df.trial_type == 'no_stim_trial'].melt(id_vars=['mouse_id', 'session_id', 'context', 'context_background', 'opto_stim_coord', 'correct_trial'],
                value_vars=['(-0.5, 0.5)_r', '(-0.5, 0.5)_shuffle_mean', '(-0.5, 0.5)_shuffle_std', '(-0.5, 0.5)_percentile', '(-0.5, 0.5)_nsigmas', 
                            '(-1.5, 0.5)_r', '(-1.5, 0.5)_shuffle_mean', '(-1.5, 0.5)_shuffle_std', '(-1.5, 0.5)_percentile', '(-1.5, 0.5)_nsigmas', 
                            '(-1.5, 3.5)_r', '(-1.5, 3.5)_shuffle_mean', '(-1.5, 3.5)_shuffle_std', '(-1.5, 3.5)_percentile', '(-1.5, 3.5)_nsigmas',
                            '(-1.5, 4.5)_r', '(-1.5, 4.5)_shuffle_mean', '(-1.5, 4.5)_shuffle_std', '(-1.5, 4.5)_percentile', '(-1.5, 4.5)_nsigmas', 
                            '(1.5, 3.5)_r', '(1.5, 3.5)_shuffle_mean', '(1.5, 3.5)_shuffle_std', '(1.5, 3.5)_percentile', '(1.5, 3.5)_nsigmas', 
                            '(0.5, 4.5)_r', '(0.5, 4.5)_shuffle_mean', '(0.5, 4.5)_shuffle_std', '(0.5, 4.5)_percentile', '(0.5, 4.5)_nsigmas', 
                            '(1.5, 1.5)_r', '(1.5, 1.5)_shuffle_mean', '(1.5, 1.5)_shuffle_std', '(1.5, 1.5)_percentile', '(1.5, 1.5)_nsigmas',
                            '(2.5, 2.5)_r', '(2.5, 2.5)_shuffle_mean', '(2.5, 2.5)_shuffle_std', '(2.5, 2.5)_percentile', '(2.5, 2.5)_nsigmas'])

    else:
        df['block_id'] = np.abs(np.diff(df.context.values, prepend=0)).cumsum()
        df['trial_count'] = np.empty(len(df), dtype=int)
        df.loc[df.trial_type == 'whisker_trial', 'trial_count'] = df.loc[df.trial_type == 'whisker_trial'].groupby(
            'block_id').cumcount()
        df.loc[df.trial_type == 'auditory_trial', 'trial_count'] = df.loc[
            df.trial_type == 'auditory_trial'].groupby(
            'block_id').cumcount()
        df.loc[df.trial_type == 'no_stim_trial', 'trial_count'] = df.loc[df.trial_type == 'no_stim_trial'].groupby(
            'block_id').cumcount()

        df = df.melt(id_vars=['mouse_id', 'session_id', 'context', 'context_background', 'block_id', 'correct_trial'],
                value_vars=['(-0.5, 0.5)_r', '(-0.5, 0.5)_shuffle_mean', '(-0.5, 0.5)_shuffle_std', '(-0.5, 0.5)_percentile', '(-0.5, 0.5)_nsigmas', 
                            '(-1.5, 3.5)_r', '(-1.5, 3.5)_shuffle_mean', '(-1.5, 3.5)_shuffle_std', '(-1.5, 3.5)_percentile', '(-1.5, 3.5)_nsigmas',
                            '(-1.5, 4.5)_r', '(-1.5, 4.5)_shuffle_mean', '(-1.5, 4.5)_shuffle_std', '(-1.5, 4.5)_percentile', '(-1.5, 4.5)_nsigmas', 
                            '(1.5, 3.5)_r', '(1.5, 3.5)_shuffle_mean', '(1.5, 3.5)_shuffle_std', '(1.5, 3.5)_percentile', '(1.5, 3.5)_nsigmas', 
                            '(0.5, 4.5)_r', '(0.5, 4.5)_shuffle_mean', '(0.5, 4.5)_shuffle_std', '(0.5, 4.5)_percentile', '(0.5, 4.5)_nsigmas', 
                            '(1.5, 1.5)_r', '(1.5, 1.5)_shuffle_mean', '(1.5, 1.5)_shuffle_std', '(1.5, 1.5)_percentile', '(1.5, 1.5)_nsigmas',
                            '(2.5, 2.5)_r', '(2.5, 2.5)_shuffle_mean', '(2.5, 2.5)_shuffle_std', '(2.5, 2.5)_percentile', '(2.5, 2.5)_nsigmas'])

        avg_df = df.groupby(by=['mouse_id', 'session_id', 'context', 'correct_trial', 'variable'])[
            'value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    return avg_df


def preprocess_pw_results(file):

    df = pd.read_parquet(file.replace("\\", "/"), use_threads=False)

    if 'opto' in file:
        
        df['opto_stim_coord'] = df.apply(lambda x: f"({x.opto_grid_ap}, {x.opto_grid_ml})",axis=1)
        df = df[df.trial_type == 'no_stim_trial'].melt(id_vars=['mouse_id', 'session_id', 'context', 'context_background', 'opto_stim_coord', 'correct_trial', 'coord_order'],
                value_vars=['(-0.5, 0.5)_r', '(-0.5, 0.5)_shuffle_mean', '(-0.5, 0.5)_shuffle_std', 
                            '(-1.5, 3.5)_r', '(-1.5, 3.5)_shuffle_mean', '(-1.5, 3.5)_shuffle_std',
                            '(-1.5, 4.5)_r', '(-1.5, 4.5)_shuffle_mean', '(-1.5, 4.5)_shuffle_std',
                            '(1.5, 3.5)_r', '(1.5, 3.5)_shuffle_mean', '(1.5, 3.5)_shuffle_std',
                            '(0.5, 4.5)_r', '(0.5, 4.5)_shuffle_mean', '(0.5, 4.5)_shuffle_std',
                            '(1.5, 1.5)_r', '(1.5, 1.5)_shuffle_mean', '(1.5, 1.5)_shuffle_std',
                            '(2.5, 2.5)_r', '(2.5, 2.5)_shuffle_mean', '(2.5, 2.5)_shuffle_std'])
        
        df['value'] = df.apply(lambda x: x.value[0], axis=1)
        df = df.explode(['coord_order', 'value'])
        avg_df = df.groupby(by=['mouse_id', 'session_id', 'context', 'correct_trial', 'opto_stim_coord', 'variable', 'coord_order'])[
            'value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    else:
        df['block_id'] = np.abs(np.diff(df.context.values, prepend=0)).cumsum()
        df['trial_count'] = np.empty(len(df), dtype=int)
        df.loc[df.trial_type == 'whisker_trial', 'trial_count'] = df.loc[df.trial_type == 'whisker_trial'].groupby(
            'block_id').cumcount()
        df.loc[df.trial_type == 'auditory_trial', 'trial_count'] = df.loc[
            df.trial_type == 'auditory_trial'].groupby(
            'block_id').cumcount()
        df.loc[df.trial_type == 'no_stim_trial', 'trial_count'] = df.loc[df.trial_type == 'no_stim_trial'].groupby(
            'block_id').cumcount()
        

        df = df.melt(id_vars=['mouse_id', 'session_id', 'context', 'context_background', 'block_id', 'correct_trial', 'coord_order'],
                value_vars=['(-0.5, 0.5)_r', '(-0.5, 0.5)_shuffle_mean', '(-0.5, 0.5)_shuffle_std', 
                            '(-1.5, 3.5)_r', '(-1.5, 3.5)_shuffle_mean', '(-1.5, 3.5)_shuffle_std',
                            '(-1.5, 4.5)_r', '(-1.5, 4.5)_shuffle_mean', '(-1.5, 4.5)_shuffle_std',
                            '(1.5, 3.5)_r', '(1.5, 3.5)_shuffle_mean', '(1.5, 3.5)_shuffle_std',
                            '(0.5, 4.5)_r', '(0.5, 4.5)_shuffle_mean', '(0.5, 4.5)_shuffle_std',
                            '(1.5, 1.5)_r', '(1.5, 1.5)_shuffle_mean', '(1.5, 1.5)_shuffle_std',
                            '(2.5, 2.5)_r', '(2.5, 2.5)_shuffle_mean', '(2.5, 2.5)_shuffle_std'])

        df['value'] = df.apply(lambda x: x.value[0], axis=1)
        df = df.explode(['coord_order', 'value'])
        avg_df = df.groupby(by=['mouse_id', 'session_id', 'context', 'correct_trial', 'variable', 'coord_order'])[
            'value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    return avg_df


def plot_avg_within_blocks(df, roi, save_path, vmin=-0.05, vmax=0.05):

    total_avg = df.groupby(by=['correct_trial', 'variable'])['value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f"{roi}_r"), 'value'].values[0],
                        title='Correct',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f"{roi}_r"), 'value'].values[0],
                        title='Incorrect',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f"{roi}_r"), 'value'].values[0] - \
            total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f"{roi}_r"), 'value'].values[0]

    plot_single_frame(im, title='Correct - Incorrect',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_r.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0],
                        title='Correct',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0],
                        title='Incorrect',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0] - \
            total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0]

    plot_single_frame(im, title='Correct - Incorrect',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0],
                        title='Correct',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0],
                        title='Incorrect',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0] - \
            total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0]

    plot_single_frame(im, title='Correct - Incorrect',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_std.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"R - shuffle")

    im_r = total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]
    im_nor = total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]

    plot_single_frame(im_r,
                        title='Correct',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(im_nor,
                        title='Incorrect',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[1])

    plot_single_frame(im_r - im_nor, title='Correct - Incorrect',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_avg.png"))

    d_palette = sns.color_palette("gnuplot2", 50)
    dprime_palette = LinearSegmentedColormap.from_list("Custom", d_palette[:-2])

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Rewarded',
                        colormap=dprime_palette, vmin=1.8, vmax=3, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap=dprime_palette, vmin=1.8, vmax=3, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0] - \
            total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]

    plot_single_frame(im, title='R+ - R-',
                        colormap=seismic_palette, vmin=-0.3, vmax=0.3, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_nsigmas.png"))

    mask_r = np.where(total_avg.loc[(total_avg.correct_trial == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_r, np.nan)
    mask_non_r = np.where(total_avg.loc[(total_avg.correct_trial == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_nor, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(mask_r,
                        title='Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(mask_non_r,
                        title='Non-Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[1])

    plot_single_frame(mask_r-mask_non_r, title='R+ - R-',
                        colormap=seismic_palette, vmin=-0.1, vmax=0.1, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_significant_pairs.png"))


def plot_avg_between_blocks(df, roi, save_path, vmin=-0.1, vmax=0.1):
    total_avg = df.groupby(by=['context', 'variable'])['value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_r"), 'value'].values[0],
                        title='Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_r"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_r"), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_r"), 'value'].values[0]

    plot_single_frame(im, title='R+ - R-',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_r.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0],
                        title='Rewarded',
                        colormap='icefire', vmin=-0.1, vmax=0.1, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap='icefire', vmin=-0.1, vmax=0.1, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0]

    plot_single_frame(im, title='R+ - R-',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0],
                        title='Rewarded',
                        colormap='viridis', vmin=0, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap='viridis', vmin=0, vmax=0.5, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0]

    plot_single_frame(im, title='R+ - R-',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_std.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"R - shuffle")

    im_r = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]
    im_nor = total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]

    plot_single_frame(im_r,
                        title='Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(im_nor,
                        title='Non-Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[1])

    plot_single_frame(im_r - im_nor, title='R+ - R-',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_avg.png"))

    d_palette = sns.color_palette("gnuplot2", 50)
    dprime_palette = LinearSegmentedColormap.from_list("Custom", d_palette[:-2])

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Rewarded',
                        colormap=dprime_palette, vmin=1.8, vmax=3, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap=dprime_palette, vmin=1.8, vmax=3, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]

    plot_single_frame(im, title='R+ - R-',
                        colormap=seismic_palette, vmin=-0.3, vmax=0.3, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_nsigmas.png"))

    mask_r = np.where(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_r, np.nan)
    mask_non_r = np.where(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_nor, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(mask_r,
                        title='Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(mask_non_r,
                        title='Non-Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.6, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]

    plot_single_frame(mask_r-mask_non_r, title='R+ - R-',
                        colormap=seismic_palette, vmin=-0.1, vmax=0.1, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_significant_pairs.png"))


def plot_reduced_correlations(df, roi, save_path):

    if not os.path.exists(os.path.join(save_path, 'red_im')):
        os.makedirs(os.path.join(save_path, 'red_im'))

    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    total_avg = df.groupby(by=['context', 'variable'])['value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    im_R = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_r"), 'value'].values[0]
    im_nR = total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_r"), 'value'].values[0]
    im_sub = im_R - im_nR

    red_im_R, coords = reduce_im_dimensions(im_R[np.newaxis, ...])
    red_im_nR, coords = reduce_im_dimensions(im_nR[np.newaxis, ...])
    red_im_sub, coords = reduce_im_dimensions(im_sub[np.newaxis, ...])
    im_R_df = generate_reduced_image_df(red_im_R, coords)
    im_nR_df = generate_reduced_image_df(red_im_nR, coords)
    im_sub_df = generate_reduced_image_df(red_im_sub, coords)

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_grid_on_allen(im_R_df, outcome='dff0', palette='viridis', result_path=None, dotsize=340, vmin=0.3, vmax=0.9, norm=None, fig=fig, ax= ax[0])
    plot_grid_on_allen(im_nR_df, outcome='dff0', palette='viridis', result_path=None, dotsize=340, vmin=0.3, vmax=0.9, norm=None, fig=fig, ax= ax[1])
    plot_grid_on_allen(im_sub_df, outcome='dff0', palette=seismic_palette, result_path=None, dotsize=340, vmin=-0.1, vmax=0.1, norm=None, fig=fig, ax= ax[2])
    fig.savefig(os.path.join(save_path, 'red_im', f'{roi}_r_reduced_images.png'))

    im_R = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_r"), 'value'].values[0] - total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0]
    im_nR = total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_r"), 'value'].values[0] - total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0]
    im_sub = im_R - im_nR

    red_im_R, coords = reduce_im_dimensions(im_R[np.newaxis, ...])
    red_im_nR, coords = reduce_im_dimensions(im_nR[np.newaxis, ...])
    red_im_sub, coords = reduce_im_dimensions(im_sub[np.newaxis, ...])
    im_R_df = generate_reduced_image_df(red_im_R, coords)
    im_nR_df = generate_reduced_image_df(red_im_nR, coords)
    im_sub_df = generate_reduced_image_df(red_im_sub, coords)

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_grid_on_allen(im_R_df, outcome='dff0', palette='viridis', result_path=None, dotsize=340, vmin=0.3, vmax=0.9, norm=None, fig=fig, ax= ax[0])
    plot_grid_on_allen(im_nR_df, outcome='dff0', palette='viridis', result_path=None, dotsize=340, vmin=0.3, vmax=0.9, norm=None, fig=fig, ax= ax[1])
    plot_grid_on_allen(im_sub_df, outcome='dff0', palette=seismic_palette, result_path=None, dotsize=340, vmin=-0.1, vmax=0.1, norm=None, fig=fig, ax= ax[2])
    fig.savefig(os.path.join(save_path, 'red_im', f'{roi}_r_corrected_reduced_images.png'))


    d_palette = sns.color_palette("gnuplot2", 50)
    dprime_palette = LinearSegmentedColormap.from_list("Custom", d_palette[:-2])

    im_R = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]
    im_nR = total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]
    im_sub = im_R - im_nR

    red_im_R, coords = reduce_im_dimensions(im_R[np.newaxis, ...])
    red_im_nR, coords = reduce_im_dimensions(im_nR[np.newaxis, ...])
    red_im_sub, coords = reduce_im_dimensions(im_sub[np.newaxis, ...])
    im_R_df = generate_reduced_image_df(red_im_R, coords)
    im_nR_df = generate_reduced_image_df(red_im_nR, coords)
    im_sub_df = generate_reduced_image_df(red_im_sub, coords)

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_grid_on_allen(im_R_df, outcome='dff0', palette=dprime_palette, result_path=None, dotsize=340, vmin=1.8, vmax=3, norm=None, fig=fig, ax= ax[0])
    plot_grid_on_allen(im_nR_df, outcome='dff0', palette=dprime_palette, result_path=None, dotsize=340, vmin=1.8, vmax=3, norm=None, fig=fig, ax= ax[1])
    plot_grid_on_allen(im_sub_df, outcome='dff0', palette=seismic_palette, result_path=None, dotsize=340, vmin=-0.1, vmax=0.1, norm=None, fig=fig, ax= ax[2])
    fig.savefig(os.path.join(save_path, 'red_im', f'{roi}_nsgimas.png'))

    im_R = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_r"), 'value'].values[0] - total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0]
    im_nR = total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_r"), 'value'].values[0] - total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0]
    im_sub = im_R - im_nR

    mask_r = np.where(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_R, np.nan)
    mask_non_r = np.where(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_nR, np.nan)
    mask_sub = np.where(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_sub, np.nan)

    red_im_R, coords = reduce_im_dimensions(mask_r[np.newaxis, ...])
    red_im_nR, coords = reduce_im_dimensions(mask_non_r[np.newaxis, ...])
    red_im_sub, coords = reduce_im_dimensions(mask_sub[np.newaxis, ...])
    im_R_df = generate_reduced_image_df(red_im_R, coords)
    im_nR_df = generate_reduced_image_df(red_im_nR, coords)
    im_sub_df = generate_reduced_image_df(red_im_sub, coords)

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_grid_on_allen(im_R_df, outcome='dff0', palette='viridis', result_path=None, dotsize=340, vmin=0.3, vmax=0.9, norm=None, fig=fig, ax= ax[0])
    plot_grid_on_allen(im_nR_df, outcome='dff0', palette='viridis', result_path=None, dotsize=340, vmin=0.3, vmax=0.9, norm=None, fig=fig, ax= ax[1])
    plot_grid_on_allen(im_sub_df, outcome='dff0', palette=seismic_palette, result_path=None, dotsize=340, vmin=-0.1, vmax=0.1, norm=None, fig=fig, ax= ax[2])
    fig.savefig(os.path.join(save_path, 'red_im', f'{roi}_significant_pairs.png'))


def plot_mouse_barplot_r_context(mouse_avg, output_path):

    if 'opto' in output_path:
        stim_list = ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']
    else:
        stim_list = ['(-5.0, 5.0)']

    for stim in stim_list:
        group = mouse_avg.loc[mouse_avg.opto_stim_coord==stim]
        if 'opto' in output_path:
            group['correct_trial']=1

        redim_df = []
        for i, row in group.iterrows():
            redim, coords = reduce_im_dimensions(row['value'][np.newaxis])
            df = generate_reduced_image_df(redim, coords)
            df['context'] = row.context
            df['mouse_id'] = row.mouse_id
            df['correct_trial'] = row.correct_trial
            df['variable'] = row.variable
            redim_df+=[df]
            
        redim_df = pd.concat(redim_df).rename(columns={'dff0': 'value'})

        redim_df['seed'] = redim_df.apply(lambda x: x.variable.split("_")[0], axis=1)
        redim_df['coord_order'] = redim_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
        redim_df = redim_df.groupby(by=['mouse_id', 'context', 'correct_trial', 'coord_order', 'seed']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r', 'coord_order': 'coord'})
        
        if 'opto' in output_path:
            save_path = os.path.join(output_path, f"{stim}_stim")
        else:
            save_path = os.path.join(output_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        all_rois_stats, all_rois_stats_correct_vs_incorrect, selected_rois_stats, selected_rois_stats_correct_vs_incorrect = compute_stats_barplot_context(redim_df, save_path)
        
        redim_df = redim_df.groupby(by=['mouse_id', 'correct_trial', 'coord', 'seed']).apply(lambda x: x.loc[x.context==1, 'r'].values[0] - x.loc[x.context==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})

        ## Plot corrected correlation r substracted R+ - R-  with correct vs incorrect trial
        if 'opto' not in output_path:
            g = sns.catplot(
                x="coord",
                y="r",
                hue="correct_trial",
                hue_order=[1,0],
                palette=['#032b22', '#da4e02'],
                row="seed",
                data=redim_df,
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=3,
                alpha=0.5,
                sharex=False)
            
            g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1,0], palette=['#032b22', '#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
            g.set_ylabels('R- <-- r-shuffle --> R+')
            g.tick_params(axis='x', rotation=30)
            for ax in g.axes.flat:
                if stim == '(-5.0, 5.0)':
                    ax.set_ylim([-0.15, 0.15])
                else:
                    ax.set_ylim([-0.3, 0.3])

                seed = ax.get_title('center').split("= ")[-1]
                stats = all_rois_stats_correct_vs_incorrect[all_rois_stats_correct_vs_incorrect.seed==seed]
                ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            g.figure.tight_layout()
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot.png'))
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot.svg'))

            g = sns.catplot(
                x="coord",
                y="r",
                hue="correct_trial",
                hue_order=[1,0],
                order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
                palette=['#032b22', '#da4e02'],
                col="seed",
                data=redim_df.loc[redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"])],
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=1,
                alpha=0.5)
            
            g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1,0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#032b22', '#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
            g.set_ylabels('R- <-- r-shuffle --> R+')
            g.tick_params(axis='x', rotation=30)
            for ax in g.axes.flat:
                if stim=='(-5.0, 5.0)':
                    ax.set_ylim([-0.15, 0.15])
                else:
                    ax.set_ylim([-0.3, 0.3])
                    
                seed = ax.get_title('center').split("= ")[-1]
                stats = selected_rois_stats_correct_vs_incorrect[selected_rois_stats_correct_vs_incorrect.seed==seed]
                ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            g.figure.tight_layout()
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_selected_rois.png'))
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_selected_rois.svg'))

        ## Plot corrected correlation r substracted R+ - R-  with correct trials
        g = sns.catplot(
            x="coord",
            y="r",
            palette=['#032b22'],
            row="seed",
            data=redim_df.loc[redim_df.correct_trial==1],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=3,
            alpha=0.5,
            sharex=False)
        
        g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1],  palette=['#032b22'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('R- <-- r-shuffle --> R+')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            ax.set_ylim([-0.15, 0.15])
            seed = ax.get_title('center').split("= ")[-1]
            stats = all_rois_stats.loc[(all_rois_stats.seed==seed) & (all_rois_stats.correct_trial==1)]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_correct.png'))
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_correct.svg'))

        g = sns.catplot(
            x="coord",
            y="r",
            palette=['#032b22'],
            col="seed",
            order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
            data=redim_df.loc[(redim_df.correct_trial==1) & (redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"]))],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=0.8,
            alpha=0.5)
        
        g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#032b22', '#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('R- <-- r-shuffle --> R+')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            ax.set_ylim([-0.15, 0.15])
            seed = ax.get_title('center').split("= ")[-1]
            stats = selected_rois_stats.loc[(selected_rois_stats.seed==seed) & (selected_rois_stats.correct_trial==1)]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()                
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_selected_rois_correct.png'))
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_selected_rois_correct.svg'))

        if 'opto' not in output_path:
            ## Plot corrected correlation r substracted R+ - R-  with incorrect trials
            g = sns.catplot(
                x="coord",
                y="r",
                palette=['#da4e02'],
                row="seed",
                data=redim_df.loc[redim_df.correct_trial==0],
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=3,
                alpha=0.5,
                sharex=False)
            
            g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[0], palette=['#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
            g.set_ylabels('R- <-- r-shuffle --> R+')
            g.tick_params(axis='x', rotation=30)
            for ax in g.axes.flat:
                ax.set_ylim([-0.15, 0.15])
                seed = ax.get_title('center').split("= ")[-1]
                stats = all_rois_stats.loc[(all_rois_stats.seed==seed) & (all_rois_stats.correct_trial==0)]
                ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            g.figure.tight_layout()
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_incorrect.png'))
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_incorrect.svg'))

            g = sns.catplot(
                x="coord",
                y="r",
                palette=['#da4e02'],
                col="seed",
                order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
                data=redim_df.loc[(redim_df.correct_trial==0) & (redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"]))],
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=0.8,
                alpha=0.5)
            
            g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
            g.set_ylabels('R- <-- r-shuffle --> R+')
            g.tick_params(axis='x', rotation=30)
            for ax in g.axes.flat:
                ax.set_ylim([-0.15, 0.15])
                seed = ax.get_title('center').split("= ")[-1]
                stats = selected_rois_stats.loc[(selected_rois_stats.seed==seed) & (selected_rois_stats.correct_trial==0)]
                ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            g.figure.tight_layout()
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_selected_rois_incorrect.png'))
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_R+-R-_barplot_selected_rois_incorrect.svg'))


def plot_connected_dot_r_context(total_avg, output_path):
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)
    viridis_palette = cm.get_cmap('viridis')

    if 'opto' in output_path:
        stim_list = ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']
    else:
        stim_list = ['(-5.0, 5.0)']

    total_df = []
    for stim in stim_list:
        group = total_avg.loc[total_avg.opto_stim_coord==stim].reset_index(drop=True)
        group['seed'] = group.apply(lambda x: x.variable.split("_")[0], axis=1)

        if 'opto' in output_path:
            group['correct_trial']=1
            group['masked_data'] = group.value
        else:
            group['masked_data'] = group.groupby(by=['opto_stim_coord', 'context', 'correct_trial', 'seed']).apply(
                lambda x: x.apply(
                    lambda y: np.where(x.loc[x.variable.str.contains('sigmas'), 'value'].values[0]>=1.8, y.value, np.nan), axis=1)).reset_index()[0]

        for i, row in group.iterrows():
            # redim, coords = reduce_im_dimensions(row['value'][np.newaxis])
            redim, coords = reduce_im_dimensions(row['masked_data'][np.newaxis])
            df = generate_reduced_image_df(redim, coords)
            df['context'] = row.context
            df['seed'] = row.seed
            # df['mouse_id'] = row.mouse_id
            df['correct_trial'] = row.correct_trial
            df['variable'] = row.variable
            df['opto_stim_coord'] = stim
            total_df+=[df]
            
    total_df = pd.concat(total_df).rename(columns={'dff0': 'value'})
    total_df['coord'] = total_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
    total_df = total_df[total_df.coord.isin(total_df.seed.unique())]
    total_df['y_dest'] = total_df.apply(lambda x: eval(x.coord)[0], axis=1)
    total_df['x_dest'] = total_df.apply(lambda x: eval(x.coord)[1], axis=1)
    total_df['y_source'] = total_df.apply(lambda x: eval(x.seed)[0], axis=1)
    total_df['x_source'] = total_df.apply(lambda x: eval(x.seed)[1], axis=1)

    r_df = total_df.groupby(by=['context', 'opto_stim_coord', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r'})
    sigma_df = total_df.groupby(by=['context', 'opto_stim_coord', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("sigma"), 'value'].values[0])).reset_index().rename(columns={0:'sigma'})
    delta_r_df = r_df.groupby(by=['opto_stim_coord', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: x.loc[x.context==1, 'r'].values[0] - x.loc[x.context==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})

    for coord in total_df['opto_stim_coord'].unique():
        stats = pd.read_csv(os.path.join(output_path, [f"{coord}_stim" if 'opto' in output_path else ''][0], 'pairwise_selected_rois_stats.csv'))
        stats['norm_d'] = np.round(np.clip((stats.d_prime.values - 0.8)/(2 - 0.8), 0, 1), 2)

        for outcome in total_df.correct_trial.unique():
            r = r_df[(r_df.correct_trial==outcome) & (r_df.opto_stim_coord==coord)]
            r = r[r.seed != coord]
            r['norm_r'] = np.round(np.clip((r.r.values- 0.3)/(0.6 - 0.3), 0, 1), 2)


            sigma = sigma_df[(sigma_df.correct_trial==outcome) & (sigma_df.opto_stim_coord==coord)]
            sigma = sigma[sigma.seed != coord]

            delta = delta_r_df[(delta_r_df.correct_trial==outcome) & (delta_r_df.opto_stim_coord==coord)]
            delta = delta[delta.seed != coord]
            delta['norm_r'] = np.round(np.clip((delta.r.values- -0.1)/(0.1 - -0.1), 0, 1), 2)

            for c in total_df.context.unique():
                fig, ax = plt.subplots(figsize=(4,4))
                fig.suptitle(f"{coord} stim r between rois")
                im=ax.scatter(r.loc[r.coord==r.seed, 'x_source'], r.loc[r.coord==r.seed, 'y_source'], s=100, c='k')
                if coord != '(-5.0, 5.0)':
                    ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=100)

                ax.scatter(0, 0, marker='+', c='gray', s=100)
                if 'opto' in output_path:
                    iterator = zip(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(2.5, 2.5)", "(1.5, 3.5)"], 
                                      ["(1.5, 1.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(1.5, 3.5)", "(0.5, 4.5)", "(0.5, 4.5)"])
                else:
                    iterator = combinations(r.seed.unique(), 2)

                for seed, dest in iterator:
                    if seed == dest:
                        continue
                    
                    sub_r = r[(r.context==c) & (r.seed.isin([seed, dest])) & (r.coord.isin([seed, dest]))]
                    sub_r = sub_r[sub_r.seed != sub_r.coord]
                    sub_sigma = sigma[(sigma.context==c) & (sigma.seed.isin([seed, dest])) & (sigma.coord.isin([seed, dest]))]
                    sub_sigma = sub_sigma[sub_sigma.seed != sub_sigma.coord]
                    if 'opto' in output_path:
                        ax.plot([sub_r.x_source.unique(), sub_r.x_dest.unique()], [sub_r.y_source.unique(), sub_r.y_dest.unique()], c=viridis_palette(sub_r.norm_r.mean()), linewidth=4)       
                    else:
                        if sub_sigma.sigma.mean()>=1.8:
                            ax.plot([sub_r.x_source.unique(), sub_r.x_dest.unique()], [sub_r.y_source.unique(), sub_r.y_dest.unique()], c=viridis_palette(sub_r.norm_r.mean()), linewidth=4)       

                ax.grid(True)
                ax.set_xticks(np.linspace(0.5,5.5,6))
                ax.set_xlim([-0.25, 6])
                ax.set_yticks(np.linspace(-3.5, 2.5,7))
                ax.set_ylim([-3.75, 2.75])
                ax.invert_xaxis()

                if 'opto' in output_path:
                    save_path = os.path.join(output_path, f"{coord}_stim")
                else:
                    save_path = os.path.join(output_path, f'{"correct" if outcome==1 else "incorrect"}')

                fig.savefig(os.path.join(save_path, f'r_summary_{["rewarded" if c else "non-rewarded"][0]}.png'))
                fig.savefig(os.path.join(save_path, f'r_summary_{["rewarded" if c else "non-rewarded"][0]}.svg'))

            fig, ax = plt.subplots(figsize=(4,4))
            fig.suptitle(f"{coord} stim r between rois")
            im=ax.scatter(delta.loc[delta.coord==delta.seed, 'x_source'], delta.loc[delta.coord==delta.seed, 'y_source'], s=100, c='k')
            if coord != '(-5.0, 5.0)':
                ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=100)

            ax.scatter(0, 0, marker='+', c='gray', s=100)
            if 'opto' in output_path:
                iterator = zip(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(2.5, 2.5)", "(1.5, 3.5)"], 
                                    ["(1.5, 1.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(1.5, 3.5)", "(0.5, 4.5)", "(0.5, 4.5)"])
            else:
                iterator = combinations(r.seed.unique(), 2)
                
            for seed, dest in iterator:
                if seed == dest:
                    continue
                
                sub_delta = delta[(delta.seed.isin([seed, dest])) & (delta.coord.isin([seed, dest]))]
                sub_delta = sub_delta[sub_delta.seed != sub_delta.coord]
                sub_sigma = sigma[(sigma.seed.isin([seed, dest])) & (sigma.coord.isin([seed, dest]))]
                sub_sigma = sub_sigma[sub_sigma.seed != sub_sigma.coord]

                if 'opto' in output_path:
                    d = stats.loc[(stats.correct_trial==outcome) & (stats.seed==seed) & (stats.coord==dest), 'd_prime'].values[0]
                    norm_d = stats.loc[(stats.correct_trial==outcome) & (stats.seed==seed) & (stats.coord==dest), 'norm_d'].values[0]
                    ax.plot([sub_delta.x_source.unique(), sub_delta.x_dest.unique()], [sub_delta.y_source.unique(), sub_delta.y_dest.unique()], c=seismic_palette(sub_delta.norm_r.mean()), linewidth=d, alpha=norm_d)       
                else:
                    if sub_sigma.sigma.mean()>=1.8:
                        d = stats.loc[(stats.correct_trial==outcome) & (stats.seed==seed) & (stats.coord==dest), 'd_prime'].values[0]
                        norm_d = stats.loc[(stats.correct_trial==outcome) & (stats.seed==seed) & (stats.coord==dest), 'norm_d'].values[0]
                        ax.plot([sub_delta.x_source.unique(), sub_delta.x_dest.unique()], [sub_delta.y_source.unique(), sub_delta.y_dest.unique()], c=seismic_palette(sub_delta.norm_r.mean()), linewidth=d, alpha=norm_d)       

            ax.grid(True)
            ax.set_xticks(np.linspace(0.5,5.5,6))
            ax.set_xlim([-0.25, 6])
            ax.set_yticks(np.linspace(-3.5, 2.5,7))
            ax.set_ylim([-3.75, 2.75])
            ax.invert_xaxis()

            if 'opto' in output_path:
                save_path = os.path.join(output_path, f"{coord}_stim")
            else:
                save_path = os.path.join(output_path, f'{"correct" if outcome==1 else "incorrect"}')

            fig.savefig(os.path.join(save_path, 'r_summary_delta.png'))
            fig.savefig(os.path.join(save_path, 'r_summary_delta.svg'))


def compute_stats_barplot_context(df, output_path):
    df=df[df.coord!='(2.5, 5.5)']
    all_rois_stats =[]
    for name, group in df.groupby(by=['correct_trial', 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group.context==1, 'r'].to_numpy(), group.loc[group.context==0, 'r'].to_numpy())
        mean_diff = (group.loc[group.context==1, 'r'].mean() - group.loc[group.context==0, 'r'].mean())
        std_diff = np.std(group.loc[group.context==1, 'r'].to_numpy() - group.loc[group.context==0, 'r'].to_numpy())

        results = {
         'correct_trial': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean_rew': group.loc[group.context==1, 'r'].mean(),
         'std_rew': group.loc[group.context==1, 'r'].std(),
         'mean_no_rew': group.loc[group.context==0, 'r'].mean(),
         'std_no_rew': group.loc[group.context==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.coord.unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.coord.unique().shape[0],
         'significant': p<(0.05/df.coord.unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
         }
        
        all_rois_stats += [results]
    all_rois_stats = pd.DataFrame(all_rois_stats)
    all_rois_stats.to_csv(os.path.join(output_path, 'pairwise_all_rois_stats.csv'))

    if 'opto' not in output_path:
        all_rois_stats_correct_vs_incorrect =[]
        for name, group in df.groupby(by=['seed', 'coord']):
            context_diff = group.groupby(by=['mouse_id', 'correct_trial']).apply(lambda x: x.loc[x.context==1, 'r'].to_numpy() - x.loc[x.context==0, 'r'].to_numpy()).reset_index().rename(columns={0:'r'})
            t, p = ttest_rel(context_diff.loc[context_diff.correct_trial==1, 'r'], context_diff.loc[context_diff.correct_trial==0, 'r'])
            mean_diff = (context_diff.loc[context_diff.correct_trial==1, 'r'].mean() - context_diff.loc[context_diff.correct_trial==0, 'r'].mean())
            std_diff = np.std(context_diff.loc[context_diff.correct_trial==1, 'r'].to_numpy() - context_diff.loc[context_diff.correct_trial==0, 'r'].to_numpy())

            results = {
            'seed': name[0],
            'coord': name[1],
            'dof': context_diff.mouse_id.unique().shape[0]-1,
            'mean_correct': context_diff.loc[context_diff.correct_trial==1, 'r'].mean(),
            'std_correct': context_diff.loc[context_diff.correct_trial==1, 'r'].std(),
            'mean_incorrect': context_diff.loc[context_diff.correct_trial==0, 'r'].mean(),
            'std_incorrect': context_diff.loc[context_diff.correct_trial==0, 'r'].std(),
            't': t,
            'p': p,
            'p_corr': p*df.coord.unique().shape[0],
            'alpha': 0.05,
            'alpha_corr': 0.05/df.coord.unique().shape[0],
            'significant': p<(0.05/df.coord.unique().shape[0]),
            'd_prime': abs(mean_diff/std_diff)
            }
            all_rois_stats_correct_vs_incorrect += [results]
        all_rois_stats_correct_vs_incorrect = pd.DataFrame(all_rois_stats_correct_vs_incorrect)
        all_rois_stats_correct_vs_incorrect.to_csv(os.path.join(output_path, 'pairwise_all_rois_stats_correct_vs_incorrect.csv'))
    else:
        all_rois_stats_correct_vs_incorrect=[]

    # df = df[df.seed!='(1.5, 3.5)']
    selected_rois_stats =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['correct_trial', 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group.context==1, 'r'].to_numpy(), group.loc[group.context==0, 'r'].to_numpy())
        mean_diff = (group.loc[group.context==1, 'r'].mean() - group.loc[group.context==0, 'r'].mean())
        std_diff = np.std(group.loc[group.context==1, 'r'].to_numpy() - group.loc[group.context==0, 'r'].to_numpy())

        results = {
         'correct_trial': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean_rew': group.loc[group.context==1, 'r'].mean(),
         'std_rew': group.loc[group.context==1, 'r'].std(),
         'mean_no_rew': group.loc[group.context==0, 'r'].mean(),
         'std_no_rew': group.loc[group.context==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
         'd_prime': abs(mean_diff/std_diff)

         }
        
        selected_rois_stats += [results]
    selected_rois_stats = pd.DataFrame(selected_rois_stats)
    selected_rois_stats.to_csv(os.path.join(output_path, 'pairwise_selected_rois_stats.csv'))

    if 'opto' not in output_path:
        selected_rois_stats_correct_vs_incorrect =[]
        for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['seed', 'coord']):
            context_diff = group.groupby(by=['mouse_id', 'correct_trial']).apply(lambda x: x.loc[x.context==1, 'r'].to_numpy() - x.loc[x.context==0, 'r'].to_numpy()).reset_index().rename(columns={0:'r'})
            t, p = ttest_rel(context_diff.loc[context_diff.correct_trial==1, 'r'], context_diff.loc[context_diff.correct_trial==0, 'r'])
            mean_diff = (context_diff.loc[context_diff.correct_trial==1, 'r'].mean() - context_diff.loc[context_diff.correct_trial==0, 'r'].mean())
            std_diff = np.std(context_diff.loc[context_diff.correct_trial==1, 'r'].to_numpy() - context_diff.loc[context_diff.correct_trial==0, 'r'].to_numpy())
            results = {
            'seed': name[0],
            'coord': name[1],
            'dof': context_diff.mouse_id.unique().shape[0]-1,
            'mean_correct': context_diff.loc[context_diff.correct_trial==1, 'r'].mean(),
            'std_correct': context_diff.loc[context_diff.correct_trial==1, 'r'].std(),
            'mean_incorrect': context_diff.loc[context_diff.correct_trial==0, 'r'].mean(),
            'std_incorrect': context_diff.loc[context_diff.correct_trial==0, 'r'].std(),
            't': t,
            'p': p,
            'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
            'alpha': 0.05,
            'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
            'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
            'd_prime': abs(mean_diff/std_diff)
            }
            selected_rois_stats_correct_vs_incorrect += [results]
        selected_rois_stats_correct_vs_incorrect = pd.DataFrame(selected_rois_stats_correct_vs_incorrect)
        selected_rois_stats_correct_vs_incorrect.to_csv(os.path.join(output_path, 'pairwise_selected_rois_stats_correct_vs_incorrect.csv'))
    else:
        selected_rois_stats_correct_vs_incorrect=[]
    return all_rois_stats, all_rois_stats_correct_vs_incorrect, selected_rois_stats, selected_rois_stats_correct_vs_incorrect


# def plot_mouse_barplot_r_from_images_2(mouse_avg, grouping, variable, palette, output_path, file_name):

#     if 'opto' in output_path:
#         stim_list = ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']
#     else:
#         stim_list = ['(-5.0, 5.0)']

#     for stim in stim_list:
#         group = mouse_avg.loc[mouse_avg.opto_stim_coord==stim]
#         if 'opto' in output_path:
#             group['correct_trial']=1

#         redim_df = []
#         for i, row in group.iterrows():
#             redim, coords = reduce_im_dimensions(row['value'][np.newaxis])
#             df = generate_reduced_image_df(redim, coords)
#             df['context'] = row.context
#             df['mouse_id'] = row.mouse_id
#             df['correct_trial'] = row.correct_trial
#             df['variable'] = row.variable
#             redim_df+=[df]
            
#         redim_df = pd.concat(redim_df).rename(columns={'dff0': 'value'})

#         redim_df['seed'] = redim_df.apply(lambda x: x.variable.split("_")[0], axis=1)
#         redim_df['coord_order'] = redim_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
#         redim_df = redim_df.groupby(by=['mouse_id', 'context', 'correct_trial', 'coord_order', 'seed']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r', 'coord_order': 'coord'})
        
#         if 'opto' in output_path:
#             save_path = os.path.join(output_path, 'from_images', f"{stim}_stim")
#         else:
#             save_path = os.path.join(output_path, 'from_images')
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         all_rois_stats, selected_rois_stats = compute_stats_barplot_2(redim_df, grouping, variable, save_path, file_name)
        
#         redim_df = redim_df.groupby(by=['mouse_id', grouping, 'coord', 'seed']).apply(lambda x: x.loc[x[f'{variable}']==1, 'r'].values[0] - x.loc[x[f'{variable}']==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})

#         ## Plot corrected correlation r substracted R+ - R-  with correct vs incorrect trial
#         g = sns.catplot(
#             x="coord",
#             y="r",
#             hue=grouping,
#             hue_order=[1,0],
#             palette=palette,
#             row="seed",
#             data=redim_df,
#             kind="bar",
#             errorbar = ('ci', 95),
#             edgecolor="black",
#             errcolor="black",
#             errwidth=1.5,
#             capsize = 0.1,
#             height=4,
#             aspect=3,
#             alpha=0.5,
#             sharex=False)
        
#         g.map(sns.stripplot, 'coord', 'r', grouping, hue_order=[1,0], palette=palette, dodge=True, alpha=0.6, ec='k', linewidth=1)
#         g.set_ylabels('R- <-- r-shuffle --> R+')
#         g.tick_params(axis='x', rotation=30)
#         for ax in g.axes.flat:
#             if stim == '(-5.0, 5.0)':
#                 ax.set_ylim([-0.15, 0.15])
#             else:
#                 ax.set_ylim([-0.3, 0.3])

#             seed = ax.get_title('center').split("= ")[-1]
#             stats = all_rois_stats[all_rois_stats.seed==seed]
#             ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
#             for label in ax.get_xticklabels():
#                 label.set_horizontalalignment('right')
#         g.figure.tight_layout()
#         g.figure.savefig(os.path.join(save_path, f'{file_name}.png'))
#         g.figure.savefig(os.path.join(save_path, f'{file_name}.svg'))

#         g = sns.catplot(
#             x="coord",
#             y="r",
#             hue=grouping,
#             hue_order=[1,0],
#             order=["(-0.5, 0.5)", "(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"],
#             palette=palette,
#             col="seed",
#             data=redim_df.loc[redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"])],
#             kind="bar",
#             errorbar = ('ci', 95),
#             edgecolor="black",
#             errcolor="black",
#             errwidth=1.5,
#             capsize = 0.1,
#             height=4,
#             aspect=1,
#             alpha=0.5)
        
#         g.map(sns.stripplot, 'coord', 'r', grouping, hue_order=[1,0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"], palette=palette, dodge=True, alpha=0.6, ec='k', linewidth=1)
#         g.set_ylabels('R- <-- r-shuffle --> R+')
#         g.tick_params(axis='x', rotation=30)
#         for ax in g.axes.flat:
#             if stim=='(-5.0, 5.0)':
#                 ax.set_ylim([-0.15, 0.15])
#             else:
#                 ax.set_ylim([-0.3, 0.3])
                
#             seed = ax.get_title('center').split("= ")[-1]
#             stats = selected_rois_stats[selected_rois_stats.seed==seed]
#             ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
#             for label in ax.get_xticklabels():
#                 label.set_horizontalalignment('right')
#         g.figure.tight_layout()
#         g.figure.savefig(os.path.join(save_path, f'{file_name}_selected_rois.png'))
#         g.figure.savefig(os.path.join(save_path, f'{file_name}_selected_rois.svg'))


# def compute_stats_barplot_2(df, grouping, variable, output_path, file_name):
    df=df[df.coord!='(2.5, 5.5)']

    all_rois_stats =[]
    for name, group in df.groupby(by=[grouping, 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group[variable]==1, 'r'].values, group.loc[group[variable]==0, 'r'].values)
        mean_diff = (group.loc[group[variable]==1, 'r'].values[0] - group.loc[group[variable]==0, 'r'].values[0])/group.loc[group[variable]==1, 'r'].shape[0]
        std_diff = np.std(mean_diff)

        results = {
         f'{grouping}': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean_1': group.loc[group[variable]==1, 'r'].mean(),
         'std_1': group.loc[group[variable]==1, 'r'].std(),
         'mean_2': group.loc[group[variable]==0, 'r'].mean(),
         'std_2': group.loc[group[variable]==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.coord.unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.coord.unique().shape[0],
         'significant': p<(0.05/df.coord.unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
         }
        
        all_rois_stats += [results]
    all_rois_stats = pd.DataFrame(all_rois_stats)
    all_rois_stats.to_csv(os.path.join(output_path, f'{file_name}.csv'))

    df = df[df.seed!='(1.5, 3.5)']
    selected_rois_stats =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=[grouping, 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group[variable]==1, 'r'].values, group.loc[group[variable]==0, 'r'])
        mean_diff = (group.loc[group[variable]==1, 'r'].values[0] - group.loc[group[variable]==0, 'r'].values[0])/group.loc[group[variable]==1, 'r'].shape[0]
        std_diff = np.std(mean_diff)

        results = {
         f'{grouping}': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean_1': group.loc[group[variable]==1, 'r'].mean(),
         'std_1': group.loc[group[variable]==1, 'r'].std(),
         'mean_2': group.loc[group[variable]==0, 'r'].mean(),
         'std_2': group.loc[group[variable]==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
         'd_prime': abs(mean_diff/std_diff)

         }
        
        selected_rois_stats += [results]
    selected_rois_stats = pd.DataFrame(selected_rois_stats)
    selected_rois_stats.to_csv(os.path.join(output_path, f'{file_name}_selected_rois_stats.csv'))

    return all_rois_stats, selected_rois_stats
def plot_mouse_barplot_r_choice(mouse_avg, output_path):

    if 'opto' in output_path:
        stim_list = ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']
    else:
        stim_list = ['(-5.0, 5.0)']

    for stim in stim_list:
        group = mouse_avg.loc[mouse_avg.opto_stim_coord==stim]
        if 'opto' in output_path:
            group['correct_trial']= group.apply(lambda x: 0 if x.opto_stim_coord=='(-5.0, 5.0)' else 1, axis=1)

        redim_df = []
        for i, row in group.iterrows():
            redim, coords = reduce_im_dimensions(row['value'][np.newaxis])
            df = generate_reduced_image_df(redim, coords)
            df['context'] = row.context
            df['mouse_id'] = row.mouse_id
            df['correct_trial'] = row.correct_trial
            df['variable'] = row.variable
            redim_df+=[df]
            
        redim_df = pd.concat(redim_df).rename(columns={'dff0': 'value'})

        redim_df['seed'] = redim_df.apply(lambda x: x.variable.split("_")[0], axis=1)
        redim_df['coord_order'] = redim_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
        redim_df = redim_df.groupby(by=['mouse_id', 'context', 'correct_trial', 'coord_order', 'seed']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r', 'coord_order': 'coord'})
        
        if 'opto' in output_path:
            save_path = os.path.join(output_path, f"{stim}_stim")
        else:
            save_path = os.path.join(output_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        all_rois_stats, all_rois_stats_rew_vs_norew, selected_rois_stats, selected_rois_stats_rew_vs_norew = compute_stats_barplot_choice(redim_df, save_path)
        
        redim_df = redim_df.groupby(by=['mouse_id', 'context', 'coord', 'seed']).apply(lambda x: x.loc[x.correct_trial==1, 'r'].values[0] - x.loc[x.correct_trial==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})

        ## Plot corrected correlation r substracted R+ - R-  with correct vs incorrect trial
        g = sns.catplot(
            x="coord",
            y="r",
            hue="context",
            hue_order=[1,0],
            palette=['#348A18', '#6E188A'],
            row="seed",
            data=redim_df,
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=3,
            alpha=0.5,
            sharex=False)
            
        g.map(sns.stripplot, 'coord', 'r', 'context', hue_order=[1,0], palette=['#348A18', '#6E188A'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('incorrect <-- --> correct')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            if stim == '(-5.0, 5.0)':
                ax.set_ylim([-0.15, 0.15])
            else:
                ax.set_ylim([-0.3, 0.3])

            seed = ax.get_title('center').split("= ")[-1]
            stats = all_rois_stats_rew_vs_norew[all_rois_stats_rew_vs_norew.seed==seed]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot.png'))
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot.svg'))

        g = sns.catplot(
            x="coord",
            y="r",
            hue="context",
            hue_order=[1,0],
            order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
            palette=['#348A18', '#6E188A'],
            col="seed",
            data=redim_df.loc[redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"])],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=1,
            alpha=0.5)
        
        g.map(sns.stripplot, 'coord', 'r', 'context', hue_order=[1,0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#348A18', '#6E188A'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('incorrect <-- --> correct')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            if stim=='(-5.0, 5.0)':
                ax.set_ylim([-0.15, 0.15])
            else:
                ax.set_ylim([-0.3, 0.3])
                
            seed = ax.get_title('center').split("= ")[-1]
            stats = selected_rois_stats_rew_vs_norew[selected_rois_stats_rew_vs_norew.seed==seed]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_selected_rois.png'))
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_selected_rois.svg'))

        ## Plot corrected correlation r substracted R+ - R-  with correct trials
        g = sns.catplot(
            x="coord",
            y="r",
            palette=['#348A18'],
            row="seed",
            data=redim_df.loc[redim_df.context==1],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=3,
            alpha=0.5,
            sharex=False)
        
        g.map(sns.stripplot, 'coord', 'r', 'context', hue_order=[1],  palette=['#348A18'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('incorrect <-- --> correct')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            ax.set_ylim([-0.15, 0.15])
            seed = ax.get_title('center').split("= ")[-1]
            stats = all_rois_stats.loc[(all_rois_stats.seed==seed) & (all_rois_stats.context==1)]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_rewarded.png'))
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_rewarded.svg'))

        g = sns.catplot(
            x="coord",
            y="r",
            palette=['#348A18'],
            col="seed",
            order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
            data=redim_df.loc[(redim_df.context==1) & (redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"]))],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=0.8,
            alpha=0.5)
        
        g.map(sns.stripplot, 'coord', 'r', 'context', hue_order=[1], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#348A18'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('incorrect <-- --> correct')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            ax.set_ylim([-0.15, 0.15])
            seed = ax.get_title('center').split("= ")[-1]
            stats = selected_rois_stats.loc[(selected_rois_stats.seed==seed) & (selected_rois_stats.context==1)]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()                
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_selected_rois_rewarded.png'))
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_selected_rois_rewarded.svg'))

        if 'opto' not in output_path:
            ## Plot corrected correlation r substracted R+ - R-  with incorrect trials
            g = sns.catplot(
                x="coord",
                y="r",
                palette=['#6E188A'],
                row="seed",
                data=redim_df.loc[redim_df.context==0],
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=3,
                alpha=0.5,
                sharex=False)
            
            g.map(sns.stripplot, 'coord', 'r', 'context', hue_order=[0], palette=['#6E188A'], dodge=True, alpha=0.6, ec='k', linewidth=1)
            g.set_ylabels('incorrect <-- --> correct')
            g.tick_params(axis='x', rotation=30)
            for ax in g.axes.flat:
                ax.set_ylim([-0.15, 0.15])
                seed = ax.get_title('center').split("= ")[-1]
                stats = all_rois_stats.loc[(all_rois_stats.seed==seed) & (all_rois_stats.context==0)]
                ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            g.figure.tight_layout()
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_nonrewarded.png'))
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_nonrewarded.svg'))

            g = sns.catplot(
                x="coord",
                y="r",
                palette=['#6E188A'],
                col="seed",
                order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
                data=redim_df.loc[(redim_df.context==0) & (redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"]))],
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=1,
                alpha=0.5)
            
            g.map(sns.stripplot, 'coord', 'r', 'context', hue_order=[0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#6E188A'], dodge=True, alpha=0.6, ec='k', linewidth=1)
            g.set_ylabels('incorrect <-- --> correct')
            g.tick_params(axis='x', rotation=30)
            for ax in g.axes.flat:
                ax.set_ylim([-0.15, 0.15])
                seed = ax.get_title('center').split("= ")[-1]
                stats = selected_rois_stats.loc[(selected_rois_stats.seed==seed) & (selected_rois_stats.context==0)]
                ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            g.figure.tight_layout()
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_selected_rois_nonrewarded.png'))
            g.figure.savefig(os.path.join(save_path, 'r-shuffle_choice_barplot_selected_rois_nonrewarded.svg'))


def compute_stats_barplot_choice(df, output_path):
    df=df[df.coord!='(2.5, 5.5)']
    all_rois_stats =[]
    for name, group in df.groupby(by=['context', 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group.correct_trial==1, 'r'].to_numpy(), group.loc[group.correct_trial==0, 'r'].to_numpy())
        mean_diff = (group.loc[group.correct_trial==1, 'r'].mean() - group.loc[group.correct_trial==0, 'r'].mean())
        std_diff = np.std(group.loc[group.correct_trial==1, 'r'].to_numpy() - group.loc[group.correct_trial==0, 'r'].to_numpy())

        results = {
         'context': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean_rew': group.loc[group.correct_trial==1, 'r'].mean(),
         'std_rew': group.loc[group.correct_trial==1, 'r'].std(),
         'mean_no_rew': group.loc[group.correct_trial==0, 'r'].mean(),
         'std_no_rew': group.loc[group.correct_trial==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.coord.unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.coord.unique().shape[0],
         'significant': p<(0.05/df.coord.unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
         }
        
        all_rois_stats += [results]
    all_rois_stats = pd.DataFrame(all_rois_stats)
    all_rois_stats.to_csv(os.path.join(output_path, 'choice_pairwise_all_rois_stats.csv'))

    if 'opto' not in output_path:
        all_rois_stats_rew_vs_norew =[]
        for name, group in df.groupby(by=['seed', 'coord']):
            choice_diff = group.groupby(by=['mouse_id', 'context']).apply(lambda x: x.loc[x.correct_trial==1, 'r'].to_numpy() - x.loc[x.correct_trial==0, 'r'].to_numpy()).reset_index().rename(columns={0:'r'})
            t, p = ttest_rel(choice_diff.loc[choice_diff.context==1, 'r'].to_numpy(), choice_diff.loc[choice_diff.context==0, 'r'].to_numpy())
            mean_diff = choice_diff.loc[choice_diff.context==1, 'r'].mean() - choice_diff.loc[choice_diff.context==0, 'r'].mean()
            std_diff = np.std(choice_diff.loc[choice_diff.context==1, 'r'].to_numpy() - choice_diff.loc[choice_diff.context==0, 'r'].to_numpy())

            results = {
            'seed': name[0],
            'coord': name[1],
            'dof': choice_diff.mouse_id.unique().shape[0]-1,
            'mean_correct': choice_diff.loc[choice_diff.context==1, 'r'].mean(),
            'std_correct': choice_diff.loc[choice_diff.context==1, 'r'].std(),
            'mean_incorrect': choice_diff.loc[choice_diff.context==0, 'r'].mean(),
            'std_incorrect': choice_diff.loc[choice_diff.context==0, 'r'].std(),
            't': t,
            'p': p,
            'p_corr': p*df.coord.unique().shape[0],
            'alpha': 0.05,
            'alpha_corr': 0.05/df.coord.unique().shape[0],
            'significant': p<(0.05/df.coord.unique().shape[0]),
            'd_prime': abs(mean_diff/std_diff)
            }
            all_rois_stats_rew_vs_norew += [results]
        all_rois_stats_rew_vs_norew = pd.DataFrame(all_rois_stats_rew_vs_norew)
        all_rois_stats_rew_vs_norew.to_csv(os.path.join(output_path, 'choice_pairwise_all_rois_stats_rew_vs_norew.csv'))
    else:
        all_rois_stats_rew_vs_norew=[]

    # df = df[df.seed!='(1.5, 3.5)']
    selected_rois_stats =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['context', 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group.correct_trial==1, 'r'].to_numpy(), group.loc[group.correct_trial==0, 'r'].to_numpy())
        mean_diff = group.loc[group.correct_trial==1, 'r'].mean() - group.loc[group.correct_trial==0, 'r'].mean()
        std_diff = np.std(group.loc[group.correct_trial==1, 'r'].to_numpy() - group.loc[group.correct_trial==0, 'r'].to_numpy())

        results = {
         'context': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean_rew': group.loc[group.correct_trial==1, 'r'].mean(),
         'std_rew': group.loc[group.correct_trial==1, 'r'].std(),
         'mean_no_rew': group.loc[group.correct_trial==0, 'r'].mean(),
         'std_no_rew': group.loc[group.correct_trial==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
         'd_prime': abs(mean_diff/std_diff)

         }
        
        selected_rois_stats += [results]
    selected_rois_stats = pd.DataFrame(selected_rois_stats)
    selected_rois_stats.to_csv(os.path.join(output_path, 'choice_pairwise_selected_rois_stats.csv'))

    if 'opto' not in output_path:
        selected_rois_stats_rew_vs_norew =[]
        for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['seed', 'coord']):
            choice_diff = group.groupby(by=['mouse_id', 'context']).apply(lambda x: x.loc[x.correct_trial==1, 'r'].to_numpy() - x.loc[x.correct_trial==0, 'r'].to_numpy()).reset_index().rename(columns={0:'r'})
            t, p = ttest_rel(choice_diff.loc[choice_diff.context==1, 'r'].to_numpy(), choice_diff.loc[choice_diff.context==0, 'r'].to_numpy())
            mean_diff = choice_diff.loc[choice_diff.context==1, 'r'].mean() - choice_diff.loc[choice_diff.context==0, 'r'].mean()
            std_diff = np.std(choice_diff.loc[choice_diff.context==1, 'r'].to_numpy() - choice_diff.loc[choice_diff.context==0, 'r'].to_numpy())
            results = {
            'seed': name[0],
            'coord': name[1],
            'dof': choice_diff.mouse_id.unique().shape[0]-1,
            'mean_correct': choice_diff.loc[choice_diff.context==1, 'r'].mean(),
            'std_correct': choice_diff.loc[choice_diff.context==1, 'r'].std(),
            'mean_incorrect': choice_diff.loc[choice_diff.context==0, 'r'].mean(),
            'std_incorrect': choice_diff.loc[choice_diff.context==0, 'r'].std(),
            't': t,
            'p': p,
            'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
            'alpha': 0.05,
            'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
            'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
            'd_prime': abs(mean_diff/std_diff)
            }
            selected_rois_stats_rew_vs_norew += [results]
        selected_rois_stats_rew_vs_norew = pd.DataFrame(selected_rois_stats_rew_vs_norew)
        selected_rois_stats_rew_vs_norew.to_csv(os.path.join(output_path, 'choice_pairwise_selected_rois_stats_rew_vs_norew.csv'))
    else:
        selected_rois_stats_rew_vs_norew=[]
    return all_rois_stats, all_rois_stats_rew_vs_norew, selected_rois_stats, selected_rois_stats_rew_vs_norew


def plot_connected_dot_r_choice(total_avg, output_path):
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)
    viridis_palette = cm.get_cmap('viridis')

    if 'opto' in output_path:
        stim_list = ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']
    else:
        stim_list = ['(-5.0, 5.0)']

    total_df = []
    for stim in stim_list:
        group = total_avg.loc[total_avg.opto_stim_coord==stim].reset_index(drop=True)

        group['seed'] = group.apply(lambda x: x.variable.split("_")[0], axis=1)
        group['masked_data'] = group.groupby(by=['opto_stim_coord', 'context', 'correct_trial', 'seed']).apply(
            lambda x: x.apply(
                lambda y: np.where(x.loc[x.variable.str.contains('sigmas'), 'value'].values[0]>=1.8, y.value, np.nan), axis=1)).reset_index()[0]

        for i, row in group.iterrows():
            # redim, coords = reduce_im_dimensions(row['value'][np.newaxis])
            redim, coords = reduce_im_dimensions(row['masked_data'][np.newaxis])
            df = generate_reduced_image_df(redim, coords)
            df['context'] = row.context
            df['seed'] = row.seed
            # df['mouse_id'] = row.mouse_id
            df['correct_trial'] = row.correct_trial
            df['variable'] = row.variable
            df['opto_stim_coord'] = stim
            total_df+=[df]
            
    # total_df = pd.concat(total_df).rename(columns={'dff0': 'value'})
    total_df = pd.concat(total_df).rename(columns={'dff0': 'value'})
    total_df['coord'] = total_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
    total_df = total_df[total_df.coord.isin(total_df.seed.unique())]
    total_df['y_dest'] = total_df.apply(lambda x: eval(x.coord)[0], axis=1)
    total_df['x_dest'] = total_df.apply(lambda x: eval(x.coord)[1], axis=1)
    total_df['y_source'] = total_df.apply(lambda x: eval(x.seed)[0], axis=1)
    total_df['x_source'] = total_df.apply(lambda x: eval(x.seed)[1], axis=1)

    r_df = total_df.groupby(by=['context', 'opto_stim_coord', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r'})
    sigma_df = total_df.groupby(by=['context', 'opto_stim_coord', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("sigma"), 'value'].values[0])).reset_index().rename(columns={0:'sigma'})
    delta_r_df = r_df.groupby(by=['opto_stim_coord', 'context', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: x.loc[x.correct_trial==1, 'r'].values[0] - x.loc[x.correct_trial==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})

    for coord in total_df['opto_stim_coord'].unique():
        stats = pd.read_csv(os.path.join(output_path, [f"{coord}_stim" if 'opto' in output_path else ''][0], 'choice_pairwise_selected_rois_stats.csv'))
        stats['norm_d'] = np.round(np.clip((stats.d_prime.values - 0.8)/(2 - 0.8), 0, 1), 2)

        for c in total_df.context.unique():
            r = r_df[(r_df.context==c) & (r_df.opto_stim_coord==coord)]
            r = r[r.seed != coord]
            r['norm_r'] = np.round(np.clip((r.r.values- 0.3)/(0.6 - 0.3), 0, 1), 2)

            sigma = sigma_df[(sigma_df.context==c) & (sigma_df.opto_stim_coord==coord)]
            sigma = sigma[sigma.seed != coord]

            delta = delta_r_df[(delta_r_df.context==c) & (delta_r_df.opto_stim_coord==coord)]
            delta = delta[delta.seed != coord]
            delta['norm_r'] = np.round(np.clip((delta.r.values- -0.1)/(0.1 - -0.1), 0, 1), 2)

            for outcome in total_df.correct_trial.unique():
                fig, ax = plt.subplots(figsize=(4,4))
                fig.suptitle(f"{coord} stim r between rois")
                im=ax.scatter(r.loc[r.coord==r.seed, 'x_source'], r.loc[r.coord==r.seed, 'y_source'], s=100, c='k')
                if coord != '(-5.0, 5.0)':
                    ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=100)

                ax.scatter(0, 0, marker='+', c='gray', s=100)
                for seed, dest in combinations(r.seed.unique(), 2):
                    if seed == dest:
                        continue
                    
                    sub_r = r[(r.correct_trial==outcome) & (r.seed.isin([seed, dest])) & (r.coord.isin([seed, dest]))]
                    sub_r = sub_r[sub_r.seed != sub_r.coord]
                    sub_sigma = sigma[(sigma.correct_trial==outcome) & (sigma.seed.isin([seed, dest])) & (sigma.coord.isin([seed, dest]))]
                    sub_sigma = sub_sigma[sub_sigma.seed != sub_sigma.coord]
                    if np.round(sub_sigma.sigma.mean(), 1)>=1.8:
                        ax.plot([sub_r.x_source.unique(), sub_r.x_dest.unique()], [sub_r.y_source.unique(), sub_r.y_dest.unique()], c=viridis_palette(sub_r.norm_r.mean()), linewidth=4)       

                ax.grid(True)
                ax.set_xticks(np.linspace(0.5,5.5,6))
                ax.set_xlim([-0.25, 6])
                ax.set_yticks(np.linspace(-3.5, 2.5,7))
                ax.set_ylim([-3.75, 2.75])
                ax.invert_xaxis()

                if 'opto' in output_path:
                    save_path = os.path.join(output_path, f"{coord}_stim")
                else:
                    save_path = os.path.join(output_path, f'{"rewarded" if c==1 else "non-rewarded"}')

                fig.savefig(os.path.join(save_path, f'r_summary_{["correct" if outcome else "non-rewarded"][0]}.png'))
                fig.savefig(os.path.join(save_path, f'r_summary_{["rewarded" if outcome else "non-rewarded"][0]}.svg'))

            fig, ax = plt.subplots(figsize=(4,4))
            fig.suptitle(f"{coord} stim r between rois")
            im=ax.scatter(delta.loc[delta.coord==delta.seed, 'x_source'], delta.loc[delta.coord==delta.seed, 'y_source'], s=100, c='k')
            if coord != '(-5.0, 5.0)':
                ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=100)

            ax.scatter(0, 0, marker='+', c='gray', s=100)
            for seed, dest in combinations(delta.seed.unique(), 2):
                if seed == dest:
                    continue
                
                sub_delta = delta[(delta.seed.isin([seed, dest])) & (delta.coord.isin([seed, dest]))]
                sub_delta = sub_delta[sub_delta.seed != sub_delta.coord]
                sub_sigma = sigma[(sigma.seed.isin([seed, dest])) & (sigma.coord.isin([seed, dest]))]
                sub_sigma = sub_sigma[sub_sigma.seed != sub_sigma.coord]

                if sub_sigma.sigma.mean()>=1.8:
                    d = stats.loc[(stats.context==c) & (stats.seed==seed) & (stats.coord==dest), 'd_prime'].values[0]
                    norm_d = stats.loc[(stats.context==c) & (stats.seed==seed) & (stats.coord==dest), 'norm_d'].values[0]
                    ax.plot([sub_delta.x_source.unique(), sub_delta.x_dest.unique()], [sub_delta.y_source.unique(), sub_delta.y_dest.unique()], c=seismic_palette(sub_delta.norm_r.mean()), linewidth=d, alpha=norm_d)       

            ax.grid(True)
            ax.set_xticks(np.linspace(0.5,5.5,6))
            ax.set_xlim([-0.25, 6])
            ax.set_yticks(np.linspace(-3.5, 2.5,7))
            ax.set_ylim([-3.75, 2.75])
            ax.invert_xaxis()

            if 'opto' in output_path:
                save_path = os.path.join(output_path, f"{coord}_stim")
            else:
                save_path = os.path.join(output_path, f'{"rewarded" if c==1 else "non-rewarded"}')

            fig.savefig(os.path.join(save_path, 'r_summary_delta.png'))
            fig.savefig(os.path.join(save_path, 'r_summary_delta.svg'))


def plot_mouse_barplot_r_opto(mouse_avg, output_path):

    stim_list = ['(-0.5, 0.5)', '(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']

    for stim in stim_list:
        if stim == "(-1.5, 0.5)":
            continue

        group = mouse_avg.loc[mouse_avg.opto_stim_coord.isin([stim, '(-5.0, 5.0)'])]

        redim_df = []
        for i, row in group.iterrows():
            redim, coords = reduce_im_dimensions(row['value'][np.newaxis])
            df = generate_reduced_image_df(redim, coords)
            df['context'] = row.context
            df['mouse_id'] = row.mouse_id
            df['opto_stim_coord'] = row.opto_stim_coord
            df['variable'] = row.variable
            redim_df+=[df]
            
        redim_df = pd.concat(redim_df).rename(columns={'dff0': 'value'})

        redim_df['seed'] = redim_df.apply(lambda x: x.variable.split("_")[0], axis=1)
        redim_df['coord_order'] = redim_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
        redim_df = redim_df.groupby(by=['mouse_id', 'opto_stim_coord', 'context', 'coord_order', 'seed']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r', 'coord_order': 'coord'})
        
        save_path = os.path.join(output_path, f"{stim}_stim")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        all_rois_stats, all_rois_stats_rew_vs_norew, selected_rois_stats, selected_rois_stats_rew_vs_norew = compute_stats_barplot_opto(redim_df, save_path)
        
        for c, group in redim_df.groupby('context'):
            g = sns.catplot(
                x="coord",
                y="r",
                hue="opto_stim_coord",
                hue_order=[stim, f"(-5.0, 5.0)"], 
                palette=['#005F60', '#FD5901'],
                row="seed",
                data=group,
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=3,
                alpha=0.5,
                sharex=False)
                
            g.map(sns.stripplot, 'coord', 'r', 'opto_stim_coord', hue_order=[stim, f"(-5.0, 5.0)"], palette=['#005F60', '#FD5901'], dodge=True, alpha=0.6, ec='k', linewidth=1)
            g.set_ylabels('R')
            g.tick_params(axis='x', rotation=30)
            for ax in g.axes.flat:
                ax.set_ylim([-1, 1])
                seed = ax.get_title('center').split("= ")[-1]
                stats = all_rois_stats.loc[(all_rois_stats.context==c) & (all_rois_stats.seed==seed)]
                ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.9, marker='*', s=100, c='k')
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            g.figure.tight_layout()
            g.figure.savefig(os.path.join(save_path, f'r-shuffle_stim_vs_control_barplot_{"rewarded" if c else "non-rewarded"}.png'))
            g.figure.savefig(os.path.join(save_path, f'r-shuffle_stim_vs_control_barplot_{"rewarded" if c else "non-rewarded"}.svg'))

            g = sns.catplot(
            x="coord",
            y="r",
            hue="opto_stim_coord",
            hue_order=[stim, f"(-5.0, 5.0)"], 
            palette=['#005F60', '#FD5901'],
            order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
            col="seed",
            data=group.loc[group.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"])],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=1,
            alpha=0.5)
        
            g.map(sns.stripplot, 'coord', 'r', 'opto_stim_coord', hue_order=[stim, f"(-5.0, 5.0)"], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#005F60', '#FD5901'], dodge=True, alpha=0.6, ec='k', linewidth=1)
            g.set_ylabels('R')
            g.tick_params(axis='x', rotation=30)
            for ax in g.axes.flat:
                ax.set_ylim([-1, 1])
                seed = ax.get_title('center').split("= ")[-1]
                stats = selected_rois_stats.loc[(selected_rois_stats.context==c) & (selected_rois_stats.seed==seed)]
                ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.9, marker='*', s=100, c='k')
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            g.figure.tight_layout()
            g.figure.savefig(os.path.join(save_path, f'r-shuffle_stim_vs_control_barplot_selected_rois_{"rewarded" if c else "non-rewarded"}.png'))
            g.figure.savefig(os.path.join(save_path, f'r-shuffle_stim_vs_control_barplot_selected_rois_{"rewarded" if c else "non-rewarded"}.svg'))

        redim_df = redim_df.groupby(by=['mouse_id', 'context', 'coord', 'seed']).apply(lambda x: x.loc[x.opto_stim_coord==stim, 'r'].values[0] - x.loc[x.opto_stim_coord=='(-5.0, 5.0)', 'r'].values[0]).reset_index().rename(columns={0:'r'})

        ## Plot corrected correlation r substracted R+ - R-  with correct vs incorrect trial
        g = sns.catplot(
            x="coord",
            y="r",
            hue="context",
            hue_order=[1,0],
            palette=['#348A18', '#6E188A'],
            row="seed",
            data=redim_df,
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=3,
            alpha=0.5,
            sharex=False)
            
        g.map(sns.stripplot, 'coord', 'r', 'context', hue_order=[1,0], palette=['#348A18', '#6E188A'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('Control <-- --> Stim')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            if stim == '(-5.0, 5.0)':
                ax.set_ylim([-0.15, 0.15])
            else:
                ax.set_ylim([-0.3, 0.3])

            seed = ax.get_title('center').split("= ")[-1]
            stats = all_rois_stats_rew_vs_norew[all_rois_stats_rew_vs_norew.seed==seed]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_stim_vs_control_barplot.png'))
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_stim_vs_control_barplot.svg'))

        g = sns.catplot(
            x="coord",
            y="r",
            hue="context",
            hue_order=[1,0],
            order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
            palette=['#348A18', '#6E188A'],
            col="seed",
            data=redim_df.loc[redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"])],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=1,
            alpha=0.5)
        
        g.map(sns.stripplot, 'coord', 'r', 'context', hue_order=[1,0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#348A18', '#6E188A'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('Control <-- --> Stim')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            if stim=='(-5.0, 5.0)':
                ax.set_ylim([-0.15, 0.15])
            else:
                ax.set_ylim([-0.3, 0.3])
                
            seed = ax.get_title('center').split("= ")[-1]
            stats = selected_rois_stats_rew_vs_norew[selected_rois_stats_rew_vs_norew.seed==seed]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_stim_vs_control_barplot_selected_rois.png'))
        g.figure.savefig(os.path.join(save_path, 'r-shuffle_stim_vs_control_barplot_selected_rois.svg'))


def compute_stats_barplot_opto(df, output_path):
    df=df[df.coord!='(2.5, 5.5)']

    all_rois_stats =[]
    for name, group in df.groupby(by=['context', 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].to_numpy(), group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].to_numpy())
        mean_diff = group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].mean() - group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].mean()
        std_diff = np.std(group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].to_numpy() - group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].to_numpy())

        results = {
         'context': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean_rew': group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].mean(),
         'std_rew': group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].std(),
         'mean_no_rew': group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].mean(),
         'std_no_rew': group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.coord.unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.coord.unique().shape[0],
         'significant': p<(0.05/df.coord.unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
         }
        
        all_rois_stats += [results]
    all_rois_stats = pd.DataFrame(all_rois_stats)
    all_rois_stats.to_csv(os.path.join(output_path, 'stim_vs_control_pairwise_all_rois_stats.csv'))

    all_rois_stats_rew_vs_norew =[]
    for name, group in df.groupby(by=['seed', 'coord']):
        stim_diff = group.groupby(by=['mouse_id', 'context']).apply(lambda x: x.loc[x.opto_stim_coord!='(-5.0, 5.0)', 'r'].to_numpy() - x.loc[x.opto_stim_coord=='(-5.0, 5.0)', 'r'].to_numpy()).reset_index().rename(columns={0:'r'})
        t, p = ttest_rel(stim_diff.loc[stim_diff.context==1, 'r'].to_numpy(), stim_diff.loc[stim_diff.context==0, 'r'].to_numpy())
        mean_diff = stim_diff.loc[stim_diff.context==1, 'r'].mean() - stim_diff.loc[stim_diff.context==0, 'r'].mean()
        std_diff = np.std(stim_diff.loc[stim_diff.context==1, 'r'].to_numpy() - stim_diff.loc[stim_diff.context==0, 'r'].to_numpy())

        results = {
        'seed': name[0],
        'coord': name[1],
        'dof': stim_diff.mouse_id.unique().shape[0]-1,
        'mean_correct': stim_diff.loc[stim_diff.context==1, 'r'].mean(),
        'std_correct': stim_diff.loc[stim_diff.context==1, 'r'].std(),
        'mean_incorrect': stim_diff.loc[stim_diff.context==0, 'r'].mean(),
        'std_incorrect': stim_diff.loc[stim_diff.context==0, 'r'].std(),
        't': t,
        'p': p,
        'p_corr': p*df.coord.unique().shape[0],
        'alpha': 0.05,
        'alpha_corr': 0.05/df.coord.unique().shape[0],
        'significant': p<(0.05/df.coord.unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
        }
        all_rois_stats_rew_vs_norew += [results]
    all_rois_stats_rew_vs_norew = pd.DataFrame(all_rois_stats_rew_vs_norew)
    all_rois_stats_rew_vs_norew.to_csv(os.path.join(output_path, 'stim_vs_control_pairwise_all_rois_stats_rew_vs_norew.csv'))

    # df = df[df.seed!='(1.5, 3.5)']
    selected_rois_stats =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['context', 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].to_numpy(), group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].to_numpy())
        mean_diff = (group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].mean() - group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].mean())
        std_diff = np.std(group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].to_numpy() - group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].to_numpy())

        results = {
         'context': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean_rew': group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].mean(),
         'std_rew': group.loc[group.opto_stim_coord!='(-5.0, 5.0)', 'r'].std(),
         'mean_no_rew': group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].mean(),
         'std_no_rew': group.loc[group.opto_stim_coord=='(-5.0, 5.0)', 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
         'd_prime': abs(mean_diff/std_diff)

         }
        
        selected_rois_stats += [results]
    selected_rois_stats = pd.DataFrame(selected_rois_stats)
    selected_rois_stats.to_csv(os.path.join(output_path, 'stim_vs_control_pairwise_selected_rois_stats.csv'))

    selected_rois_stats_rew_vs_norew =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['seed', 'coord']):
        stim_diff = group.groupby(by=['mouse_id', 'context']).apply(lambda x: x.loc[x.opto_stim_coord!='(-5.0, 5.0)', 'r'].to_numpy() - x.loc[x.opto_stim_coord=='(-5.0, 5.0)', 'r'].to_numpy()).reset_index().rename(columns={0:'r'})
        t, p = ttest_rel(stim_diff.loc[stim_diff.context==1, 'r'].to_numpy(), stim_diff.loc[stim_diff.context==0, 'r'].to_numpy())
        mean_diff = (stim_diff.loc[stim_diff.context==1, 'r'].mean() - stim_diff.loc[stim_diff.context==0, 'r'].mean())
        std_diff = np.std(stim_diff.loc[stim_diff.context==1, 'r'].to_numpy() - stim_diff.loc[stim_diff.context==0, 'r'].to_numpy())
        
        results = {
        'seed': name[0],
        'coord': name[1],
        'dof': stim_diff.mouse_id.unique().shape[0]-1,
        'mean_correct': stim_diff.loc[stim_diff.context==1, 'r'].mean(),
        'std_correct': stim_diff.loc[stim_diff.context==1, 'r'].std(),
        'mean_incorrect': stim_diff.loc[stim_diff.context==0, 'r'].mean(),
        'std_incorrect': stim_diff.loc[stim_diff.context==0, 'r'].std(),
        't': t,
        'p': p,
        'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
        'alpha': 0.05,
        'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
        'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
        'd_prime': abs(mean_diff.sum()/std_diff)
        }
        selected_rois_stats_rew_vs_norew += [results]
    selected_rois_stats_rew_vs_norew = pd.DataFrame(selected_rois_stats_rew_vs_norew)
    selected_rois_stats_rew_vs_norew.to_csv(os.path.join(output_path, 'stim_vs_control_pairwise_selected_rois_stats_rew_vs_norew.csv'))

    return all_rois_stats, all_rois_stats_rew_vs_norew, selected_rois_stats, selected_rois_stats_rew_vs_norew


def plot_stim_control_comparison(df, roi, save_path, vmin=-0.05, vmax=0.05):

    stim = df.opto_stim_coord.unique()
    stim = stim[stim!='(-5.0, 5.0)'][0]

    total_avg = df.groupby(by=['opto_stim_coord', 'variable'])['value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_r"), 'value'].values[0],
                        title=stim,
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_r"), 'value'].values[0],
                        title='Control',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_r"), 'value'].values[0] - \
            total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_r"), 'value'].values[0]

    plot_single_frame(im, title=f'{stim} - Control',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_r.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0],
                        title=stim,
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0],
                        title='Control',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0] - \
            total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0]

    plot_single_frame(im, title=f'{stim} - Control',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0],
                        title=stim,
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0],
                        title='Control',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0] - \
            total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0]

    plot_single_frame(im, title=f'{stim} - Control',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_std.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"R - shuffle")

    im_r = total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]
    im_nor = total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]

    plot_single_frame(im_r,
                        title=stim,
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(im_nor,
                        title='Control',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])

    plot_single_frame(im_r - im_nor, title=f'{stim} - Control',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_avg.png"))

    d_palette = sns.color_palette("gnuplot2", 50)
    dprime_palette = LinearSegmentedColormap.from_list("Custom", d_palette[:-2])

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Rewarded',
                        colormap=dprime_palette, vmin=1.8, vmax=3, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap=dprime_palette, vmin=1.8, vmax=3, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0] - \
            total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]

    plot_single_frame(im, title='R+ - R-',
                        colormap=seismic_palette, vmin=-0.3, vmax=0.3, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_nsigmas.png"))

    mask_r = np.where(total_avg.loc[(total_avg.opto_stim_coord != '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_r, np.nan)
    mask_non_r = np.where(total_avg.loc[(total_avg.opto_stim_coord == '(-5.0, 5.0)') & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_nor, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(mask_r,
                        title='Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.9, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(mask_non_r,
                        title='Non-Rewarded',
                        colormap='viridis', vmin=0.3, vmax=0.9, norm=False, fig=fig, ax=ax[1])

    plot_single_frame(mask_r-mask_non_r, title='R+ - R-',
                        colormap=seismic_palette, vmin=-0.1, vmax=0.1, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_significant_pairs.png"))


def plot_connected_dot_r_stim_vs_control(total_avg, output_path):
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)
    viridis_palette = cm.get_cmap('viridis')

    stim_list = ["(-5.0, 5.0)", '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']

    total_df = []
    for stim in stim_list:
        group = total_avg.loc[total_avg.opto_stim_coord==stim].reset_index(drop=True)
        group['seed'] = group.apply(lambda x: x.variable.split("_")[0], axis=1)
        group['masked_data'] = group.value

        for i, row in group.iterrows():
            # redim, coords = reduce_im_dimensions(row['value'][np.newaxis])
            redim, coords = reduce_im_dimensions(row['masked_data'][np.newaxis])
            df = generate_reduced_image_df(redim, coords)
            df['context'] = row.context
            df['seed'] = row.seed
            # df['mouse_id'] = row.mouse_id
            df['variable'] = row.variable
            df['opto_stim_coord'] = stim
            total_df+=[df]
            
    # total_df = pd.concat(total_df).rename(columns={'dff0': 'value'})
    total_df = pd.concat(total_df).rename(columns={'dff0': 'value'})
    total_df['coord'] = total_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
    total_df = total_df[total_df.coord.isin(total_df.seed.unique())]
    total_df['y_dest'] = total_df.apply(lambda x: eval(x.coord)[0], axis=1)
    total_df['x_dest'] = total_df.apply(lambda x: eval(x.coord)[1], axis=1)
    total_df['y_source'] = total_df.apply(lambda x: eval(x.seed)[0], axis=1)
    total_df['x_source'] = total_df.apply(lambda x: eval(x.seed)[1], axis=1)

    r_df = total_df.groupby(by=['context', 'opto_stim_coord', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r'})
    sigma_df = total_df.groupby(by=['context', 'opto_stim_coord', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("sigma"), 'value'].values[0])).reset_index().rename(columns={0:'sigma'})
    delta_r_df = r_df.groupby(by=['context', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: x.loc[x.opto_stim_coord==stim, 'r'].values[0] - x.loc[x.opto_stim_coord!=stim, 'r'].values[0]).reset_index().rename(columns={0:'r'})

    for coord in total_df['opto_stim_coord'].unique():
        if coord == "(-5.0, 5.0)":
            continue

        stats = pd.read_csv(os.path.join(output_path, f"{coord}_stim", 'stim_vs_control_pairwise_selected_rois_stats.csv'))
        stats['norm_d'] = np.round(np.clip((stats.d_prime.values - 0.8)/(2 - 0.8), 0, 1), 2)

        for c in total_df.context.unique():
            r = r_df[(r_df.context==c) & (r_df.opto_stim_coord.isin(["(-5.0, 5.0)", coord]))]
            r = r[r.seed != coord]
            r['norm_r'] = np.round(np.clip((r.r.values- 0.3)/(0.6 - 0.3), 0, 1), 2)

            sigma = sigma_df[(sigma_df.context==c) & (sigma_df.opto_stim_coord.isin(["(-5.0, 5.0)", coord]))]
            sigma = sigma[sigma.seed != coord]

            delta = delta_r_df[(delta_r_df.context==c)]
            delta = delta[delta.seed != coord]
            delta['norm_r'] = np.round(np.clip((delta.r.values- -0.1)/(0.1 - -0.1), 0, 1), 2)

            for stim in [coord, '(-5.0, 5.0)']:
                fig, ax = plt.subplots(figsize=(4,4))
                fig.suptitle(f"{coord} stim r between rois")
                im=ax.scatter(r.loc[r.coord==r.seed, 'x_source'], r.loc[r.coord==r.seed, 'y_source'], s=100, c='k')
                if coord != '(-5.0, 5.0)':
                    ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=100)

                ax.scatter(0, 0, marker='+', c='gray', s=100)
                for seed, dest in zip(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(2.5, 2.5)", "(1.5, 3.5)"], 
                                      ["(1.5, 1.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(1.5, 3.5)", "(0.5, 4.5)", "(0.5, 4.5)"]):
                    
                    sub_r = r[(r.opto_stim_coord==stim) & (r.seed.isin([seed, dest])) & (r.coord.isin([seed, dest]))]
                    sub_r = sub_r[sub_r.seed != sub_r.coord]
                    sub_sigma = sigma[(r.opto_stim_coord==stim) & (sigma.seed.isin([seed, dest])) & (sigma.coord.isin([seed, dest]))]
                    sub_sigma = sub_sigma[sub_sigma.seed != sub_sigma.coord]
                    ax.plot([sub_r.x_source.unique(), sub_r.x_dest.unique()], [sub_r.y_source.unique(), sub_r.y_dest.unique()], c=viridis_palette(sub_r.norm_r.mean()), linewidth=4)       

                ax.grid(True)
                ax.set_xticks(np.linspace(0.5,5.5,6))
                ax.set_xlim([-0.25, 6])
                ax.set_yticks(np.linspace(-3.5, 2.5,7))
                ax.set_ylim([-3.75, 2.75])
                ax.invert_xaxis()

                save_path = os.path.join(output_path, f"{coord}_stim")

                fig.savefig(os.path.join(save_path, f'{"rewarded" if c else "non-rewarded"}_r_summary_{["stim" if stim!="(-5.0, 5.0)" else "control"][0]}.png'))
                fig.savefig(os.path.join(save_path, f'{"rewarded" if c else "non-rewarded"}_r_summary_{["stim" if stim!="(-5.0, 5.0)" else "control"][0]}.svg'))

            fig, ax = plt.subplots(figsize=(4,4))
            fig.suptitle(f"{coord} stim r between rois")
            im=ax.scatter(delta.loc[delta.coord==delta.seed, 'x_source'], delta.loc[delta.coord==delta.seed, 'y_source'], s=100, c='k')
            if coord != '(-5.0, 5.0)':
                ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=100)

            ax.scatter(0, 0, marker='+', c='gray', s=100)
            for seed, dest in zip(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(2.5, 2.5)", "(1.5, 3.5)"], 
                                  ["(1.5, 1.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(1.5, 3.5)", "(0.5, 4.5)", "(0.5, 4.5)"]):
                
                sub_delta = delta[(delta.seed.isin([seed, dest])) & (delta.coord.isin([seed, dest]))]
                sub_delta = sub_delta[sub_delta.seed != sub_delta.coord]
                sub_sigma = sigma[(sigma.seed.isin([seed, dest])) & (sigma.coord.isin([seed, dest]))]
                sub_sigma = sub_sigma[sub_sigma.seed != sub_sigma.coord]

                d = stats.loc[(stats.context==c) & (stats.seed==seed) & (stats.coord==dest), 'd_prime'].values[0]
                norm_d = stats.loc[(stats.context==c) & (stats.seed==seed) & (stats.coord==dest), 'norm_d'].values[0]
                ax.plot([sub_delta.x_source.unique(), sub_delta.x_dest.unique()], [sub_delta.y_source.unique(), sub_delta.y_dest.unique()], c=seismic_palette(sub_delta.norm_r.mean()), linewidth=d, alpha=norm_d)       

            ax.grid(True)
            ax.set_xticks(np.linspace(0.5,5.5,6))
            ax.set_xlim([-0.25, 6])
            ax.set_yticks(np.linspace(-3.5, 2.5,7))
            ax.set_ylim([-3.75, 2.75])
            ax.invert_xaxis()

            save_path = os.path.join(output_path, f"{coord}_stim")

            fig.savefig(os.path.join(save_path, f'{"rewarded" if c else "non-rewarded"}_r_summary_delta.png'))
            fig.savefig(os.path.join(save_path, f'{"rewarded" if c else "non-rewarded"}_r_summary_delta.svg'))


def main(data, output_path):
    ## plot
    if 'opto' not in output_path:
        # data.trial_count = data.trial_count.map({0: 1, 1: 2, 2: 3, 3: 4, 4: -4, 5: -3, 6: -2, 7: -1})
        data['opto_stim_coord'] = '(-5.0, 5.0)'
        data.value = data.apply(lambda x: x.value[0] if 'percentile' not in x.variable else x.value, axis=1)
        mouse_avg = data.groupby(by=['mouse_id', 'opto_stim_coord', 'context', 'correct_trial', 'variable'])['value'].apply(
            lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
        mouse_avg['value'] = mouse_avg['value'].apply(lambda x: np.array(x).reshape(125, -1))

        total_avg = mouse_avg.groupby(by=['opto_stim_coord', 'context', 'correct_trial', 'variable'])['value'].apply(
            lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()

        # plot total avg
        for roi in ['(-0.5, 0.5)', '(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)']:
            if roi == '(-1.5, 0.5)':
                continue
            print(f"Plotting total averages for roi {roi}")
            save_path = os.path.join(output_path, 'context', 'correct', roi)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plot_avg_between_blocks(total_avg.loc[total_avg.correct_trial==1], roi, save_path)
            plot_reduced_correlations(total_avg.loc[total_avg.correct_trial==1], roi, save_path)
            save_path = os.path.join(output_path, 'context', 'incorrect', roi)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plot_avg_between_blocks(total_avg.loc[total_avg.correct_trial==0], roi, save_path)
            plot_reduced_correlations(total_avg.loc[total_avg.correct_trial==0], roi, save_path)

            save_path = os.path.join(output_path, 'choice', 'rewarded', roi)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plot_avg_within_blocks(total_avg.loc[total_avg.context==1], roi, save_path)
            
            save_path = os.path.join(output_path, 'choice', 'non-rewarded', roi)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plot_avg_within_blocks(total_avg.loc[total_avg.context==0], roi, save_path)

        plot_mouse_barplot_r_context(mouse_avg, os.path.join(output_path, 'context'))
        plot_connected_dot_r_context(total_avg, os.path.join(output_path, 'context'))

        plot_mouse_barplot_r_choice(mouse_avg, os.path.join(output_path, 'choice'))
        plot_connected_dot_r_choice(total_avg, os.path.join(output_path, 'choice'))

    else:
        data.value = data.apply(lambda x: x.value[0] if 'percentile' not in x.variable else x.value, axis=1)
        data['correct_trial'] = data.apply(lambda x: 1 if x.opto_stim_coord!="(-5.0, 5.0)" else x.correct_trial, axis=1)
        data = data[data.correct_trial==1]
        mouse_avg = data.groupby(by=['mouse_id', 'context', 'opto_stim_coord', 'variable'])['value'].apply(
            lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
        mouse_avg['value'] = mouse_avg['value'].apply(lambda x: np.array(x).reshape(125, -1))

        total_avg = mouse_avg.groupby(by=['context', 'opto_stim_coord', 'variable'])['value'].apply(
            lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
        
        # for coord, group in total_avg.groupby('opto_stim_coord'):
        for coord in ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']: 

            group = total_avg.loc[total_avg.opto_stim_coord==coord]
            for roi in ['(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)']:#'A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2', 'RSC'
                print(f"Plotting total averages for roi {roi}, stim coord {coord}")
                save_path = os.path.join(output_path, 'context', f"{coord}_stim", f"{roi}_seed")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if coord=='(-5.0, 5.0)':
                    plot_avg_between_blocks(group, roi, save_path)
                else:
                    plot_avg_between_blocks(group, roi, save_path, vmin=-0.1, vmax=0.1)

                if coord =='(-5.0, 5.0)':
                    continue

                group = total_avg.loc[total_avg.opto_stim_coord.isin(['(-5.0, 5.0)', coord])]
                save_path = os.path.join(output_path, 'stim_vs_control', 'rewarded', f"{coord}_stim", f"{roi}_seed")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plot_stim_control_comparison(group.loc[group.context==1], roi, vmin=-0.2, vmax=0.2, save_path=save_path)

                save_path = os.path.join(output_path, 'stim_vs_control', 'non-rewarded', f"{coord}_stim", f"{roi}_seed")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plot_stim_control_comparison(group.loc[group.context==0], roi, vmin=-0.5, vmax=0.5, save_path=save_path)

        plot_mouse_barplot_r_context(mouse_avg, os.path.join(output_path, 'context'))
        plot_connected_dot_r_context(total_avg, os.path.join(output_path, 'context'))

        plot_mouse_barplot_r_opto(mouse_avg, os.path.join(output_path, 'stim_vs_control'))
        plot_connected_dot_r_stim_vs_control(total_avg, os.path.join(output_path, 'stim_vs_control'))


if __name__ == "__main__":

    # root = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_trial_based_corr_mar2025"
    # root = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_correlation_opto_wf"
    root = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_correlations_20250604"
    root = haas_pathfun(root)
    # for dataset in os.listdir(root):
    for dtype in [ 'wf_opto_controls', 'controls_tdtomato_expert','gcamp_expert', 'gfp_expert']: # 'jrgeco_expert', 'wf_opto', 
        dataset = f'pixel_cross_correlation_{dtype}'

        print(f"Analyzing {dataset}")

        result_folder = os.path.join(root, dataset)
        result_folder = haas_pathfun(result_folder)
        if not os.path.exists(os.path.join(result_folder, 'results')):
            os.makedirs(os.path.join(result_folder, 'results'), exist_ok=True)

        load_data=True

        if load_data==True and os.path.exists(os.path.join(result_folder, 'results', "combined_avg_correlation_results.json")):
            data = pd.read_json(os.path.join(result_folder, 'results',"combined_avg_correlation_results.json"))
            data['value'] = data.value.apply(lambda x: np.asarray(x, dtype=float))

            print(f'{dataset} results loaded')

        else:
            data = []
            all_files = glob.glob(os.path.join(result_folder, "**", "*", "correlation_table.parquet.gzip"))

            for file in tqdm(all_files):
                session_data = preprocess_corr_results(file)
                data += [session_data]

            data = pd.concat(data, axis=0, ignore_index=True)
            if not os.path.exists(os.path.join(result_folder, 'results')):
                os.makedirs(os.path.join(result_folder, 'results'))

            data.to_json(os.path.join(result_folder, 'results', "combined_avg_correlation_results.json"))
            data = pd.read_json(os.path.join(result_folder, 'results',"combined_avg_correlation_results.json"))
            data['value'] = data.value.apply(lambda x: np.asarray(x, dtype=float))
 
        main(data, os.path.join(result_folder, 'results'))
