import os
import sys
sys.path.append(os.getcwd())
import yaml
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


def plot_avg_between_blocks(df, roi, save_path, vmin=-0.03, vmax=0.03):
    total_avg = df.groupby(by=['context', 'variable'])['value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_r"), 'value'].values[0],
                        title='Rewarded',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_r"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_r"), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_r"), 'value'].values[0]

    plot_single_frame(im, title='R+ - R-',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_r.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0],
                        title='Rewarded',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_shuffle_mean"), 'value'].values[0]

    plot_single_frame(im, title='R+ - R-',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0],
                        title='Rewarded',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_shuffle_std"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])
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
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(im_nor,
                        title='Non-Rewarded',
                        colormap='icefire', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])

    plot_single_frame(im_r - im_nor, title='R+ - R-',
                        colormap=seismic_palette, vmin=vmin, vmax=vmax, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_avg.png"))

    d_palette = sns.color_palette("gnuplot2", 50)
    dprime_palette = LinearSegmentedColormap.from_list("Custom", d_palette[:-2])

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Rewarded',
                        colormap=dprime_palette, vmin=0.5, vmax=2, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap=dprime_palette, vmin=0.5, vmax=2, norm=False, fig=fig, ax=ax[1])
    im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]

    plot_single_frame(im, title='R+ - R-',
                        colormap=seismic_palette, vmin=-0.2, vmax=0.2, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_nsigmas.png"))


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
    plot_grid_on_allen(im_R_df, outcome='dff0', palette='icefire', result_path=None, dotsize=340, vmin=-0.5, vmax=0.5, norm=None, fig=fig, ax= ax[0])
    plot_grid_on_allen(im_nR_df, outcome='dff0', palette='icefire', result_path=None, dotsize=340, vmin=-0.5, vmax=0.5, norm=None, fig=fig, ax= ax[1])
    plot_grid_on_allen(im_sub_df, outcome='dff0', palette=seismic_palette, result_path=None, dotsize=340, vmin=-0.03, vmax=0.03, norm=None, fig=fig, ax= ax[2])
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
    plot_grid_on_allen(im_R_df, outcome='dff0', palette='icefire', result_path=None, dotsize=340, vmin=-0.5, vmax=0.5, norm=None, fig=fig, ax= ax[0])
    plot_grid_on_allen(im_nR_df, outcome='dff0', palette='icefire', result_path=None, dotsize=340, vmin=-0.5, vmax=0.5, norm=None, fig=fig, ax= ax[1])
    plot_grid_on_allen(im_sub_df, outcome='dff0', palette=seismic_palette, result_path=None, dotsize=340, vmin=-0.03, vmax=0.03, norm=None, fig=fig, ax= ax[2])
    fig.savefig(os.path.join(save_path, 'red_im', f'{roi}_r_corrected_reduced_images.png'))



def plot_trial_based_correlations(df, roi, save_path):

    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    R = df.loc[(df.variable == f'{roi}_r')]
    shuffle = df.loc[(df.variable == f'{roi}_shuffle_mean')]

    fig_r, ax_r = plt.subplots(1, 8, figsize=(20, 15))
    fig_r.suptitle(f"{roi} R in whisker trials before-after context")

    fig_b, ax_b = plt.subplots(1, 8, figsize=(20, 15))
    fig_b.suptitle(f"{roi} R-shuffle in whisker trials before-after context")

    fig_s, ax_s = plt.subplots(1, 8, figsize=(20, 15))
    fig_s.suptitle(f"{roi} nsigmas in whisker trials before-after context")

    for i, count in enumerate(R.trial_count.unique()):

        plot_single_frame(R.loc[(R.context == 1) & (R.trial_count == count), 'value'].values[0] - R.loc[(R.context == 0) & (R.trial_count == count), 'value'].values[0], title=str(count),
                            colormap=seismic_palette, vmin=-0.05, vmax=0.05, norm=False, fig=fig_r, ax=ax_r[i])

        image_rew = R.loc[(R.context == 1) & (R.trial_count == count), 'value'].values[0] - shuffle.loc[(shuffle.context == 1) & (shuffle.trial_count == count), 'value'].values[0]
        image_no_rew = R.loc[(R.context == 0) & (R.trial_count == count), 'value'].values[0] - shuffle.loc[(shuffle.context == 0) & (shuffle.trial_count == count), 'value'].values[0]
        plot_single_frame(image_rew - image_no_rew, title=str(count), colormap=seismic_palette, vmin=-0.05, vmax=0.05,
                          norm=False, fig=fig_b, ax=ax_b[i])
        
        image_rew = df.loc[(df.variable == f'{roi}_nsigmas') & (df.context == 1) & (df.trial_count == count), 'value'].values[0]
        image_no_rew = df.loc[(df.variable == f'{roi}_nsigmas') & (df.context == 0) & (df.trial_count == count), 'value'].values[0]
        plot_single_frame(image_rew - image_no_rew, title=str(count), colormap=seismic_palette, vmin=-0.2, vmax=0.2,
                          norm=False, fig=fig_s, ax=ax_s[i])        

    fig_r.tight_layout()
    fig_r.savefig(os.path.join(save_path, f'{roi}_r_by_wh_trial.png'))

    fig_b.tight_layout(pad=0.05)
    fig_b.savefig(os.path.join(save_path, f'{roi}_corrected_by_wh_trial.png'))

    fig_s.tight_layout(pad=0.05)
    fig_s.savefig(os.path.join(save_path, f'{roi}_nsigmas_by_wh_trial.png'))


def plot_trial_based_correlations_reduced(df, roi, save_path):
    if not os.path.exists(os.path.join(save_path, 'red_im')):
        os.makedirs(os.path.join(save_path, 'red_im'))

    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    R = df.loc[(df.variable == f'{roi}_r')]
    shuffle = df.loc[(df.variable == f'{roi}_shuffle_mean')]

    fig_r, ax_r = plt.subplots(1, 8, figsize=(20, 15))
    fig_r.suptitle(f"{roi} R in whisker trials before-after context")

    fig_b, ax_b = plt.subplots(1, 8, figsize=(20, 15))
    fig_b.suptitle(f"{roi} R-shuffle in whisker trials before-after context")

    fig_s, ax_s = plt.subplots(1, 8, figsize=(20, 15))
    fig_s.suptitle(f"{roi} nsigmas in whisker trials before-after context")

    for i, count in enumerate(R.trial_count.unique()):
        im_sub = R.loc[(R.context == 1) & (R.trial_count == count), 'value'].values[0] - R.loc[(R.context == 0) & (R.trial_count == count), 'value'].values[0]
        red_im_sub, coords = reduce_im_dimensions(im_sub[np.newaxis, ...])
        im_sub_df = generate_reduced_image_df(red_im_sub, coords)
        plot_grid_on_allen(im_sub_df, outcome='dff0', palette=seismic_palette, result_path=None, dotsize=340, vmin=-0.04, vmax=0.04, norm=None, fig=fig_r, ax= ax_r[i])


        image_rew = R.loc[(R.context == 1) & (R.trial_count == count), 'value'].values[0] - shuffle.loc[(shuffle.context == 1) & (shuffle.trial_count == count), 'value'].values[0]
        image_no_rew = R.loc[(R.context == 0) & (R.trial_count == count), 'value'].values[0] - shuffle.loc[(shuffle.context == 0) & (shuffle.trial_count == count), 'value'].values[0]
        im_sub = image_rew - image_no_rew
        red_im_sub, coords = reduce_im_dimensions(im_sub[np.newaxis, ...])
        im_sub_df = generate_reduced_image_df(red_im_sub, coords)
        plot_grid_on_allen(im_sub_df, outcome='dff0', palette=seismic_palette, result_path=None, dotsize=340, vmin=-0.04, vmax=0.04, norm=None, fig=fig_b, ax= ax_b[i])

        
        image_rew = df.loc[(df.variable == f'{roi}_nsigmas') & (df.context == 1) & (df.trial_count == count), 'value'].values[0]
        image_no_rew = df.loc[(df.variable == f'{roi}_nsigmas') & (df.context == 0) & (df.trial_count == count), 'value'].values[0]
        im_sub = image_rew - image_no_rew
        red_im_sub, coords = reduce_im_dimensions(im_sub[np.newaxis, ...])
        im_sub_df = generate_reduced_image_df(red_im_sub, coords)
        plot_grid_on_allen(im_sub_df, outcome='dff0', palette=seismic_palette, result_path=None, dotsize=340, vmin=-0.04, vmax=0.04, norm=None, fig=fig_s, ax= ax_s[i])

    fig_r.tight_layout(pad=0.05)
    fig_r.savefig(os.path.join(save_path, 'red_im', f'{roi}_r_by_wh_trial_reduced.png'))

    fig_b.tight_layout(pad=0.05)
    fig_b.savefig(os.path.join(save_path, 'red_im', f'{roi}_corrected_by_wh_trial_reduced.png'))

    fig_s.tight_layout(pad=0.05)
    fig_s.savefig(os.path.join(save_path, 'red_im', f'{roi}_nsigmas_by_wh_trial_reduced.png'))


def plot_mouse_barplot_r_from_images(mouse_avg, output_path):

    if 'opto' in output_path:
        stim_list = ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']
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
            save_path = os.path.join(output_path, 'from_images', f"{stim}_stim")
        else:
            save_path = os.path.join(output_path, 'from_images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        all_rois_stats, all_rois_stats_correct_vs_incorrect, selected_rois_stats, selected_rois_stats_correct_vs_incorrect = compute_stats_barplot(redim_df, save_path)
        
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
                order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"],
                palette=['#032b22', '#da4e02'],
                col="seed",
                data=redim_df.loc[redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"])],
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=1,
                alpha=0.5)
            
            g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1,0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"], palette=['#032b22', '#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
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
            order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"],
            data=redim_df.loc[(redim_df.correct_trial==1) & (redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"]))],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=1,
            alpha=0.5)
        
        g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"], palette=['#032b22', '#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
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
                order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"],
                data=redim_df.loc[(redim_df.correct_trial==0) & (redim_df.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"]))],
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=1,
                alpha=0.5)
            
            g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"], palette=['#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
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
        

def compute_stats_barplot_from_images(redim_df, output_path):
    all_rois_stats_to_0 =[]
    for name, group in redim_df.groupby(by=['correct_trial', 'seed', 'coord']):
        t, p = ttest_1samp(group.dff0, 0)
        results = {
         'correct_trial': name[0],
         'seed': name[1],
         'coord': name[2], 
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean': group.dff0.mean(),
         'std': group.dff0.std(),
         't': t,
         'p': p,
         'p_corr': p*redim_df.coord.unique().shape[0],
         'alpha': 0.05,
         'significant': p*redim_df.coord.unique().shape[0]<0.05,
         'd_prime': abs(group.dff0.mean()/group.dff0.std())
         }
        all_rois_stats_to_0 += [results]
    all_rois_stats_to_0 = pd.DataFrame(all_rois_stats_to_0)
    all_rois_stats_to_0.to_csv(os.path.join(output_path, 'all_rois_stats_to_0.csv'))

    if 'opto' not in output_path:
        all_rois_stats_correct_vs_incorrect =[]
        for name, group in redim_df.groupby(by=['seed', 'coord']):
            t, p = ttest_rel(group.loc[group.correct_trial==1, 'dff0'], group.loc[group.correct_trial==0, 'dff0'])
            results = {
            'seed': name[0],
            'coord': name[1],
            'dof': group.mouse_id.unique().shape[0]-1,
            'mean_correct': group.loc[group.correct_trial==1, 'dff0'].mean(),
            'std_correct': group.loc[group.correct_trial==1, 'dff0'].std(),
            'mean_incorrect': group.loc[group.correct_trial==0, 'dff0'].mean(),
            'std_incorrect': group.loc[group.correct_trial==0, 'dff0'].std(),
            't': t,
            'p': p,
            'p_corr': p*redim_df.coord.unique().shape[0],
            'alpha': 0.05,
            'significant': p*redim_df.coord.unique().shape[0]<0.05,
            'd_prime': t/np.sqrt(group.mouse_id.unique().shape[0]),
            }
            all_rois_stats_correct_vs_incorrect += [results]
        all_rois_stats_correct_vs_incorrect = pd.DataFrame(all_rois_stats_correct_vs_incorrect)
        all_rois_stats_correct_vs_incorrect.to_csv(os.path.join(output_path, 'all_rois_stats_correct_vs_incorrect.csv'))
    else:
        all_rois_stats_correct_vs_incorrect=[]

    selected_rois_stats_to_0 =[]
    for name, group in redim_df.loc[redim_df.coord.isin(redim_df.seed.unique())].groupby(by=['correct_trial', 'seed', 'coord']):
        t, p = ttest_1samp(group.dff0, 0)
        results = {
         'correct_trial': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean': group.dff0.mean(),
         'std': group.dff0.std(),
         't': t,
         'p': p,
         'p_corr': p*redim_df.loc[redim_df.coord.isin(redim_df.seed.unique()), 'coord'].unique().shape[0],
         'alpha': 0.05,
         'significant': p*redim_df.loc[redim_df.coord.isin(redim_df.seed.unique()), 'coord'].unique().shape[0]<0.05,
         'd_prime': abs(group.dff0.mean()/group.dff0.std())
         }
        selected_rois_stats_to_0 += [results]
    selected_rois_stats_to_0 = pd.DataFrame(selected_rois_stats_to_0)
    selected_rois_stats_to_0.to_csv(os.path.join(output_path, 'selected_rois_stats_to_0.csv'))

    if 'opto' not in output_path:
        selected_rois_stats_correct_vs_incorrect =[]
        for name, group in redim_df.loc[redim_df.coord.isin(redim_df.seed.unique())].groupby(by=['seed', 'coord']):
            t, p = ttest_rel(group.loc[group.correct_trial==1, 'dff0'], group.loc[group.correct_trial==0, 'dff0'])
            results = {
            'seed': name[0],
            'coord': name[1],
            'dof': group.mouse_id.unique().shape[0]-1,
            'mean_correct': group.loc[group.correct_trial==1, 'dff0'].mean(),
            'std_correct': group.loc[group.correct_trial==1, 'dff0'].std(),
            'mean_incorrect': group.loc[group.correct_trial==0, 'dff0'].mean(),
            'std_incorrect': group.loc[group.correct_trial==0, 'dff0'].std(),
            't': t,
            'p': p,
            'p_corr': p*redim_df.loc[redim_df.coord.isin(redim_df.seed.unique()), 'coord'].unique().shape[0],
            'alpha': 0.05,
            'significant': p*redim_df.loc[redim_df.coord.isin(redim_df.seed.unique()), 'coord'].unique().shape[0]<0.05,
            'd_prime': t/np.sqrt(group.mouse_id.unique().shape[0]),
            }
            selected_rois_stats_correct_vs_incorrect += [results]
        selected_rois_stats_correct_vs_incorrect = pd.DataFrame(selected_rois_stats_correct_vs_incorrect)
        selected_rois_stats_correct_vs_incorrect.to_csv(os.path.join(output_path, 'selected_rois_stats_correct_vs_incorrect.csv'))
    else:
        selected_rois_stats_correct_vs_incorrect=[]
    return all_rois_stats_to_0, all_rois_stats_correct_vs_incorrect, selected_rois_stats_to_0, selected_rois_stats_correct_vs_incorrect


def plot_connected_dot_r_from_images(mouse_avg, output_path):
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)
    if 'opto' in output_path:
        stim_list = ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']
    else:
        stim_list = ['(-5.0, 5.0)']

    total_df = []
    for stim in stim_list:
        group = mouse_avg.loc[mouse_avg.opto_stim_coord==stim]
        if 'opto' in output_path:
            group['correct_trial']=1

        for i, row in group.iterrows():
            redim, coords = reduce_im_dimensions(row['value'][np.newaxis])
            df = generate_reduced_image_df(redim, coords)
            df['context'] = row.context
            df['mouse_id'] = row.mouse_id
            df['correct_trial'] = row.correct_trial
            df['variable'] = row.variable
            df['opto_stim_coord'] = stim
            total_df+=[df]
            
    total_df = pd.concat(total_df).rename(columns={'dff0': 'value'})

    total_df['seed'] = total_df.apply(lambda x: x.variable.split("_")[0], axis=1)
    total_df['coord_order'] = total_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
    total_df = total_df.groupby(by=['mouse_id', 'context', 'opto_stim_coord', 'correct_trial', 'coord_order', 'seed']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r', 'coord_order': 'coord'})
    total_df = total_df.groupby(by=['mouse_id', 'opto_stim_coord', 'correct_trial', 'coord', 'seed']).apply(lambda x: x.loc[x.context==1, 'r'].values[0] - x.loc[x.context==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})

    total_df['y_dest'] = total_df.apply(lambda x: eval(x.coord)[0], axis=1)
    total_df['x_dest'] = total_df.apply(lambda x: eval(x.coord)[1], axis=1)
    total_df['y_source'] = total_df.apply(lambda x: eval(x.seed)[0], axis=1)
    total_df['x_source'] = total_df.apply(lambda x: eval(x.seed)[1], axis=1)

    for coord in total_df['opto_stim_coord'].unique():
        stats = pd.read_csv(os.path.join(output_path, 'from_images', [f"{coord}_stim" if 'opto' in output_path else ''][0], 'pairwise_selected_rois_stats.csv'))
        stats['norm_d'] = np.round(np.clip((stats.d_prime.values - 0.8)/(2 - 0.8), 0, 1), 2)

        for outcome in total_df.correct_trial.unique():
            if coord=="(-0.5, 0.5)":
                continue
            group = total_df[(total_df.correct_trial==outcome) & (total_df.opto_stim_coord==coord)]
            group = group[group.seed != coord]
            group = group[group.seed != "(-0.5, 0.5)"]
            if coord == '(-5.0, 5.0)':
                group['norm_r'] = np.round(np.clip((group.r.values- -0.03)/(0.03 - -0.03), 0, 1), 2)
            else:
                group['norm_r'] = np.round(np.clip((group.r.values- -0.1)/(0.1 - -0.1), 0, 1), 2)


            group = group[group.coord.isin(group.seed.unique())]

            fig, ax = plt.subplots(figsize=(4,4))
            fig.suptitle(f"{coord} stim r between rois")
            im=ax.scatter(group.loc[group.coord==group.seed, 'x_source'], group.loc[group.coord==group.seed, 'y_source'], s=100, c='k')
            if coord != '(-5.0, 5.0)':
                ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=100)

            ax.scatter(0, 0, marker='+', c='gray', s=100)
            for i, sub in group.groupby(by=['seed', 'coord']):
                # if stats.loc[(stats.correct_trial==outcome) & (stats.seed==i[0]) & (stats.coord==i[1]), 'significant'].values:
                #     ax.plot([sub.x_source.unique(), sub.x_dest.unique()], [sub.y_source.unique(), sub.y_dest.unique()], c=seismic_palette(sub.norm_r.mean()), linewidth=2)                if stats.loc[(stats.correct_trial==outcome) & (stats.seed==i[0]) & (stats.coord==i[1]), 'significant'].values:
                if stats.loc[(stats.correct_trial==outcome) & (stats.seed==i[0]) & (stats.coord==i[1]), 'd_prime'].values>=0.8:
                    d = stats.loc[(stats.correct_trial==outcome) & (stats.seed==i[0]) & (stats.coord==i[1]), 'd_prime'].values[0]
                    norm_d = stats.loc[(stats.correct_trial==outcome) & (stats.seed==i[0]) & (stats.coord==i[1]), 'norm_d'].values[0]
                    ax.plot([sub.x_source.unique(), sub.x_dest.unique()], [sub.y_source.unique(), sub.y_dest.unique()], c=seismic_palette(sub.norm_r.mean()), linewidth=d, alpha=norm_d)       

            ax.grid(True)
            ax.set_xticks(np.linspace(0.5,5.5,6))
            ax.set_xlim([-0.25, 6])
            ax.set_yticks(np.linspace(-3.5, 2.5,7))
            ax.set_ylim([-3.75, 2.75])
            ax.invert_xaxis()

            if 'opto' in output_path:
                save_path = os.path.join(output_path, 'from_images', f"{coord}_stim")
            else:
                save_path = os.path.join(output_path, 'from_images')

            fig.savefig(os.path.join(save_path, f'{"correct" if outcome==1 else "incorrect"}_r_summary_dprime.png'))
            fig.savefig(os.path.join(save_path, f'{"correct" if outcome==1 else "incorrect"}_r_summary_dprime.svg'))


def plot_connected_dot_r(pw_mouse_avg, output_path):
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)
    total_df = []
    if 'opto' in output_path:
        stim_list = ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']
    else:
        stim_list = ['(-5.0, 5.0)']

    for stim in stim_list:
        group = pw_mouse_avg.loc[pw_mouse_avg.opto_stim_coord==stim]
        if 'opto' in output_path:
            group['correct_trial']=1

        group['seed'] = group.apply(lambda x: x.variable.split("_")[0], axis=1)
        group = group.groupby(by=['mouse_id', 'context', 'correct_trial', 'opto_stim_coord', 'coord_order', 'seed']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r', 'coord_order': 'coord'})
        
        group = group.groupby(by=['mouse_id', 'opto_stim_coord', 'correct_trial', 'coord', 'seed']).apply(lambda x: np.nan_to_num(x.loc[x.context==1, 'r'].values[0]) - np.nan_to_num(x.loc[x.context==0, 'r'].values[0])).reset_index().rename(columns={0:'r'})
        total_df +=[group]

    total_df = pd.concat(total_df).reset_index(drop=True)
    total_df['y_dest'] = total_df.apply(lambda x: eval(x.coord)[0], axis=1)
    total_df['x_dest'] = total_df.apply(lambda x: eval(x.coord)[1], axis=1)
    total_df['y_source'] = total_df.apply(lambda x: eval(x.seed)[0], axis=1)
    total_df['x_source'] = total_df.apply(lambda x: eval(x.seed)[1], axis=1)

    for coord in total_df['opto_stim_coord'].unique():
        stats = pd.read_csv(os.path.join(output_path, [f"{stim}_stim" if 'opto' in output_path else ''][0], 'pairwise_selected_rois_stats.csv'))
        for outcome in total_df.correct_trial.unique():
            # if coord=="(-0.5, 0.5)":
            #     continue
            group = total_df[(total_df.correct_trial==outcome) & (total_df.opto_stim_coord==coord)]
            group = group[group.seed != coord]
            # group = group[group.seed != "(-0.5, 0.5)"]
            if coord == '(-5.0, 5.0)':
                group['norm_r'] = np.round(((group.r.values- -0.03)/(0.03 - -0.03)), 2)
            else:
                group['norm_r'] = np.round(((group.r.values- -0.1)/(0.1 - -0.1)), 2)

            group.loc[group.norm_r>1.0, 'norm_r'] = 1
            group.loc[group.norm_r<0, 'norm_r'] = 0
            group = group[group.coord.isin(group.seed.unique())]

            fig, ax = plt.subplots(figsize=(4,4))
            fig.suptitle(f"{coord} stim r between rois")
            im=ax.scatter(group.loc[group.coord==group.seed, 'x_source'], group.loc[group.coord==group.seed, 'y_source'], s=100, c='k')
            if coord != '(-5.0, 5.0)':
                ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=100)

            ax.scatter(0, 0, marker='+', c='gray', s=100)
            for i, sub in group.groupby(by=['seed', 'coord']):
                if stats.loc[(stats.correct_trial==outcome) & (stats.seed==i[0]) & (stats.coord==i[1]), 'significant'].values:
                    ax.plot([sub.x_source.unique(), sub.x_dest.unique()], [sub.y_source.unique(), sub.y_dest.unique()], c=seismic_palette(sub.norm_r.mean()), linewidth=2)
            ax.grid(True)
            ax.set_xticks(np.linspace(0.5,5.5,6))
            ax.set_xlim([-0.25, 6])
            ax.set_yticks(np.linspace(-3.5, 2.5,7))
            ax.set_ylim([-3.75, 2.75])
            ax.invert_xaxis()

            if 'opto' in output_path:
                save_path = os.path.join(output_path, f"{coord}_stim")
            else:
                save_path = output_path

            fig.savefig(os.path.join(save_path, f'{"correct" if outcome==1 else "incorrect"}_r_summary.png'))
            fig.savefig(os.path.join(save_path, f'{"correct" if outcome==1 else "incorrect"}_r_summary.svg'))


def plot_mouse_barplot_r(pw_mouse_avg, output_path):

    if 'opto' in output_path:
        stim_list = ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']
    else:
        stim_list = ['(-5.0, 5.0)']

    for stim in stim_list:
        group = pw_mouse_avg.loc[pw_mouse_avg.opto_stim_coord==stim]
        if 'opto' in output_path:
            group['correct_trial']=1

        group['seed'] = group.apply(lambda x: x.variable.split("_")[0], axis=1)
        group = group.groupby(by=['mouse_id', 'context', 'correct_trial', 'opto_stim_coord', 'coord_order', 'seed']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r', 'coord_order': 'coord'})
        
        all_rois_stats, all_rois_stats_correct_vs_incorrect, selected_rois_stats, selected_rois_stats_correct_vs_incorrect = compute_stats_barplot(group, output_path)

        group = group.groupby(by=['mouse_id', 'correct_trial', 'coord', 'seed']).apply(lambda x: x.loc[x.context==1, 'r'].values[0] - x.loc[x.context==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})
        
        ## Plot corrected correlation r substracted R+ - R-  with correct vs incorrect trial
        g = sns.catplot(
            x="coord",
            y="r",
            hue="correct_trial",
            hue_order=[1,0],
            palette=['#032b22', '#da4e02'],
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
        
        g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1,0], palette=['#032b22', '#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('R- <-- r-shuffle --> R+')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            ax.set_ylim([-0.15, 0.15])
            seed = ax.get_title('center').split("= ")[-1]
            stats = all_rois_stats_correct_vs_incorrect[all_rois_stats_correct_vs_incorrect.seed==seed]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()
        g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot.png'))
        g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot.svg'))

        g = sns.catplot(
            x="coord",
            y="r",
            hue="correct_trial",
            hue_order=[1,0],
            order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"],
            palette=['#032b22', '#da4e02'],
            col="seed",
            data=group.loc[group.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"])],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=1,
            alpha=0.5)
        
        g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1,0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"], palette=['#032b22', '#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
        g.set_ylabels('R- <-- r-shuffle --> R+')
        g.tick_params(axis='x', rotation=30)
        for ax in g.axes.flat:
            ax.set_ylim([-0.15, 0.15])
            seed = ax.get_title('center').split("= ")[-1]
            stats = selected_rois_stats_correct_vs_incorrect[selected_rois_stats_correct_vs_incorrect.seed==seed]
            ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        g.figure.tight_layout()
        g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_selected_rois.png'))
        g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_selected_rois.svg'))

        ## Plot corrected correlation r substracted R+ - R-  with correct trials
        g = sns.catplot(
            x="coord",
            y="r",
            palette=['#032b22'],
            row="seed",
            data=group.loc[group.correct_trial==1],
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
        g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_correct.png'))
        g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_correct.svg'))

        g = sns.catplot(
            x="coord",
            y="r",
            palette=['#032b22'],
            col="seed",
            order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"],
            data=group.loc[(group.correct_trial==1) & (group.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"]))],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=1,
            alpha=0.5)
        
        g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"], palette=['#032b22', '#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
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
        g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_selected_rois_correct.png'))
        g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_selected_rois_correct.svg'))

        if 'opto' not in output_path:
            ## Plot corrected correlation r substracted R+ - R-  with incorrect trials
            g = sns.catplot(
                x="coord",
                y="r",
                palette=['#da4e02'],
                row="seed",
                data=group.loc[group.correct_trial==0],
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
            g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_incorrect.png'))
            g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_incorrect.svg'))

            g = sns.catplot(
                x="coord",
                y="r",
                palette=['#da4e02'],
                col="seed",
                order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"],
                data=group.loc[(group.correct_trial==0) & (group.coord.isin(["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"]))],
                kind="bar",
                errorbar = ('ci', 95),
                edgecolor="black",
                errcolor="black",
                errwidth=1.5,
                capsize = 0.1,
                height=4,
                aspect=1,
                alpha=0.5)
            
            g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[0], order=["(-0.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)"], palette=['#da4e02'], dodge=True, alpha=0.6, ec='k', linewidth=1)
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
            g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_selected_rois_incorrect.png'))
            g.figure.savefig(os.path.join(output_path, 'r-shuffle_R+-R-_barplot_selected_rois_incorrect.svg'))


def compute_stats_barplot(df, output_path):
    df=df[df.coord!='(2.5, 5.5)']
    all_rois_stats =[]
    for name, group in df.groupby(by=['correct_trial', 'seed', 'coord']):
        # t, p = ttest_1samp(group.dff0, 0)
        t, p = ttest_rel(group.loc[group.context==1, 'r'].values, group.loc[group.context==0, 'r'].values)
        mean_diff = (group.loc[group.context==1, 'r'].values[0] - group.loc[group.context==0, 'r'].values[0])/group.loc[group.context==1, 'r'].shape[0]
        std_diff = np.std(mean_diff)

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
        #  'd_prime': abs(t/np.sqrt(df.mouse_id.unique().shape[0]))
        'd_prime': abs(mean_diff/std_diff)
         }
        
        all_rois_stats += [results]
    all_rois_stats = pd.DataFrame(all_rois_stats)
    all_rois_stats.to_csv(os.path.join(output_path, 'pairwise_all_rois_stats.csv'))

    if 'opto' not in output_path:
        all_rois_stats_correct_vs_incorrect =[]
        for name, group in df.groupby(by=['seed', 'coord']):
            context_diff = group.groupby(by=['mouse_id', 'correct_trial']).apply(lambda x: x.loc[x.context==1, 'r'].values[0] - x.loc[x.context==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})
            t, p = ttest_rel(context_diff.loc[context_diff.correct_trial==1, 'r'], context_diff.loc[context_diff.correct_trial==0, 'r'])
            mean_diff = (context_diff.loc[context_diff.correct_trial==1, 'r'].values[0] - context_diff.loc[context_diff.correct_trial==0, 'r'].values[0])/context_diff.loc[context_diff.correct_trial==0, 'r'].shape[0]
            std_diff = np.std(mean_diff)

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
            # 'd_prime': abs(t/np.sqrt(context_diff.mouse_id.unique().shape[0])),
            'd_prime': abs(mean_diff/std_diff)
            }
            all_rois_stats_correct_vs_incorrect += [results]
        all_rois_stats_correct_vs_incorrect = pd.DataFrame(all_rois_stats_correct_vs_incorrect)
        all_rois_stats_correct_vs_incorrect.to_csv(os.path.join(output_path, 'pairwise_all_rois_stats_correct_vs_incorrect.csv'))
    else:
        all_rois_stats_correct_vs_incorrect=[]

    df = df[df.seed!='(1.5, 3.5)']
    selected_rois_stats =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['correct_trial', 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group.context==1, 'r'].values, group.loc[group.context==0, 'r'])
        mean_diff = (group.loc[group.context==1, 'r'].values[0] - group.loc[group.context==0, 'r'].values[0])/group.loc[group.context==1, 'r'].shape[0]
        std_diff = np.std(mean_diff)

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
        #  'd_prime': abs(t/np.sqrt(df.mouse_id.unique().shape[0]))
         'd_prime': abs(mean_diff/std_diff)

         }
        
        selected_rois_stats += [results]
    selected_rois_stats = pd.DataFrame(selected_rois_stats)
    selected_rois_stats.to_csv(os.path.join(output_path, 'pairwise_selected_rois_stats.csv'))

    if 'opto' not in output_path:
        selected_rois_stats_correct_vs_incorrect =[]
        for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['seed', 'coord']):
            context_diff = group.groupby(by=['mouse_id', 'correct_trial']).apply(lambda x: x.loc[x.context==1, 'r'].values[0] - x.loc[x.context==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})
            t, p = ttest_rel(context_diff.loc[context_diff.correct_trial==1, 'r'], context_diff.loc[context_diff.correct_trial==0, 'r'])
            mean_diff = (context_diff.loc[context_diff.correct_trial==1, 'r'].values[0] - context_diff.loc[context_diff.correct_trial==0, 'r'].values[0])/context_diff.loc[context_diff.correct_trial==1, 'r'].shape[0]
            std_diff = np.std(mean_diff)
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
            # 'd_prime': abs(t/np.sqrt(context_diff.mouse_id.unique().shape[0])),
            'd_prime': abs(mean_diff/std_diff)
            }
            selected_rois_stats_correct_vs_incorrect += [results]
        selected_rois_stats_correct_vs_incorrect = pd.DataFrame(selected_rois_stats_correct_vs_incorrect)
        selected_rois_stats_correct_vs_incorrect.to_csv(os.path.join(output_path, 'pairwise_selected_rois_stats_correct_vs_incorrect.csv'))
    else:
        selected_rois_stats_correct_vs_incorrect=[]
    return all_rois_stats, all_rois_stats_correct_vs_incorrect, selected_rois_stats, selected_rois_stats_correct_vs_incorrect



def main(data, pw_data, output_path):
    ## plot
    if 'opto' not in output_path:
        # data.trial_count = data.trial_count.map({0: 1, 1: 2, 2: 3, 3: 4, 4: -4, 5: -3, 6: -2, 7: -1})
        data['opto_stim_coord'] = '(-5.0, 5.0)'
        data.value = data.apply(lambda x: x.value[0] if 'percentile' not in x.variable else x.value, axis=1)
        mouse_avg = data.groupby(by=['mouse_id', 'opto_stim_coord', 'context', 'correct_trial', 'variable'])['value'].apply(
            lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
        mouse_avg['value'] = mouse_avg['value'].apply(lambda x: np.array(x).reshape(125, -1))

        total_avg = mouse_avg.groupby(by=['context', 'correct_trial', 'variable'])['value'].apply(
            lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()

        # plot total avg
        # for roi in ['(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)']:

        #     print(f"Plotting total averages for roi {roi}")
        #     save_path = os.path.join(output_path, roi)
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     plot_avg_between_blocks(total_avg.loc[total_avg.correct_trial==1], roi, save_path)
        #     plot_reduced_correlations(total_avg.loc[total_avg.correct_trial==1], roi, save_path)
        #     save_path = os.path.join(output_path, 'incorrect', roi)
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     plot_avg_between_blocks(total_avg.loc[total_avg.correct_trial==0], roi, save_path)
        #     plot_reduced_correlations(total_avg.loc[total_avg.correct_trial==0], roi, save_path)

        # plot_mouse_barplot_r_from_images(mouse_avg, output_path)
        plot_connected_dot_r_from_images(mouse_avg, output_path)

        # pw_data['opto_stim_coord'] = '(-5.0, 5.0)'
        # pw_mouse_avg = pw_data.groupby(by=['mouse_id', 'context', 'opto_stim_coord', 'correct_trial', 'coord_order', 'variable'])['value'].apply(
        #     lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
      
        # plot_mouse_barplot_r(pw_mouse_avg, output_path)
        # plot_connected_dot_r(pw_mouse_avg, output_path)

    else:
        data.value = data.apply(lambda x: x.value[0] if 'percentile' not in x.variable else x.value, axis=1)
        mouse_avg = data.groupby(by=['mouse_id', 'context', 'opto_stim_coord', 'variable'])['value'].apply(
            lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
        mouse_avg['value'] = mouse_avg['value'].apply(lambda x: np.array(x).reshape(125, -1))

        total_avg = mouse_avg.groupby(by=['context', 'opto_stim_coord', 'variable'])['value'].apply(
            lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
        
        # for coord, group in total_avg.groupby('opto_stim_coord'):
        for coord in ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-2.5, 5.5)', '(-2.5, 1.5)']:
            if coord =='(-5.0, 5.0)':
                continue
            group = total_avg.loc[total_avg.opto_stim_coord==coord]
            # for roi in ['(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)']:#'A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2', 'RSC'
            #     print(f"Plotting total averages for roi {roi}, stim coord {coord}")
            #     save_path = os.path.join(output_path, f"{coord}_stim", f"{roi}_seed")
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     if coord=='(-5.0, 5.0)':
            #         plot_avg_between_blocks(group, roi, save_path)
            #     else:
            #         plot_avg_between_blocks(group, roi, save_path, vmin=-0.1, vmax=0.1)

        plot_mouse_barplot_r_from_images(mouse_avg, output_path)
        plot_connected_dot_r_from_images(mouse_avg, output_path)

        # pw_data['opto_stim_coord'] = '(-5.0, 5.0)'
        # pw_mouse_avg = pw_data.groupby(by=['mouse_id', 'context', 'opto_stim_coord', 'correct_trial', 'coord_order', 'variable'])['value'].apply(
        #     lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
        # pw_mouse_avg['value'] = mouse_avg['value'].apply(lambda x: np.array(x).reshape(125, -1))

        # pw_avg = pw_mouse_avg.groupby(by=['context', 'opto_stim_coord', 'variable'])['value'].apply(
        #     lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
        
        # # for coord, group in mouse_avg.groupby('opto_stim_coord'):
        # for coord in ['(-5.0, 5.0)', '(-0.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)']:
        #     group = mouse_avg.loc[mouse_avg.opto_stim_coord==coord]
        #     context_diff = group.groupby(by=['mouse_id', 'variable']).apply(lambda x: np.nan_to_num(x.loc[x.context==1, 'value']) - np.nan_to_num(x.loc[x.context==0, 'value'])).reset_index().rename(columns={0:'value'})
        #     context_diff['correct_trial']=1
        #     save_path = os.path.join(output_path, f"{coord}_stim")
        #     plot_mouse_barplot_r(context_diff, save_path)


if __name__ == "__main__":

    # root = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_trial_based_corr_mar2025"
    # root = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_correlation_opto_wf"
    root = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_correlations_20250604"
    root = haas_pathfun(root)
    for dataset in os.listdir(root):
        if '25_75' in dataset:
            continue
        if 'opto' not in dataset:
            continue

        print(f"Analyzing {dataset}")

        result_folder = os.path.join(root, dataset)
        result_folder = haas_pathfun(result_folder)
        if not os.path.exists(os.path.join(result_folder, 'results')):
            os.makedirs(os.path.join(result_folder, 'results'), exist_ok=True)

        load_data = True

        if load_data==True and os.path.exists(os.path.join(result_folder, 'results', "combined_avg_correlation_results.json")):
            data = pd.read_json(os.path.join(result_folder, 'results',"combined_avg_correlation_results.json"))
            data['value'] = data.value.apply(lambda x: np.asarray(x, dtype=float))
            pw_data = pd.read_json(os.path.join(result_folder, 'results',"combined_pw_correlation_results.json"))
            pw_data['value'] = pw_data.value.apply(lambda x: np.asarray(x, dtype=float))

            print(f'{dataset} results loaded')

        else:
            # data = []
            # all_files = glob.glob(os.path.join(result_folder, "**", "*", "correlation_table.parquet.gzip"))

            # for file in tqdm(all_files):
            #     session_data = preprocess_corr_results(file)
            #     data += [session_data]

            # data = pd.concat(data, axis=0, ignore_index=True)
            # if not os.path.exists(os.path.join(result_folder, 'results')):
            #     os.makedirs(os.path.join(result_folder, 'results'))

            # data.to_json(os.path.join(result_folder, 'results', "combined_avg_correlation_results.json"))
            data = pd.read_json(os.path.join(result_folder, 'results',"combined_avg_correlation_results.json"))
            data['value'] = data.value.apply(lambda x: np.asarray(x, dtype=float))
            
            pw_data = []
            all_files = glob.glob(os.path.join(result_folder, "**", "*", "pairwise_correlation.parquet.gzip"))
            for file in tqdm(all_files):
                session_data = preprocess_pw_results(file)
                pw_data += [session_data]

            pw_data = pd.concat(pw_data, axis=0, ignore_index=True)
            pw_data.to_json(os.path.join(result_folder, 'results', "combined_pw_correlation_results.json"))
 
        main(data, pw_data, os.path.join(result_folder, 'results'))
