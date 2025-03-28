import os
import sys
sys.path.append(os.getcwd())
import yaml
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.haas_utils import *
from utils.wf_plotting_utils import plot_single_frame, reduce_im_dimensions, plot_grid_on_allen, generate_reduced_image_df


def preprocess_corr_results(file):

    df = pd.read_parquet(file.replace("\\", "/"))
    df['block_id'] = np.abs(np.diff(df.context.values, prepend=0)).cumsum()
    df['trial_count'] = np.empty(len(df), dtype=int)
    df.loc[df.trial_type == 'whisker_trial', 'trial_count'] = df.loc[df.trial_type == 'whisker_trial'].groupby(
        'block_id').cumcount()
    df.loc[df.trial_type == 'auditory_trial', 'trial_count'] = df.loc[
        df.trial_type == 'auditory_trial'].groupby(
        'block_id').cumcount()
    df.loc[df.trial_type == 'no_stim_trial', 'trial_count'] = df.loc[df.trial_type == 'no_stim_trial'].groupby(
        'block_id').cumcount()

    df = df.loc[df.trial_type=='whisker_trial'].melt(id_vars=['mouse_id', 'session_id', 'context', 'context_background', 'block_id', 'trial_count'],
            value_vars=['A1_r', 'A1_shuffle_mean', 'A1_shuffle_std', 'A1_percentile', 'A1_nsigmas', 
                        'ALM_r', 'ALM_shuffle_mean', 'ALM_shuffle_std', 'ALM_percentile', 'ALM_nsigmas',
                        'RSC_r', 'RSC_shuffle_mean', 'RSC_shuffle_std', 'RSC_percentile', 'RSC_nsigmas', 
                        'tjM1_r', 'tjM1_shuffle_mean', 'tjM1_shuffle_std', 'tjM1_percentile', 'tjM1_nsigmas', 
                        'tjS1_r', 'tjS1_shuffle_mean', 'tjS1_shuffle_std', 'tjS1_percentile', 'tjS1_nsigmas', 
                        'wM1_r', 'wM1_shuffle_mean', 'wM1_shuffle_std', 'wM1_percentile', 'wM1_nsigmas',
                        'wM2_r', 'wM2_shuffle_mean', 'wM2_shuffle_std', 'wM2_percentile', 'wM2_nsigmas', 
                        'wS1_r', 'wS1_shuffle_mean', 'wS1_shuffle_std', 'wS1_percentile', 'wS1_nsigmas', 
                        'wS2_r', 'wS2_shuffle_mean', 'wS2_shuffle_std', 'wS2_percentile', 'wS2_nsigmas'])

    # df['value'] = df['value'].apply(lambda x: np.asarray(x, dtype=float))

    avg_df = df.groupby(by=['mouse_id', 'session_id', 'context', 'trial_count', 'variable'])[
        'value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    return avg_df


def plot_avg_between_blocks(df, roi, save_path):
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
                        colormap=seismic_palette, vmin=-0.03, vmax=0.03, norm=False, fig=fig, ax=ax[2])
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
                        colormap=seismic_palette, vmin=-0.03, vmax=0.03, norm=False, fig=fig, ax=ax[2])
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
                        colormap=seismic_palette, vmin=-0.03, vmax=0.03, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_std.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"R - shuffle")

    im_r = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]
    im_nor = total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]

    plot_single_frame(im_r,
                        title='Rewarded',
                        colormap='icefire', vmin=-0.2, vmax=0.2, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(im_nor,
                        title='Non-Rewarded',
                        colormap='icefire', vmin=-0.2, vmax=0.2, norm=False, fig=fig, ax=ax[1])

    plot_single_frame(im_r - im_nor, title='R+ - R-',
                        colormap=seismic_palette, vmin=-0.03, vmax=0.03, norm=False, fig=fig, ax=ax[2])
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_avg.png"))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")
    plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Rewarded',
                        colormap='icefire', vmin=-1, vmax=1, norm=False, fig=fig, ax=ax[0])
    plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                        title='Non-Rewarded',
                        colormap='icefire', vmin=-1.5, vmax=1.5, norm=False, fig=fig, ax=ax[1])
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


if __name__ == "__main__":

    root = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_trial_based_corr_mar2025"
    root = haas_pathfun(root)
    for dataset in os.listdir(root):
        result_folder = fr"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_trial_based_corr_mar2025/{dataset}"
        result_folder = haas_pathfun(result_folder)
        if not os.path.exists(os.path.join(result_folder, 'results')):
            os.makedirs(os.path.join(result_folder, 'results'), exist_ok=True)

        load_data = True

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
            data.to_json(os.path.join(result_folder, 'results', "combined_avg_correlation_results.json"))

        ## plot
        data.trial_count = data.trial_count.map({0: 1, 1: 2, 2: 3, 3: 4, 4: -4, 5: -3, 6: -2, 7: -1})
        data.value = data.apply(lambda x: x.value[0] if 'percentile' not in x.variable else x.value, axis=1)
        mouse_avg = data.groupby(by=['mouse_id', 'context', 'trial_count', 'variable'])['value'].apply(
            lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()
        mouse_avg['value'] = mouse_avg['value'].apply(lambda x: np.array(x).reshape(125, -1))

        total_avg = mouse_avg.groupby(by=['context', 'trial_count', 'variable'])['value'].apply(
            lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()


        ## plot total avg
        for roi in ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2', 'RSC']:
            print(f"Plotting total averages for roi {roi}")
            save_path = os.path.join(result_folder, 'results', roi)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # plot_avg_between_blocks(total_avg, roi, save_path)
            # plot_trial_based_correlations(total_avg, roi, save_path=save_path)
            plot_reduced_correlations(total_avg, roi, save_path)
            plot_trial_based_correlations_reduced(total_avg, roi, save_path=save_path)

    ## plot mouse avg
    # for state in ['naive', 'expert']:
    #     print(f"Plotting single mouse data for state {state}")
    #     for mouse in tqdm(mouse_avg.loc[mouse_avg.state==state, 'mouse_id'].unique()):
    #         for roi in ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2', 'RSC']:
    #             save_path = os.path.join(result_folder, state, mouse, roi)
    #             if not os.path.exists(save_path):
    #                 os.makedirs(save_path)
    #         plot_avg_between_blocks(mouse_avg.loc[(mouse_avg.state==state) & (mouse_avg.mouse_id==mouse)], roi, save_path)
    #         plot_trial_based_correlations(mouse_avg.loc[(mouse_avg.state==state) & (mouse_avg.mouse_id==mouse)], roi, save_path=save_path)