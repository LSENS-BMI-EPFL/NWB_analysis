import os
import yaml
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.wf_plotting_utils import plot_single_frame


def preprocess_corr_results(file):

    df = pd.read_json(file.replace("\\", "/"))
    df['state'] = state
    df['block_id'] = np.abs(np.diff(df.context.values, prepend=0)).cumsum()
    df['trial_count'] = np.empty(len(df), dtype=int)
    df.loc[df.trial_type == 'whisker_trial', 'trial_count'] = df.loc[df.trial_type == 'whisker_trial'].groupby(
        'block_id').cumcount()
    df.loc[df.trial_type == 'auditory_trial', 'trial_count'] = df.loc[
        df.trial_type == 'auditory_trial'].groupby(
        'block_id').cumcount()
    df.loc[df.trial_type == 'no_stim_trial', 'trial_count'] = df.loc[df.trial_type == 'no_stim_trial'].groupby(
        'block_id').cumcount()

    df = df.loc[df.trial_type=='whisker_trial'].melt(id_vars=['mouse_id', 'session_id', 'state', 'context', 'context_background', 'block_id', 'trial_count'],
            value_vars=['A1_r', 'A1_all_trial_shuffle', 'A1_block_shuffle', 'A1_within_block_shuffle',
                        'ALM_r', 'ALM_all_trial_shuffle', 'ALM_block_shuffle','ALM_within_block_shuffle',
                        'tjM1_r', 'tjM1_all_trial_shuffle', 'tjM1_block_shuffle', 'tjM1_within_block_shuffle',
                        'tjS1_r', 'tjS1_all_trial_shuffle', 'tjS1_block_shuffle', 'tjS1_within_block_shuffle',
                        'wM1_r', 'wM1_all_trial_shuffle', 'wM1_block_shuffle', 'wM1_within_block_shuffle',
                        'wM2_r', 'wM2_all_trial_shuffle', 'wM2_block_shuffle', 'wM2_within_block_shuffle',
                        'wS1_r', 'wS1_all_trial_shuffle', 'wS1_block_shuffle', 'wS1_within_block_shuffle',
                        'wS2_r', 'wS2_all_trial_shuffle', 'wS2_block_shuffle', 'wS2_within_block_shuffle'])

    df['value'] = df['value'].apply(lambda x: np.asarray(x, dtype=float))

    avg_df = df.groupby(by=['mouse_id', 'session_id', 'state', 'context', 'trial_count', 'variable'])[
        'value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    return avg_df


def plot_avg_between_blocks(df, roi, save_path):
    total_avg = df.groupby(by=['context', 'variable'])['value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    data_types =[f'{roi}_r', f"{roi}_all_trial_shuffle", f"{roi}_block_shuffle", f"{roi}_within_block_shuffle"]
    for data in data_types:
        fig, ax = plt.subplots(1, 3, figsize=(8, 4))
        fig.suptitle(f"{data} block average")
        plot_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == data), 'value'].values[0],
                          title='Rewarded',
                          colormap='plasma', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
        plot_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == data), 'value'].values[0],
                          title='Non-Rewarded',
                          colormap='plasma', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])
        im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == data), 'value'].values[0] - \
             total_avg.loc[(total_avg.context == 0) & (total_avg.variable == data), 'value'].values[0]

        plot_single_frame(im, title='R+ - R-',
                          colormap='seismic', vmin=-0.1, vmax=0.1, norm=False, fig=fig, ax=ax[2])
        fig.savefig(os.path.join(save_path, f"{data}_block_avg.png"))

    data_types = [f"{roi}_all_trial_shuffle", f"{roi}_block_shuffle", f"{roi}_within_block_shuffle"]
    for data in data_types:
        fig, ax = plt.subplots(1, 3, figsize=(8, 4))
        fig.suptitle(f"R - {data} block average")

        im_r = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
             total_avg.loc[(total_avg.context == 1) & (total_avg.variable == data), 'value'].values[0]
        im_nor = total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
             total_avg.loc[(total_avg.context == 0) & (total_avg.variable == data), 'value'].values[0]

        plot_single_frame(im_r,
                          title='Rewarded',
                          colormap='plasma', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[0])
        plot_single_frame(im_nor,
                          title='Non-Rewarded',
                          colormap='plasma', vmin=-0.5, vmax=0.5, norm=False, fig=fig, ax=ax[1])

        plot_single_frame(im_r - im_nor, title='R+ - R-',
                          colormap='seismic', vmin=-0.1, vmax=0.1, norm=False, fig=fig, ax=ax[2])
        fig.savefig(os.path.join(save_path, f"R-{data}_block_avg.png"))


def plot_trial_based_correlations(df, roi, save_path):
    R = df.loc[(df.variable == f'{roi}_r')]
    all_trial_shuffle = df.loc[(df.variable == f'{roi}_all_trial_shuffle')]
    block_shuffle = df.loc[(df.variable == f'{roi}_block_shuffle')]
    within_block_shuffle = df.loc[(df.variable == f'{roi}_within_block_shuffle')]

    fig_r, ax_r = plt.subplots(3, 8, figsize=(20, 15))
    fig_r.suptitle(f"{roi} R in whisker trials before-after context")

    fig_t, ax_t = plt.subplots(3, 8, figsize=(20, 15))
    fig_t.suptitle(f"{roi} R-all_trial_shuffle in whisker trials before-after context")

    fig_b, ax_b = plt.subplots(3, 8, figsize=(20, 15))
    fig_b.suptitle(f"{roi} R-block_shuffle in whisker trials before-after context")

    fig_w, ax_w = plt.subplots(3, 8, figsize=(20, 15))
    fig_w.suptitle(f"{roi} R-within_block_shuffle in whisker trials before-after context")

    for i, count in enumerate(R.trial_count.unique()):
        for c in [1, 0]:
            plot_single_frame(R.loc[(R.context == c) & (R.trial_count == count), 'value'].values[0], title=str(count),
                              colormap='plasma', vmin=-1, vmax=1, norm=False, fig=fig_r, ax=ax_r[c, i])

            image = R.loc[(R.context == c) & (R.trial_count == count), 'value'].values[0] - all_trial_shuffle.loc[
                (all_trial_shuffle.context == c) & (all_trial_shuffle.trial_count == count), 'value'].values[0]
            plot_single_frame(image, title=str(count), colormap='plasma', vmin=-0.5, vmax=0.5, norm=False, fig=fig_t,
                              ax=ax_t[c, i])

            image = R.loc[(R.context == c) & (R.trial_count == count), 'value'].values[0] - block_shuffle.loc[
                (block_shuffle.context == c) & (block_shuffle.trial_count == count), 'value'].values[0]
            plot_single_frame(image, title=str(count), colormap='plasma', vmin=-0.5, vmax=0.5, norm=False, fig=fig_b,
                              ax=ax_b[c, i])

            image = R.loc[(R.context == c) & (R.trial_count == count), 'value'].values[0] - within_block_shuffle.loc[
                (within_block_shuffle.context == c) & (within_block_shuffle.trial_count == count), 'value'].values[0]
            plot_single_frame(image, title=str(count), colormap='seismic', vmin=-0.5, vmax=0.5, norm=False, fig=fig_w,
                              ax=ax_w[c, i])

        R_dif = R.loc[(R.context == 1) & (R.trial_count == count), 'value'].values[0] - \
                R.loc[(R.context == 0) & (R.trial_count == count), 'value'].values[0]
        plot_single_frame(R.loc[(R.context == c) & (R.trial_count == count), 'value'].values[0], title=str(count),
                          colormap='seismic', vmin=-1, vmax=1, norm=False, fig=fig_r, ax=ax_r[2, i])

        image_rew = R.loc[(R.context == 1) & (R.trial_count == count), 'value'].values[0] - all_trial_shuffle.loc[
            (all_trial_shuffle.context == 1) & (all_trial_shuffle.trial_count == count), 'value'].values[0]
        image_no_rew = R.loc[(R.context == 0) & (R.trial_count == count), 'value'].values[0] - all_trial_shuffle.loc[
            (all_trial_shuffle.context == 0) & (all_trial_shuffle.trial_count == count), 'value'].values[0]
        plot_single_frame(image_rew - image_no_rew, title=str(count), colormap='seismic', vmin=-0.1, vmax=0.1,
                          norm=False, fig=fig_t, ax=ax_t[2, i])

        image_rew = R.loc[(R.context == 1) & (R.trial_count == count), 'value'].values[0] - block_shuffle.loc[
            (block_shuffle.context == 1) & (block_shuffle.trial_count == count), 'value'].values[0]
        image_no_rew = R.loc[(R.context == 0) & (R.trial_count == count), 'value'].values[0] - block_shuffle.loc[
            (block_shuffle.context == 0) & (block_shuffle.trial_count == count), 'value'].values[0]
        plot_single_frame(image_rew - image_no_rew, title=str(count), colormap='seismic', vmin=-0.1, vmax=0.1,
                          norm=False, fig=fig_b, ax=ax_b[2, i])

        image_rew = R.loc[(R.context == 1) & (R.trial_count == count), 'value'].values[0] - within_block_shuffle.loc[
            (within_block_shuffle.context == 1) & (within_block_shuffle.trial_count == count), 'value'].values[0]
        image_no_rew = R.loc[(R.context == 0) & (R.trial_count == count), 'value'].values[0] - within_block_shuffle.loc[
            (within_block_shuffle.context == 0) & (within_block_shuffle.trial_count == count), 'value'].values[0]
        plot_single_frame(image_rew - image_no_rew, title=str(count), colormap='seismic', vmin=-0.1, vmax=0.1,
                          norm=False, fig=fig_w, ax=ax_w[2, i])
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(save_path, f'{roi}_r'))

    fig_t.tight_layout(pad=0.05)
    fig_t.savefig(os.path.join(save_path, f'{roi}_all_trial_shuffle.png'))
    fig_b.tight_layout(pad=0.05)
    fig_b.savefig(os.path.join(save_path, f'{roi}_block_shuffle.png'))
    fig_w.tight_layout(pad=0.05)
    fig_w.savefig(os.path.join(save_path, f'{roi}_within_block_shuffle.png'))


if __name__ == "__main__":

    result_folder = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_trial_based_correlation_jrgeco"
    if not os.path.exists(os.path.join(result_folder)):
        os.makedirs(os.path.join(result_folder), exist_ok=True)

    load_data = True

    if load_data==True and os.path.exists(os.path.join(result_folder, "combined_avg_correlation_results.json")):
        data = pd.read_json(os.path.join(result_folder, "combined_avg_correlation_results.json"))
        data['value'] = data.value.apply(lambda x: np.asarray(x, dtype=float))
        print('Results loaded')

    else:
        data = []
        for state in ['naive', 'expert']:
            data_folder = fr"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_cross_correlation_jrgeco_{state}"
            all_files = glob.glob(os.path.join(data_folder, "**", "*", "cross_corr_results_trial_based.json"))

            for file in tqdm(all_files):
                session_data = preprocess_corr_results(file)
                data += [session_data]

        data = pd.concat(data, axis=0, ignore_index=True)
        data.to_json(os.path.join(result_folder, "combined_avg_correlation_results.json"))

    ## plot
    data.trial_count = data.trial_count.map({0: 1, 1: 2, 2: 3, 3: 4, 4: -4, 5: -3, 6: -2, 7: -1})
    mouse_avg = data.groupby(by=['mouse_id', 'state', 'context', 'trial_count', 'variable'])['value'].apply(
        lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()
    mouse_avg['value'] = mouse_avg['value'].apply(lambda x: np.array(x).reshape(125, -1))

    total_avg = mouse_avg.groupby(by=['state', 'context', 'trial_count', 'variable'])['value'].apply(
        lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()


    ## plot total avg
    for state in ['expert']:#'naive',
        for roi in ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2']:#['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2']:
            print(f"Plotting total averages for state {state} and roi {roi}")
            save_path = os.path.join(result_folder, state, roi)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plot_avg_between_blocks(total_avg.loc[total_avg.state==state], roi, save_path)
            plot_trial_based_correlations(total_avg.loc[total_avg.state==state], roi, save_path=save_path)


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