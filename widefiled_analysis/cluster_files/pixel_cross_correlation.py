import os
import sys
sys.path.append("/home/bechvila/NWB_analysis")

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nwb_wrappers.nwb_reader_functions as nwb_read

from tqdm import tqdm
from utils.haas_utils import *
from nwb_utils import utils_misc
from scipy.signal import correlate2d, correlation_lags, butter, filtfilt
from numba import njit, prange


def highpass_filter(data, cutoff=5, fs=100, order=4, axis=-1):
    """
    Apply a high-pass Butterworth filter to a 2D array along a specified axis.

    Parameters:
    - data: 2D NumPy array (shape: [n_channels, n_samples])
    - cutoff: High-pass filter cutoff frequency (Hz)
    - fs: Sampling frequency (Hz)
    - order: Order of the Butterworth filter

    Returns:
    - Filtered 2D array
    """
    # Design high-pass Butterworth filter
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # Apply filter along axis=1
    return filtfilt(b, a, data, axis=1)


def get_roi_frames_by_epoch(nwb_file, n_trials, rrs_keys, rrs_ts, start, end):
    # Get activity and brain region ROIs
    rrs_array = nwb_read.get_roi_response_serie_data(nwb_file, keys=rrs_keys)
    rrs_cell_type_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file, keys=rrs_keys)
    n_cell_types = len(rrs_cell_type_dict.keys())

    # Get all quiet windows
    quiet_window_frames = []
    for trial in range(n_trials):
        start_frame = utils_misc.find_nearest(rrs_ts, start[trial])
        end_frame = utils_misc.find_nearest(rrs_ts, end[trial])
        quiet_window_frames.append(np.arange(start_frame, end_frame))

    # Get average activity in each ROIs for each trial
    data_roi = np.zeros((n_trials, n_cell_types, 200))
    for trial in range(n_trials):
        rrs_quiet = rrs_array[:, quiet_window_frames[trial]]
        if rrs_quiet.shape[1] > 200:
            rrs_quiet = rrs_quiet[:, :200]
        elif rrs_quiet.shape[1] == 199:
            rrs_quiet = np.pad(rrs_quiet, (0,1), 'edge')
            
        rrs_quiet_filt = highpass_filter(rrs_quiet, cutoff=5, fs=100, order=4, axis=-1)
        # rrs_quiet_filt = filtfilt(b, a, rrs_quiet, axis=-1)
        data_roi[trial, :, :] = rrs_quiet_filt

    return rrs_cell_type_dict, data_roi


def get_frames_by_epoch(nwb_file, n_trials, wf_timestamps, start=-200, stop=200):
    frames = []

    for trial in range(n_trials):
        start_frame = utils_misc.find_nearest(wf_timestamps, start[trial])
        end_frame = utils_misc.find_nearest(wf_timestamps, stop[trial])
        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], start_frame, end_frame)
        if data.shape[0] > 200:
            data = data[:200, :]
        elif data.shape[0] == 199:
            data = np.pad(data, (1,0, 0), 'edge')
        
        data_filt = highpass_filter(data.reshape(200,-1).T, cutoff=5, fs=100, order=4, axis=-1)
        frames.append(data_filt)

    data_frames = np.array(frames)
    # data_frames = np.stack(data_frames, axis=0)
    return data_frames


def plot_corr_results(df, result_path, show=True):
    agg_df = df.groupby(by=['mouse_id', 'roi', 'context'])['r'].apply(lambda x: np.nanmean(np.stack(x), axis=0).squeeze()).reset_index()
    total_agg_df = agg_df.groupby(by=['roi', 'context'])['r'].apply(lambda x: np.nanmean(np.stack(x), axis=0).squeeze()).reset_index()

    for roi, group in total_agg_df.groupby('roi'):
        make_figures(group, roi, result_path=os.path.join(result_path, f'{roi}_all_avg_crosscorr'), show=False)

    for (mouse_id, roi), group in agg_df.groupby(by=['mouse_id', 'roi']):
        save_path = os.path.join(result_path, mouse_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        make_figures(group, f'{mouse_id} {roi}', result_path=os.path.join(save_path, f'{mouse_id}_{roi}_avg_crosscorr'), show=False)

    return


def make_figures(data, title, result_path, save=True, show=True):
    from utils.wf_plotting_utils import plot_single_frame

    fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    fig.suptitle(title)
    plot_single_frame(data.loc[data.context == 'Rewarded', 'r'].to_numpy()[0].reshape(125, 160), 'Rewarded',
                      colormap='plasma', fig=fig, ax=ax[0], vmin=-0.75, vmax=0.75, show=False)
    plot_single_frame(data.loc[data.context == 'Non-Rewarded', 'r'].to_numpy()[0].reshape(125, 160), 'Non-Rewarded',
                      colormap='plasma', fig=fig, ax=ax[1], vmin=-0.75, vmax=0.75, show=False)
    plot_single_frame(data.loc[data.context == 'Rewarded', 'r'].to_numpy()[0].reshape(125, 160) -
                      data.loc[data.context == 'Non-Rewarded', 'r'].to_numpy()[0].reshape(125, 160),
                      'Rewarded - Non-Rewarded', colormap='seismic', fig=fig, ax=ax[2], vmin=-0.25, vmax=0.25,
                      show=False)
    for subax in ax:
        subax.spines['right'].set_visible(False)
        subax.spines['top'].set_visible(False)
        subax.spines['left'].set_visible(False)
        subax.spines['bottom'].set_visible(False)

    if show:
        fig.show()
    if save:
        fig.savefig(result_path + ".png")
        # fig.savefig(result_path + ".svg")
    return


@njit()
def correlate(x, y):
    return np.corrcoef(x, y)


@njit(parallel=True)
def compute_corr_numpy(template, target, r):

    for trial in prange(template.shape[0]):
        for px in range(target.shape[0]):
            r[trial, px] = correlate(template[trial, :], target[px, trial, :])[1, 0]

    return r


def trial_based_correlation(mouse_id, session_id, trial_table, dict_roi, data_roi, data_frames, output_path):

    for roi in dict_roi.keys():
        row = dict_roi[roi][0]

        print(f'Computing {roi} correlation')
        template = data_roi[:, row, :]

        target = np.rollaxis(data_frames, axis=1)

        # Correlate data
        corr = compute_corr_numpy(template, target, r=np.zeros([template.shape[0], target.shape[0]]))
        trial_table[f'{roi}_r'] = [[corr[im]] for im in range(corr.shape[0])]

        # Shuffle blocks
        block_shuffle = []
        for i in range(10):
            if 'COMPUTERNAME' not in os.environ.keys():
                if i % 100 == 0:
                    output = f"Block shuffle {i} iterations"
                    os.system("echo " + output)

            block_id = np.abs(np.diff(trial_table.context.values, prepend=0)).cumsum()
            shuffle = np.hstack([np.where(block_id == block) for block in np.random.permutation(np.unique(block_id))])[0]
            block_shuffle += [compute_corr_numpy(template[shuffle], target, r=np.zeros([template.shape[0], target.shape[0]]))]

        block_shuffle = np.stack(block_shuffle)
        np.save(Path(output_path, f"{roi}_shuffle.npy"), block_shuffle)
        
        shuffle_mean = np.nanmean(block_shuffle, axis=0)
        trial_table[f'{roi}_shuffle_mean'] = [[shuffle_mean[im]] for im in range(shuffle_mean.shape[0])]

        shuffle_std = np.nanstd(block_shuffle, axis=0)
        trial_table[f'{roi}_shuffle_std'] = [[shuffle_std[im]] for im in range(shuffle_std.shape[0])]

        percentile = []
        for i, row in trial_table.iterrows():
            percentile += [np.sum(row[f'{roi}_r'][0] >= block_shuffle[:, i, :], axis=0) / block_shuffle.shape[0]]
        trial_table[f"{roi}_percentile"] = percentile
        
        n_sigmas = (corr - shuffle_mean)/shuffle_std
        trial_table[f"{roi}_nsigmas"] = [[n_sigmas[im]] for im in range(n_sigmas.shape[0])]


    trial_table['mouse_id'] = mouse_id
    trial_table['session_id'] = session_id

    trial_table.to_parquet(path=Path(output_path, "correlation_table.parquet.gzip"), compression='gzip')

    return trial_table


def context_block_correlation(mouse_id, session_id, trial_table, dict_roi, data_roi, data_frames):
    df =[]
    for roi in dict_roi.keys():
        row = dict_roi[roi][0]
        for idx, context in enumerate(['Non-Rewarded', 'Rewarded']):

            print(f'Computing {roi} {context} correlation')

            template = data_roi[trial_table.context == idx, row, :].flatten()

            target = np.rollaxis(data_frames[trial_table.context == idx], axis=1).reshape(20000, -1)
            r = []
            for px in range(target.shape[0]):
                r += [np.corrcoef(template, target[px, :])[1, 0]]

            result_dict = {
                'mouse_id': mouse_id,
                'session_id': session_id,
                'roi': roi,
                'context': context,
                'r': [r]
            }
            df += [result_dict]

    return df
    

def main(nwb_files, result_path, trial_based=True, correct_trials=True):
    df = []
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)

        mouse_id = session_id[0:5]
        print(f"Mouse : {mouse_id}, session : {session_id}")

        # Get trial table
        trial_table = nwb_read.get_trial_table(nwb_file)
        # trial_table['correct_trial'] = trial_table.lick_flag == trial_table.reward_available
        correct = []
        for i, x in trial_table.iterrows():
            if x['trial_type'] == 'auditory_trial' and x['lick_flag'] == 1:
                correct += [1]
            elif x['trial_type'] == 'whisker_trial' and x['context'] == 1 and x['lick_flag'] == 1:
                correct += [1]
            elif x['trial_type'] == 'whisker_trial' and x['context'] == 0 and x['lick_flag'] == 0:
                correct += [1]
            elif x['trial_type'] == 'no_stim_trial' and x['lick_flag'] == 0:
                correct += [1]
            else:
                correct += [0]

        trial_table['correct_trial'] = correct
        n_trials = len(trial_table)
        rrs_keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']
        wf_timestamps = nwb_read.get_roi_response_serie_timestamps(nwb_file, keys=rrs_keys)

        dict_roi, data_roi = get_roi_frames_by_epoch(nwb_file, n_trials, rrs_keys, wf_timestamps, trial_table.start_time - 2, trial_table.start_time)

        data_frames = get_frames_by_epoch(nwb_file, n_trials, wf_timestamps, start=trial_table.start_time - 2, stop=trial_table.start_time)

        if correct_trials == True:
            data_roi = data_roi[trial_table.correct_trial==1]
            data_frames = data_frames[trial_table.correct_trial==1]
            trial_table = trial_table.loc[trial_table.correct_trial==1]

        if trial_based:
            # results = trial_based_cross_correlation(mouse_id, session_id, trial_table, dict_roi, data_roi, data_frames)
            results = trial_based_correlation(mouse_id, session_id, trial_table, dict_roi, data_roi, data_frames)
            df += results

        else:
            results = context_block_correlation(mouse_id, session_id, trial_table, dict_roi, data_roi, data_frames)
            df += [results]

    if 'COMPUTERNAME' in os.environ.keys():
        df = pd.DataFrame(df)
    else:
        df = df[0]

    if trial_based:
        df.to_json(os.path.join(result_path, 'cross_corr_results_trial_based.json'))

    return


if __name__ == '__main__':

    if 'COMPUTERNAME' in os.environ.keys():
        for state in ['expert']:
            config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Session_list/context_sessions_jrgeco_{state}.yaml"
            config_file = haas_pathfun(config_file)
            with open(config_file, 'r', encoding='utf8') as stream:
                config_dict = yaml.safe_load(stream)

            nwb_files = [haas_pathfun(p.replace("\\", '/')) for p in config_dict['Session path']]

            result_path = f'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/pixel_trial_based_corr_gcamp_{state}'
            result_path = haas_pathfun(result_path)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            main(files, result_path=result_path, correct_trials=False)

    else:
        file = sys.argv[1]
        output_path = sys.argv[2]

        main([file], result_path=output_path, trial_based=True, correct_trials=False)
