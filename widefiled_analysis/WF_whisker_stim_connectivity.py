import os
import yaml
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from nwb_utils.utils_misc import find_nearest
from nwb_wrappers import nwb_reader_functions as nwb_read


def highpass_filter(data, cutoff=10, fs=1000, order=4):
    """
    Apply a high-pass Butterworth filter to a 2D array along axis=1.

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


def cross_correlation(x, y, max_lag, n_shuffles, do_shuffle):
    """
    Compute the cross-correlation (Pearson correlation) between x and y
    for lags from -max_lag to +max_lag using np.corrcoef.
    """
    assert len(x) == len(y), "Signals must have the same length"
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags))
    shuffled_corr = np.zeros((len(lags), n_shuffles))

    for i, lag in enumerate(lags):
        if lag < 0:
            x_segment = x[:n+lag]
            y_segment = y[-lag:n]
        else:
            x_segment = x[lag:n]
            y_segment = y[:n-lag]

        # Compute Pearson correlation
        corr[i] = np.corrcoef(x_segment - np.mean(x_segment), y_segment - np.mean(y_segment))[0, 1]
        if do_shuffle:
            for i_shuffle in range(n_shuffles):
                x_shuffle = np.copy(x_segment)
                np.random.shuffle(x_shuffle)
                shuffled_corr[i, i_shuffle] = np.corrcoef(x_shuffle - np.mean(x_shuffle), y_segment - np.mean(y_segment))[0, 1]

    return corr, np.mean(shuffled_corr, axis=1)


mouse_line = 'gcamp'  # controls_gfp, gcamp, jrgeco, controls_td_tomato

config_file = f'//sv-nas1.rcp.epfl.ch/Petersen-lab/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_{mouse_line}_expert.yaml'

with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
nwb_files = config_dict['Session path']

root_folder = r'Z:\analysis\Robin_Dard\Pop_results\Context_behaviour\test_connectvity_analysis'

rrs_keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']
# rrs_keys = ['ophys', 'brain_grid_fluorescence', 'dff0_grid_traces']
only_correct_trials = False
post_stim_connectivity = False
iti_connectivity = True
filter_traces = True
fz_cut = 2
zero_order_corr = False
do_xcorr = True
do_shuffle = False
n_shuffles = 100

output_folder = os.path.join(root_folder, '20250307', f'{fz_cut}Hz_filtered')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

rew_data_dict = dict()
non_rew_data_dict = dict()
for file in nwb_files:
    session = nwb_read.get_session_id(file)
    mouse = session[0:5]
    print(' ')
    print(f"Mouse: {mouse}, Session: {session}")
    saving_folder = os.path.join(output_folder)
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    fig_folder = os.path.join(saving_folder, f'{mouse}')
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # Extract neural data
    traces = nwb_read.get_roi_response_serie_data(nwb_file=file, keys=rrs_keys)
    rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file=file, keys=rrs_keys)
    if len(rrs_ts) > traces.shape[1]:
        rrs_ts = rrs_ts[0: traces.shape[1]]

    # Extract area names
    area_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file=file, keys=rrs_keys)
    sorted_areas = sorted(area_dict, key=area_dict.get)

    # Get trial table
    trial_table = nwb_read.get_trial_table(nwb_file=file)

    # Look at connectivity post stimulus
    if post_stim_connectivity:
        # Keep only correct trials
        if only_correct_trials:
            trial_table = trial_table.loc[trial_table.correct_trials == True]
            trial_table = trial_table.reset_index(drop=True)
        # Keep only whisker trials
        trial_table = trial_table.loc[trial_table.trial_type == 'whisker_trial']
        # Loop on each trail for connectivity analysis
        for trial in range(len(trial_table)):
            stim_time = trial_table.iloc[trial].start_time
            start_frame = int(find_nearest(rrs_ts, stim_time))
            end_frame = int(find_nearest(rrs_ts, stim_time + 0.250))
            trial_data = traces[:, start_frame: end_frame]
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            for i, area in enumerate(sorted_areas):
                ax.plot(np.arange(0, end_frame - start_frame) / 100, trial_data[i, :], label=sorted_areas[i])
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            ax.set_title(f'{session}, whisker trial : {trial}, context : {trial_table.iloc[trial].context}, '
                         f'lick : {trial_table.iloc[trial].lick_flag}')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            fig.tight_layout()

            area_pairs = list(combinations(list(np.arange(0, trial_data.shape[0])), 2))

            fig, axes = plt.subplots(int(np.sqrt(len(area_pairs))), int(np.sqrt(len(area_pairs))), figsize=(9, 9),
                                     sharex=True, sharey=True)
            for idx, pair in enumerate(area_pairs):
                print(f"Area 1 : {sorted_areas[pair[0]]}, Area 2 : {sorted_areas[pair[1]]}")
                vec1 = trial_data[pair[0], :]
                vec2 = trial_data[pair[1], :]
                xcorr = np.correlate(vec1 - np.mean(vec1), vec2 - np.mean(vec2), "full")
                axes.flatten()[idx].plot(np.arange(-len(xcorr) // 2, len(xcorr) // 2) / 100, xcorr)
                axes.flatten()[idx].set_title(f'{sorted_areas[pair[0]]}, {sorted_areas[pair[1]]}', fontsize=8)
                axes.flatten()[idx].axvline(x=0, ymin=0, ymax=1, c='r', linestyle='--')
                axes.flatten()[idx].spines["right"].set_visible(False)
                axes.flatten()[idx].spines["top"].set_visible(False)
                axes.flatten()[idx].set_xlabel('', fontsize=6)
                axes.flatten()[idx].set_ylabel('', fontsize=6)
                axes.flatten()[idx].tick_params(axis='both', which='both', labelsize=6)
            fig.suptitle(f'{session}, whisker trial : {trial}, context : {trial_table.iloc[trial].context}, '
                         f'lick : {trial_table.iloc[trial].lick_flag}')
            fig.tight_layout()

    # Look at connectivity out of trial with stim:
    if iti_connectivity:
        # Keep 2d array ROIs x frames for frames out of trial for each context
        iti_rew_frames = np.ones(traces.shape[1]).astype(bool)
        iti_non_rew_frames = np.ones(traces.shape[1]).astype(bool)

        # Put at False the other context
        rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='rewarded')
        non_rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='non-rewarded')

        rewarded_epoch_frames = []
        for epoch in range(rewarded_times.shape[1]):
            start_frame = int(find_nearest(rrs_ts, rewarded_times[0][epoch]))
            end_frame = int(find_nearest(rrs_ts, rewarded_times[1][epoch]))
            rewarded_epoch_frames.extend(np.arange(start_frame, end_frame))
        rewarded_epoch_frames = np.array(rewarded_epoch_frames)
        rewarded_epoch_frames = rewarded_epoch_frames[rewarded_epoch_frames > 0]
        rewarded_epoch_frames = rewarded_epoch_frames[rewarded_epoch_frames <= traces.shape[1]]
        iti_non_rew_frames[rewarded_epoch_frames] = False

        non_rewarded_epoch_frames = []
        for epoch in range(non_rewarded_times.shape[1]):
            start_frame = int(find_nearest(rrs_ts, non_rewarded_times[0][epoch]))
            end_frame = int(find_nearest(rrs_ts, non_rewarded_times[1][epoch]))
            non_rewarded_epoch_frames.extend(np.arange(start_frame, end_frame))
        non_rewarded_epoch_frames = np.array(non_rewarded_epoch_frames)
        non_rewarded_epoch_frames = non_rewarded_epoch_frames[non_rewarded_epoch_frames >= 0]
        non_rewarded_epoch_frames = non_rewarded_epoch_frames[non_rewarded_epoch_frames <= traces.shape[1]]
        iti_rew_frames[non_rewarded_epoch_frames] = False

        # Get frames within trial
        stim_table = trial_table.loc[trial_table.trial_type.isin(['whisker_trial', 'auditory_trial'])]
        stim_table = stim_table.reset_index(drop=True)
        stim_frames = []
        for trial in range(len(stim_table)):
            stim_time = stim_table.iloc[trial].start_time
            start_frame = int(find_nearest(rrs_ts, stim_time))
            end_frame = int(find_nearest(rrs_ts, stim_time + 5))
            stim_frames.extend(np.arange(start_frame, end_frame))
        stim_frames = np.array(stim_frames)

        # Remove them
        iti_rew_frames[stim_frames] = False
        iti_non_rew_frames[stim_frames] = False

        # Quick figure for frames selection
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(0, len(iti_rew_frames)) / 100, iti_rew_frames, color='green')
        ax.plot(np.arange(0, len(iti_non_rew_frames)) / 100, iti_non_rew_frames, color='darkmagenta')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_title(f'{session} : ITI frames')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_folder, f'frame_selection_{session}.pdf'))
        plt.close()

        # Select traces and high pass filter at cut frequency Hz
        rew_iti_traces = traces[:, iti_rew_frames]
        non_rew_iti_traces = traces[:, iti_non_rew_frames]
        if filter_traces:
            filt_rew_iti_traces = highpass_filter(rew_iti_traces, cutoff=fz_cut, fs=100)
            filt_non_rew_iti_traces = highpass_filter(non_rew_iti_traces, cutoff=fz_cut, fs=100)
        else:
            filt_rew_iti_traces = rew_iti_traces
            filt_non_rew_iti_traces = non_rew_iti_traces

        # Just n frames for plot
        rew_len = filt_rew_iti_traces.shape[1]
        nn_rew_len = filt_non_rew_iti_traces.shape[1]

        # Quick plot of dff0 and filtered version
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 7))
        for i in range(filt_rew_iti_traces.shape[0]):
            axes.flatten()[0].plot(np.arange(0, rew_len) / 100, rew_iti_traces[i, :])
            axes.flatten()[1].plot(np.arange(0, rew_len) / 100, filt_rew_iti_traces[i, :])
            axes.flatten()[2].plot(np.arange(0, nn_rew_len) / 100, non_rew_iti_traces[i, :])
            axes.flatten()[3].plot(np.arange(0, nn_rew_len) / 100, filt_non_rew_iti_traces[i, :], label=f'{sorted_areas[i]}')
        for ax in axes.flatten():
            ax.spines[['right', 'top']].set_visible(False)
        axes.flatten()[0].set_title(f'{session} : dff0 R+ ITI')
        axes.flatten()[1].set_title(f'{session} : {fz_cut}Hz high-pass filtered dff0 R+ ITI')
        axes.flatten()[2].set_title(f'{session} : dff0 R- ITI')
        axes.flatten()[3].set_title(f'{session} : {fz_cut}Hz high-pass filtered dff0 R- ITI')
        axes.flatten()[3].legend(loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(os.path.join(fig_folder, f'traces_selection_{session}.pdf'))
        plt.close()

        # Correlation & x-correlation
        area_pairs = list(combinations(list(np.arange(0, len(sorted_areas))), 2))
        area_pair_names = [(sorted_areas[pair[0]], sorted_areas[pair[1]]) for pair in area_pairs]
        for pair_name in area_pair_names:
            rew_data_dict.setdefault(pair_name, {})
            rew_data_dict[pair_name].setdefault(mouse, [])
            non_rew_data_dict.setdefault(pair_name, {})
            non_rew_data_dict[pair_name].setdefault(mouse, [])

        if zero_order_corr:
            max_lag = 0
            for idx, area_pair in enumerate(area_pairs):
                pair_name = (sorted_areas[area_pair[0]], sorted_areas[area_pair[1]])
                print(f"Area pair : {pair_name}")
                vec1 = filt_rew_iti_traces[area_pair[0], :]
                vec2 = filt_rew_iti_traces[area_pair[1], :]
                vec3 = filt_non_rew_iti_traces[area_pair[0], :]
                vec4 = filt_non_rew_iti_traces[area_pair[1], :]

                rew_corr, rew_corr_shuffle = cross_correlation(vec1, vec2, max_lag, n_shuffles, do_shuffle)
                non_rew_corr, non_rew_corr_shuffle = cross_correlation(vec3, vec4, max_lag, n_shuffles, do_shuffle)

                rew_data_dict[pair_name][mouse].append(rew_corr)
                non_rew_data_dict[pair_name][mouse].append(non_rew_corr)

                print(f'R+ corr: {rew_corr}')
                print(f'R- corr: {non_rew_corr}')
                print(f"Diff : {rew_corr - non_rew_corr}")
                print(f'R+ corr avg shuffle: {np.mean(rew_corr_shuffle)}')
                print(f'R- corr avg shuffle: {np.mean(non_rew_corr_shuffle)}')

        if do_xcorr:
            max_lag = 5
            t = np.arange(-max_lag, max_lag + 1) / 100
            fig, axes = plt.subplots(int(np.sqrt(len(area_pairs))), int(np.sqrt(len(area_pairs))), sharex=True,
                                     sharey=True, figsize=(9, 9))

            for idx, area_pair in enumerate(area_pairs):
                pair_name = (sorted_areas[area_pair[0]], sorted_areas[area_pair[1]])
                print(f"Area pair : {pair_name}")
                vec1 = filt_rew_iti_traces[area_pair[0], :]
                vec2 = filt_rew_iti_traces[area_pair[1], :]
                vec3 = filt_non_rew_iti_traces[area_pair[0], :]
                vec4 = filt_non_rew_iti_traces[area_pair[1], :]

                rew_xcorr, rew_xcorr_shuffle = cross_correlation(vec1, vec2, max_lag, n_shuffles, do_shuffle)
                nn_rew_xcorr, nn_rew_xcorr_shuffle = cross_correlation(vec3, vec4, max_lag, n_shuffles, do_shuffle)

                rew_data_dict[pair_name][mouse].append(rew_xcorr)
                non_rew_data_dict[pair_name][mouse].append(nn_rew_xcorr)

                axes.flatten()[idx].plot(t, rew_xcorr, color='green')
                axes.flatten()[idx].plot(t, nn_rew_xcorr, color='darkmagenta')
                axes.flatten()[idx].plot(t, rew_xcorr_shuffle, color='green', linestyle='--')
                axes.flatten()[idx].plot(t, nn_rew_xcorr_shuffle, color='darkmagenta', linestyle='--')
                axes.flatten()[idx].set_title(f'{sorted_areas[area_pair[0]]} vs {sorted_areas[area_pair[1]]}', fontsize=6)
                axes.flatten()[idx].set_xlabel('Lag', fontsize=6)
                axes.flatten()[idx].set_ylabel('Corr', fontsize=6)
                axes.flatten()[idx].tick_params(axis='both', which='both', labelsize=6)
                axes.flatten()[idx].spines[['right', 'top']].set_visible(False)
                axes.flatten()[idx].axvline(x=0, ymin=0, ymax=1, c='r', linestyle='--')
            fig.suptitle(f'{session}, xcorr, filtered : {filter_traces}, @ {fz_cut}Hz')
            fig.tight_layout()
            fig.savefig(os.path.join(fig_folder, f'xcorr_{session}.pdf'))
            plt.close()


np.save(os.path.join(saving_folder, f'{mouse_line}_rew_data.npy'), rew_data_dict)
np.save(os.path.join(saving_folder, f'{mouse_line}_non_rew_data.npy'), non_rew_data_dict)





