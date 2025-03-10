import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.statespace.varmax import VARMAX
from nwb_utils.utils_misc import find_nearest
from nwb_wrappers import nwb_reader_functions as nwb_read


config_file = r"Z:\z_LSENS\Share\Pol_Bech\Session_list\context_sessions_gcamp_expert.yaml"
# config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
# sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
# sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
# sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
# files = [session[1] for session in sessions]
nwb_files = config_dict['Session path']

mouse_idx = [39, 43, 45, 49, 54]
mouse_ids = ['RD0' + str(i) for i in mouse_idx]
nwb_files = [path for path in nwb_files if any(keyword in path for keyword in mouse_ids)]

root_folder = r'Z:\analysis\Robin_Dard\Pop_results\Context_behaviour\test_connectvity_analysis'
output_folder = os.path.join(root_folder, '20250228')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
save_fig = False
rrs_keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']
use_granger_causality = False
use_mutual_information = False
xcorr_to_rew = True


def rolling_granger_causality(data, exog_data, window_size, step_size, maxlags):
    regions = data.columns
    n_regions = len(regions)

    # Prepare to store results for each rolling window
    results = []

    for start in range(0, len(data) - window_size + 1, step_size):
        # Define the rolling window
        end = start + window_size
        rolling_data = data.iloc[start:end]
        rolling_exog = exog_data.iloc[start:end]

        # Matrix to store p-values for the current window
        p_values_matrix = np.zeros((n_regions, n_regions))

        # Loop over all pairs of regions (i, j)
        for i in range(n_regions):
            for j in range(n_regions):
                if i == j:
                    # Skip self-causality tests
                    p_values_matrix[i, j] = np.nan
                    continue

                # Full model: all regions included
                full_model = VARMAX(rolling_data, exog=rolling_exog, order=(maxlags, 0))
                full_model_fitted = full_model.fit(disp=False)
                ll_full = full_model_fitted.llf

                # Restricted model: exclude region i (the "causing" region)
                restricted_data = rolling_data.drop(columns=[regions[i]])
                restricted_model = VARMAX(restricted_data, exog=rolling_exog, order=(5, 0))
                restricted_model_fitted = restricted_model.fit(disp=False)
                ll_restricted = restricted_model_fitted.llf

                # Perform likelihood ratio test
                lr_stat = -2 * (ll_restricted - ll_full)
                df = data.shape[1] - restricted_data.shape[1]  # Degrees of freedom = 1
                p_value = stats.chi2.sf(lr_stat, df)

                # Store p-value in the matrix
                p_values_matrix[i, j] = p_value

        # Convert the p-value matrix to a DataFrame for the current window
        p_values_df = pd.DataFrame(p_values_matrix, index=regions, columns=regions)

        # Store the results for this window
        results.append(p_values_df)

    return results


# Function to calculate mutual information for two variables
def mutual_information(x, y):
    """Calculate mutual information between two continuous variables."""
    # Discretize the continuous variables
    # Here, we use a simple binning approach
    x_bins = np.histogram_bin_edges(x, bins='auto')
    y_bins = np.histogram_bin_edges(y, bins='auto')

    # Calculate MI using mutual_info_score
    return mutual_info_score(np.digitize(x, x_bins), np.digitize(y, y_bins))


def rolling_window_mutual_information(data, window_size, step_size):
    regions = data.columns
    n_regions = len(regions)

    # Prepare to store results for each rolling window
    results = []

    # Loop through the data with the defined rolling window
    for start in range(0, len(data) - window_size + 1, step_size):
        # Define the rolling window
        end = start + window_size
        rolling_data = data.iloc[start:end]

        # Matrix to store mutual information values for the current window
        mi_matrix = np.zeros((n_regions, n_regions))

        # Loop over all pairs of regions (i, j)
        for i in range(n_regions):
            for j in range(n_regions):
                if i == j:
                    # Skip self-dependency
                    mi_matrix[i, j] = np.nan
                    continue

                # Calculate mutual information between region i and j
                mi_value = mutual_information(rolling_data[regions[i]], rolling_data[regions[j]])
                mi_matrix[i, j] = mi_value

        # Convert the mutual information matrix to a DataFrame for the current window
        mi_df = pd.DataFrame(mi_matrix, index=regions, columns=regions)
        # Store the results for this window
        results.append(mi_df)

    return results


for file in nwb_files:
    session = nwb_read.get_session_id(file)
    mouse = session[0:5]
    print(' ')
    print(f"Mouse: {mouse}, Session: {session}")
    saving_folder = os.path.join(output_folder, f'{mouse}', f'{session}')
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Extract neural data
    traces = nwb_read.get_roi_response_serie_data(nwb_file=file, keys=rrs_keys)
    rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file=file, keys=rrs_keys)
    if len(rrs_ts) > traces.shape[1]:
        rrs_ts = rrs_ts[0: traces.shape[1]]

    # Extract area names
    area_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file=file, keys=rrs_keys)
    sorted_areas = sorted(area_dict, key=area_dict.get)

    # Deal with exogenous factors : context, whisker and auditory stimuli
    # Get the time of rewarded context
    epochs = nwb_read.get_behavioral_epochs_names(nwb_file=file)
    rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='rewarded')
    non_rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='non-rewarded')

    # Assign context to each imaging frame
    rewarded_img_frame = [0 for i in range(len(rrs_ts))]
    for i, ts in enumerate(rrs_ts):
        for epoch in range(rewarded_times.shape[1]):
            if (ts >= rewarded_times[0][epoch]) and (ts <= rewarded_times[1][epoch]):
                rewarded_img_frame[i] = 1
                break
            else:
                continue

    # Get aligned activity around transition to R+ and to R-
    if rewarded_times[0, 0] == 0:
        to_rewarded_times = rewarded_times[0, 1: -1]
    else:
        to_rewarded_times = rewarded_times[0, :]
    if non_rewarded_times[0, 0] == 0:
        to_non_rewarded_times = non_rewarded_times[0, 1: -1]
    else:
        to_non_rewarded_times = non_rewarded_times[0, :]

    rew_transitions = [find_nearest(rrs_ts, t, True) for t in to_rewarded_times]
    to_rew_aligned_data = np.zeros((traces.shape[0], 400, len(rew_transitions)))
    for idx, rew_transition in enumerate(rew_transitions):
        to_rew_aligned_data[:, :, idx] = traces[:, int(rew_transition - 100): int(rew_transition + 300)]

    nn_rew_transitions = [find_nearest(rrs_ts, t, True) for t in to_non_rewarded_times]
    to_nn_rew_aligned_data = np.zeros((traces.shape[0], 400, len(nn_rew_transitions)))
    for idx, nn_rew_transition in enumerate(nn_rew_transitions):
        to_nn_rew_aligned_data[:, :, idx] = traces[:, int(nn_rew_transition - 100): int(nn_rew_transition + 300)]

    # Extract frame indices of whisker stim and auditory stim
    # Get trial table
    trial_table = nwb_read.get_trial_table(nwb_file=file)
    whisker_timestamps = trial_table.loc[trial_table.trial_type == 'whisker_trial'].whisker_stim_time.values[:]
    auditory_timestamps = trial_table.loc[trial_table.trial_type == 'auditory_trial'].auditory_stim_time.values[:]
    rewarded_context_trial_starts = trial_table.loc[trial_table.context == 1].start_time.values[:]
    non_rewarded_context_trial_starts = trial_table.loc[trial_table.context == 0].start_time.values[:]

    whisker_stim_frames = np.zeros(len(rrs_ts))
    for wh_ts in whisker_timestamps:
        whisker_stim_frames[find_nearest(rrs_ts, wh_ts, True)] = 1

    auditory_stim_frames = np.zeros(len(rrs_ts))
    for aud_ts in auditory_timestamps:
        auditory_stim_frames[find_nearest(rrs_ts, aud_ts, True)] = 1

    # Array of rewarded baseline
    rewarded_bs = np.zeros((traces.shape[0], 200, len(rewarded_context_trial_starts))) * np.nan
    for bs_idx, rew_trial in enumerate(rewarded_context_trial_starts):
        start_frame = find_nearest(rrs_ts, rew_trial - 2, True)
        stop_frame = find_nearest(rrs_ts, rew_trial, True)
        tmp_data = traces[:, start_frame: stop_frame]
        if tmp_data.shape != rewarded_bs[:, :, bs_idx].shape:
            continue
        else:
            rewarded_bs[:, :, bs_idx] = tmp_data

    # Array of non-rewarded baseline
    non_rewarded_bs = np.zeros((traces.shape[0], 200, len(non_rewarded_context_trial_starts))) * np.nan
    for bs_idx, nn_rew_trial in enumerate(non_rewarded_context_trial_starts):
        start_frame = find_nearest(rrs_ts, nn_rew_trial - 2, True)
        stop_frame = find_nearest(rrs_ts, nn_rew_trial, True)
        tmp_data = traces[:, start_frame: stop_frame]
        if tmp_data.shape != non_rewarded_bs[:, :, bs_idx].shape:
            continue
        else:
            non_rewarded_bs[:, :, bs_idx] = tmp_data

    # Figure :
    ax_titles = ['R+ baseline', 'R- baseline', 'To R+', 'To R-']
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for i in range(len(sorted_areas)):
        axes.flatten()[0].plot(np.nanmean(rewarded_bs[i, :, :], axis=1))
        axes.flatten()[1].plot(np.nanmean(non_rewarded_bs[i, :, :], axis=1))
        axes.flatten()[2].plot(np.nanmean(to_rew_aligned_data[i, :, :], axis=1))
        axes.flatten()[3].plot(np.nanmean(to_nn_rew_aligned_data[i, :, :], axis=1), label=sorted_areas[i])
    axes.flatten()[2].axvline(x=100, ymin=0, ymax=1, c='k', linestyle='--')
    axes.flatten()[3].axvline(x=100, ymin=0, ymax=1, c='k', linestyle='--')
    axes.flatten()[3].legend(loc="upper left", bbox_to_anchor=(1, 1))
    for ax_idx, ax in enumerate(axes.flatten()):
        ax.set_title(ax_titles[ax_idx])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.suptitle(f'{session}')
    fig.savefig(os.path.join(saving_folder, f'{session}_baseline.png'))

    # Connectivity test
    if use_granger_causality:
        exogenous_dict = {'context': rewarded_img_frame, 'auditory stim': auditory_stim_frames,
                          'whisker stim': whisker_stim_frames}

        exogenous_data = pd.DataFrame(exogenous_dict)

        maxlag = 10  # Choose the lag order based on AIC/BIC
        gc_results = rolling_granger_causality(data=pd.DataFrame(np.transpose(traces), columns=sorted_areas),
                                               exog_data=exogenous_data,
                                               window_size=5000,
                                               step_size=1000,
                                               maxlags=maxlag)
        window_idx = 10
        plt.figure(figsize=(8, 6))
        sns.heatmap(gc_results[window_idx], annot=True, cmap="coolwarm", cbar=True)
        plt.title(f'Granger Causality Matrix at Window {window_idx}')
        plt.show()

    if use_mutual_information:
        print('Use MI')
        t_start = 4
        frame_start = int(100 * t_start)
        t_stop = 9
        frame_stop = int(100 * t_stop)
        winsize = 1 * 100
        step = 1 * 100
        data_df = pd.DataFrame(np.transpose(traces[:, frame_start:frame_stop]), columns=sorted_areas)
        mi_results = rolling_window_mutual_information(data=data_df,
                                                       window_size=winsize,
                                                       step_size=step)
        windows = range(frame_start, frame_stop, step)

        fig, axes = plt.subplots(3, 6, figsize=(16, 8))
        for window_idx in range(5):
            sns.heatmap(mi_results[window_idx], cmap="coolwarm", cbar=True, ax=axes.flatten()[window_idx])
            axes.flatten()[window_idx].set_title(f'MI window {windows[window_idx] / 100} - '
                                                 f'{(windows[window_idx] + winsize) / 100} sec')
        fig.tight_layout()
        plt.show()

    if xcorr_to_rew:
        pairs = list(combinations(list(np.arange(0, to_rew_aligned_data.shape[0])), 2))
        for trial in range(to_rew_aligned_data.shape[2]):
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            for i, area in enumerate(sorted_areas):
                ax.plot(np.arange(0, 300) / 100, to_rew_aligned_data[i, 100:, trial],
                        label=sorted_areas[i])
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            ax.set_title(f'{session}, R+ transition {trial}')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            fig.tight_layout()

            fig, axes = plt.subplots(int(np.sqrt(len(pairs))), int(np.sqrt(len(pairs))), figsize=(9, 9),
                                     sharex=True, sharey=True)
            for idx, pair in enumerate(pairs):
                print(f"Area 1 : {sorted_areas[pair[0]]}, Area 2 : {sorted_areas[pair[1]]}")
                vec1 = to_rew_aligned_data[pair[0], 100:, trial]
                vec2 = to_rew_aligned_data[pair[1], 100:, trial]
                xcorr = np.correlate(vec1 - np.mean(vec1), vec2 - np.mean(vec2), "full")
                axes.flatten()[idx].plot(np.arange(-len(xcorr) // 2, len(xcorr) // 2) / 100, xcorr)
                axes.flatten()[idx].set_title(f'{sorted_areas[pair[0]]}, {sorted_areas[pair[1]]}', fontsize=8)
                axes.flatten()[idx].axvline(x=0, ymin=0, ymax=1, c='r', linestyle='--')
                axes.flatten()[idx].spines["right"].set_visible(False)
                axes.flatten()[idx].spines["top"].set_visible(False)
                axes.flatten()[idx].set_xlabel('', fontsize=6)
                axes.flatten()[idx].set_ylabel('', fontsize=6)
            fig.suptitle(f'{session}, R+ transition {trial}')
            fig.tight_layout()




