import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
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

root_folder = r'Z:\analysis\Robin_Dard\Pop_results\Context_behaviour'
output_folder = os.path.join(root_folder, 'test_PCA_analysis', '20241018', 'trial_average_pca_and_proj_baseline_kept')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

save_fig = True
save_traces = False
remove_baseline = False
rrs_keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']
components_to_plot = [0, 4]
n_frames_before_stim = 100
n_frames_after_stim = 25
session_data_dict = dict()
color_dict = {'RWD-Wh-hit': ['darkgreen', 'limegreen'], 'RWD-Wh-miss': ['black', 'dimgrey'],
              'NN-RWD-Wh-hit': ['darkred', 'red'], 'NN-RWD-Wh-miss': ['darkblue', 'blue']}
session_jaw_data_dict = dict()

for file in nwb_files:
    session = nwb_read.get_session_id(file)
    mouse = session[0:5]
    session_dict = dict()
    print(' ')
    print(f"Mouse: {mouse}, Session: {session}")
    result_folder = os.path.join(output_folder, f'{mouse}', f'{session}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Get the data and associated timestamps
    traces = nwb_read.get_roi_response_serie_data(nwb_file=file, keys=rrs_keys)
    rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file=file, keys=rrs_keys)

    # Extract area names
    area_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file=file, keys=rrs_keys)
    sorted_areas = sorted(area_dict, key=area_dict.get)

    # Get trial table
    trial_table = nwb_read.get_trial_table(nwb_file=file)
    n_trials = len(trial_table)

    # Get full traces around stim
    selected_frames = []
    outcome = []
    pre_stim = n_frames_before_stim
    post_stim = n_frames_after_stim
    nf = pre_stim + post_stim
    for trial in range(n_trials):
        if trial_table.iloc[trial].trial_type != 'whisker_trial':
            continue
        stim_time = trial_table.iloc[trial].start_time
        stim_frame = find_nearest(rrs_ts, stim_time)
        start_frame = stim_frame - pre_stim
        end_frame = stim_frame + post_stim
        selected_frames.append(np.arange(start_frame, end_frame))
        epoch = trial_table.iloc[trial].context
        lick = trial_table.iloc[trial].lick_flag
        if epoch == 1:
            if lick == 1:
                outcome.append(np.zeros(len(np.arange(start_frame, end_frame))))
            else:
                outcome.append(np.ones(len(np.arange(start_frame, end_frame))))
        else:
            if lick == 1:
                outcome.append(np.zeros(len(np.arange(start_frame, end_frame))) + 3)
            else:
                outcome.append(np.zeros(len(np.arange(start_frame, end_frame))) + 2)

    # Full data
    selected_frames = np.concatenate(selected_frames, axis=0)
    selected_frames_traces = traces[:, selected_frames]
    outcome = np.concatenate(outcome, axis=0)

    # Get concatenation by trial type all trials
    rwd_hits = selected_frames_traces[:, np.where(outcome == 0)[0]]
    rwd_hits = np.reshape(rwd_hits, (rwd_hits.shape[0], int(rwd_hits.shape[1] / nf), nf))
    session_data_dict['RWD-Wh-hit'] = rwd_hits

    rwd_miss = selected_frames_traces[:, np.where(outcome == 1)[0]]
    rwd_miss = np.reshape(rwd_miss, (rwd_miss.shape[0], int(rwd_miss.shape[1] / nf), nf))
    session_data_dict['RWD-Wh-miss'] = rwd_miss

    nn_rwd_hits = selected_frames_traces[:, np.where(outcome == 3)[0]]
    nn_rwd_hits = np.reshape(nn_rwd_hits, (nn_rwd_hits.shape[0], int(nn_rwd_hits.shape[1] / nf), nf))
    session_data_dict['NN-RWD-Wh-hit'] = nn_rwd_hits

    nn_rwd_miss = selected_frames_traces[:, np.where(outcome == 2)[0]]
    nn_rwd_miss = np.reshape(nn_rwd_miss, (nn_rwd_miss.shape[0], int(nn_rwd_miss.shape[1] / nf), nf))
    session_data_dict['NN-RWD-Wh-miss'] = nn_rwd_miss

    if save_traces:
        np.save(os.path.join(f"{result_folder}", f"{session}_whisker_psths.npy"), session_data_dict)

    # Figure for trial averaged activity
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    ax_to_use = 0
    for key, data in session_data_dict.items():
        avg_data = np.mean(data, axis=1)
        if remove_baseline:
            avg_data -= np.nanmean(avg_data[:, 0:n_frames_before_stim], axis=1, keepdims=True)
        for i in range(avg_data.shape[0]):
            axes.flatten()[ax_to_use].plot(avg_data[i, :], label=f'{sorted_areas[i]}')
            axes.flatten()[ax_to_use].axvline(x=n_frames_before_stim, linestyle='--', color='grey')
            axes.flatten()[ax_to_use].spines[['right', 'top']].set_visible(False)
            axes.flatten()[ax_to_use].legend(ncol=2, loc="upper left")
            axes.flatten()[ax_to_use].set_xlabel('Time (in frames)')
            axes.flatten()[ax_to_use].set_ylabel('df/f')
        axes.flatten()[ax_to_use].set_title(f'{key} ({data.shape[1]} trials)')
        ax_to_use += 1
    fig.suptitle(f'{session} Average trial response')
    fig.tight_layout()
    if save_fig:
        save_formats = ['png', 'svg']
        for save_format in save_formats:
            fig.savefig(os.path.join(result_folder, f'{session}_average_trial_response.{save_format}'))
    else:
        plt.show()

    # Concatenate trial averaged data
    concatenated_trial_avg_data = []
    for key, data in session_data_dict.items():
        avg_data = np.mean(data, axis=1)
        if remove_baseline:
            avg_data -= np.nanmean(avg_data[:, 0:n_frames_before_stim], axis=1, keepdims=True)
        concatenated_trial_avg_data.append(avg_data)
    concatenated_trial_avg_data = np.concatenate(concatenated_trial_avg_data, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i in range(concatenated_trial_avg_data.shape[0]):
        ax.plot(concatenated_trial_avg_data[i, :], label=f'{sorted_areas[i]}')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel(f'Time (in frames)')
    ax.set_ylabel(f'df/f')
    ax.legend(ncol=2, loc='upper left')
    fig.suptitle(f'{session} Concatenated trial averaged response input to PCA')
    if save_fig:
        save_formats = ['png', 'svg']
        for save_format in save_formats:
            fig.savefig(os.path.join(result_folder, f'{session}_input_to_pca.{save_format}'))
    else:
        plt.show()

    # Scale the trial averaged data
    scaler = StandardScaler()
    avg_data_for_pca = scaler.fit_transform(np.transpose(concatenated_trial_avg_data))

    # Apply PCA
    pca = PCA(n_components=8)
    results = pca.fit(avg_data_for_pca)
    principal_components = pca.transform(avg_data_for_pca)

    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(8)])
    n_tot_f = len(pca_df)

    # Assign rewarded frames on averaged data
    rewarded_img_frame = np.zeros(n_tot_f)
    rewarded_img_frame[0: n_tot_f // 2] = 1

    # Get the jaw opening
    dlc_ts = nwb_read.get_dlc_timestamps(nwb_file=file, keys=['behavior', 'BehavioralTimeSeries'])
    if dlc_ts is not None:
        jaw_angle = nwb_read.get_dlc_data(nwb_file=file, keys=['behavior', 'BehavioralTimeSeries'], part='jaw_angle')
        dlc_frames = []
        for img_ts in rrs_ts:
            dlc_frames.append(find_nearest(array=dlc_ts[0], value=img_ts, is_sorted=True))
        aligned_jaw_angle = jaw_angle[dlc_frames]
        jaw_filt = gaussian_filter1d(input=aligned_jaw_angle, sigma=20, axis=-1, order=0)
    else:
        jaw_filt = np.empty(traces.shape[1])
        jaw_filt[:] = np.nan
    selected_jaw_filt = jaw_filt[selected_frames]

    jaw_rwd_hits = selected_jaw_filt[np.where(outcome == 0)[0]]
    jaw_rwd_hits = np.reshape(jaw_rwd_hits, (int(jaw_rwd_hits.shape[0] / nf), nf))
    session_jaw_data_dict['RWD-Wh-hit'] = jaw_rwd_hits

    jaw_rwd_miss = selected_jaw_filt[np.where(outcome == 1)[0]]
    jaw_rwd_miss = np.reshape(jaw_rwd_miss, (int(jaw_rwd_miss.shape[0] / nf), nf))
    session_jaw_data_dict['RWD-Wh-miss'] = jaw_rwd_miss

    jaw_nn_rwd_hits = selected_jaw_filt[np.where(outcome == 3)[0]]
    jaw_nn_rwd_hits = np.reshape(jaw_nn_rwd_hits, (int(jaw_nn_rwd_hits.shape[0] / nf), nf))
    session_jaw_data_dict['NN-RWD-Wh-hit'] = jaw_nn_rwd_hits

    jaw_nn_rwd_miss = selected_jaw_filt[np.where(outcome == 2)[0]]
    jaw_nn_rwd_miss = np.reshape(jaw_nn_rwd_miss, (int(jaw_nn_rwd_miss.shape[0] / nf), nf))
    session_jaw_data_dict['NN-RWD-Wh-miss'] = jaw_nn_rwd_miss

    concatenated_trial_jaw_avg_data = []
    for key, data in session_jaw_data_dict.items():
        avg_data = np.nanmean(data, axis=0)
        concatenated_trial_jaw_avg_data.append(avg_data)
    concatenated_trial_jaw_avg_data = np.concatenate(concatenated_trial_jaw_avg_data)

    # Figure 1 : loadings and explained variance
    color = iter(cm.rainbow(np.linspace(0, 1, 8)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(components_to_plot[0], components_to_plot[1]):
        c = next(color)
        ax0.plot(np.abs(results.components_[i]), label=f'Principal Component {i}', color=c, marker='o')
        ax0.set_xticklabels(['-10'] + sorted_areas)
        ax0.set_ylabel('PC Loadings (absolute value)')
        ax0.set_xlabel('Area')
    ax0.legend(loc='upper left')
    ax0.set_ylim([-0.05, 1])
    ax1.plot(np.cumsum(results.explained_variance_ratio_), marker='o')
    ax1.set_ylabel('Explained variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylim([0.50, 1.1])
    ax1.axhline(y=0.95, linestyle='--', color='black')
    ax0.spines[['right', 'top']].set_visible(False)
    ax1.spines[['right', 'top']].set_visible(False)
    fig.suptitle(f'{session} PCA on concatenated trial averaged data')
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(result_folder, f'{session}_PC_loadings_and_variance.png'))
    else:
        plt.show()

    # Figure 2 : PC time courses with jaw opening and context
    color = iter(cm.rainbow(np.linspace(0, 1, 8)))
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(15, 9))
    for i in range(components_to_plot[0], components_to_plot[1]):
        c = next(color)
        axes[i].plot(range(n_tot_f), pca_df[f'PC{i}'], label=f'Principal Component {i}', color=c)
        axes[i].set_ylabel(f'PC{i}')
        axes[i].fill_between(range(n_tot_f), 0, 1, where=np.array(rewarded_img_frame).astype(bool), alpha=0.4,
                             color='green', transform=axes[i].get_xaxis_transform(), label='RWD context')
        axes[i].plot(range(n_tot_f), concatenated_trial_jaw_avg_data, color='red', label='Average jaw angle filtered')
        axes[i].legend(loc='upper right')
        axes[i].spines[['right', 'top']].set_visible(False)
    axes[3].set_xlabel('Time (in frames)')
    axes[0].set_title('Temporal PCA - Principal Components Over Time')
    fig.suptitle(f'{session}')
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(result_folder, f'{session}_PC_time_courses.png'))
    else:
        plt.show()

    # Transform each trial type average with PC loadings and plot projection of trial average
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    for key, data in session_data_dict.items():
        avg_data = np.mean(data, axis=1)
        if remove_baseline:
            avg_data -= np.nanmean(avg_data[:, 0:n_frames_before_stim], axis=1, keepdims=True)
        reduced_data = np.transpose(pca.transform(np.transpose(avg_data)))
        ax0.plot(reduced_data[0, :], reduced_data[1, :], c=color_dict[key][0], label=f'{key}')
        ax0.scatter(reduced_data[0, pre_stim], reduced_data[1, pre_stim], c='gold', marker='*', s=40,
                    zorder=5)
        ax1.scatter(reduced_data[0, :], reduced_data[1, :], c=color_dict[key][0], label=f'{key}',
                    alpha=np.linspace(0.1, 1, reduced_data.shape[1]))
        ax1.scatter(reduced_data[0, pre_stim], reduced_data[1, pre_stim], c='gold', marker='*', s=40,
                    zorder=5)
    ax0.spines[['right', 'top']].set_visible(False)
    ax0.set_xlabel(f'PC #0')
    ax0.set_ylabel(f'PC #1')
    ax0.legend(ncol=2, loc='upper left')
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.set_xlabel(f'PC #0')
    ax1.set_ylabel(f'PC #1')
    ax1.legend(ncol=2, loc='upper left')
    fig.suptitle(f'{session} Projection of trial averaged activity')
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(result_folder, f'{session}_trial_averaged_activity_projected.png'))
    else:
        plt.show()


