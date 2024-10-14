import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
output_folder = os.path.join(root_folder, 'pca_stim_evoked', '20241009')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

save_fig = False
save_traces = False
rrs_keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']
components_to_plot = [0, 4]
n_frames_before_stim = 20
n_frames_after_stim = 25
n_trials_dict = dict()

for file in nwb_files:
    session = nwb_read.get_session_id(file)
    mouse = session[0:5]
    session_dict = dict()
    print(' ')
    print(f"Mouse: {mouse}, Session: {session}")

    # Get the data and associated timestamps
    traces = nwb_read.get_roi_response_serie_data(nwb_file=file, keys=rrs_keys)
    rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file=file, keys=rrs_keys)

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
    # Rewarded whisker hits
    # Get the traces
    rwd_whisker_hits = selected_frames_traces[:, np.where(outcome == 0)[0]]
    # Get number of trial
    n_rwd_whisker_hits = int(rwd_whisker_hits.shape[1] / nf)
    # Baseline on each trial
    rwd_whisker_hits_bs_frames = [np.arange(i * nf, i * nf + pre_stim) for i in range(n_rwd_whisker_hits)]
    rwd_whisker_hits_bs = np.zeros((rwd_whisker_hits.shape[0], rwd_whisker_hits.shape[1]))
    for i in range(n_rwd_whisker_hits):
        bs_values = np.nanmean(rwd_whisker_hits[:, rwd_whisker_hits_bs_frames[i]], axis=1, keepdims=True)
        rwd_whisker_hits_bs[:, np.arange(i * nf, (i + 1) * nf)] = bs_values
    # Subtract trial baseline
    rwd_whisker_hits -= rwd_whisker_hits_bs
    # Color
    rwd_wh_map = cm.get_cmap('Greens')
    rwd_wh_colors = [rwd_wh_map(i) for i in np.linspace(0.1, 1, nf)] * n_rwd_whisker_hits

    # Rewarded whisker miss
    rwd_whisker_miss = selected_frames_traces[:, np.where(outcome == 1)[0]]
    n_rwd_whisker_miss = int(rwd_whisker_miss.shape[1] / nf)
    rwd_whisker_miss_bs_frames = [np.arange(i * nf, i * nf + pre_stim) for i in range(n_rwd_whisker_miss)]
    rwd_whisker_miss_bs = np.zeros((rwd_whisker_miss.shape[0], rwd_whisker_miss.shape[1]))
    for i in range(n_rwd_whisker_miss):
        bs_values = np.nanmean(rwd_whisker_miss[:, rwd_whisker_miss_bs_frames[i]], axis=1, keepdims=True)
        rwd_whisker_miss_bs[:, np.arange(i * nf, (i + 1) * nf)] = bs_values
    rwd_whisker_miss -= rwd_whisker_miss_bs
    rwd_wm_map = cm.get_cmap('Greys')
    rwd_wm_colors = [rwd_wm_map(i) for i in np.linspace(0.1, 1, nf)] * n_rwd_whisker_miss

    # Non-rewarded whisker hits
    nn_rwd_whisker_hits = selected_frames_traces[:, np.where(outcome == 3)[0]]
    n_nn_rwd_whisker_hits = int(nn_rwd_whisker_hits.shape[1] / nf)
    nn_rwd_whisker_hits_bs_frames = [np.arange(i * nf, i * nf + pre_stim) for i in range(n_nn_rwd_whisker_hits)]
    nn_rwd_whisker_hits_bs = np.zeros((nn_rwd_whisker_hits.shape[0], nn_rwd_whisker_hits.shape[1]))
    for i in range(n_nn_rwd_whisker_hits):
        bs_values = np.nanmean(nn_rwd_whisker_hits[:, nn_rwd_whisker_hits_bs_frames[i]], axis=1, keepdims=True)
        nn_rwd_whisker_hits_bs[:, np.arange(i * nf, (i + 1) * nf)] = bs_values
    nn_rwd_whisker_hits -= nn_rwd_whisker_hits_bs
    nn_rwd_wh_map = cm.get_cmap('Reds')
    nn_rwd_wh_colors = [nn_rwd_wh_map(i) for i in np.linspace(0.1, 1, nf)] * n_nn_rwd_whisker_hits

    # Non-rewarded whisker miss
    nn_rwd_whisker_miss = selected_frames_traces[:, np.where(outcome == 2)[0]]
    n_nn_rwd_whisker_miss = int(nn_rwd_whisker_miss.shape[1] / nf)
    nn_rwd_whisker_miss_bs_frames = [np.arange(i * nf, i * nf + pre_stim) for i in range(n_nn_rwd_whisker_miss)]
    nn_rwd_whisker_miss_bs = np.zeros((nn_rwd_whisker_miss.shape[0], nn_rwd_whisker_miss.shape[1]))
    for i in range(n_nn_rwd_whisker_miss):
        bs_values = np.nanmean(nn_rwd_whisker_miss[:, nn_rwd_whisker_miss_bs_frames[i]], axis=1, keepdims=True)
        nn_rwd_whisker_miss_bs[:, np.arange(i * nf, (i + 1) * nf)] = bs_values
    nn_rwd_whisker_miss -= nn_rwd_whisker_miss_bs
    nn_rwd_wm_map = cm.get_cmap('Blues')
    nn_rwd_wm_colors = [nn_rwd_wm_map(i) for i in np.linspace(0.1, 1, nf)] * n_nn_rwd_whisker_miss

    # Save number of trials:
    session_dict['rwd_wh_hits'] = n_rwd_whisker_hits
    session_dict['rwd_wh_miss'] = n_rwd_whisker_miss
    session_dict['nn_rwd_wh_hits'] = n_nn_rwd_whisker_hits
    session_dict['nn_rwd_wh_miss'] = n_nn_rwd_whisker_miss
    n_wh_trials = n_rwd_whisker_hits + n_rwd_whisker_miss + n_nn_rwd_whisker_hits + n_nn_rwd_whisker_miss
    session_dict['n_whisker_trials'] = n_wh_trials
    n_trials_dict[f'{session}'] = session_dict

    # Concatenate session data
    data = np.concatenate((rwd_whisker_hits, rwd_whisker_miss, nn_rwd_whisker_hits, nn_rwd_whisker_miss), axis=1)
    colors = ['g'] * n_rwd_whisker_hits + ['black'] * n_rwd_whisker_miss + ['r'] * n_nn_rwd_whisker_hits + ['b'] * n_nn_rwd_whisker_miss
    full_colors = rwd_wh_colors + rwd_wm_colors + nn_rwd_wh_colors + nn_rwd_wm_colors

    if save_traces:
        result_folder = os.path.join(output_folder, f'{mouse}')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        np.save(os.path.join(f"{result_folder}", f"{session}_concat_psths.npy"), data)

    # ¨Plot traces
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    for i in range(rwd_whisker_hits.shape[0]):
        axes[0, 0].plot(rwd_whisker_hits[i, :])
        axes[0, 0].set_ylabel('df/f')
        axes[0, 0].set_title('RWD WH-hit')
    for i in range(rwd_whisker_miss.shape[0]):
        axes[0, 1].plot(rwd_whisker_miss[i, :])
        axes[0, 1].set_title('RWD WH-miss')
    for i in range(nn_rwd_whisker_hits.shape[0]):
        axes[1, 0].plot(nn_rwd_whisker_hits[i, :])
        axes[1, 0].set_ylabel('df/f')
        axes[1, 0].set_xlabel('Time (frames)')
        axes[1, 0].set_title('NN-RWD WH-hit')
    for i in range(nn_rwd_whisker_miss.shape[0]):
        axes[1, 1].plot(nn_rwd_whisker_miss[i, :])
        axes[1, 1].set_xlabel('Time (frames)')
        axes[1, 1].set_title('NN-RWD WH-miss')
    plt.suptitle(f'{session} all trials traces')
    fig.tight_layout()
    result_folder = os.path.join(output_folder, f'{mouse}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_formats = ['png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(result_folder, f'{session}_traces_all_trials.{save_format}'))
    plt.close('all')

    # Scale the data
    scaler = StandardScaler()
    data_for_pca = scaler.fit_transform(np.transpose(data))

    # Apply PCA
    pca = PCA(n_components=3)  # We will keep the top 3 principal components for visualization
    pca.fit(data_for_pca)

    # Project each trial onto the top 3 principal components
    # Resulting shape: (M * T, 3), where 3 is the number of components
    reduced_data = pca.transform(data_for_pca)

    # Reshape the reduced data back to (M, T, 3) to separate trials
    reduced_data_by_trial = reduced_data.reshape(n_wh_trials, nf, 3)

    # Plot each trial's trajectory in the space of the first two principal components
    plt.figure(figsize=(9, 9))
    for trial in range(n_wh_trials):
        # Extract the trajectory for the current trial in PC space (T time points, 3 PCs)
        trial_trajectory = reduced_data_by_trial[trial, :, :]

        # Plot the trajectory for the current trial in PC1 vs. PC2
        plt.plot(trial_trajectory[:, 0], trial_trajectory[:, 1], label=f'Trial {trial + 1}', c=colors[trial])

    # Add labels and legend
    plt.title(f'{session} Neural Trajectories for Individual Trials in PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    result_folder = os.path.join(output_folder, f'{mouse}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_formats = ['png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(result_folder, f'{session}_all_trials_trajectories.{save_format}'))
    plt.close('all')

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    for trial in range(n_rwd_whisker_hits):
        trial_trajectory = reduced_data_by_trial[trial, :, :]
        axes[0, 0].plot(trial_trajectory[:, 0], trial_trajectory[:, 1], label=f'Trial {trial + 1}', c='g')
        axes[0, 0].set_title('RWD Wh-hits')
        axes[0, 0].set_ylabel('Principal Component 2')
    for trial in range(n_rwd_whisker_hits, n_rwd_whisker_hits + n_rwd_whisker_miss):
        trial_trajectory = reduced_data_by_trial[trial, :, :]
        axes[0, 1].plot(trial_trajectory[:, 0], trial_trajectory[:, 1], label=f'Trial {trial + 1}', c='black')
        axes[0, 1].set_title('RWD Wh-miss')
    for trial in range(n_rwd_whisker_hits + n_rwd_whisker_miss,
                       n_rwd_whisker_hits + n_rwd_whisker_miss + n_nn_rwd_whisker_hits):
        trial_trajectory = reduced_data_by_trial[trial, :, :]
        axes[1, 0].plot(trial_trajectory[:, 0], trial_trajectory[:, 1], label=f'Trial {trial + 1}', c='r')
        axes[1, 0].set_title('NNRWD Wh-hits')
        axes[1, 0].set_ylabel('Principal Component 2')
        axes[1, 0].set_xlabel('Principal Component 1')
    for trial in range(n_rwd_whisker_hits + n_rwd_whisker_miss + n_nn_rwd_whisker_hits,
                       n_rwd_whisker_hits + n_rwd_whisker_miss + n_nn_rwd_whisker_hits + n_nn_rwd_whisker_miss):
        trial_trajectory = reduced_data_by_trial[trial, :, :]
        axes[1, 1].plot(trial_trajectory[:, 0], trial_trajectory[:, 1], label=f'Trial {trial + 1}', c='b')
        axes[1, 1].set_title('NNRWD Wh-miss')
        axes[1, 1].set_xlabel('Principal Component 1')
    plt.suptitle(f'{session} Neural Trajectories for Individual Trials in PCA Space')
    result_folder = os.path.join(output_folder, f'{mouse}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_formats = ['png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(result_folder, f'{session}_all_trials_trajectories_by_type.{save_format}'))
    plt.close('all')

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    for trial in range(n_rwd_whisker_hits):
        trial_trajectory = reduced_data_by_trial[trial, :, :]
        axes[0, 0].scatter(trial_trajectory[:, 0], trial_trajectory[:, 1], label=f'Trial {trial + 1}', c=rwd_wh_colors[0:nf])
        axes[0, 0].set_title('RWD Wh-hits')
        axes[0, 0].set_ylabel('Principal Component 2')
    for trial in range(n_rwd_whisker_hits, n_rwd_whisker_hits + n_rwd_whisker_miss):
        trial_trajectory = reduced_data_by_trial[trial, :, :]
        axes[0, 1].scatter(trial_trajectory[:, 0], trial_trajectory[:, 1], label=f'Trial {trial + 1}', c=rwd_wm_colors[0:nf])
        axes[0, 1].set_title('RWD Wh-miss')
    for trial in range(n_rwd_whisker_hits + n_rwd_whisker_miss,
                       n_rwd_whisker_hits + n_rwd_whisker_miss + n_nn_rwd_whisker_hits):
        trial_trajectory = reduced_data_by_trial[trial, :, :]
        axes[1, 0].scatter(trial_trajectory[:, 0], trial_trajectory[:, 1], label=f'Trial {trial + 1}', c=nn_rwd_wh_colors[0:nf])
        axes[1, 0].set_title('NNRWD Wh-hits')
        axes[1, 0].set_ylabel('Principal Component 2')
        axes[1, 0].set_xlabel('Principal Component 1')
    for trial in range(n_rwd_whisker_hits + n_rwd_whisker_miss + n_nn_rwd_whisker_hits,
                       n_rwd_whisker_hits + n_rwd_whisker_miss + n_nn_rwd_whisker_hits + n_nn_rwd_whisker_miss):
        trial_trajectory = reduced_data_by_trial[trial, :, :]
        axes[1, 1].scatter(trial_trajectory[:, 0], trial_trajectory[:, 1], label=f'Trial {trial + 1}', c=nn_rwd_wm_colors[0:nf])
        axes[1, 1].set_title('NNRWD Wh-miss')
        axes[1, 1].set_xlabel('Principal Component 1')
    plt.suptitle(f'{session} Neural Trajectories for Individual Trials in PCA Space')
    result_folder = os.path.join(output_folder, f'{mouse}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_formats = ['png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(result_folder, f'{session}_all_trials_trajectories_by_type_dots.{save_format}'))
    plt.close('all')

    # ---------- Trial averaged data ---------- #
    # Keep only baseline subtracted average across trials for each trial type:
    rwd_whisker_hits = selected_frames_traces[:, np.where(outcome == 0)[0]]
    rwd_whisker_hits = np.reshape(rwd_whisker_hits,
                                  (rwd_whisker_hits.shape[0], int(rwd_whisker_hits.shape[1] / nf), nf))
    rwd_whisker_hits = np.nanmean(rwd_whisker_hits, axis=1)
    rwd_whisker_hits -= np.nanmean(rwd_whisker_hits[:, 0:pre_stim], axis=1, keepdims=True)

    rwd_whisker_miss = selected_frames_traces[:, np.where(outcome == 1)[0]]
    rwd_whisker_miss = np.reshape(rwd_whisker_miss,
                                  (rwd_whisker_miss.shape[0], int(rwd_whisker_miss.shape[1] / nf), nf))
    rwd_whisker_miss = np.nanmean(rwd_whisker_miss, axis=1)
    rwd_whisker_miss -= np.nanmean(rwd_whisker_miss[:, 0:pre_stim], axis=1, keepdims=True)

    nn_rwd_whisker_hits = selected_frames_traces[:, np.where(outcome == 3)[0]]
    nn_rwd_whisker_hits = np.reshape(nn_rwd_whisker_hits,
                                     (nn_rwd_whisker_hits.shape[0], int(nn_rwd_whisker_hits.shape[1] / nf), nf))
    nn_rwd_whisker_hits = np.nanmean(nn_rwd_whisker_hits, axis=1)
    nn_rwd_whisker_hits -= np.nanmean(nn_rwd_whisker_hits[:, 0:pre_stim], axis=1, keepdims=True)

    nn_rwd_whisker_miss = selected_frames_traces[:, np.where(outcome == 2)[0]]
    nn_rwd_whisker_miss = np.reshape(nn_rwd_whisker_miss,
                                     (nn_rwd_whisker_miss.shape[0], int(nn_rwd_whisker_miss.shape[1] / nf), nf))
    nn_rwd_whisker_miss = np.nanmean(nn_rwd_whisker_miss, axis=1)
    nn_rwd_whisker_miss -= np.nanmean(nn_rwd_whisker_miss[:, 0:pre_stim], axis=1, keepdims=True)

    # Concatenate averaged data in order
    trial_avg_data = np.concatenate((rwd_whisker_hits, rwd_whisker_miss, nn_rwd_whisker_hits, nn_rwd_whisker_miss), axis=1)

    scaler = StandardScaler()
    avg_data_for_pca = scaler.fit_transform(np.transpose(trial_avg_data))

    # Apply PCA
    pca = PCA(n_components=3)  # We will keep the top 3 principal components for visualization
    pca.fit(avg_data_for_pca)
    # Project each trial onto the top 3 principal components
    avg_reduced_data = pca.transform(avg_data_for_pca)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    avg_colors = ['g', 'black', 'r', 'b']
    for i in range(4):
        ax.plot(avg_reduced_data[i * nf: (i + 1) * nf, 0], avg_reduced_data[i * nf: (i + 1) * nf, 1], c=avg_colors[i])
        ax.set_xlabel('PC #1')
        ax.set_ylabel('PC #2')
    plt.title(f'{session} Neural Trajectories for Trials Average activity in PCA Space')
    result_folder = os.path.join(output_folder, f'{mouse}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_formats = ['png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(result_folder, f'{session}_trial_average_trajectories.{save_format}'))
    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    avg_time_colors = rwd_wh_colors[0:nf] + rwd_wm_colors[0:nf] + nn_rwd_wh_colors[0:nf] + nn_rwd_wm_colors[0:nf]
    for i in range(4):
        ax.scatter(avg_reduced_data[i * nf: (i + 1) * nf, 0], avg_reduced_data[i * nf: (i + 1) * nf, 1],
                   c=avg_time_colors[i * nf: (i + 1) * nf])
        ax.set_xlabel('PC #1')
        ax.set_ylabel('PC #2')
    plt.title(f'{session} Neural Trajectories for Trials Average activity in PCA Space')
    result_folder = os.path.join(output_folder, f'{mouse}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_formats = ['png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(result_folder, f'{session}_trial_average_trajectories_dots.{save_format}'))
    plt.close('all')

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    avg_colors = ['g', 'black', 'r', 'b']
    for i in range(4):
        axes.flatten()[i].plot(avg_reduced_data[i * nf: (i + 1) * nf, 0], avg_reduced_data[i * nf: (i + 1) * nf, 1],
                               c=avg_colors[i])
    axes[0, 0].set_title('RWD Wh-hits')
    axes[0, 0].set_ylabel('Principal Component 2')
    axes[0, 1].set_title('RWD Wh-miss')
    axes[1, 0].set_title('NNRWD Wh-hits')
    axes[1, 0].set_ylabel('Principal Component 2')
    axes[1, 0].set_xlabel('Principal Component 1')
    axes[1, 1].set_title('NNRWD Wh-miss')
    axes[1, 1].set_xlabel('Principal Component 1')
    plt.suptitle(f'{session} Neural Trajectories for Trials Average activity in PCA Space')
    result_folder = os.path.join(output_folder, f'{mouse}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_formats = ['png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(result_folder, f'{session}_trial_average_trajectories_by_type.{save_format}'))
    plt.close('all')

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    avg_colors = ['g', 'black', 'r', 'b']
    for i in range(4):
        axes.flatten()[i].scatter(avg_reduced_data[i * nf: (i + 1) * nf, 0], avg_reduced_data[i * nf: (i + 1) * nf, 1],
                                  c=avg_time_colors[i * nf: (i + 1) * nf])
    axes[0, 0].set_title('RWD Wh-hits')
    axes[0, 0].set_ylabel('Principal Component 2')
    axes[0, 1].set_title('RWD Wh-miss')
    axes[1, 0].set_title('NNRWD Wh-hits')
    axes[1, 0].set_ylabel('Principal Component 2')
    axes[1, 0].set_xlabel('Principal Component 1')
    axes[1, 1].set_title('NNRWD Wh-miss')
    axes[1, 1].set_xlabel('Principal Component 1')
    plt.suptitle(f'{session} Neural Trajectories for Trials Average activity in PCA Space')
    result_folder = os.path.join(output_folder, f'{mouse}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_formats = ['png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(result_folder, f'{session}_trial_average_trajectories_by_type_dots.{save_format}'))
    plt.close('all')

    # ---------- T-SNE on average data ---------- #
    # tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
    # tsne_results = tsne.fit_transform(avg_data_for_pca)  # For averaged data
    #
    # # Plot the t-SNE results
    # plt.figure(figsize=(9, 9))
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=avg_time_colors)
    # plt.title('t-SNE of Stimulus-Evoked Neural Responses')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.legend()
    # plt.grid(False)
    # plt.show()
