import nwb_wrappers.nwb_reader_functions as nwb_read
import matplotlib.pyplot as plt
import matplotlib.colors as matcol
import matplotlib
import seaborn as sns
import umap
import os
import yaml
import numpy as np
import pandas as pd
from nwb_utils import utils_misc, server_path
from analysis.psth_analysis import make_events_aligned_data_table
from sklearn.preprocessing import StandardScaler


EXPERIMENTER_MAP = {
    'AR': 'Anthony_Renard',
    'RD': 'Robin_Dard',
    'AB': 'Axel_Bisi',
    'MP': 'Mauro_Pulin',
    'PB': 'Pol_Bech',
    'MM': 'Meriam_Malekzadeh',
    'LS': 'Lana_Smith',
    'GF': 'Anthony_Renard',
    'MI': 'Anthony_Renard',
}


def get_experimenter_nwb_folder(experimenter_initials):
    experimenter = EXPERIMENTER_MAP[experimenter_initials]
    nwb_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWB')

    return nwb_folder


def from_session_list_to_path_list(sessions_list):
    sessions_path = []
    for session_id in sessions_list:
        experimenter = session_id[0:2]
        nwb_folder = get_experimenter_nwb_folder(experimenter)
        files = os.listdir(nwb_folder)
        nwb_files = [os.path.join(nwb_folder, name) for name in files if 'nwb' in name]
        for nwb_path in nwb_files:
            if session_id in nwb_path:
                sessions_path.append(nwb_path)
                break
            else:
                continue

    return sessions_path


def umap_dimensionality_reduction(ci_traces, title, color_arg=None, alpha=None):
    reducer = umap.UMAP()
    scaled_traces = StandardScaler().fit_transform(ci_traces)
    embedding = reducer.fit_transform(scaled_traces)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if color_arg is not None:
        if len(color_arg[0]) > 1:
            rgb = color_arg
        else:
            rgb = [sns.color_palette()[x] for x in color_arg]
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=rgb,
            alpha=1)
        if alpha is not None:
            print("here")
            rgba = [matcol.to_rgba(c=rgb[i], alpha=alpha[i]) for i in range(len(rgb))]
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=rgba)
    else:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c='b',
            alpha=1)
    plt.title(f'UMAP {title}', fontsize=14)

    return fig, ax


def create_umap_color_code(n_data_points):
    # Rewarded whisker hits (correct)
    rwd_wh_map = matplotlib.cm.get_cmap('Greens')
    rwd_wh_colors = [rwd_wh_map(i) for i in np.linspace(0.1, 1, n_data_points)]
    # Rewarded whisker miss (incorrect)
    rwd_wm_map = matplotlib.cm.get_cmap('Greys')
    rwd_wm_colors = [rwd_wm_map(i) for i in np.linspace(0.1, 1, n_data_points)]
    # Non-rewarded whisker hit (incorrect)
    nnrwd_wh_map = matplotlib.cm.get_cmap('Reds')
    nnrwd_wh_colors = [nnrwd_wh_map(i) for i in np.linspace(0.1, 1, n_data_points)]
    # Non-rewarded whisker miss (correct)
    nnrwd_wm_map = matplotlib.cm.get_cmap('Blues')
    nnrwd_wm_colors = [nnrwd_wm_map(i) for i in np.linspace(0.1, 1, n_data_points)]

    # Build list of colors that is the same length than list of data-points
    data_colors = rwd_wh_colors + rwd_wm_colors + nnrwd_wh_colors + nnrwd_wm_colors

    return data_colors


def return_events_aligned_wf_table(nwb_files, rrs_keys, trials_dict, trial_names, epochs, time_range, subtract_baseline):
    """

    :param nwb_files: list of path to nwb files to analyse
    :param rrs_keys: list of keys to access traces from different brain regions in nwb file
    :param trials_dict: list of dictionaries describing the trial to get from table
    :param trial_names: list of trial names
    :param epochs: list of epochs
    :param time_range: time range for psth
    :return: a dataframe with activity aligned and trial info
    """

    full_df = []
    for index, trial_dict in enumerate(trials_dict):
        for epoch_index, epoch in enumerate(epochs):
            print(f" ")
            print(f"Trial selection : {trials_dict[index]} (Trial name : {trial_names[index]})")
            print(f"Epoch : {epoch}")
            data_table = make_events_aligned_data_table(nwb_list=nwb_files,
                                                        rrs_keys=rrs_keys,
                                                        time_range=time_range,
                                                        trial_selection=trials_dict[index],
                                                        epoch=epoch,
                                                        subtract_baseline=subtract_baseline)
            data_table['trial_type'] = trial_names[index]
            data_table['epoch'] = epochs[epoch_index]
            full_df.append(data_table)
    full_df = pd.concat(full_df, ignore_index=True)

    return full_df


def single_session_wf_umap_whisker_stim(nwb_files, rrs_keys, n_frames_before_stim, n_frames_after_stim, output_path,
                                        save_tr):
    for file in nwb_files:
        session = nwb_read.get_session_id(file)
        mouse = session[0:5]
        print(f"Mouse: {mouse}, Session: {session}")

        # Get the data and associated timestamps
        traces = nwb_read.get_roi_response_serie_data(nwb_file=file, keys=rrs_keys)
        rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file=file, keys=rrs_keys)

        # Get trial table
        trial_table = nwb_read.get_trial_table(nwb_file=file)
        n_trials = len(trial_table)

        # Get full quiet windows traces
        selected_frames = []
        outcome = []
        pre_stim = n_frames_before_stim
        post_stim = n_frames_after_stim
        nf = pre_stim + post_stim
        for trial in range(n_trials):
            if trial_table.iloc[trial].trial_type != 'whisker_trial':
                continue
            stim_time = trial_table.iloc[trial].start_time
            stim_frame = utils_misc.find_nearest(rrs_ts, stim_time)
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

        # Keep only baseline subtracted average across trials for each trial type:
        rwd_whisker_hits = selected_frames_traces[:, np.where(outcome == 0)[0]]
        rwd_whisker_hits = np.reshape(rwd_whisker_hits, (rwd_whisker_hits.shape[0], int(rwd_whisker_hits.shape[1] / nf), nf))
        rwd_whisker_hits = np.nanmean(rwd_whisker_hits, axis=1)
        rwd_whisker_hits -= np.nanmean(rwd_whisker_hits[:, 0:pre_stim], axis=1, keepdims=True)

        rwd_whisker_miss = selected_frames_traces[:, np.where(outcome == 1)[0]]
        rwd_whisker_miss = np.reshape(rwd_whisker_miss, (rwd_whisker_miss.shape[0], int(rwd_whisker_miss.shape[1] / nf), nf))
        rwd_whisker_miss = np.nanmean(rwd_whisker_miss, axis=1)
        rwd_whisker_miss -= np.nanmean(rwd_whisker_miss[:, 0:pre_stim], axis=1, keepdims=True)

        nn_rwd_whisker_hits = selected_frames_traces[:, np.where(outcome == 3)[0]]
        nn_rwd_whisker_hits = np.reshape(nn_rwd_whisker_hits, (nn_rwd_whisker_hits.shape[0], int(nn_rwd_whisker_hits.shape[1] / nf), nf))
        nn_rwd_whisker_hits = np.nanmean(nn_rwd_whisker_hits, axis=1)
        nn_rwd_whisker_hits -= np.nanmean(nn_rwd_whisker_hits[:, 0:pre_stim], axis=1, keepdims=True)

        nn_rwd_whisker_miss = selected_frames_traces[:, np.where(outcome == 2)[0]]
        nn_rwd_whisker_miss = np.reshape(nn_rwd_whisker_miss, (nn_rwd_whisker_miss.shape[0], int(nn_rwd_whisker_miss.shape[1] / nf), nf))
        nn_rwd_whisker_miss = np.nanmean(nn_rwd_whisker_miss, axis=1)
        nn_rwd_whisker_miss -= np.nanmean(nn_rwd_whisker_miss[:, 0:pre_stim], axis=1, keepdims=True)

        # Concatenate averaged data in order
        data = np.concatenate((rwd_whisker_hits, rwd_whisker_miss, nn_rwd_whisker_hits, nn_rwd_whisker_miss), axis=1)

        # Create color map that will combine trial type and time
        data_colors = create_umap_color_code(n_data_points=nf)

        # save PSTHs to reuse in other tests
        if save_tr:
            result_folder = os.path.join(output_path, f'{mouse}')
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            np.save(os.path.join(f"{result_folder}", f"{session}_avg_concat_psths.npy"), data)

        # Plot the mean baseline subtracted psths
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
        time = (np.arange(0, nn_rwd_whisker_miss.shape[1]) - pre_stim) / 100
        for cell in range(nn_rwd_whisker_miss.shape[0]):
            axes[0, 0].plot(time, rwd_whisker_hits[cell, :], lw=3)
            axes[0, 0]. set_title('RWD context - Whit')
            axes[0, 1].plot(time, rwd_whisker_miss[cell, :], lw=3)
            axes[0, 1].set_title('RWD context - Wmiss')
            axes[1, 0].plot(time, nn_rwd_whisker_hits[cell, :], lw=3)
            axes[1, 0].set_title('nn-RWD context - Whit')
            axes[1, 1].plot(time, nn_rwd_whisker_miss[cell, :], lw=3)
            axes[1, 1].set_title('nn-RWD context - Wmiss')
        for ax in axes.reshape(-1):
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('dff')
        fig.suptitle(f'{nwb_read.get_session_id(file)}', fontsize=14)
        fig.tight_layout()
        result_folder = os.path.join(output_path, f'{mouse}')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        save_formats = ['png', 'svg']
        for save_format in save_formats:
            fig.savefig(os.path.join(result_folder, f'{session}_traces.{save_format}'))
            plt.close()

        # umap_dimensionality_reduction(ci_traces=np.transpose(quiet_window_traces), color_arg=colors.astype(int))
        fig, ax = umap_dimensionality_reduction(ci_traces=np.transpose(data),
                                                title=nwb_read.get_session_id(file),
                                                color_arg=data_colors)
        fig.tight_layout()
        result_folder = os.path.join(output_path, f'{mouse}')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        save_formats = ['png', 'svg']
        for save_format in save_formats:
            fig.savefig(os.path.join(result_folder, f'{session}_umap.{save_format}'))


def trial_based_wf_umap_whisker_stim(nwb_files, rrs_keys, n_frames_before_stim, n_frames_after_stim, save_traces,
                                     output_path):
    n_trials_dict = dict()
    for file in nwb_files:
        session_dict = dict()
        session = nwb_read.get_session_id(file)
        mouse = session[0:5]
        print(f"Mouse: {mouse}, Session: {session}")

        # Get the data and associated timestamps
        traces = nwb_read.get_roi_response_serie_data(nwb_file=file, keys=rrs_keys)
        rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file=file, keys=rrs_keys)

        # Get trial table
        trial_table = nwb_read.get_trial_table(nwb_file=file)
        n_trials = len(trial_table)

        # Get full quiet windows traces
        selected_frames = []
        outcome = []
        pre_stim = n_frames_before_stim
        post_stim = n_frames_after_stim
        nf = pre_stim + post_stim
        for trial in range(n_trials):
            if trial_table.iloc[trial].trial_type != 'whisker_trial':
                continue
            stim_time = trial_table.iloc[trial].start_time
            stim_frame = utils_misc.find_nearest(rrs_ts, stim_time)
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
        rwd_wh_map = matplotlib.cm.get_cmap('Greens')
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
        rwd_wm_map = matplotlib.cm.get_cmap('Greys')
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
        nn_rwd_wh_map = matplotlib.cm.get_cmap('Reds')
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
        nn_rwd_wm_map = matplotlib.cm.get_cmap('Blues')
        nn_rwd_wm_colors = [nn_rwd_wm_map(i) for i in np.linspace(0.1, 1, nf)] * n_nn_rwd_whisker_miss

        # Save number of trials:
        session_dict['rwd_wh_hits'] = n_rwd_whisker_hits
        session_dict['rwd_wh_miss'] = n_rwd_whisker_miss
        session_dict['nn_rwd_wh_hits'] = n_nn_rwd_whisker_hits
        session_dict['nn_rwd_wh_miss'] = n_nn_rwd_whisker_miss

        n_trials_dict[f'{session}'] = session_dict

        # Concatenate session data
        data = np.concatenate((rwd_whisker_hits, rwd_whisker_miss, nn_rwd_whisker_hits, nn_rwd_whisker_miss), axis=1)
        umap_colors = rwd_wh_colors + rwd_wm_colors + nn_rwd_wh_colors + nn_rwd_wm_colors
        if save_traces:
            result_folder = os.path.join(output_path, f'{mouse}')
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            np.save(os.path.join(f"{result_folder}", f"{session}_concat_psths.npy"), data)

        # ¨Plot traces
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i in range(rwd_whisker_hits.shape[0]):
            axes[0, 0].plot(rwd_whisker_hits[i, :])
        for i in range(rwd_whisker_miss.shape[0]):
            axes[0, 1].plot(rwd_whisker_miss[i, :])
        for i in range(nn_rwd_whisker_hits.shape[0]):
            axes[1, 0].plot(nn_rwd_whisker_hits[i, :])
        for i in range(nn_rwd_whisker_miss.shape[0]):
            axes[1, 1].plot(nn_rwd_whisker_miss[i, :])
        fig.tight_layout()
        result_folder = os.path.join(output_path, f'{mouse}')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        save_formats = ['png', 'svg']
        for save_format in save_formats:
            fig.savefig(os.path.join(result_folder, f'{session}_traces_all_trials.{save_format}'))
        plt.close('all')

        # umap_dimensionality_reduction(ci_traces=np.transpose(quiet_window_traces), color_arg=colors.astype(int))
        fig, ax = umap_dimensionality_reduction(ci_traces=np.transpose(data),
                                                title=nwb_read.get_session_id(file),
                                                color_arg=umap_colors)
        fig.tight_layout()
        result_folder = os.path.join(output_path, f'{mouse}')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        save_formats = ['png', 'svg']
        for save_format in save_formats:
            fig.savefig(os.path.join(result_folder, f'{session}_umap_all_trials.{save_format}'))
        plt.close('all')

    with open(os.path.join(output_path, "n_trials.yaml"), 'w') as f:
        yaml.dump(n_trials_dict, f, default_flow_style=False, explicit_start=True)


def mice_average_wf_umap_whisker_stim(nwb_files, rrs_keys, trial_dict, trial_names, epochs, t_range, output_path):

    data_frame = return_events_aligned_wf_table(nwb_files=nwb_files,
                                                rrs_keys=rrs_keys,
                                                trials_dict=trial_dict,
                                                trial_names=trial_names,
                                                epochs=epochs,
                                                time_range=t_range,
                                                subtract_baseline=True)

    # Group data by sessions
    print(' ')
    print('Average data by session')
    data_frame = data_frame.drop(["behavior_day", "behavior_type", "roi", "event"], axis=1)
    session_avg_data = data_frame.groupby(["mouse_id", "session_id", "trial_type", "epoch",
                                          "cell_type", "time"],
                                          as_index=False).agg(np.nanmean)
    # Group session data by mice
    print(' ')
    print('Average data by mouse')
    mice_avg_data = session_avg_data.drop(['session_id'], axis=1)
    mice_avg_data = mice_avg_data.groupby(["mouse_id", "trial_type", "epoch",
                                           "cell_type", "time"],
                                          as_index=False).agg(np.nanmean)
    # General average
    print(' ')
    print('Average data across mice')
    general_average_data = mice_avg_data.drop(['mouse_id'], axis=1)
    general_average_data = general_average_data.groupby(["trial_type", "epoch", "cell_type", "time"],
                                                        as_index=False).agg(np.nanmean)
    #  Plot PSTHs
    fig = sns.relplot(data=general_average_data, x='time', y='activity', hue='cell_type', col='trial_type',
                      row='epoch', kind='line')
    fig.set_titles(template='{col_name}')
    fig.tight_layout()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig.savefig(os.path.join(output_path, f'psths.png'))

    # From table to 2D array
    # Rewarded whisker hits
    rwd_hits_activity = general_average_data.loc[(general_average_data.epoch == 'rewarded') &
                                                 (general_average_data.trial_type == 'whisker_hit')].activity.values[:]
    rwd_hits_array = np.reshape(rwd_hits_activity, (8, int(len(rwd_hits_activity)/8)))

    # Rewarded whisker miss
    rwd_miss_activity = general_average_data.loc[(general_average_data.epoch == 'rewarded') &
                                                 (general_average_data.trial_type == 'whisker_miss')].activity.values[:]
    rwd_miss_array = np.reshape(rwd_miss_activity, (8, int(len(rwd_miss_activity)/8)))

    # Non-rewarded whisker hits
    nn_rwd_hits_activity = general_average_data.loc[(general_average_data.epoch == 'non-rewarded') &
                                                    (general_average_data.trial_type == 'whisker_hit')].activity.values[:]
    nn_rwd_hits_array = np.reshape(nn_rwd_hits_activity, (8, int(len(nn_rwd_hits_activity)/8)))

    # Non-rewarded whisker miss
    nn_rwd_miss_activity = general_average_data.loc[(general_average_data.epoch == 'non-rewarded') &
                                                    (general_average_data.trial_type == 'whisker_miss')].activity.values[:]
    nn_rwd_miss_array = np.reshape(nn_rwd_miss_activity, (8, int(len(nn_rwd_miss_activity)/8)))

    # Concatenate data:
    data = np.concatenate((rwd_hits_array, rwd_miss_array, nn_rwd_hits_array, nn_rwd_miss_array), axis=1)

    # Color code :
    data_colors = create_umap_color_code(n_data_points=int(data.shape[1] / 4))

    # UMAP:
    fig, ax = umap_dimensionality_reduction(ci_traces=np.transpose(data),
                                            title='UMAP on general averaged traces',
                                            color_arg=data_colors)
    fig.tight_layout()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig.savefig(os.path.join(output_path, f'umap.png'))


config_file = r"Z:\z_LSENS\Share\Pol_Bech\Session_list\context_sessions_gcamp_expert.yaml"
# config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
# sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
# sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
# sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
# files = [session[1] for session in sessions]
files = config_dict['Session path']

# session_to_do = ["RD049_20240529_132745",
#                  "RD049_20240530_125954",
#                  "RD049_20240601_125108"]
# files = from_session_list_to_path_list(session_to_do)

output_root = os.path.join(f'{server_path.get_experimenter_saving_folder_root("RD")}',
                           'Pop_results', 'Context_behaviour')
folder = 'UMAP_20240709_all_trials'
saving_folder = os.path.join(output_root, folder)
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']

# Single session UMAP
# single_session_wf_umap_whisker_stim(nwb_files=files, rrs_keys=keys, n_frames_before_stim=20, n_frames_after_stim=25,
#                                     output_path=saving_folder, save_tr=True)

# Trial based UMAP:
trial_based_wf_umap_whisker_stim(nwb_files=files, rrs_keys=keys, n_frames_before_stim=20, n_frames_after_stim=25,
                                 save_traces=True, output_path=saving_folder)

# Average within mice then across mice and plot
# trial_selection_dict = [{'whisker_stim': [1], 'lick_flag': [1]}, {'whisker_stim': [1], 'lick_flag': [0]}]
# trial_name = ['whisker_hit', 'whisker_miss']
# epoch = ['rewarded', 'non-rewarded']
# range_t = (0.25, 0.5)
# mice_average_wf_umap_whisker_stim(nwb_files=files, rrs_keys=keys, trial_dict=trial_selection_dict,
#                                   trial_names=trial_name, epochs=epoch, t_range=range_t, output_path=saving_folder)
