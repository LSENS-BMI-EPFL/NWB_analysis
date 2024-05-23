import nwb_wrappers.nwb_reader_functions as nwb_read
import matplotlib.pyplot as plt
import matplotlib.colors as matcol
import matplotlib
import seaborn as sns
import umap
import os
import yaml
import numpy as np
from nwb_utils import utils_misc, server_path
from sklearn.preprocessing import StandardScaler


def umap_dimensionality_reduction(ci_traces, title, color_arg=None, alpha=None):
    reducer = umap.UMAP()
    scaled_traces = StandardScaler().fit_transform(ci_traces)
    embedding = reducer.fit_transform(scaled_traces)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
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


def wf_umap_whisker_stim(nwb_files, rrs_keys, n_frames_before_stim, n_frames_after_stim, output_path):
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
        # Rewarded whisker hit (correct)
        rwd_wh_map = matplotlib.cm.get_cmap('Greens')
        rwd_wh_colors = [rwd_wh_map(i) for i in np.linspace(0.1, 1, nf)]
        # Rewarded whisker miss (incorrect)
        rwd_wm_map = matplotlib.cm.get_cmap('Greys')
        rwd_wm_colors = [rwd_wm_map(i) for i in np.linspace(0.1, 1, nf)]
        # Non-rewarded whisker hit (incorrect)
        nnrwd_wh_map = matplotlib.cm.get_cmap('Reds')
        nnrwd_wh_colors = [nnrwd_wh_map(i) for i in np.linspace(0.1, 1, nf)]
        # Non-rewarded whisker miss (correct)
        nnrwd_wm_map = matplotlib.cm.get_cmap('Blues')
        nnrwd_wm_colors = [nnrwd_wm_map(i) for i in np.linspace(0.1, 1, nf)]

        # Build list of colors that is the same length than list of data-points
        data_colors = rwd_wh_colors + rwd_wm_colors + nnrwd_wh_colors + nnrwd_wm_colors

        # Plot the mean baseline subtracted psths
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        for cell in range(nn_rwd_whisker_miss.shape[0]):
            axes[0, 0].plot(rwd_whisker_hits[cell, :])
            axes[0, 0]. set_title('RWD context - Whit')
            axes[0, 1].plot(rwd_whisker_miss[cell, :])
            axes[0, 1].set_title('RWD context - Wmiss')
            axes[1, 0].plot(nn_rwd_whisker_hits[cell, :])
            axes[1, 0].set_title('nn-RWD context - Whit')
            axes[1, 1].plot(nn_rwd_whisker_miss[cell, :])
            axes[1, 1].set_title('nn-RWD context - Wmiss')
        fig.suptitle(f'{nwb_read.get_session_id(file)}', fontsize=14)
        fig.tight_layout()
        result_folder = os.path.join(output_path, f'{mouse}')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        fig.savefig(os.path.join(result_folder, f'{session}_traces.png'))
        plt.close()

        # umap_dimensionality_reduction(ci_traces=np.transpose(quiet_window_traces), color_arg=colors.astype(int))
        fig, ax = umap_dimensionality_reduction(ci_traces=np.transpose(data),
                                                title=nwb_read.get_session_id(file),
                                                color_arg=data_colors)
        fig.tight_layout()
        result_folder = os.path.join(output_path, f'{mouse}')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        fig.savefig(os.path.join(result_folder, f'{session}_umap.png'))
        plt.close()


config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
# sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
# sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
files = [session[1] for session in sessions]

output_root = os.path.join(f'{server_path.get_experimenter_saving_folder_root("RD")}',
                           'Pop_results', 'Context_behaviour')
folder = 'UMAP_20240523'
saving_folder = os.path.join(output_root, folder)
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']

wf_umap_whisker_stim(nwb_files=files, rrs_keys=keys, n_frames_before_stim=25, n_frames_after_stim=50,
                     output_path=saving_folder)
