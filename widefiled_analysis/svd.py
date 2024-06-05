import os
import matplotlib.colors
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import nwb_wrappers.nwb_reader_functions as nwb_read

from scipy.linalg import svd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF, TruncatedSVD
from utils.wf_plotting_utils import plot_single_frame
from nwb_utils import server_path, utils_misc, utils_behavior


def get_frames_by_epoch(nwb_file, trials, wf_timestamps):
    frames = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - 200, frame + 200)
        if data.shape != (400, 125, 160):
            continue
        frames.append(data)

    data_frames = np.array(frames)
    data_frames = np.stack(data_frames, axis=0)
    return data_frames


def plot_variance_explained(s, ncomps, output_path):
    # ncomps = 20
    varS = np.power(s, 2) / sum(np.power(s, 2))
    fig = plt.figure(dpi=120)
    plt.bar(x=[i for i in range(1, ncomps + 1)], height=varS[:ncomps] * 100, color='y', label='explained variance')
    plt.plot(range(1, ncomps + 1), np.cumsum(varS[:ncomps]) * 100, 'ro-', label='cummulative explained variance')
    plt.axhline(80, linestyle='--')  # Line marking 80% of variance explained
    plt.xlabel('# Components')
    plt.ylim(0, 105)
    plt.xticks(range(1, ncomps + 1, 2))
    plt.ylabel('variance explained')
    plt.legend()
    fig.show()
    fig.savefig(os.path.join(output_path, "variance_explained.png"))
    return

def compute_svd(data):
    return


def plot_component(trial_table, pxdata, wf_timestamps, U, s, v, component, output_path):

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig = plt.figure()

    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax0.imshow(U[:, component].reshape(125, 160))
    ax1.plot(wf_timestamps[:10000], pxdata, label='data')
    min, max = np.nanmin(pxdata), np.nanmax(pxdata)
    ax1.vlines(trial_table.loc[trial_table.start_time < 100, 'start_time'], ymin=min, ymax=max, color='r',
                 linestyle='--', label='trial start')
    ax1.vlines(trial_table.loc[trial_table.lick_time < 100, 'lick_time'], ymin=min, ymax=max, color='k',
                 linestyle='--', label='first lick')

    min, max = np.nanmin(s[component]*v[component, :]), np.nanmax(s[component]*v[component, :])
    ax2.plot(wf_timestamps[:10000], s[component]*v[component, :], label=f'component {component + 1}')
    ax2.vlines(trial_table.loc[trial_table.start_time < 100, 'start_time'], ymin=-0.05, ymax=0.15, color='r',
                 linestyle='--', label='trial start')
    ax2.vlines(trial_table.loc[trial_table.lick_time < 100, 'lick_time'], ymin=-0.05, ymax=0.15, color='k',
                 linestyle='--', label='first lick')
    fig.legend()
    fig.show()
    fig.savefig(os.path.join(output_path, f"random_pixel_component_{component}.png"))

def compute_svd_and_plot(nwb_files, output_path, nframes=10000):

    for nwb_file in nwb_files:
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        trial_table = nwb_read.get_trial_table(nwb_file)

        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        wf_data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], 0, nframes)
        utils_misc.find_nearest()
        U, s, v = svd(np.nan_to_num(wf_data.reshape(-1, 20000).T, 0), full_matrices=False, compute_uv=True)
        print(f"U matrix has shape {U.shape}, s {s.shape} and Vt {v.shape}" )




    return
if __name__ == '__main__':
    config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
    session_to_do = [session[0] for session in sessions]

    subject_ids = list(np.unique([session[0:5] for session in session_to_do]))

    experimenter_initials = subject_ids[0][0:2]

    root_path = server_path.get_experimenter_nwb_folder(experimenter_initials)

    output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root("PB")}',
                               'Pop_results', 'Context_behaviour', 'SVD')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_nwb_names = os.listdir(root_path)

    session_dit = {'Sessions': session_to_do}
    with open(os.path.join(output_path, "session_to_do.yaml"), 'w') as stream:
        yaml.dump(session_dit, stream, default_flow_style=False, explicit_start=True)

    nwb_files = []
    for subject_id in subject_ids:
        nwb_names = [name for name in all_nwb_names if subject_id in name]
        for session in session_to_do:
            nwb_files += [os.path.join(root_path, name) for name in nwb_names if session in name]
        print(" ")
        print(f"nwb_files : {nwb_files}")

    compute_svd_and_plot(nwb_files, output_path)