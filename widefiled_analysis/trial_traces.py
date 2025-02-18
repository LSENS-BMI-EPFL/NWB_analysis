import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns

import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import server_path, utils_misc, utils_behavior


def save_pixel_traces(nwb_file, epoch, trial_type, frames_before, frames_after):
    session_id = nwb_read.get_session_id(nwb_file)
    print(f"Session: {session_id}")
    mouse_id = session_id[0: 5]
    trials = nwb_read.get_behavioral_events_times(nwb_file, trial_type)[0]
    epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, epoch)
    trials_kept = utils_behavior.filter_events_based_on_epochs(events_ts=trials, epochs=epoch_times)

    wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

    data = []
    stims = []
    for idx, tstamp in enumerate(trials_kept):
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        stims.append((idx + 1) * frames_before + idx * frames_after)
        data.append(nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - frames_before, frame + frames_after))
    data = np.concatenate(data, axis=0)
    pixel_traces = np.reshape(data, newshape=(data.shape[1] * data.shape[2], data.shape[0]))

    root_saving_folder = server_path.get_experimenter_saving_folder_root('RD')
    output_path = os.path.join(root_saving_folder, 'Pop_results',
                               'Context_behaviour', 'pixel_traces_20240507')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.save(os.path.join(output_path, f'{session_id}_{epoch}_{trial_type}_px_traces.npy'), pixel_traces)
    np.save(os.path.join(output_path, f'{session_id}_{epoch}_{trial_type}_stims.npy'), stims)


config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
# sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
# sessions = config_dict['NWB_CI_LSENS']['Context_good_params']
# sessions = config_dict['NWB_CI_LSENS']['context_expert_widefield']
# sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
files = [session[1] for session in sessions]

nwb_file = files[0]
save_pixel_traces(nwb_file, epoch='rewarded', trial_type='whisker_hit_trial', frames_before=150, frames_after=100)


