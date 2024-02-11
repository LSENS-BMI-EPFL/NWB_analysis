import math
import os

# import dask.array as da
import numpy as np
import nwb_wrappers.nwb_reader_functions as nwb_read
from utils import server_path, utils_misc, utils_behavior
import tifffile as tiff
import yaml


def plot_wf_activity(nwb_files, output_path):
    for nwb_file in nwb_files:
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        session_type = nwb_read.get_session_type(nwb_file)
        if 'wf' not in session_type:
            print(f"{session_id} is not a widefield session")
            continue

        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

        for epoch in nwb_read.get_behavioral_epochs_names(nwb_file):
            epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, epoch)
            for trial_type in nwb_read.get_behavioral_events_names(nwb_file):
                print(f"Trial type : {trial_type}")
                trials = nwb_read.get_behavioral_events_times(nwb_file, trial_type)[0]
                print(f"Total of {len(trials)} trials")
                trials_kept = utils_behavior.filter_events_based_on_epochs(events_ts=trials, epochs=epoch_times)
                print(f"Total of {len(trials_kept)} trials in {epoch} epoch")
                frames = []
                for tstamp in trials:
                    frame = utils_misc.find_nearest(wf_timestamps, tstamp)
                    data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame-200, frame+200)
                    frames.append(data)

                data_frames = np.array(frames)
                data_frames = np.stack(data_frames, axis=0)
                avg_data = np.nanmean(data_frames, axis=0)
                save_path = os.path.join(output_path, f"{mouse_id}", f"{session_id}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                tiff.imwrite(os.path.join(save_path, f'{trial_type}_{epoch}.tiff'), avg_data)


if __name__ == "__main__":
    experimenter_initials = "PB"

    root_path = server_path.get_experimenter_nwb_folder(experimenter_initials)

    output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root(experimenter_initials)}',
                               'Pop_results', 'Context_behaviour')
    all_nwb_names = os.listdir(root_path)

    subject_ids = ['RD039', 'PB000']
    # plots_to_do = ['single_session']
    session_to_do = ["RD039_20240205_150044", "PB000_20240205_181158"]

    for subject_id in subject_ids:
        nwb_names = [name for name in all_nwb_names if subject_id in name]
        nwb_files = []
        for session in session_to_do:
            nwb_files += [os.path.join(root_path, name) for name in nwb_names if session in name]

        plot_wf_activity(nwb_files, output_path)
