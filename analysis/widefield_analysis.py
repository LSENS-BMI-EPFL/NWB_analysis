import math
import os

# import dask.array as da
import numpy as np
import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import server_path, utils_misc, utils_behavior
from scripts.psth_analysis import return_events_aligned_data_table
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

            frames = []
            for tstamp in epoch_times[0]:
                if tstamp < 10:
                    continue
                frame = utils_misc.find_nearest(wf_timestamps, tstamp)
                data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - 200, frame + 200)
                frames.append(data)

            data_frames = np.array(frames)
            data_frames = np.stack(data_frames, axis=0)
            avg_data = np.nanmean(data_frames, axis=0)
            save_path = os.path.join(output_path, f"{mouse_id}", f"{session_id}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            tiff.imwrite(os.path.join(save_path, f'to_{epoch}.tiff'), avg_data)


if __name__ == "__main__":
    experimenter_initials = "PB"

    root_path = server_path.get_experimenter_nwb_folder(experimenter_initials)

    output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root(experimenter_initials)}',
                               'Pop_results', 'Context_behaviour')
    all_nwb_names = os.listdir(root_path)

    subject_ids = ['RD039']
    # plots_to_do = ['single_session']
    session_to_do = ["RD039_20240208_143129"]

    for subject_id in subject_ids:
        nwb_names = [name for name in all_nwb_names if subject_id in name]
        nwb_files = []
        for session in session_to_do:
            nwb_files += [os.path.join(root_path, name) for name in nwb_names if session in name]

        print(f"nwb_files : {nwb_files}")

        # plot_wf_activity(nwb_files, output_path)

        trials_dict = [{'whisker_stim': [1], 'lick_flag': [1]},
                       {'whisker_stim': [1], 'lick_flag': [0]},
                       {'auditory_stim': [1], 'lick_flag': [1]},
                       {'auditory_stim': [1], 'lick_flag': [0]}]

        trial_name = ['whisker_hit',
                      'whisker_miss',
                      'auditory_hit',
                      'auditory_miss']

        epochs = ['rewarded', 'non-rewarded']

        full_df = []
        for index, trial_dict in enumerate(trials_dict):
            for epoch_index, epoch in enumerate(epochs):
                data_table = return_events_aligned_data_table(nwb_list=nwb_files,
                                                              rrs_keys=['ophys', 'brain_area_fluorescence', 'dff0_traces'],
                                                              time_range=(2, 4),
                                                              trial_selection=trials_dict[index],
                                                              epoch=epoch)
                data_table['trial_type'] = trial_name[index]
                data_table['epoch'] = epochs[epoch_index]
                full_df.append(data_table)
        full_df = pd.concat(full_df, ignore_index=True)
        fig = plt.subplots()
        data_to_plot = full_df.loc[(full_df.cell_type == 'wS1') & (full_df.trial_type.isin(['auditory_hit', 'auditory_miss']))]
        sns.lineplot(data=data_to_plot, x='time', y='activity', hue='epoch', style='trial_type')
        plt.show()
