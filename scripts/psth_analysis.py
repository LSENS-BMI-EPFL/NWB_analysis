import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import server_path, utils_two_photons, utils_behavior


nwb_list =  [
            # 'AR103_20230823_102029.nwb',
            #  'AR103_20230826_173720.nwb',
            #  'AR103_20230825_190303.nwb',
            #  'AR103_20230824_100910.nwb',
            #  'AR103_20230827_180738.nwb',
             'GF333_21012021_125450.nwb'
             ]


rrs_name = 'dcnv'
time_range = (1,5)
trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
epoch_name = 'unmotivated'

nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = [os.path.join(nwb_path, nwb) for nwb in nwb_list]

dfs = []
for nwb_file in nwb_list:
    mouse_id = nwb_read.get_mouse_id(nwb_file)
    session_id = nwb_read.get_session_id(nwb_file)
    behavior_type, behavior_day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)

    # Load trial events, activity, time stamps, cell type and epochs.
    events = nwb_read.get_trial_timestamps_from_table(nwb_file, trial_selection)[0]
    # Keep start time.
    events = events[0]
    activity = nwb_read.get_roi_response_serie_data(nwb_file, rrs_name)
    activity_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, rrs_name)
    sampling_rate = np.round(nwb_read.get_rrs_sampling_rate(nwb_file, rrs_name))
    cell_type_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file, rrs_name)
    epochs = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_name)
    print('Loaded data')

    # Filter events based on epochs.
    if epochs:
        events = utils_behavior.filter_events_based_on_epochs(events, epochs)        

    if cell_type_dict:
        for cell_type, rois in cell_type_dict.items():
            # Filter cells.
            activity_filtered = activity[rois]
            # Get data organized around events.
            df = utils_two_photons.select_activity_around_events(activity_filtered, activity_ts, rois, events,
                                                                 time_range, sampling_rate,
                                                                 mouse_id=mouse_id, session_id=session_id,
                                                                 behavior_type=behavior_type,
                                                                 behavior_day=behavior_day,
                                                                 cell_type=cell_type)
            dfs.append(df)
    else:
        # Filter cells.
        rois = np.arange(activity.shape[0])
        # Get data organized around events.
        df = utils_two_photons.select_activity_around_events(activity, activity_ts, rois, events,
                                                             time_range, sampling_rate,
                                                             mouse_id=mouse_id, session_id=session_id,
                                                             behavior_type=behavior_type,
                                                             behavior_day=behavior_day)
        dfs.append(df)
dfs = pd.concat(dfs, ignore_index=True)

temp = dfs.groupby(['mouse_id', 'session_id', 'roi', 'time', 'cell_type', 'behavior_type', 'behavior_day'], as_index=False).agg(np.nanmean)
temp = temp.astype({'roi': str})
f = plt.figure()
sns.lineplot(data=temp.loc[temp.roi.isin(['10','38','44'])], x='time', y='activity', hue='roi', style='session_id', n_boot=100)

temp = dfs.groupby(['mouse_id', 'session_id', 'roi', 'time', 'behavior_type', 'behavior_day'], as_index=False).agg(np.nanmean)
temp = temp.astype({'roi': str})
f = plt.figure()
sns.lineplot(data=temp.loc[temp.roi.isin(['10','38','44'])], x='time', y='activity', hue='roi', style='session_id', n_boot=100)

temp.roi.unique()

temp.loc[temp.roi==15, 'trace'].plot()

dfs.loc[dfs.roi==0].plot(y='activity', use_index=False)
dfs.loc[dfs.roi==0].plot(y='activity', use_index=True)

dfs.roi.unique()

dfs.iloc[:190]

a = np.concatenate((np.ones(3)*0, np.ones(3)*1, np.ones(3)*2)).reshape((3,3))

a.flatten()

for icell in temp.loc[temp.cell_type=='wM1', 'roi']
temp.loc[temp.cell_type=='wM1', 'roi'].unique()


data = dfs.loc[dfs.roi==10]

plt.figure()
for ievent in range(50):
    plt.plot(data.loc[data.event==ievent, 'activity'].to_numpy()+ievent*10)
plt.axvline(90, color='k')
plt.axvline(271, color='k')

