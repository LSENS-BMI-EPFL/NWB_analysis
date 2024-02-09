import os

import numpy as np
import pandas as pd
import seaborn as sns

import nwb_wrappers.nwb_reader_functions as nwb_read
from utils import server_path, utils_two_photons, utils_behavior


nwb_list =  ['AR103_20230827_180738.nwb',
            #  'AR103_20230826_173720.nwb',
            #  'AR103_20230825_190303.nwb',
            #  'AR103_20230824_100910.nwb',
             'AR103_20230823_102029.nwb']

time_range = (2,4)
trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
epoch_name = 'unmotivated'
cell_types = ['wS2', 'wM1']

nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = [os.path.join(nwb_path, nwb) for nwb in nwb_list]

dfs = []
for nwb_file in nwb_list:
    mouse_id = nwb_read.get_mouse_id(nwb_file)
    session_id = nwb_read.get_session_id(nwb_file)
    behavior_type, behavior_day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)

    # Load trial events, traces, time stamps, cell type and epochs.
    events_ts = nwb_read.get_trial_timestamps_from_table(nwb_file, trial_selection)[0]
    # Keep start time.
    events_ts = events_ts[0]
    traces = nwb_read.get_roi_response_serie_data(nwb_file, 'dff')
    traces_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, 'dff')
    sampling_rate = np.round(nwb_read.get_rrs_sampling_rate(nwb_file, 'dff'))
    cell_type_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file, 'dff')
    epochs = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_name)

    print('Loaded data')
    # Filter and reshape data to pandas dataframe.
    events_filtered = utils_behavior.filter_events_based_on_epochs(events_ts, epochs)

    for cell_type, rois in cell_type_dict.items():
        print(cell_type)
        traces_filtered = traces[rois]
        activity_aligned = utils_two_photons.center_rrs_on_events(traces_filtered, traces_ts,
                                                                events_filtered, time_range,
                                                                sampling_rate)
        n_cells, n_events, n_t = activity_aligned.shape
        print(n_t)
        time_stamps_vect = np.linspace(-time_range[0], time_range[1], n_t)
        time_stamps_vect = np.tile(time_stamps_vect, n_cells * n_events)
        event_vect = np.tile(np.repeat(np.arange(n_events), n_t), n_cells)
        rois_vect = np.repeat(rois, n_events * n_t)
        activity_reshaped = activity_aligned.flatten()
        
        # TODO: change dtypes.
        df = dict({'trace':activity_reshaped, 'time':time_stamps_vect, 'event':event_vect, 'roi':rois_vect})
        df = pd.DataFrame.from_dict(df)
        df['cell_type'] = cell_type
        df['mouse_id'] = str(mouse_id)
        df['session_id'] = session_id
        df['behavior_type'] = behavior_type
        df['behavior_day'] = behavior_day
        dfs.append(df)
dfs = pd.concat(dfs)

temp = dfs.groupby(['session_id', 'roi', 'time', 'cell_type', 'behavior_type', 'behavior_day'], as_index=False).agg(np.mean)
sns.lineplot(data=dfs, x='time', y='trace', hue='cell_type', n_boot=100)
        
