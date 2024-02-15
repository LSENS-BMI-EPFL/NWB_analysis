import numpy as np
import pandas as pd

import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import utils_two_photons, utils_behavior


def return_events_aligned_data_table(nwb_list, rrs_keys, time_range, trial_selection, epoch):
    """

    :param nwb_list: list of NWBs file to analyze
    :param rrs_keys: list of successive keys to access a given rois response serie
    :param time_range: tuple defining time to keep around selected events
    :param trial_selection: dictionary used to filter out trial table
    :param epoch: use to keep event only if they occur in the selected epoch
    :return:
    """

    dfs = []
    for nwb_file in nwb_list:
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        behavior_type, behavior_day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)

        # Load trial events, activity, time stamps, cell type and epochs.
        events = nwb_read.get_trial_timestamps_from_table(nwb_file, trial_selection)[0]
        # Keep start time.
        events = events[0]
        activity = nwb_read.get_roi_response_serie_data(nwb_file, rrs_keys)
        activity_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, rrs_keys)
        sampling_rate = np.round(nwb_read.get_rrs_sampling_rate(nwb_file, rrs_keys))
        cell_type_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file, rrs_keys)
        epochs = nwb_read.get_behavioral_epochs_times(nwb_file, epoch)
        print('Loaded data')

        # Filter events based on epochs.
        if len(epochs) > 0:
            events = utils_behavior.filter_events_based_on_epochs(events, epochs)
        print(f"{len(events)} events")

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
    
    return dfs
