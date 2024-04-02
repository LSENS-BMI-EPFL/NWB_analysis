import numpy as np
import pandas as pd

import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import utils_two_photons, utils_behavior


def make_events_aligned_data_table(nwb_list, rrs_keys, time_range, trial_selection, epoch):
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
        print(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        print(f"Session ID : {session_id}")
        behavior_type, behavior_day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)

        # Load trial events, activity, time stamps, cell type and epochs.
        events = nwb_read.get_trial_timestamps_from_table(nwb_file, trial_selection)[0]
        # Keep start time.
        events = events[0]
        activity = nwb_read.get_roi_response_serie_data(nwb_file, rrs_keys)
        if activity is None:
            print(f'Session {session_id} has no rrs - skipping.')
            continue
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
                df = utils_two_photons.select_activity_around_events_pd(activity_filtered, activity_ts, rois, events,
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
            df = utils_two_photons.select_activity_around_events_pd(activity, activity_ts, rois, events,
                                                                 time_range, sampling_rate,
                                                                 mouse_id=mouse_id, session_id=session_id,
                                                                 behavior_type=behavior_type,
                                                                 behavior_day=behavior_day)
            dfs.append(df)
    dfs = pd.concat(dfs, ignore_index=True)
    
    return dfs


def make_events_aligned_array(nwb_list, rrs_keys, time_range, trial_selection, epoch):

    activity_dict = {}
    for nwb_file in nwb_list:
        print(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        behavior_type, behavior_day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)
        
        # Load trial events, activity, time stamps, cell type and epochs.
        events = nwb_read.get_trial_timestamps_from_table(nwb_file, trial_selection)[0]
        # Keep start time.
        events = events[0]
        activity = nwb_read.get_roi_response_serie_data(nwb_file, rrs_keys)
        if activity is None:
            print(f'Session {session_id} has no rrs - skipping.')
            continue
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
            arrays = []
            for _, rois in cell_type_dict.items():
                # Filter cells.
                activity_filtered = activity[rois]
                # Get data organized around events.
                activity_aligned = utils_two_photons.center_rrs_on_events(activity_filtered, activity_ts,
                                                                          events, time_range,
                                                                          sampling_rate)
                arrays.append(activity_aligned)
            # Join cell_type arrays into commun array of shape (n_types, n_cells, n_events, n_t).
            n = np.stack([a.shape for a in arrays])
            n = np.max(n, axis=0)
            # Add dim for n_cell_types.
            n = np.concatenate(([len(arrays)], n))
            # Initialize commun array of right size padded with nan's.
            data = np.full(n, np.nan)
            for itype, a in enumerate(arrays):
                s = a.shape
                data[itype, :s[0], :s[1], :s[2]] = a
            metadata = {'mouse_id': mouse_id,
                        'behavior_type': behavior_type,
                        'behavior_day': behavior_day,
                        'cell_type': dict(enumerate(cell_type_dict.keys()))}
            activity_dict[session_id] = {'data': data,
                                         'metadata': metadata}
            
        else:
            # Get data organized around events.
            activity_aligned = utils_two_photons.center_rrs_on_events(activity, activity_ts,
                                                    events, time_range,
                                                    sampling_rate)
            metadata = {'mouse_id': mouse_id,
                        'behavior_type': behavior_type,
                        'behavior_day': behavior_day,
                        'cell_type': {0:'all_cells'}}
            activity_dict[session_id] = {'data': activity_aligned[np.newaxis],
                                    'metadata': metadata}

    # Join session arrays into commun 6d array.
    # Find dims and preallocate .
    mouse_ids = [data['metadata']['mouse_id'] for _, data in activity_dict.items()]
    n_mice = len(np.unique(mouse_ids))
    n_session_per_mouse = max([mouse_ids.count(m) for m in np.unique(mouse_ids)])
    dims = []
    for session, data in activity_dict.items():
        dims.append(data['data'].shape)
    dims = np.max(dims, axis=0)
    dims = np.concatenate([[n_mice], [n_session_per_mouse], dims])
    array_6d = np.full(dims, np.nan)

    for session, data in activity_dict.items():
        mouse = data['metadata']['mouse_id']
        mouse_idx = list(np.unique(mouse_ids)).index(mouse)
        session_idx = [session for session in list(activity_dict.keys())
                    if mouse in session]
        session_idx = session_idx.index(session)
        s = data['data'].shape
        array_6d[mouse_idx, session_idx, :s[0], :s[1], :s[2], :s[3]] = data['data']

    return array_6d


