"""
This file defines NWB reader functions (inspired from CICADA NWB_wrappers).
"""

import ast

import numpy as np
from pynwb import NWBHDF5IO
from pynwb.base import TimeSeries


def get_subject_info(nwb_file):
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()
    subject = nwb_data.subject

    return subject


def get_mouse_id(nwb_file):
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()
    mouse_id = nwb_data.subject.subject_id

    return mouse_id


def get_session_id(nwb_file):
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()
    session_id = nwb_data.session_id

    return session_id


def get_nwb_file_metadata(nwb_file):
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()
    session_metadata = nwb_data.subject

    return session_metadata


def get_session_metadata(nwb_file):
    """Get session-level metadata.
     Converts string of dictionary into a dictionary."""
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()
    session_metadata = ast.literal_eval(nwb_data.experiment_description.replace("nan", "None"))
    
    # Adding session type and day.
    # Read behaviour_type and day from session_description, encoded at creation as behavior_type_<day>
    description = nwb_data.session_description.split('_')
    if description[0] == 'free':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    elif description[1] == 'psy':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    elif description[1] == 'on':
        behavior_type = description[0] + '_' + description[1] + '_' + description[2]
        day = int(description[2])
    elif description[1] == 'off':
        behavior_type = description[0] + '_' + description[1]
        # day = int(description[2]) todo : fix to add a day also for whisker_off
        day = int(1)
    elif description[1] == 'context':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    else:
        behavior_type = description[0]
        day = int(description[1])
        
    session_metadata['behavior_type'] = behavior_type
    session_metadata['day'] = day

    return session_metadata


def get_session_type(nwb_file):
    """Get session-type from metadata.
     """
    session_metadata = get_session_metadata(nwb_file)

    return session_metadata['session_type']


# TODO: output string for day
def get_bhv_type_and_training_day_index(nwb_file):
    """
    This function extracts the behavior type and training day index, relative to whisker training start, from a NWB file.
    :param nwb_file:
    :return:
    """
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    # Read behaviour_type and day from session_description, encoded at creation as behavior_type_<day>
    description = nwb_data.session_description.split('_')
    if description[0] == 'free':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    elif description[1] == 'psy':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    elif description[1] == 'on':
        behavior_type = description[0] + '_' + description[1] + '_' + description[2]
        day = int(description[2])
    elif description[1] == 'off':
        behavior_type = description[0] + '_' + description[1]
        # day = int(description[2]) todo : fix to add a day also for whisker_off
        day = int(1)
    elif description[1] == 'context':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    else:
        behavior_type = description[0]
        day = int(description[1])
    
    return behavior_type, day


def get_trial_table(nwb_file):
    """
    This function extracts the trial table from a NWB file.
    :param nwb_file:
    :return:
    """
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()
    nwb_objects = nwb_data.objects
    objects_list = [data for key, data in nwb_objects.items()]
    data_to_take = None

    # Iterate over NWB objects but keep "trial"
    for obj in objects_list:
        if 'trial' in obj.name:
            data = obj
            if isinstance(data, TimeSeries):
                continue
            else:
                data_to_take = data
                break
        else:
            continue
    trial_data_frame = data_to_take.to_dataframe()

    return trial_data_frame


def get_roi_response_serie(nwb_file, keys):
    """

    Args:
        keys: list of string allowing to get the roi repsonse series wanted

    Returns:

    """

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if len(keys) < 3:
        return None
    if keys[0] not in nwb_data.modules:
        return None

    if keys[1] not in nwb_data.modules[keys[0]].data_interfaces:
        return None

    if keys[2] not in nwb_data.modules[keys[0]].data_interfaces[keys[1]].roi_response_series:
        return None

    return nwb_data.modules[keys[0]].data_interfaces[keys[1]].roi_response_series[keys[2]]


def get_roi_response_serie_data(nwb_file, keys):
    """_summary_

    Args:
        nwb_file (_type_): _description_
        keys (_type_): list of successive keys to access rrs

    Returns:
        _type_: _description_
    """

    rrs = get_roi_response_serie(nwb_file, keys)
    
    if rrs is None:
        return None
    else:
        return np.transpose(rrs.data[:])


def get_roi_response_serie_timestamps(nwb_file, keys, verbose=True):

    rrs = get_roi_response_serie(nwb_file, keys)
    rrs_ts = rrs.timestamps[:]

    if rrs_ts is not None:
        if verbose:
            print(f"Timestamps directly provided for this RoiResponseSeries ({keys[2]})")
        return rrs_ts
    else:
        # In case rate rather than timestamps.
        if verbose:
            print(f"Timestamps not directly provided for this RoiResponseSeries ({keys[2]})")
        rrs_start_time = rrs.starting_time
        rrs_rate = rrs.rate
        if (rrs_rate is not None) and (rrs_rate < 1):
            if verbose:
                print(f"Found a rate of {np.round(rrs_rate, decimals=3)} Hz, assume it is in fact 1/rate. "
                        f"New rate: {np.round(1 / rrs_rate, decimals=2)} Hz.")
            rrs_rate = 1 / rrs_rate
        if (rrs_start_time is not None) and (rrs_rate is not None):
            if verbose:
                print(f"Build timestamps from starting time ({rrs_start_time} s) and "
                        f"rate ({np.round(rrs_rate, decimals=2)} Hz)")
            n_times = rrs.data.shape[0]
            rrs_ts = (np.arange(0, n_times) * (1/rrs_rate)) + rrs_start_time
            return rrs_ts
        else:
            if verbose:
                print("Starting time and rate not provided neither, no timestamps can be returned")
            return None


def get_behavioral_events_names(nwb_file):

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if 'behavior' not in nwb_data.processing:
        if 'Behavior' not in nwb_data.processing:
            return []

    try:
        behavior_nwb_module = nwb_data.processing['behavior']
    except KeyError:
        behavior_nwb_module = nwb_data.processing['Behavior']

    try:
        behavioral_events = behavior_nwb_module.get(name='BehavioralEvents')
    except KeyError:
        return []

    # a dictionary containing the TimeSeries in this BehavioralEvents container
    time_series = behavioral_events.time_series

    return list(time_series.keys())


def get_behavioral_events_times(nwb_file, event_name):

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if 'behavior' not in nwb_data.processing:
        if 'Behavior' not in nwb_data.processing:
            return []

    try:
        behavior_nwb_module = nwb_data.processing['behavior']
    except KeyError:
        behavior_nwb_module = nwb_data.processing['Behavior']

    try:
        behavioral_events = behavior_nwb_module.get(name='BehavioralEvents')
    except KeyError:
        return []
    # a dictionary containing the TimeSeries in this BehavioralEvents container
    time_series = behavioral_events.time_series
    if event_name not in time_series.keys():
        return None
    events_time_serie = time_series[event_name]
    one_d_event_timestamps = events_time_serie.timestamps[:]
    # Event only have one value - Match epoch (start, stop) structure.
    # (just duplicate first value).
    event_timestamps = np.tile(one_d_event_timestamps, (2, 1))

    return event_timestamps


def get_behavioral_epochs_names(nwb_file):
    """
    The name of the different behavioral
    Returns:

    """
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if 'behavior' not in nwb_data.processing:
        if 'Behavior' not in nwb_data.processing:
            return []

    try:
        behavior_nwb_module = nwb_data.processing['behavior']
    except KeyError:
        behavior_nwb_module = nwb_data.processing['Behavior']

    try:
        behavioral_epochs = behavior_nwb_module.get(name='BehavioralEpochs')
    except KeyError:
        return []
    # a dictionary containing the IntervalSeries in this BehavioralEpochs container
    interval_series = behavioral_epochs.interval_series

    return list(interval_series.keys())


def get_behavioral_epochs_times(nwb_file, epoch_name):
    """
    Return an interval times (start and stop in seconds) as a numpy array of 2*n_times.
    Args:
        epoch_name: Name of the interval to retrieve

    Returns: None if the interval doesn't exists or a 2d array

    Stores intervals of data. The timestamps field stores the beginning and end of intervals.
    The data field stores whether the interval just started (>0 value) or ended (<0 value).
    Different interval types can be represented in the same series by using multiple key values
    (eg, 1 for feature A, 2 for feature B, 3 for feature C, etc). The field data stores an 8-bit integer.
    This is largely an alias of a standard TimeSeries but that is identifiable as representing
    time intervals in a machine-readable way.

    """

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if 'behavior' not in nwb_data.processing:
        if 'Behavior' not in nwb_data.processing:
            return []

    try:
        behavior_nwb_module = nwb_data.processing['behavior']
    except KeyError:
        behavior_nwb_module = nwb_data.processing['Behavior']

    try:
        behavioral_epochs = behavior_nwb_module.get(name='BehavioralEpochs')
    except KeyError:
        return None
    # a dictionary containing the IntervalSeries in this BehavioralEpochs container
    interval_series = behavioral_epochs.interval_series

    if epoch_name not in interval_series:
        return None

    interval_serie = interval_series[epoch_name]

    # data: >0 if interval started, <0 if interval ended.
    # timestamps: Timestamps for samples stored in data
    # so far we use only one type of integer, but otherwise as describe in the doc:
    data = interval_serie.data
    time_stamps = interval_serie.timestamps

    data = np.zeros((2, int(len(time_stamps) / 2)))
    index_data = 0
    for i in np.arange(0, len(time_stamps), 2):
        data[0, index_data] = time_stamps[i]
        data[1, index_data] = time_stamps[i+1]
        index_data += 1

    return data


def get_rrs_sampling_rate(nwb_file, keys):
    """

    Args:

    Returns: (float) sampling rate of the movie, return None if no sampling rate is found

    """

    rrs = get_roi_response_serie(nwb_file, keys)

    sampling_rate = rrs.rate
    if sampling_rate is not None:
        print(f"Sampling rate is directly provided in {keys[2]}")
        print(f"Sampling rate: {sampling_rate} Hz")
        return sampling_rate
    else:
        print(f"Sampling rate is not directly provided in {keys[2]}:"
              f" Estimate rate from timestamps of first 100 frames")
        rrs_ts_sample = rrs.timestamps[0: 100]
        if rrs_ts_sample is None:
            print("Found neither rate nor timestamps")
            return None
        else:
            sampling_rate = np.round(1 / (np.nanmedian(np.diff(rrs_ts_sample))), decimals=2)
            print(f"Sampling rate: {sampling_rate} Hz")
            return sampling_rate


def get_widefield_dff0(nwb_file, keys, start, stop):
    """

    Args:
        keys: lsit of string allowing to get the roi repsonse series wanted

    Returns:

    """

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if len(keys) < 2:
        return None

    if keys[0] not in nwb_data.modules:
        return None

    if keys[1] not in nwb_data.modules[keys[0]].data_interfaces:
        return None

    return nwb_data.modules[keys[0]].data_interfaces[keys[1]].data[start:stop, :, :]


def get_widefield_dff0_traces(nwb_file, keys):
    """

    Args:
        keys: lsit of string allowing to get the roi repsonse series wanted

    Returns:

    """

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if len(keys) < 3:
        return None

    if keys[0] not in nwb_data.modules:
        return None

    if keys[1] not in nwb_data.modules[keys[0]].data_interfaces:
        return None

    if keys[2] not in nwb_data.modules[keys[0]].data_interfaces[keys[1]].roi_response_series:
        return None

    return np.transpose(nwb_data.modules[keys[0]].data_interfaces[keys[1]].roi_response_series[keys[2]].data[:])


def get_widefield_timestamps(nwb_file, keys):
    """

    Args:
        keys: lsit of string allowing to get the roi repsonse series wanted

    Returns:

    """

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if len(keys) < 2:
        return None

    if keys[0] not in nwb_data.modules:
        return None

    if keys[1] not in nwb_data.modules[keys[0]].data_interfaces:
        return None

    return np.array(nwb_data.modules[keys[0]].data_interfaces[keys[1]].timestamps)


def get_widefield_raw_acquisition_path(nwb_file, acquisition_name):
    """

    Args:
        acquisition_name:

    Returns:

    """

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if acquisition_name not in nwb_data.acquisition:
        return None

    wf_one_photon_series = nwb_data.acquisition[acquisition_name]

    if wf_one_photon_series.external_file is None:
        return None

    return wf_one_photon_series.external_file[:]


def has_trial_table(nwb_file):

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    nwb_objects = nwb_data.objects
    objects_list = [data for _, data in nwb_objects.items()]
    data_to_take = None
    for obj in objects_list:
        if 'trial' in obj.name:
            data = obj
            if isinstance(data, TimeSeries):
                continue
            else:
                data_to_take = data
                break
        else:
            continue
    if data_to_take is None:
        return False
    else:
        return True


def get_trial_names_from_table(nwb_file):

    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    if nwb_data.self.has_trial_table() is False:
        return None
    else:
        # Get the trial table object
        nwb_objects = nwb_data.objects
        objects_list = [data for key, data in nwb_objects.items()]
        data_to_take = None
        trial_dicts = []
        for obj in objects_list:
            if 'trial' in obj.name:
                data = obj
                if isinstance(data, TimeSeries):
                    continue
                else:
                    data_to_take = data
                    break
            else:
                continue

        column_names = data_to_take.colnames
        variable_columns = []
        variable_columns_values = []
        for col_index, col_name in enumerate(column_names):
            if ("time" in col_name) or col_name in ['perf', 'trial_type']:
                continue
            col_val = data_to_take.get(col_name)
            col_val_data = col_val.data[:]
            if isinstance(col_val_data[0], str):
                unique_col_val_data = np.unique(col_val_data)
            else:
                unique_col_val_data = np.unique(col_val_data[~np.isnan(col_val_data)])
            is_variable = len(unique_col_val_data) > 1
            is_index = len(unique_col_val_data) == len(col_val.data[:])
            is_times_series = 'timeseries' in col_name
            unique_col_val_data = list(unique_col_val_data)
            if is_variable and (not is_index) and (not is_times_series):
                variable_columns.append(col_name)
                variable_columns_values.append(unique_col_val_data)
            trial_dict = dict(zip(variable_columns, variable_columns_values))
            trial_dicts.append(trial_dict)
        return trial_dicts[-1]


def get_trial_timestamps_from_table(nwb_file, requirements_dict):

    if not has_trial_table(nwb_file):
        return None
    else:
        io = NWBHDF5IO(path=nwb_file, mode='r')
        nwb_data = io.read()
        # Get the trial table object
        nwb_objects = nwb_data.objects
        objects_list = [data for key, data in nwb_objects.items()]
        data_to_take = None
        for ind, obj in enumerate(objects_list):
            if 'trial' in obj.name:
                data = obj
                if isinstance(data, TimeSeries):
                    continue
                else:
                    data_to_take = data
                    break
            else:
                continue
        trial_data_frame = data_to_take.to_dataframe()
        trial_data_frame = trial_data_frame.reset_index()  # Get trial_id as column.
        for column_name, column_requirements in requirements_dict.items():
            column_values_type = type(trial_data_frame[column_name].values[0])
            column_requirements = [column_values_type(requirement) for requirement in column_requirements]
            trial_data_frame = trial_data_frame.loc[trial_data_frame[column_name].isin(column_requirements)]
        if trial_data_frame.empty:
            print("No trial meets the selection criteria")
            return None
        n_event = len(trial_data_frame.index)
        event_timestamps = np.zeros((2, n_event))
        event_timestamps[0, :] = np.array(trial_data_frame.get('start_time').values[:])
        event_timestamps[1, :] = np.array(trial_data_frame.get('stop_time').values[:])

        # Try to see if it is really timestamps and not frames
        epoch_start_timestamps_frac = [value % 1 for value in event_timestamps[0, :]]
        is_all_zero_starts = all(number == 0 for number in epoch_start_timestamps_frac)
        epoch_stop_timestamps_frac = [value % 1 for value in event_timestamps[1, :]]
        is_all_zero_stops = all(number == 0 for number in epoch_stop_timestamps_frac)

        if is_all_zero_starts and is_all_zero_stops:
            print("Seems like 'start_time' and 'stop_time' in trial table are given in frames instead of seconds")
            time_unit = "frames"
        else:
            time_unit = "seconds"

        return event_timestamps, time_unit


def get_cell_indices_by_cell_type(nwb_file, keys):
    """
    Return a dict with key the cell_type name and value an array of int representing the cell indices of this type
    Args:
        roi_serie_keys:

    Returns:

    """

    rrs = get_roi_response_serie(nwb_file, keys)
    if rrs is None:
        return {}

    # rrs.control is an array (uint8) as long as n_cells, with a code for each cell type
    # rrs.control_description is the list of cell type names, as long as n_cells
    if rrs.control_description is not None:
        cell_type_names = list(set(rrs.control_description))
    else:
        cell_type_names = []
    code_by_cell_type = dict()
    for cell_type_name in cell_type_names:
        index = list(rrs.control_description).index(cell_type_name)
        code_by_cell_type[cell_type_name] = rrs.control[index]

    cell_type_names.sort()

    cell_indices_by_cell_type = dict()
    for cell_type_name in cell_type_names:
        code = code_by_cell_type[cell_type_name]
        cell_indices_by_cell_type[cell_type_name] = np.where(np.array(rrs.control) == code)[0]

    return cell_indices_by_cell_type

