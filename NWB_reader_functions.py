from pynwb import NWBHDF5IO
from pynwb.base import TimeSeries
import ast
import numpy as np

"""
This file define NWB reader functions (inspired from CICADA NWB_wrappers).
The goal is that a function is used to extract one specific element from a NWB file to pass it to any analysis
"""


def get_mouse_id(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    mouse_id = nwb_data.subject.subject_id

    return mouse_id


def get_session_id(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    session_id = nwb_data.session_id

    return session_id

def get_nwb_file_metadata(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    session_metadata = nwb_data.subject

    return session_metadata

def get_session_metadata(nwb_file):
    """Get session-level metadata.
     Converts string of dictionary into a dictionary."""
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    session_metadata = ast.literal_eval(nwb_data.experiment_description)

    return session_metadata




def get_bhv_type_and_training_day_index(nwb_file):
    """
    This function extracts the behavior type and training day index, relative to whisker training start, from a NWB file.
    :param nwb_file:
    :return:
    """
    io = NWBHDF5IO(nwb_file, 'r')
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
        day = int(description[3])
    elif description[1] == 'off':
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
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    nwb_objects = nwb_data.objects
    objects_list = [data for key, data in nwb_objects.items()]
    data_to_take = None

    # Iterate over NWB objects but keep "trial"
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
    return trial_data_frame

def get_behavioral_events(nwb_file):
    """
    This function extracts the behavioral events from a NWB file.
    :param nwb_file:
    :return:
    """

    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    event_keys = nwb_data.processing['behavior']['BehavioralEvents'].time_series.keys()
    beh_event_dict = {}
    for key in event_keys:
        event_ts = nwb_data.processing['behavior']['BehavioralEvents'].time_series[key].timestamps
        beh_event_dict[key] = np.array(event_ts)

    return beh_event_dict



